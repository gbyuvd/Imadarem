from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

@dataclass
class ImplicitRefinementConfig:
    vocab_size: int = 100
    hidden_size: int = 128
    num_layers: int = 3
    num_heads: int = 4
    max_seq_len: int = 20
    max_refinement_steps: int = 6
    dropout: float = 0.1
    use_self_cond: bool = True
    stop_threshold: float = 0.02
    min_refine_uncertainty: float = 0.1
    ema_decay: float = 0.995
    diversity_weight: float = 0.05
    sampling_temperature: float = 1.2


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, L, device):
        return self.pe[:L].unsqueeze(0).to(device)


class AdaptivePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.base_pe = SinusoidalPositionalEmbedding(dim, max_seq_len)
        self.decay_logit = nn.Parameter(torch.zeros(max_seq_len))

    def forward(self, x):
        B, L = x.shape[:2]
        pe = self.base_pe(L, x.device)
        decay = torch.sigmoid(self.decay_logit[:L])
        return pe * decay.unsqueeze(-1)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, 
            batch_first=True, dropout=config.dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.hidden_size, config.hidden_size)
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ImplicitRefinementModel(nn.Module):
    def __init__(self, config: ImplicitRefinementConfig, tokenizer=None):
        super().__init__()
        self.config = config

        # Tokenizer-aware special tokens
        if tokenizer is not None:
            self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
            self.mask_token_id = getattr(tokenizer, "mask_token_id", None)
            # Try eos_token_id, then sep_token_id
            self.eos_token_id = getattr(tokenizer, "eos_token_id", 
                                       getattr(tokenizer, "sep_token_id", None))
            
            if self.mask_token_id is None:
                raise ValueError("Tokenizer must define mask_token_id")
            if self.pad_token_id is None:
                print("[Warning] pad_token_id not set — using 0")
                self.pad_token_id = 0
            if self.eos_token_id is None:
                print("[Warning] No eos_token_id or sep_token_id — will use implicit stopping")
        else:
            # Fallback for testing
            self.pad_token_id = 0
            self.mask_token_id = config.vocab_size - 1
            self.eos_token_id = config.vocab_size - 2

        # Validate token IDs don't collide
        special_tokens = [self.pad_token_id, self.mask_token_id]
        if self.eos_token_id is not None:
            special_tokens.append(self.eos_token_id)
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError(f"Special token collision: pad={self.pad_token_id}, "
                           f"mask={self.mask_token_id}, eos={self.eos_token_id}")

        # Model components
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = AdaptivePositionalEmbedding(config.hidden_size, config.max_seq_len)
        self.time_emb = nn.Linear(1, config.hidden_size)

        if config.use_self_cond:
            self.self_cond_proj = nn.Linear(config.vocab_size, config.hidden_size)
        else:
            self.self_cond_proj = None

        self.transformer = nn.Sequential(*[
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size)

        # EMA teacher (not a submodule)
        self.teacher = None
        self.ema_decay = config.ema_decay
        self.register_buffer("ema_step", torch.tensor(0.0))

    def init_teacher(self):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self)
            self.teacher.teacher = None  # prevent recursion
            for p in self.teacher.parameters():
                p.requires_grad = False

    def update_teacher(self):
        if self.teacher is None:
            self.init_teacher()
            return

        self.ema_step += 1
        decay = min(self.ema_decay, (self.ema_step - 1) / (self.ema_step + 1))
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.parameters()):
                t_param.data.mul_(decay).add_(s_param.data, alpha=1 - decay)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_self_cond: Optional[torch.Tensor] = None):
        B, L = x_t.shape
        x = self.token_emb(x_t)
        x = x + self.pos_emb(x_t)
        time_fea = self.time_emb(t.float().unsqueeze(1)).unsqueeze(1)
        x = x + time_fea

        if x_self_cond is not None and self.self_cond_proj is not None:
            x = x + self.self_cond_proj(x_self_cond)

        x = self.transformer(x)
        return self.to_logits(x)

    def _get_uncertainty_from_teacher(self, x_t, t, x_self_cond):
        model = self.teacher if self.teacher is not None else self
        with torch.no_grad():
            logits = model(x_t, t, x_self_cond)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return entropy / math.log(self.config.vocab_size)

    @torch.no_grad()
    def sample(self, batch_size: int, max_len: int, device: torch.device) -> List[List[int]]:
        x_t = torch.full((batch_size, max_len), self.mask_token_id, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        x_prev = x_t.clone()
        x_self_cond = None

        for step in range(self.config.max_refinement_steps):
            t = torch.full((batch_size,), step, dtype=torch.float, device=device)
            logits = self(x_t, t, x_self_cond)
            uncertainty = self._get_uncertainty_from_teacher(x_t, t, x_self_cond)

            probs = F.softmax(logits / self.config.sampling_temperature, dim=-1)
            pred_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(batch_size, max_len)

            # Determine which tokens to refine
            needs_refine = (uncertainty > self.config.min_refine_uncertainty)

            # Handle EOS: freeze positions at/after first EOS
            if self.eos_token_id is not None:
                eos_mask = (x_t == self.eos_token_id)
                for b in range(batch_size):
                    if finished[b]:
                        needs_refine[b] = False
                        continue
                    eos_pos = eos_mask[b].nonzero(as_tuple=True)[0]
                    if eos_pos.numel() > 0:
                        first_eos = eos_pos.min().item()
                        needs_refine[b, first_eos:] = False
                        finished[b] = True

            x_t = torch.where(needs_refine, pred_tokens, x_t)

            # Early stopping
            changed = (x_t != x_prev)
            if self.eos_token_id is not None:
                active = ~finished.unsqueeze(1).expand(-1, max_len)
                change_ratio = (changed & active).float().sum() / (active.float().sum() + 1e-8)
            else:
                change_ratio = changed.float().mean()

            if change_ratio < self.config.stop_threshold or finished.all():
                print(f"✅ Stopped at step {step+1} (change: {change_ratio:.2%})")
                break

            x_prev = x_t.clone()
            if self.config.use_self_cond:
                x_self_cond = F.softmax(logits, dim=-1).detach()

        # Post-process: trim at EOS
        outputs = []
        for b in range(batch_size):
            seq = x_t[b].cpu().tolist()
            if self.eos_token_id is not None:
                try:
                    eos_idx = seq.index(self.eos_token_id)
                    seq = seq[:eos_idx + 1]  # include EOS
                except ValueError:
                    pass  # no EOS found
            outputs.append(seq)
        return outputs

    def loss(self, x_0: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        B, L = x_0.shape
        device = x_0.device

        # Create padding mask
        if lengths is not None:
            positions = torch.arange(L, device=device).unsqueeze(0)
            padding_mask = positions < lengths.unsqueeze(1)
        else:
            padding_mask = (x_0 != self.pad_token_id)

        t = torch.randint(0, self.config.max_refinement_steps, (B,), device=device)
        mask_rate = 1.0 - (t.float() / self.config.max_refinement_steps)
        rand_mask = torch.rand(B, L, device=device) < mask_rate.unsqueeze(1)
        mask = rand_mask & padding_mask
        x_t = torch.where(mask, self.mask_token_id, x_0)

        # Self-conditioning
        x_self_cond = None
        if self.config.use_self_cond:
            with torch.no_grad():
                t_prev = torch.clamp(t + 1, max=self.config.max_refinement_steps - 1)
                rand_mask_prev = torch.rand(B, L, device=device) < (1.0 - (t_prev.float() / self.config.max_refinement_steps)).unsqueeze(1)
                mask_prev = rand_mask_prev & padding_mask
                x_t_prev = torch.where(mask_prev, self.mask_token_id, x_0)
                logits_init = self(x_t_prev, t_prev.float())
                x_self_cond = F.softmax(logits_init / 1.5, dim=-1).detach()

        logits = self(x_t, t.float(), x_self_cond)
        recon_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x_0.view(-1), reduction='none')
        recon_loss = recon_loss.view(B, L)
        recon_loss = (recon_loss * mask.float()).sum() / (mask.float().sum() + 1e-8)

        # Diversity loss
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        diversity_loss = torch.relu(0.5 - entropy) * mask.float()
        diversity_loss = diversity_loss.sum() / (mask.float().sum() + 1e-8)

        total_loss = recon_loss + self.config.diversity_weight * diversity_loss
        return {"total": total_loss, "recon": recon_loss, "diversity": diversity_loss}

from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gbyuvd/bionat-selfies-gen-tokenizer-wordlevel")

    config = ImplicitRefinementConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_layers=2,
        max_seq_len=512,
        max_refinement_steps=5,
        stop_threshold=0.03,
        diversity_weight=0.1,
        sampling_temperature=1.3
    )

    model = ImplicitRefinementModel(config, tokenizer=tokenizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Prepare data with EOS
    texts = ["Hello, world!", "Adaptive refinement is working.", "This is a test."]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    x = inputs["input_ids"]  # already includes EOS at end of each sequence
    attention_mask = inputs["attention_mask"]
    lengths = attention_mask.sum(dim=1)  # includes EOS

    print("Training V+ with real tokenizer...")
    for step in range(20):
        optimizer.zero_grad()
        losses = model.loss(x, lengths=lengths)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_teacher()

        if step % 5 == 0:
            print(f"Step {step}: total={losses['total']:.4f}")

    print("\nSampling (no target_lengths!)...")
    samples = model.sample(batch_size=2, max_len=20, device='cpu')
    for i, seq in enumerate(samples):
        decoded = tokenizer.decode(seq, skip_special_tokens=False)
        print(f"Sample {i+1}: {repr(decoded)}")