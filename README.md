# Imadarem
Implicitly Adaptive Refinement Model

> A lightweight, iterative-refinement language model that learns to “fill-in-the-blanks” starting from a fully-masked sequence.  
> Works with any tokenizer that supplies `mask_token_id`, `pad_token_id` and (optionally) `eos_token_id`.

---

## 1. High-level idea
Instead of left-to-right generation, the model treats text generation as a **denoising** process:

1. Start with every token = `[MASK]`
2. Run a small, shared transformer for ≤ K steps
3. At each step, re-predict only the positions whose uncertainty (entropy) is still high
4. Freeze tokens once an `[EOS]` is sampled; stop early when < τ tokens change

The training objective is a **masked-language-modeling** loss whose corruption schedule is **time-dependent**:  
mask-rate(t) = 1 − t / K.

```text
🔍 Refinement Trajectory (max_steps=10)

t=0: [MASK] [MASK] [MASK] [MASK] [MASK]
        ↑     ↑      ↑      ↑      ↑  ← High entropy everywhere (max uncertainty)
t=1: [[12]] [ [5]] [ [5]] [ [5]] [[10]]
        ↑      ↑      ↑      ↑      ↑  ← High uncertainty at pos 0, 1, 2, 3, 4
t=2: [ [7]] [ [5]] [ [5]] [ [5]] [ EOS]
        ↑      ↑      ↑      ↑      ↑  ← High uncertainty at pos 0, 1, 2, 3, 4
t=3: [ [7]] [ [5]] [[18]] [ [9]] [ EOS]
        ↑      ↑      ↑                ← High uncertainty at pos 1, 2, 3
                                       ← change_ratio = 0.0% < 2% → ✅ Early stopping triggered

Final output: '[Ring1] [C] [S] [O] </s>'
```

Note, for NLP task you might need more refinement steps:
```text
🔍 Refinement Trajectory (max_steps=10)

t=0: [MASK] [MASK] [MASK] [MASK] [MASK]
t=1: [[2014]] [[2016]] [[1012]] [[2014]] [[1996]]
t=2: [[5953]] [[9732]] [[9325]] [[29532]] [[2059]]
t=3: [[28107]] [[6957]] [[29263]] [[2242]] [[1998]]
t=4: [[12422]] [[9519]] [[20008]] [[2481]] [[13263]]
t=5: [[2096]] [[14265]] [[9668]] [[11757]] [[15375]]
t=6: [[3547]] [[30363]] [[23634]] [[7887]] [[27003]]
t=7: [[24989]] [[27104]] [[2588]] [[2018]] [[29863]]
t=8: [[19563]] [[1653]] [[28125]] [[30112]] [[20188]]
t=9: [[6267]] [[2077]] [[2132]] [[2001]] [[27228]]
t=10: [[2008]] [[2492]] [[2045]] [[2056]] [[3566]]
                                                 → ⏹️  Max steps reached
Final output: 'that field there said mom'
```

---

## 2. Architecture snapshot
| Component | Purpose | Key hyper-params |
|-----------|---------|------------------|
| `TokenEmbedding` | learned | `vocab_size`, `hidden_size` |
| `AdaptivePositionalEmbedding` | sinusoidal PE × learned decay | `max_seq_len` |
| `TimeEmbedding` | 1-step MLP | `hidden_size` |
| `Self-condition projection` | soft prev-logits → residual | optional |
| `Transformer blocks` | full self-attention | `num_layers`, `num_heads`, `dropout` |
| `Teacher (EMA)` | exponential moving average | `ema_decay` |

## 4. Sampling algorithm
| Hyper-param | Meaning | Default |
|-------------|---------|---------|
| `max_refinement_steps` | hard cap | 6 |
| `sampling_temperature` | softmax T | 1.2 |
| `min_refine_uncertainty` | entropy threshold | 0.1 |
| `stop_threshold` | % changed tokens | 0.02 |

## 5. Tokenizer contract
Required special IDs (auto-detected):
```
tokenizer.mask_token_id   # must exist
tokenizer.pad_token_id    # fallback 0
tokenizer.eos_token_id    # fallback sep_token_id, else None
```
Collision check is performed at init.


## 6. Typical config (quick start)
```python
ImplicitRefinementConfig(
    vocab_size=100,
    hidden_size=128,
    num_layers=3,
    num_heads=4,
    max_seq_len=20,
    max_refinement_steps=6,
    dropout=0.1,
    use_self_cond=True,
    stop_threshold=0.02,
    min_refine_uncertainty=0.1,
    ema_decay=0.995,
    diversity_weight=0.05,
    sampling_temperature=1.2
)
model = ImplicitRefinementModel(config, tokenizer=tokenizer)
```

## 7. Strengths & limitations
✅ **Pros**  
- Non-autoregressive → parallel sampling  
- Early-stopping gives variable-length outputs  
- EMA teacher stabilizes uncertainty estimation  
- Works with any sub-word or character tokenizer  

❌ **Cons**  
- Length capped by `max_seq_len`  
- No explicit coverage mechanism for long inputs  


## 8. Citations
- Ranger21 Optimizer:
```bibtex
@article{wright2021ranger21,
      title={Ranger21: a synergistic deep learning optimizer}, 
      author={Wright, Less and Demeure, Nestor},
      year={2021},
      journal={arXiv preprint arXiv:2106.13731},
}
```
