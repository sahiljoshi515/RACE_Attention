# Hybrid decode: incremental KV cache (softmax layers) + RACE running state

_Makes hybrid decode incremental so it avoids recomputation; verified correct three ways._

## Problem

The original eval decode ran the hybrid with `use_cache=False` and re-ran the **full
sequence every token** (RACE had no cache, so the whole model was forced to recompute).
Decode cost ≈ one prefill per token (614 ms/tok @32K, 1470 ms/tok @64K) → 1.6 / 0.68 tok/s.

## Change

RACE's causal scan reduces, at each step, to two fixed-size running prefix sums per layer:
`A = Σ_t probsK_t  [N,S]` and `B = Σ_t probsK_t⊗v_t  [N,S,hd]`, with
`out_t = probsQ_t · (B/(A+eps))`. So:
- **softmax (kept) layers** use a normal HF KV cache;
- **RACE layers** carry `(A,B)` state — prefill captures `A_final/B_final`, each decode
  step advances them by the new token. `O(1)` in context length, pure torch (no kernel change).

Implemented in `race_llama_attention.py` (`enable_decode_cache`, `reset_decode_state`, the
`T==1` decode branch + prefill capture; batched Q/K soft-hash to cut launch count) and wired
in `eval_ruler.py` (`--decode-mode cache|recompute`, default `cache`). Training is untouched:
state capture is gated `_capture_decode_state` (default off, only the eval harness enables it);
`_dec_A/_dec_B` are plain attributes (never in `state_dict`).

Design constraint: cache decode requires **layer 0 to be softmax** (HF reads sequence length
+ causal-mask size from cache slot 0; RACE layers don't write the cache). ARRR/AR keep layer 0
softmax; `build_model` asserts this.

## Verification (3 independent ways)

1. **Math** (agent + standalone numeric test): the incremental recurrence and prefill capture
   reproduce `race_common.race_prefix_ref` exactly — 0.0 max-abs-diff in fp64.
2. **Training-safety / numerics** (agent): training forward byte-identical (gated off);
   fp32 accumulation; new attributes never enter `state_dict`.
3. **Empirical** (`validate_decode_cache.py`, on GPU): cache vs recompute produce
   **token-identical** greedy output through the real model + HF KV cache.

## Results (H200, ARRR S=8, `synthetic_decode_probe.py`)

Decode latency (steady-state, warmup-excluded) and peak memory vs context length:

| ctx | teacher ms/tok | hybrid ms/tok | hybrid/teacher | teacher mem | hybrid mem |
|---|:--:|:--:|:--:|:--:|:--:|
| 8K | 13.70 | 18.80 | 0.73× | 8.0 GB | 7.3 GB |
| 32K | 13.72 | 18.79 | 0.73× | 12.7 | 9.8 |
| 64K | 13.74 | 18.80 | 0.73× | 18.8 | 13.2 |
| 131K | 17.35 | 18.78 | 0.92× | 31.2 | 20.0 |
| 262K | 28.23 | **18.75** | **1.51×** | 56.0 | 33.4 |
| 524K | OOM | 18.71 | — | — | 60.4 |

- **Hybrid decode is flat (~18.7 ms/tok) from 8K→524K** — incremental, O(1) in T.
- **Teacher decode is flat to 64K then rises** (KV-read dominates), and OOMs by 524K.
- **Crossover ≈131K; hybrid is 1.5× faster at 262K** and runs where the teacher can't.
- **Hybrid memory is always lower, gap widening.**
- vs the old recompute path: cache decode is **6× faster at 8K** and flat instead of blowing up.

## Why not faster below ~128K

At ≤64K on a 3B model, decode is **weight-bandwidth-bound**, not attention-bound (the KV read
is ~0.4 ms of ~13.7 ms; the rest is streaming 6 GB of weights). Replacing attention saves
almost nothing there, while 21 RACE layers add ~5 ms of small-kernel launch overhead → 0.73×.
The win appears only once attention (teacher KV-read) grows to dominate, ~128K+. CUDA-graph /
`torch.compile` capture of the decode step would cut the launch overhead and lower the crossover,
but the regime where RACE's linear decode structurally wins is long context (consistent with the
kernel-level RACE-vs-FA3 crossover in `../progress.md`).
