# Fused single-token decode CUDA kernel for RACE Attention

_Collapses each RACE layer's ~9 per-token decode ops into one launch; correct (token-identical)
and 1.2–1.25× faster decode at every context length. H100 NVL, ARRR S=8._

## Problem

Hybrid decode (the `T==1` branch in `race_llama_attention.py`) is O(1) in context but
**launch-bound**: per RACE layer it fired `_soft_hash` (matmul→tanh→/scale→matmul→softmax ≈5
launches) + the state update (`A+=pk`, `B+=pk⊗v`, divide, einsum ≈4 launches). ×21 RACE layers
≈ ~200 tiny launches/token → ~5 ms overhead → hybrid was **0.73× the teacher** below ~128K
(see `REPORT_decode_cache.md`). Decode is inference-only, so no backward is needed.

## Change

A **standalone** CUDA extension `race_decode` (kept fully separate from the training
`race_cuda` fwd/bwd kernels):

- `kernels/gpu/decode_kernel.cu` — one fused kernel, **grid = N blocks, block = head_dim**.
  Per row it does soft-hash of q & k (block-reduced projections → per-L-group max-subtracted
  softmax) + in-place advance of the running state `A [N,S]`, `B [N,S,hd]` + readout
  `out = Σ_s pq[s]·B[s,d]/(A[s]+eps)`. One token ⇒ each `(s,d)` cell is touched by exactly
  one thread ⇒ **atomic-free**; `A[s]` is added once per `s` (the one race hazard, fixed).
- `kernels/gpu/race_decode_binding.cpp`, `kernels/gpu/race_decode_build.py` — own pybind
  module + JIT loader (`name="race_decode"`, sm_90, no `--use_fast_math`).
- `scaling/race_decode_cuda.py` — `race_decode_step(q,k,v,planes_T,protos_T,A,B,out,scale,eps,L,Kbits)`
  wrapper (no autograd; decode is `torch.no_grad`). `out` preallocated; `A,B` mutated in place.
- `race_llama_attention.py` — gated fast-path in the `T==1` branch behind `_use_decode_kernel`
  (default on when the ext imports; auto-disables once on any failure → torch fallback). The
  q/k/v/o projections and RoPE stay in torch. `scale = sqrt(hd)·exp(log_temp)` is **cached
  once** (`_dec_scale`) — `log_temp` is frozen at inference, so we avoid a `.item()` D2H sync
  per layer per token (those syncs erased the win at long context; see below). Prefill capture
  and the layer-0-must-be-softmax constraint are unchanged.

## Verification (`run_decode_kernel.sbatch`, H100 NVL, race_vit_env / torch 2.10)

1. **Unit step** (`scaling/correctness_decode_step.py`) — kernel vs torch `_soft_hash`+einsum
   over PARAM_SETS {(2,2),(3,3),(4,4)}, log_temp ∈ {0, 0.3}: **out relerr ≤ 3.4e-7, A/B ≤ 1e-7.**
2. **Sequence** (64 incremental steps) vs cumsum `race_prefix_ref` AND chunked-scan
   `RaceCausalFn`: **relerr ≤ 3.4e-7.** All checks pass.
3. **Token-identity** (`validate_decode_cache.py`, RULER niah, 8K): `recompute vs cache(kernel)`
   **TOKEN-IDENTICAL** (greedy).

## Results — decode ms/tok (steady-state, warmup-excluded; `synthetic_decode_probe.py`)

| ctx | teacher | torch decode | **kernel** | kernel vs teacher |
|---|:--:|:--:|:--:|:--:|
| 8K | 11.7 | 15.82 | **12.72** | 0.73× → **0.92×** |
| 32K | 11.7 | 15.82 | **12.67** | 0.73× → **0.92×** |
| 131K | 20.9 | 15.84 | **12.65** | 1.31× → **1.65×** |
| 262K | 36.3 | 15.75 | **14.20** | 2.30× → **2.55×** |

- Kernel decode is **~12.65 ms/tok flat** 8K→131K (1.25× faster than the torch path) and beats
  it at 262K too (14.20 vs 15.75). The slight rise at 262K is the kept-softmax KV read growing
  (fundamental to the hybrid), not the kernel.
- **The per-token D2H sync mattered.** First run (scale via `.item()` every layer/token) was
  faster only at short ctx and *regressed* at 262K (18.50 ms/tok) — the 21 syncs/token
  serialized against the growing softmax work. Caching `scale` removed the regression.

## Limits / follow-ups

- Win is largest where decode was launch-bound (short/mid ctx). Below ~128K the model is still
  weight-bandwidth-bound, so 0.92× (not >1×) vs teacher — but the hybrid runs where the teacher
  OOMs and is 2.55× faster at 262K.
- Occupancy: N≈24 blocks underfill the GPU but the kernel is launch-bound; B>1 self-mitigates.
- Possible next steps: CUDA-graph capture of the whole decode step (removes remaining launch
  overhead incl. projections); fuse the redundant torch `_soft_hash` skip already done on the
  kernel path; fuse all RACE layers' decode into one launch.
