# Scaling module (`scaling/`)

The standalone, kernel-backed RACE used for the extreme-length scaling study and correctness tests.
Cleaner than `misc/race.py` and the place to look when wiring RACE into a new backend.

## Files

- `scaling/race_common.py` - shared building blocks:
  - `build_planes_protos(d_k, Kbits, L, M, device="cuda", share_planes=True, ...)` (L14) - creates
    random hyperplanes `W` and the $R = 2^{\text{Kbits}}$ corner prototypes $v_r$.
  - `soft_hash_probs(Q, K, V, planes_T, protos_T, L, Kbits, M, share_planes=True)` (L36) - the
    **soft bucketization** of Algorithm 1 step 3: $\tanh(W\cdot x)$ aligned to corners, $\mathrm{softmax}$ over
    corners $\to$ `probsQ`, `probsK`. (Inner helper `packM` at L49 packs the M-ensemble layout.)
  - `race_prefix_ref(probsK, probsQ, V2, eps=1e-6)` (L76) - the **pure-PyTorch cumsum reference**
    (ground truth) the CUDA kernels are tested against.
- `scaling/race_causal_cuda.py` - CUDA-backed module:
  - `fwd_chunk(T)` (L33) - heuristic picking the kernel chunk size from sequence length T.
  - `RaceCausalFn(Function)` (L47) - autograd `Function`; `forward` (L51) calls `race_fused_fwd`,
    backward calls `race_backward` (`codebase/gpu-kernels`).
  - `RaceCausalCuda(nn.Module)` (L76) - full module: soft-hash (`race_common`) + CUDA scan.
    `__init__(d_k, Kbits, L, M, device="cuda", share_planes=True, eps=1e-6, seed=0)`;
    `forward(Q,K,V)` (L94) takes `[B,H,T,D]` $\to$ `[B,H,T,D]`.
- `scaling/race_torch_cumsum.py` - `RaceCumsumCausal` (L14): same API as `RaceCausalCuda` but pure
  `cumsum`; memory-inefficient (materializes `B_pref[N,T,S,D]`), OOMs on long T. Used as reference.

## Relationship

`RaceCausalCuda` = `soft_hash_probs` (math from `paper/03-algorithm-noncausal`) + `RaceCausalFn`
(kernels from `codebase/gpu-kernels`). Tests pin both against `race_prefix_ref` / `RaceCumsumCausal`
in fp64. See `codebase/tests-benchmarks`. For the vLLM backend, `RaceCausalCuda.forward(Q,K,V)` with
`[B,H,T,D]` is the natural integration seam (`codebase/vllm-backend`).

---
Source: scaling/race_common.py (L14, L36, L49, L76), race_causal_cuda.py (L33, L47, L51, L76, L94), race_torch_cumsum.py (L14). Verified against HEAD c620cdc.
