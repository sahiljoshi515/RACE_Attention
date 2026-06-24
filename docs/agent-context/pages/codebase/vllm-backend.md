# vLLM RACE Attention backend - status & integration notes

**Branch:** `feat/vllm-race-attention-backend`. **Status (HEAD c620cdc):** the branch has **no
unique commits over `main`** (`git log main..HEAD` is empty) - the backend is **not yet
implemented**. This page is the working scratchpad for that effort.

## What exists to build on

- `RaceCausalCuda.forward(Q, K, V)` with `[B, H, T, D]` (`scaling/race_causal_cuda.py:76`) - the
  natural attention seam to call from a vLLM backend.
- The chunked-scan CUDA kernels `race_fused_fwd` / `race_backward` (`codebase/gpu-kernels`),
  JIT-loaded via `kernels/gpu/race_cuda_build.py:load_ext()` (targets `sm_90`).
- Soft-hash front end `soft_hash_probs` + `build_planes_protos` (`scaling/race_common.py`).
- Correctness harness `scaling/test_kernels.py` (extend with a vLLM-path parity test).

## Open design questions (carry into the build)

These come straight from the paper's gaps and the review (`crosswalk/open-questions`,
`crosswalk/concerns-tracker`):

1. **Causal correctness has no proof.** Theorem 1 is non-causal only (`paper/06-theory`). The
   decode path is inherently causal $\to$ lean on `test_kernels.py` parity, not theory.
2. **Decode = single-token append.** Algorithm 2 is a left-to-right prefix scan; for autoregressive
   decode the per-table cumulative stats `A_cum`, `B_cum` are exactly a **KV-cache** ($R + R\cdot d$ per
   table per head), updated by one token at a time. This is the paper's hinted "efficient KV
   caching" direction (conclusion, `paper/08-scaling`). The current kernels are full-sequence scans,
   not incremental-append - a decode kernel is new work.
3. **FlashAttention parity / version.** Reviewer eQBU asked which FlashAttention version was
   benchmarked; for a vLLM backend, define the accuracy/throughput parity target explicitly.
4. **Framing.** The 12M/75M numbers are single-primitive limits, not model context (`paper/08-scaling`).
   A real model adds embeddings, FFN, multiple layers, and KV-cache memory.

## Suggested first steps (not yet done)

- Stand up a vLLM custom attention backend that calls `RaceCausalCuda` for prefill.
- Add an incremental-decode kernel (append one token to `A_cum`/`B_cum`) or document why prefill-only.
- Parity test vs `race_prefix_ref` and vs softmax on a small model.

---
Source: `git log main..HEAD` (empty) at HEAD c620cdc; integration seams verified in scaling/ and kernels/gpu/. Update this page as the branch advances (re-run /rdocs refresh for the inventory).
