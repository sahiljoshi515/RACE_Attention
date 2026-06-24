# Tests & benchmarks (`scaling/`)

## Correctness - `scaling/test_kernels.py`

Compares the CUDA kernels against the fp64 PyTorch `cumsum` reference (`race_prefix_ref` /
`RaceCumsumCausal`). Covers: forward across chunk sizes (1024, 4096, 8192); backward gradients
(`gradProbsK`, `gradProbsQ`, `gradV`); determinism; and end-to-end module gradients. Shapes swept
include $N \in \{1,2,4\}$, T up to 65536, $D \in \{16,64,96,128,130,256\}$. Tolerances ~`2e-5` (forward),
~3-5e-3 (gradients). Run: `python scaling/test_kernels.py` (needs a CUDA GPU; exit 0 = pass).

This is the regression guard for any kernel change - the causal backward bug fixed in HEAD c620cdc
is exactly what this catches. **For the vLLM backend, this is the harness to extend** with a
parity test against `RaceCausalCuda`.

## Performance - `scaling/benchmark_time.py`

Measures wall-clock time + memory for a single forward-backward pass vs sequence length, for RACE
(CUDA) against baselines. It defines its own attention implementations for apples-to-apples timing:
`softmax_attention` (L14), `flash_attention` (L24), `angular_attention` (L52, exponent default 8.0),
`linformer_attention` (L62), plus a local `BatchedACE` (L77) and `race_attention(...)` (L168).
Supports causal and non-causal modes and can emit the scaling plots in `paper/figures`. Run:
`python scaling/benchmark_time.py [options]`.

Note: `benchmark_time.py` carries a **second** `BatchedACE` distinct from `misc/race.py`'s - keep
them straight when reading.

---
Source: scaling/test_kernels.py; scaling/benchmark_time.py (softmax L14, flash L24, angular L52, linformer L62, BatchedACE L77, race_attention L168). Verified against HEAD c620cdc.
