# CPU kernels (`kernels/cpu/`)

OpenMP C++ extensions implementing the causal RACE prefix-mean in double precision, loaded via
`torch.utils.cpp_extension`. This is the CPU realization of Algorithm 2 (`paper/09-causal-algorithm`).

## Files

- `kernels/cpu/race_pref.cpp` - the causal RACE prefix-mean kernel.
  - `race_prefix_mean_flat(probsK_flat, V_flat, eps)` (defined L36) - **forward**. Computes the
    causal prefix mean per independent stream ($NS = N\cdot S$ streams flattened):
$$
\begin{aligned}
A_t &= \sum_{\tau \le t} w_\tau \\
B_t &= \sum_{\tau \le t} w_\tau \cdot v_\tau \\
E_t &= \frac{B_t}{A_t + \varepsilon}
\end{aligned}
$$
    OpenMP-parallel over the NS dimension; double-precision accumulation for numerical stability.
  - `race_prefix_mean_flat_bw(...)` (defined L104) - **backward**.
  - pybind exports at L224-225: `race_prefix_mean_flat`, `race_prefix_mean_flat_bw`.
- `kernels/cpu/linear_pref.cpp` - Linear-Attention prefix baseline (for benchmark comparison).
- `kernels/cpu/race_ext.py` - JIT loader (`torch.utils.cpp_extension.load`); handles OpenMP
  linking on macOS (homebrew libomp) and Linux.
- `kernels/cpu/setup.py` - legacy build config; the JIT `load_ext` path is preferred.

## How it's called

The flat layout (`probsK[NS,T]`, `V[NS,T,D]`) means the Python side reshapes `[N, T, S]` bucket
probabilities + `[N, T, D]` values into independent `(stream, time)` sequences before the call.
The math matches the running-prefix form of Algorithm 2; the GPU equivalent is the chunked scan in
`codebase/gpu-kernels`. The pure-PyTorch ground truth for both is `race_prefix_ref`
(`codebase/scaling-module`).

---
Source: kernels/cpu/race_pref.cpp (L36, L104, L224-225), kernels/cpu/race_ext.py, verified against HEAD c620cdc.
