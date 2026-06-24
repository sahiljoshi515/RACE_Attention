# GPU / CUDA kernels (`kernels/gpu/`)

The CUDA realization of causal RACE (Algorithm 2, `paper/09-causal-algorithm`): a chunked parallel
scan over the time axis that never materializes the full prefix tensor. Default chunk size 8192;
built for `sm_90` (Hopper / H200). This is the kernel that enables the 12M-token GPU scaling result.

## Forward - `forward_kernel.cu`

`race_fused_fwd(probsK, probsQ, V2, eps, chunk)` computes, causally:
$$
\begin{aligned}
A(t)[s] &= \sum_{\tau \le t} \mathrm{probsK}[\tau,s] \\
B(t)[s,d] &= \sum_{\tau \le t} \mathrm{probsK}[\tau,s]\cdot \mathrm{V2}[\tau,d] \\
\mathrm{out}[t,d] &= \sum_s \mathrm{probsQ}[t,s]\cdot \frac{B(t)[s,d]}{A(t)[s] + \varepsilon}
\end{aligned}
$$
Three-phase chunked scan:
- `racefwd_phase1` (L26) - per-chunk partial sums `cA`, `cB`.
- `racefwd_phase2` (L50) - exclusive prefix scan over chunk totals (in place).
- `racefwd_phase3` (L75) - per-chunk readout with atomicAdd into `out`.
Launches at L128/131/133 (`<<<g1,blk>>>`, `<<<g2,256>>>`, `<<<g3,blk>>>`).

## Backward - `backward_kernels.cu`

`race_backward(probsK, probsQ, V2, grad_out, eps, chunk)` $\to$ `{gradProbsK, gradProbsQ, gradV}`.
Two-pass: a forward scan rebuilds the A/B prefix states (never stored) and computes `gradProbsQ`
and `gradA`; a reverse/suffix scan produces `gradProbsK` and `gradV`. Reconstruction (rather than
caching A/B) keeps memory linear and gradients numerically exact. This is the file the
[backward-bug fix + kernel optimization](git log c620cdc) touched.

## Glue

- `race_cuda_binding.cpp` - pybind module: declares both functions (L12-13) and exports them
  (`m.def` at L18, L21) as `race_fused_fwd` and `race_backward`.
- `race_cuda_build.py` - `load_ext(verbose=True)` (L17) JIT-compiles the `.cu` + binding via
  `torch.utils.cpp_extension.load`, targeting `sm_90`, cached in `TORCH_EXTENSIONS_DIR`.

The autograd `Function` wrapping these is `RaceCausalFn` (`codebase/scaling-module`).

---
Source: kernels/gpu/forward_kernel.cu (phases L26/50/75, launches L128-133), backward_kernels.cu, race_cuda_binding.cpp (L12-13, L18, L21), race_cuda_build.py (L17). Verified against HEAD c620cdc.
