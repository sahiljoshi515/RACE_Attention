# Crosswalk: paper construct → code

Maps each piece of the RACE Attention method to where it lives in the implementation. Line numbers
verified against HEAD c620cdc; paper anchors are `arXiv-2510.04008v5/main.tex`.

| Paper construct | Paper ref | Code location |
| --- | --- | --- |
| Soft assignment $\phi(x)_r$ (Alg 1 step 3): $\tanh(Wx)$ aligned to corners, $\mathrm{softmax}$ over $R=2^P$ | L1203-1206 | `scaling/race_common.py:soft_hash_probs` (L36); planes/corners from `build_planes_protos` (L14) |
| Random hyperplanes $W^{(\ell)} \sim N(0,I)$, corner set $\{\pm 1\}^P$ | Alg 1 steps 1-2, L1199-1200 | `scaling/race_common.py:build_planes_protos` (L14) |
| Bucket stats $A^{(\ell)}$ (mass), $B^{(\ell)}$ (value-sum) | Alg 1 step 5, L1207-1212 | accumulated inside the prefix scan: `kernels/gpu/forward_kernel.cu` (`cA`,`cB`), `kernels/cpu/race_pref.cpp` |
| Non-causal output $\widehat{O} = \operatorname{diag}(\mathrm{Den})^{-1} \mathrm{Num}$ | Alg 1 steps 7-8, L1215-1217 | `scaling/race_common.py:race_prefix_ref` (L76, reference); `misc/race.py:BatchedACE` (L26) |
| Causal running prefix (Alg 2): `A_cum`,`B_cum` updated left-to-right | Alg 2, L2625-2670 | **forward** `kernels/gpu/forward_kernel.cu:race_fused_fwd` (3-phase scan: `racefwd_phase1/2/3` L26/50/75); CPU `kernels/cpu/race_pref.cpp:race_prefix_mean_flat` (L36) |
| Causal backward (gradProbsK/Q, gradV) | (impl of Alg 2) | `kernels/gpu/backward_kernels.cu:race_backward`; CPU `race_prefix_mean_flat_bw` (L104) |
| Autograd wrapper around kernels | - | `scaling/race_causal_cuda.py:RaceCausalFn` (L47), `RaceCausalCuda` (L76, `forward` L94) |
| Drop-in multi-head module | "drop-in replacement", L1151 | `misc/race.py:RACEAttention` (L144), `RACEBlock` (L176) |
| Params P, L, $\beta$ | Alg 1 inputs, L1193-1194 | `K` (=P), `L`, `M` ensembles in `RACEAttention`/`RaceCausalCuda`; $\beta$ = trainable temperature; $S = L\cdot 2^K$ |
| Angular kernel $(1 - \cos^{-1}(\cdot)/\pi)^\gamma$, $\gamma = 8$ | Eq exp-angular-sim, L1252-1256 | `scaling/benchmark_time.py:angular_attention` (L52, exponent=8.0) |
| Complexity $\mathcal{O}(LNRd)$ time / $\mathcal{O}(L(NR+Rd))$ space | L1295 | realized by bucket-summary compression (never builds $N\times N$); see `codebase/scaling-module` |
| Baselines (softmax/flash/linformer/performer/sigmoid/yoso) | §Experiments | `scaling/benchmark_time.py` (`softmax_attention` L14, `flash_attention` L24, `linformer_attention` L62, ...) |
| Pure-PyTorch reference (ground truth) | Alg 1/2 semantics | `scaling/race_common.py:race_prefix_ref` (L76), `scaling/race_torch_cumsum.py:RaceCumsumCausal` (L14) |
| Correctness test (kernel vs fp64 ref) | - | `scaling/test_kernels.py` |

## Notation bridge (paper ↔ code)

`P` (hyperplanes) = `K`/`Kbits` in code; $R = 2^P$ corners; `L` tables; `M` ensembles (code-only,
not in the paper algorithm box); $S = L\cdot R$ = bucket summaries per token; $\beta$ = trainable softmax
temperature in `soft_hash_probs`.

---
Source: arXiv-2510.04008v5/main.tex (anchors above) + live code at HEAD c620cdc (symbols grepped and verified). See `codebase/*` for narrative.
