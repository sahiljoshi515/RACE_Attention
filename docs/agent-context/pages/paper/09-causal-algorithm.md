# Algorithm 2 - RACE Attention (causal)

The causal variant turns the per-table bucket statistics into **running prefix sums** so token t
attends only to tokens $\le t$. Same inputs/outputs as Algorithm 1.

For each table $\ell = 1..L$:
- Draw $W^{(\ell)} \in \mathbb{R}^{P\times d}$; corner set $V = \{\pm 1\}^P$ ($R = 2^P$).
- Build $\Phi_Q^{(\ell)}, \Phi_K^{(\ell)} \in \mathbb{R}^{N\times R}$ with the same soft-assignment rows as Algorithm 1.
- Initialize cumulative stats $A_{\mathrm{cum}}^{(\ell)} \leftarrow \mathbf{0}_R$, $B_{\mathrm{cum}}^{(\ell)} \leftarrow \mathbf{0}_{R\times d}$.
- For $t = 1..N$ (left-to-right scan):
  - $A_{\mathrm{cum}}^{(\ell)} \mathrel{+}= \Phi_K^{(\ell)}[t,:]^\top$  ($\in \mathbb{R}^R$)
  - $B_{\mathrm{cum}}^{(\ell)} \mathrel{+}= \Phi_K^{(\ell)}[t,:]^\top \cdot V_t$  ($\in \mathbb{R}^{R\times d}$)
  - $\mathrm{num}_t^{(\ell)} = \Phi_Q^{(\ell)}[t,:] \cdot B_{\mathrm{cum}}^{(\ell)}$  ($\in \mathbb{R}^d$)
  - $\mathrm{den}_t^{(\ell)} = \Phi_Q^{(\ell)}[t,:] \cdot A_{\mathrm{cum}}^{(\ell)}$  ($\in \mathbb{R}$)

Then per token t:
$\mathrm{Num}_t = (1/L) \sum_\ell \mathrm{num}_t^{(\ell)}$, $\mathrm{Den}_t = (1/L) \sum_\ell \mathrm{den}_t^{(\ell)}$, $\widehat{O}_t = \mathrm{Num}_t / \mathrm{Den}_t$.

## Implementation note

Implemented with **OpenMP/CUDA** parallelization, not a naive nested loop: each hash table runs in
its own thread with its own cumulative arrays, updated incrementally in a **single left-to-right
scan**. This avoids redundant recomputation that a `torch.cumsum()` materialization would incur and
keeps CPU/GPU execution parallel with negligible synchronization.

- The `torch.cumsum` reference (memory-heavy, materializes `B_pref[N,T,S,D]`, OOMs on long T) →
  `codebase/scaling-module` (`RaceCumsumCausal`).
- The streaming kernels → `codebase/gpu-kernels` (3-phase chunked scan forward,
  two-pass backward) and `codebase/cpu-kernels` (`race_prefix_mean_flat`).

**Caveat:** Algorithm 2 has **no approximation guarantee** - Theorem 1 is non-causal only
(`paper/06-theory`). Central for the causal vLLM backend (`crosswalk/open-questions`).

---
Source: arXiv-2510.04008v5/main.tex Algorithm 2 L2625-2670 (impl note L2672-2673), §Causal RACE Attention L2623.
