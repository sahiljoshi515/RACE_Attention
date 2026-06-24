# Algorithm 1 - RACE Attention (non-causal)

**Inputs:** $Q, K, V \in \mathbb{R}^{N\times d}$; `L` hash tables; `P` hyperplanes; temperature $\beta > 0$.
**Output:** $\widehat{O} \in \mathbb{R}^{N\times d}$.

For each table $\ell = 1..L$:

1. Draw $W^{(\ell)} \in \mathbb{R}^{P\times d}$ with rows $w_p \sim \text{i.i.d.}\ N(0, I_d)$.
2. Define corner set $V = \{\pm 1\}^P$ (so $R = 2^P$ corners), $v_r \in \{\pm 1\}^P$.
3. Build $\Phi_Q^{(\ell)}, \Phi_K^{(\ell)} \in \mathbb{R}^{N\times R}$ with the **soft assignment** rows
   (softmax over R corners of the soft-sign alignment):

   $$[\phi^{(\ell)}(x)]_r = \frac{\exp\{\beta\cdot(\tanh(W^{(\ell)} x))^\top v_r\}}{\sum_{r'} \exp\{\beta\cdot(\tanh(W^{(\ell)} x))^\top v_{r'}\}}, \quad x \in \{Q_i, K_j\}$$
4. Per-table bucket statistics:
   - $A^{(\ell)} = (\Phi_K^{(\ell)})^\top \cdot \mathbf{1}_N \in \mathbb{R}^R$   (soft mass per bucket)
   - $B^{(\ell)} = (\Phi_K^{(\ell)})^\top \cdot V \in \mathbb{R}^{R\times d}$ (value-sum per bucket)

Then aggregate across tables and normalize:

5. $\mathrm{Num} = \frac{1}{L} \sum_\ell \Phi_Q^{(\ell)} B^{(\ell)}$ ; $\mathrm{Den} = \frac{1}{L} \sum_\ell \Phi_Q^{(\ell)} A^{(\ell)}$.
6. Return $\widehat{O} = \operatorname{diag}(\mathrm{Den})^{-1} \cdot \mathrm{Num}$.

## Three stages (the mental model)

1. **Soft bucketization** (steps 2-4): each query/key is randomly projected through P
   hyperplanes and **softly** assigned to $R = 2^P$ hypercube corners with distribution
   $\phi^{(\ell)}(x)$. This is the differentiable replacement for classical RACE's hard $\operatorname{sign}(Wx)$.
2. **Bucket aggregation** (step 5 / stat eqs): per table, accumulate key mass $A^{(\ell)}[r]$ and
   weighted value sums $B^{(\ell)}[r,:]$. Keys/values are compressed into $R$ bucket summaries.
3. **Global normalization** (steps 7-8): average over the L tables to reduce variance, then
   reconstruct $\widehat{O} = \operatorname{diag}(\mathrm{Den})^{-1} \mathrm{Num}$.

## Kernel view (why it works)

In classical RACE, $h(x) = \operatorname{sign}(W^{(\ell)}x)$ and two vectors collide with probability
$\Pr[h(Q_i)=h(K_j)] = S_{ij} = \mathrm{sim}(Q_i,K_j)$, exactly the **P-powered angular kernel** (Eq. for
`sim` with $\gamma = P$). Soft RACE keeps this geometry but swaps the hard sign for $\tanh(W^{(\ell)}x)$ and a
softmax over the $R$ corner sign-patterns $v_r$, so the per-table inner product
$\phi^{(\ell)}(Q_i)^\top \phi^{(\ell)}(K_j)$ is a **smooth approximation** to the P-powered angular similarity. Near-
aligned vectors still put most mass in the same buckets.

Never materializes the $N\times N$ matrix: each query mixes with $S = L\cdot R$ bucket summaries, not all N
keys. The causal variant (running prefix sums) is Algorithm 2 - see `paper/09-causal-algorithm`.
Implementation: `codebase/python-api`, `codebase/scaling-module`, `codebase/gpu-kernels`.

---
Source: arXiv-2510.04008v5/main.tex Algorithm 1 L1190-1220; three stages L1281-1284; kernel view L1286-1288.
