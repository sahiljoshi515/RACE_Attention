# Computational complexity

Per-table runtime of Algorithm 1, by step:

- Step 2 (random projections): $\mathcal{O}(N d P)$
- Step 3 (logits over $R = 2^P$ corners): $\mathcal{O}(N P R)$
- Step 5 (bucket aggregation): $\mathcal{O}(N R d)$
- Step 7 global accumulation: adds $\mathcal{O}(N R d)$ per table

**Per-table:** $\mathcal{O}(NdP + NPR + NRd) = \mathcal{O}(N R d)$ time, $\mathcal{O}(NR + Rd)$ space.
**Across L tables:** **$\mathcal{O}(L N R d)$ time**, **$\mathcal{O}(L(NR + Rd))$ space**.

## Versus FlashAttention-2/3

| | Time | Space |
| --- | --- | --- |
| FlashAttention-2/3 (exact softmax) | $\mathcal{O}(N^2 d)$ | $\mathcal{O}(Nd)$ |
| RACE Attention | $\mathcal{O}(L N R d)$ | $\mathcal{O}(L(NR + Rd))$ |

RACE wins because $R, L \ll N$ and $R, L \ll d$, even for moderate N and d. FlashAttention removes the
quadratic **memory** of the score matrix via tiling but still computes all key-query interactions
and must store token-level Q, K, V, O activations + gradients at $\mathcal{O}(B\cdot H\cdot N\cdot d)$; for large N that
linear footprint alone exceeds GPU HBM. RACE compresses Q/K into R bucket summaries per table,
reducing activation memory to $\mathcal{O}(B\cdot H\cdot L\cdot(NR + Rd))$.

Key knobs: `P` (hyperplanes $\Rightarrow$ $R = 2^P$ buckets, kernel sharpening $\gamma = P$), `L` (tables, variance),
$\beta$ (temperature, bias). Tradeoffs are formalized in `paper/06-theory`.

---
Source: arXiv-2510.04008v5/main.tex §"Computational Complexity" L1291-1296; memory footprint L1602.
