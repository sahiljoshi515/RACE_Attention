# Appendix: proof of Theorem 1 (structure)

Full proof in `arXiv-2510.04008v5/main.tex` Appendix §"Proof of Theorem 1" (L1853+). The argument
decomposes the output error into a **bias** (finite-$\beta$ soft bucketization) and a **variance**
(finite-L averaging) term, then propagates a kernel-level bound through attention normalization.

## Lemma / theorem chain (anchors in main.tex)

| Result | Label | Line | Role |
| --- | --- | --- | --- |
| Bounds for a single ensemble | `lem:phibounds-paper` | L1915 | $\lVert \widehat{S}^{(\ell)}\rVert_F \le N$ type norm bounds on one table's kernel |
| (helper bound on X) | `lem:boundX-paper` | L1931 | per-entry bound feeding the concentration step |
| Matrix Bernstein | `lem:bernstein-paper` | L1982 | matrix-concentration tool (RandNLA) |
| Kernel deviation w/ explicit constants | `thm:kernel-deviation-paper` | L1991 | $\lVert \widehat{S}-S\rVert_2 \le \lVert \tilde{B}\rVert_2 + \mathcal{O}\!\left(\frac{N}{\sqrt{L}}\sqrt{\log(N/\delta)}\right) + \mathcal{O}\!\left(\frac{N}{L}\log(N/\delta)\right)$ (bias + variance) |
| Probability-vector inequality | `lem:ineq` | L2077 | $\lvert p^\top q - \mathbf{1}\{a=b\}\rvert \le (1-p_a)+(1-q_b)$ for prob. vectors |
| Bounding the bias term | (lemma) | L2125 | $\lVert \tilde{B}\rVert_2 \le \frac{4}{\sqrt{2\pi}}\cdot\frac{NP}{\beta} + 2C_1\cdot NP\cdot e^{-c\beta}$, $c = 2\cdot\tanh(1)$ |
| Row-sum & inverse-diagonal control | `lem:rowsum-paper` | L2291 | controls $\operatorname{diag}(\mathrm{Den})^{-1}$ under (A1) |
| Concentration bound for E | `lem:E-concentration` | L2354 | concentration of the normalization error |
| Exact perturbation identity & bound | `lem:perturb-paper` | L2415 | exact identity for $\widehat{A} = \widehat{D}^{-1}\widehat{S}$ perturbation |
| Attention deviation | `thm:attn-deviation-paper` | L2461 | kernel error $\to$ attention-weight error |
| End-to-end output error | `thm:output-paper` | L2509 | assembles the final $\lVert \widehat{O}-O\rVert_{\mathrm{rms}}$ bound (Theorem 1) |

## Takeaways for using the result

- The **bias** scales like $P/\beta$ (plus an exponentially small $e^{-c\beta}$ term) - so $\beta$ must grow
  with P to keep soft bucketization faithful.
- The **variance** scales like $\mathcal{O}\!\left(\sqrt{\frac{\log(N/\delta)}{L}}\right)$ - $L = \Theta(\log N)$ suffices to control it.
- Everything is **non-causal**; the causal masking case is explicitly left open (`paper/06-theory`).

Appendix also contains additional experiment tables (hyperparameters `tab:exp-hparams`, FashionMNIST
`tab:fashion`, Tiny Stories `tab:tiny`, LRA `tab:long_range_arena`, Yahoo/IMDB/SST-2
`tab:yahoo_imdb_sst2`) and the causal Algorithm 2 (`paper/09-causal-algorithm`).

---
Source: arXiv-2510.04008v5/main.tex Appendix L1853-2520; line anchors grepped from \label{...} in the source.
