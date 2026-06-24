# Crosswalk: reviewer concerns → status → where addressed

ICLR 2026 Submission 22728, decision **Accept (Poster)**. Reviews: W9FA=2, eQBU=4, if3U=6.
Status legend: **Addressed** (authors resolved + revised), **Partial** (revised but reviewer not
fully satisfied), **Open** (acknowledged gap, future work). Sources: `reviews/*` (full text).

| # | Concern | Raised by | Status | Where addressed |
| --- | --- | --- | --- | --- |
| 1 | **YOSO similarity not discussed/contrasted** (same powered-angular kernel + LSH) | W9FA(W1), if3U(W3) | **Partial** | Manuscript lines 51-59, 263-272 (soft-vs-hard hashing, num+den vs numerator-only, formal bias/variance, causal support, 64K vs YOSO's 4K). See `paper/05-angular-kernel`. if3U still felt novelty was oversold (see #8). |
| 2 | **Accuracy only on short seqs (<8K besides Arxiv)** | W9FA(W2) | **Addressed** | Added Arxiv 16K/32K/64K (Table 7) + Food-101 ViT @16K (Table 9), matched hyperparameters. `paper/07-experiments`. |
| 3 | **Efficiency plots meaningless without paired accuracy** (any linear attn can be fast with degenerate hyperparams) | W9FA(W3,W4) | **Addressed** | Clarified scaling uses the **same** hyperparameters as accuracy runs (Figs 4/5/6), per HyperAttention/Performer practice. |
| 4 | **Missing baselines: FlashAttention-3, Sigmoid Attention** | eQBU(W1) | **Addressed** | Added FA3 + Sigmoid to accuracy + scaling (GH200). CPU-RACE up to 20× faster, GPU-RACE up to ~2500× faster than FA3 @4M. `paper/07-experiments`, `paper/08-scaling`. |
| 5 | **Algorithm 1 seems to re-implement softmax; $\phi\leftrightarrow$angular link unclear** | eQBU(W2), eQBU(Q on derivation) | **Addressed** | Clarified the softmax in Alg 1 is **local smoothing for differentiable LSH bucketization**, not the Transformer softmax; added full derivation $\phi \leftarrow$ P-powered angular kernel (lines 300-313). `paper/03-algorithm-noncausal`. |
| 6 | **How to choose $\gamma$? sensitivity?** | eQBU(Q1,Q2) | **Addressed** | $\gamma$ is **not tunable**: fixed by hyperplanes, $\gamma = P$. Added Frobenius-error-vs-$\gamma$ plot (Fig 2); modest degree (~8) suffices. `paper/05-angular-kernel`. |
| 7 | **Title oversells: 75M tokens = single attention layer, not full-model context** | if3U(W1,Summary) | **Partial** | Authors confirmed it is one layer's forward-backward pass; revised title to scope it to the attention primitive. if3U **maintained** the framing oversells real capability (memory compounds across layers). `paper/08-scaling`, `crosswalk/open-questions`. |
| 8 | **Novelty oversold vs prior LSH-attention; superficial YOSO engagement** | if3U(post-rebuttal) | **Open/Partial** | Authors expanded YOSO contrast + added concentration bounds. if3U's follow-up **reduced support**: views the core novelty as making discrete bucketing differentiable via soft assignment, with standard concentration proof machinery; wanted honest "makes prior LSH-attention practical" framing. Reviewer could not respond again before discussion closed. See `reviews/meta-review`. |
| 9 | **Page-8 tables confusing: is Angular expected to beat RACE?** | if3U(W2) | **Addressed** | Angular ($\gamma = 8$) is the exact powered-angular kernel = the target RACE *approximates* (upper bound). `paper/07-experiments`. |
| 10 | **Is adapting the analysis to causal masking easy, or problematic?** | if3U(Q1) | **Open** | Causal Algorithm 2 is **implemented** (Appendix), but the **theory is non-causal only**; rigorous causal analysis is future work. `paper/06-theory`, `paper/09-causal-algorithm`. |
| 11 | **Cite related work** (Zeng et al. 2021 / YOSO; IWSLT 2022 LSH paper) | if3U(W3,Q2) | **Addressed** | Positioning expanded; YOSO contrast added. |

## Outcome (meta-review, AC u14c)

Most concerns addressed in rebuttal; the **framing vs prior work** (rows 1/7/8) was the lingering
issue. eQBU explicitly raised their score; if3U's support dropped post-clarification but could not
reply again before discussion closed (truncated by an OpenReview incident). AC gave benefit of the
doubt on the strength of the systems + experimental work → Accept (Poster), with an explicit
caution to frame the context-length claims judiciously (the abstract still reads ambiguously).

---
Source: reviews/{reviewer-w9fa,reviewer-eqbu,reviewer-if3u,rebuttal-if3u,author-summaries,meta-review}.md (OpenReview forum RR8Lh8RHgA), cross-referenced to paper sections. Verified against the rendered review pages.
