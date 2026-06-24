# RACE Attention - paper overview

**Title:** RACE Attention: A Strictly Linear-Time Attention Layer for Training on Outrageously Large Contexts
**Venue:** ICLR 2026 (Accept, Poster). **arXiv:** 2510.04008v5.
**Authors:** Sahil Joshi, Agniva Chowdhury (equal contribution), Amar Kanakamedala, Ekam Singh, Evan Tu, Anshumali Shrivastava - Rice University, Dept. of Computer Science.
**Code:** https://github.com/sahiljoshi515/RACE_Attention

## One-paragraph summary

Softmax Attention is $\mathcal{O}(N^2 d)$ in sequence length, so even FlashAttention-2/3 cannot finish a
single forward-backward pass of one attention layer beyond ~4M tokens on an NVIDIA GH200 (96 GB).
**RACE** (**R**epeated **A**rrays-of-**C**ount **E**stimators) **Attention** is a kernel-inspired
drop-in replacement that is **strictly linear** in both sequence length $N$ and embedding size $d$.
It replaces the exponential softmax kernel with a **sharpened angular similarity** kernel and
approximates the attention output via **Gaussian random projections + soft (differentiable) LSH**,
never materializing the $N\times N$ attention matrix. In a controlled single-layer scaling study it
processes up to **12M tokens on GPU** and **75M tokens on CPU** in one forward-backward pass, and
matches or beats strong baselines up to 64K sequence length.

## Key contributions (paper's own list)

- **I. Long-context scaling:** 75M tokens (CPU) / 12M tokens (GPU) for a single attention layer's
  forward-backward pass, using the same hyperparameters as the accuracy evaluations.
- **II. Trainable RACE:** a differentiable sketch - hard hashing replaced by smooth soft
  assignments over hypercube corners - enabling end-to-end training.
- **III. CPU/GPU pre-training:** supports causal (autoregressive) and non-causal (bidirectional)
  training via custom OpenMP/CUDA kernels computing causal prefix ops in a single streaming pass.
- **IV. Theoretical insights:** LSH-framework approximation guarantee (Theorem 1) relating L
  (hash tables), P (hyperplanes), and $\beta$ (temperature) to the variance-accuracy tradeoff.

## How the pieces relate (page map)

- Method: `paper/03-algorithm-noncausal` (Algorithm 1), `paper/09-causal-algorithm` (Algorithm 2).
- Why angular kernel + YOSO contrast: `paper/05-angular-kernel`.
- Cost: `paper/04-complexity`. Guarantee: `paper/06-theory`. Proof structure: `paper/10-appendix-proofs`.
- Results: `paper/07-experiments`, `paper/08-scaling`. Figures: `paper/figures`.
- Implementation of all the above: `codebase/*`. How they line up: `crosswalk/paper-to-code`.

## Caveats surfaced in review (see crosswalk/concerns-tracker)

- The **75M / 12M token** numbers are for a **single attention primitive**, not a full model.
- The **causal** algorithm has **no approximation proof** (theory is non-causal only).
- Same powered-angular kernel as **YOSO**; the paper differentiates on soft-vs-hard hashing,
  differentiability, linear-in-d cost, and a formal guarantee.

---
Source: arXiv-2510.04008v5/main.tex (abstract L1100-1103, contributions L1152-1157). Provenance verified against the LaTeX source.
