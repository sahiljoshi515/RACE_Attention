# Introduction & related work

## Problem

The Transformer's core primitive, Softmax Attention, scales **quadratically** in context length.
Even FlashAttention-2/3 (exact, fused, GPU-optimized softmax) cannot complete a single
forward-backward pass of a single attention layer (1 batch, 4 heads, d=128) beyond ~4M tokens on
an NVIDIA GH200 (96 GB). Reaching "outrageously long" contexts (hundreds of millions of tokens)
needs a fundamentally different primitive that is accurate, fast, and memory-efficient.

## Related work (how RACE positions itself)

- **Linear / kernel-feature attention** - Linear Attention ($\phi(x)=\mathrm{elu}(x)+1$, reorders into
  associative sums; linear but often degrades accuracy); **Performer** (Random Fourier Features to
  approximate the exponential; **quadratic in embedding size $d$**, and RFF needs high-dim features
  for accuracy $\to$ poor practical scaling).
- **Low-rank** - **Linformer** (learned length-wise K/V projections), **Nyströmformer** (Nyström
  landmarks, rank-k). Reduce $\mathcal{O}(N^2 d)\to\mathcal{O}(Nkd)$ but require tuning/growing $k$ and **do not support
  autoregressive tasks**. RACE beats Linformer in accuracy despite Linformer having ~13% more
  params.
- **YOSO Attention** - the closest prior work: same powered-angular kernel, but uses **hard LSH**
  (Bernoulli collision indicators) with a non-standard post-hoc $\ell_2$ normalization, is
  non-differentiable (needs surrogate gradients), and is **quadratic in $d$**. Detailed contrast in
  `paper/05-angular-kernel`.
- **Sparsity** (Longformer, BigBird, Reformer, HyperAttention) - **complementary**; exploits
  structural priors rather than making the primitive itself cheaper. Out of scope; combinable in
  future work.

The paper's framing: despite many approximations, softmax stays dominant because prior methods
lack a rigorous framework linking efficiency knobs to downstream accuracy.

## Key idea

Standard attention (Eq. 1):  $O = \mathrm{softmax}(QK^\top/\sqrt{d})\, V$ - row-wise, weights $\ge 0$ and sum to 1, and
the exponential sharply amplifies small score differences.

RACE replaces the exponential with a higher-degree monomial of an **angular** (cosine-geometry)
kernel (Eq. 2, informal):  $O = (1 - \cos^{-1}(QK^\top)/\pi)^\gamma V$.

For large $\gamma$ this mimics softmax sharpness, **and** (unlike softmax) it admits a **linear-time**
approximation via RACE sketches (see `paper/05-angular-kernel`, `paper/02-background`). RACE
Attention is a **drop-in replacement** for softmax, evaluated on causal LM, masked LM, and
text/image classification. Because each query mixes with a fixed bank of $S = L\cdot R$ bucket
summaries (not all $N$ keys), the working set stays compact and activation memory drops.

---
Source: arXiv-2510.04008v5/main.tex §Introduction L1105-1158 (related work L1120-1130; key idea + Eqs 1-2 L1134-1148).
