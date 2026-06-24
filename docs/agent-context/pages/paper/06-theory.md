# Theoretical analysis (Theorem 1)

## Kernel-approximation view

Each table $\ell$ induces a random feature map $\phi^{(\ell)}: \mathbb{R}^d \to \mathbb{R}^R$ ($R = 2^P$ corners) and an
approximate kernel $\widehat{S}^{(\ell)}_{ij} = \phi^{(\ell)}(Q_i)^\top \phi^{(\ell)}(K_j)$. Averaging over L tables gives
$\widehat{S} = \frac{1}{L} \sum_\ell \widehat{S}^{(\ell)}$, which replaces the target angular kernel S ($\gamma = P$). Because each $\phi^{(\ell)}$
is a softmax distribution over corners, $\widehat{S}$ inherits concentration from the Gaussian projections,
analyzable with RandNLA / matrix-concentration tools.

## Assumptions

- **(A1) Row sums bounded away from 0:** $s_{\min} := \min_i (S\cdot 1)_i \ge C_1\cdot N$, $C_1 > 0$. Ensures stable
  row-normalization; rules out a query with vanishing similarity to all keys. Mild for learned reps.
- **(A2) Spectral norm bounded:** $\lVert S\rVert_2 \le C_2\cdot N$. Always holds with $C_2 = 1$ since $S_{ij} \in [0,1]$
  (worst case is the all-ones matrix with $\lVert J_N\rVert_2 = N$).

## Theorem 1 (quality of approximation)

For parameters L, P, $\beta$ under (A1)-(A2), the estimator $\widehat{O}$ from Algorithm 1 satisfies, with
probability $\ge 1 - \delta$:

$$\lVert \widehat{O}-O\rVert_{\mathrm{rms}} = \mathcal{O}\!\left(\frac{P}{\beta} + \sqrt{\frac{\log(N/\delta)}{L}}\right) \cdot \lVert V\rVert_F$$

where $O$ uses Eqs. (O_i) and (exp-angular-sim) with $\gamma = P$, and
$\lVert \widehat{O}-O\rVert_{\mathrm{rms}} := \sqrt{\frac{1}{N} \sum_i \lVert \widehat{O}_i - O_i\rVert_2^2}$ is the **per-token RMS error**.

## Reading the bound

- **Bias term $\mathcal{O}(P/\beta)$** - from finite-$\beta$ soft bucketization. Larger $\beta \to$ smaller bias. Powering by
  P sharpens collisions but smoothing adds bias, so **$\beta$ should scale with P**.
- **Variance term $\mathcal{O}(\sqrt{\log(N/\delta)/L})$** - from finite L tables. Larger L $\to$ smaller variance.
- As $\beta, L \to \infty$, error vanishes. Taking $L = \Theta(\log N)$ keeps variance from exploding.
- Net: L, P, $\beta$ jointly govern the accuracy-efficiency tradeoff (a precise RandNLA lens).

## Causal masking caveat (important)

The LM experiments (WikiText-103, PTB) use **causal** RACE (Algorithm 2, OpenMP/CUDA). **The
theory covers only the non-causal setting.** Extending Theorem 1's bias-variance guarantee to the
causal case is an **open problem** - the cumulative-sum constraint interacts non-trivially with
the random-feature construction. This gap is flagged by reviewers (see `crosswalk/open-questions`)
and matters directly for a vLLM/decode backend, which is inherently causal.

Full proof + lemmas: `paper/10-appendix-proofs`.

---
Source: arXiv-2510.04008v5/main.tex §"Theoretical Analysis" L1298-1334 (A1/A2 L1307-1312; Theorem 1 L1321-1331; reading the bound L1333-1334; causal remark L1337).
