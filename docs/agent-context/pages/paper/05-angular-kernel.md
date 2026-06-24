# The angular kernel, sharpening $\gamma$, and the YOSO contrast

## Generalized similarity attention

Attention weights can come from **any** non-negative similarity `sim`:

$$O_i = \frac{\sum_j \mathrm{sim}(Q_i, K_j) V_j}{\sum_j \mathrm{sim}(Q_i, K_j)} \qquad (\text{Eq. for } O_i)$$

Softmax's exponential gives (i) non-negativity, (ii) weights summing to 1, and a strong
non-linearity that amplifies small score differences. RACE seeks a softmax-like similarity that
**also** admits accurate linear-time estimation.

## The (sharpened) angular kernel

Raw angular similarity (norm-invariant, depends only on the angle):
$\mathrm{sim}(Q_i,K_j) = 1 - \cos^{-1}\!\left(\frac{Q_i^\top K_j}{\lVert Q_i\rVert\lVert K_j\rVert}\right) / \pi$.

It is well-behaved, normalized, and **LSHable**, but **flat** near high similarity, so it
discriminates poorly among nearly aligned vectors. Fix: **exponentiate** with a sharpening
parameter $\gamma$:

$$\mathrm{sim}(Q_i,K_j) = \left(1 - \cos^{-1}\!\left(\frac{Q_i^\top K_j}{\lVert Q_i\rVert\lVert K_j\rVert}\right) / \pi\right)^\gamma \qquad (\text{Eq. exp-angular-sim})$$

For sufficiently large $\gamma$ this becomes almost indistinguishable from the softmax kernel (a
high-degree monomial like $x^{12}$ behaves like an exponential). The Frobenius error between Angular
and Softmax Attention drops sharply with $\gamma$; **softmax-level sharpness is reached by $\gamma \approx 8$**. Any
constant power of the angular kernel belongs to the family that RACE sketches can estimate in
linear time (see `paper/02-background`).

## RACE vs YOSO (same kernel, different estimator)

Both operate on the **same** powered-angular kernel (Eq. exp-angular-sim), but:

| | YOSO | RACE |
| --- | --- | --- |
| Hashing | **Hard** LSH $\to$ Bernoulli collision indicators | **Soft** differentiable assignments (tanh + softmax over corners) |
| Estimates | unbiased **numerator** only, then post-hoc **$\ell_2$** normalization (non-standard) | numerator **and** denominator $\to$ standard attention normalization |
| Gradients | non-differentiable $\to$ surrogate lower-bound gradients from extra Bernoulli samples | direct differentiation of the kernel |
| Cost in d | **quadratic in d** $\to$ poor end-to-end scaling | **linear in d** |
| Guarantees | none on approximation quality; no causal mechanism | Theorem 1 bound (`paper/06-theory`); causal Algorithm 2 |

Empirically YOSO's CUDA kernel fails beyond 32K tokens (memory) while RACE keeps scaling
(Fig. yoso_vs_race). This contrast is the single most-debated point in review - see
`crosswalk/concerns-tracker` and `reviews/reviewer-w9fa`.

---
Source: arXiv-2510.04008v5/main.tex §"Softmax-Like Similarities…" L1222-1273 (similarity attention L1237-1244; angular kernel L1246-1256; γ≈8 L1259; YOSO contrast L1272-1273).
