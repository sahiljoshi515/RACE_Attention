# Background: LSH and the RACE sketch

## Locality-Sensitive Hashing (LSH)

An LSH family $H$ for a similarity $\mathrm{Sim}$ makes near pairs collide more often than far pairs.
Formally $H$ is $(S_0, cS_0, p_1, p_2)$-sensitive (with $p_1 > p_2$, $c < 1$) if:

- $\mathrm{Sim}(x,y) \ge S_0 \Rightarrow \Pr[h(x)=h(y)] \ge p_1$
- $\mathrm{Sim}(x,y) \le cS_0 \Rightarrow \Pr[h(x)=h(y)] \le p_2$

A convenient sufficient condition, satisfied by **SimHash** and **WTA hashing**, is that the
collision probability is a **monotone increasing function of similarity**:
$\Pr[h(x)=h(y)] = f(\mathrm{Sim}(x,y))$. This is what lets a collision count stand in for a similarity sum.

## RACE sketch (Repeated Arrays-of-Count Estimators)

RACE (Coleman & Shrivastava 2020) shows any similarity expressible as a non-negative linear
combination of LSH collision kernels can be sketched with **ACE-style** estimation. It gives an
**unbiased** estimator of kernel-density sums $\sum_x k(x,q)^p$ and their powers: hash items into
counters, then read the counter addressed by the query. Averaging across **L** independent rows
reduces variance.

**Lemma (Thm 1 of Coleman & Shrivastava 2020).** Given dataset $D$, an LSH family $H$ with finite
range $[1, R]$, and parameter $p$, build $h(x) \to [1, R^p]$ by concatenating $p$ independent hashes.
Let $A$ be an ACE array built from $h(x)$. Then for any query $q$:

$$
\mathbb{E}[A[h(q)]] = \sum_{x\in D} k(x,q)^p.
$$

This is the engine RACE Attention rides: the **p-powered angular kernel** is exactly an
LSH collision kernel powered by $p$, so its density sums (the attention numerator/denominator) are
estimable in linear time. The remaining hurdle - classical RACE uses a non-differentiable
$\operatorname{sign}(Wx)$ hash - is solved by the **soft** relaxation in `paper/03-algorithm-noncausal`.

---
Source: arXiv-2510.04008v5/main.tex §Background L1159-1185 (LSH L1160-1169; RACE sketch + Lemma L1171-1185).
