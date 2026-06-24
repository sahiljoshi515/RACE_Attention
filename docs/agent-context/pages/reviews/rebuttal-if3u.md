# Rebuttal thread - Reviewer_if3U

Threaded discussion between the authors and Reviewer_if3U (chronological).

## Response (1/2) to Reviewer if3U  — Authors

> Q1. "I am a bit confused by the numerous instances of "stress test" and therefore it unclear... The most reasonable reading of the title suggests full model capability."

A1. We thank the reviewer for raising this important point. To clarify unambiguously, our scaling experiments stress-test only the attention mechanism itself, not the end-to-end training of a full Transformer model. This experimental design follows well-established practice in prior work on scalable attention mechanisms, including HyperAttention [1] and Performer [2].

Our stress tests benchmark a single forward–backward pass of a multi-head attention layer under fixed hyperparameters (the same ones used in our accuracy evaluations). Thus, the reported 75M-token capability refers solely to the maximum sequence length for which the attention primitive remains computationally feasible, not for training a full Transformer model at that length.

We do not train any end-to-end model at 75M tokens. Rather, the purpose of these experiments is to evaluate the scalability of the attention operation in isolation, which is the dominant computational bottleneck in long-context models.

This decomposition is deliberate and directly motivated by the behavior of long-context architectures. At extremely large sequence lengths:

- The attention operation dominates both memory and compute.

- Many existing attention variants fail or become infeasible far earlier.

- Demonstrating that the attention mechanism itself can scale to tens of millions of tokens is a necessary prerequisite before any full model can be trained in this regime.

Lastly, when we refer to “outrageously large context windows,” we are referring specifically to the attainable context range of the attention mechanism under practical forward–backward compute. While end-to-end Transformer training at 75M tokens is not yet standard practice, our results indicate that our proposed RACE Attention mechanism indeed brings such regimes significantly closer to feasibility.

**[1] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruff, Amir Zandieh. HyperAttention: Long-Context Attention in Near-Linear Time. ICLR 2024.**

**[2] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller. Rethinking Attention with Performers. ICLR 2021.**

> Q2. Clarifications for Tables on page 8.

A2. Yes, Angular is expected to be slightly better than RACE in general, because RACE approximates the sharpened angular kernel.

> Q3. Comparison with YOSO [3] and LSH Attention [4].

A3. We thank the reviewer for highlighting these related works. Both papers are indeed relevant, and we appreciate the opportunity to clarify our positioning relative to them. We discuss and outline the key differences between RACE and YOSO Attention in Section 1 (lines 53–61) and further elaborate in our response to Reviewer W9FA (Q1). Lastly, we do note that the LSH Attention paper [4] naturally fits within structural-sparsity approaches, and we now explicitly cite it in the "Sparsity is Complementary" subsection of Section 1. This work applies Reformer-style LSH attention to cross-attention within neural machine translation models. Its goal is to leverage sparsity patterns to reduce attention cost in encoder-decoder architectures, and is therefore complementary to ours. Our approach focuses on improving the efficiency of the core dense self-attention mechanism itself in a mathematically principled way. For this reason we view such sparsity-based methods, including [4], as orthogonal to our method.

**[3] Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling. ICML 2021.**

**[4] Frithjof Petrick, Jan Rosendahl, Christian Herold, Hermann Ney. Locality-sensitive hashing for long context neural machine translation. IWSLT 2022.**

## Response (2/2) to Reviewer if3U  — Authors

> Q4. Challenges in extending the current analysis to causal settings.

A4. Thank you for raising this important question. Algorithmically, adapting RACE Attention to causal masking is straightforward, and we already employ a prefix-scan (causal) version in all autoregressive experiments (see Algorithm 2 in the Appendix and Tables 6, 12). The modification concerns only how bucket summaries are accumulated. 

In the non-causal setting (Algorithm 1), each hash table $\ell$ uses global bucket summaries
$$
A^{(\ell)} = \sum_{j=1}^N \phi^{(\ell)}(K_j), 
\qquad
B^{(\ell)} = \sum_{j=1}^N \phi^{(\ell)}(K_j)V_j,
$$
which are shared across all query positions.

However,  under causal masking, these become prefix-cumulative quantities,
$$
A_{\mathrm{cum}}^{(\ell)}(t) = \sum_{j \le t} \phi^{(\ell)}(K_j),
\qquad
B_{\mathrm{cum}}^{(\ell)}(t) = \sum_{j \le t} \phi^{(\ell)}(K_j)V_j,
$$
and the attention estimate at time $t$ uses
$$
\text{Numerator: } 
\phi(Q_t)^\top B_{\mathrm{cum}}^{(\ell)}(t),
\qquad
\text{Denominator: }
\phi(Q_t)^\top A_{\mathrm{cum}}^{(\ell)}(t).
$$
The feature maps $\phi(Q_i), \phi(K_j)$, the Soft-RACE hashing, and the kernel estimator all remain unchanged; only the accumulation rule differs. Thus, the causal variant is a light modification of the non-causal algorithm. The difficulty lies in the theory rather than in the implementation. In the non-causal case, all queries share the same normalization mass $\phi(Q_i)^\top A^{(\ell)}$, enabling a uniform variance analysis. In contrast, under causal masking, the normalization term $\phi(Q_t)^\top A_{\mathrm{cum}}^{(\ell)}(t)$ grows with $t$, and both numerator and denominator depend on the same evolving cumulative buckets. This introduces position-dependent variance and temporal dependencies that are absent in the non-causal setting, preventing a direct extension of the proof of Theorem 2. Therefore, a rigorous analysis of causal RACE Attention is an important direction for future work.

## framing, positioning and experiments  — Reviewer_if3U

Thanks for your clarification. Let me comment briefly on a few points, which after reading your response, have reduced my support for this paper. 

I am quite concerned by two framing issues. 

First, your title in my opinion misrepresents what is really demonstrated. A context window to me refers to what a complete model can process end-to-end. This does not mean an isolated primitive. You have stress-tested a single attention layer at 75M tokens. This is impressive but at the same time, very different from training a model at that scale where memory compounds across layers. In your response, you cite HyperAttention/Performer as precedent for primitive benchmarking. I agree. But neither of these papers makes such an aggressive claim by using an attention grabbing title. Your defense is that this is a necessary prerequisite. Again I agree. But showing a prerequisite does not line up with your title. The framing oversells the practical capability and what systems can actually be built today. 

I am also disappointed with the somewhat superficial engagement with the YOSO work in your response. I think that the novelty is oversold in your paper. Does YOSO also not use a similar kernel as its target and discuss unbiased estimation?  Your work is adding concentration bounds via standard proof machinery. Essentially more tables and higher temperature is better. In my reading, this paper is making the discrete bucketing differentiable via soft assignments. This is good but the positioning of claiming algorithmic novelty while dismissing prior work even though it was pointed out in two separate reviews is confusing.  

The systems-level work is impressive and represents the main strength. But this would need, at a minimum, an honest framing that this paper makes LSH-attention already presented in earlier works practical and highlighting the main contribution by situating it correctly with the prior works, rather than overselling theory. This would be a very different paper.

## Response to Reviewer if3U on Framing, Positioning and Experiments  — Authors

> Q5. Discussion on Framing, Positioning and Experiments.

We sincerely thank the reviewer for clearly articulating the mismatch between our original title and the actual scope of our contributions. Our intention was to highlight both the aspects: Time and Accuracy in long-context training and Scalability at the primitive level. We now fully understand that the term “context windows” in the title can indeed be misleading, as it may suggest full end-to-end model capability at tens of millions of tokens, whereas our scaling experiments stress-test only the attention mechanism itself at such extreme lengths. We appreciate the reviewer’s guidance in identifying this framing issue. In response, and to better align the title with the precise scope of our work, we have revised the title to: **"RACE Attention: A Linear-Time Attention Mechanism for Long-Sequence Training with Extreme-Length Attention-Layer Scaling"**

This revised title avoids implying full-model training at extreme context lengths, while still capturing the two key aspects we intended to highlight:
- The ability of our method to train effectively on long sequences in strictly linear time, and
- The ability of the underlying attention mechanism to scale to extreme sequence lengths.

We believe that framing it this way more accurately reflects the balance of algorithmic, theoretical, and systems-level contributions in the paper.

> Q6. Detailed comparison with YOSO.

We appreciate the reviewer’s emphasis on the need for a more thorough discussion of YOSO. We recognize that our earlier response did not clearly convey the relationship between the two methods. The reviewer is correct that both YOSO and our approach operate on the same LSH-induced powered angular kernel and that YOSO discusses unbiased estimation of this kernel. We fully acknowledge this connection and have revised the paper (lines 51-59 and 263-272) to make it more explicit.

At the same time, while the underlying kernel family is shared, the two approaches differ significantly in what is being estimated, how the estimation is performed, and the resulting properties of the attention output. Standard attention's formula (Eq. 3 in our paper) can be written as:
$$
O_i=\frac{\sum_{j=1}^N sim\left(Q_i, K_j\right) V_j}{\sum_{j=1}^N sim \left(Q_i, K_j\right)}
$$
and for both YOSO and RACE, the similarity is given by the LSH-induced powered angular kernel, 
$$
\mathrm{sim}(Q_i, K_j) = \left(1 - \frac{\cos^{-1}\left(\frac{Q_i^\top K_j}{\lVert Q_i\rVert\, \lVert K_j\rVert}\right)}{\pi}\right)^\gamma.
$$
YOSO constructs a Bernoulli collision matrix $B$ via hard LSH hashing, where $(BV)\_i$ is an unbiased estimator of the _unnormalized_ kernel numerator $\sum_{j=1}^N sim(Q_i, K_j)V_j$. However, YOSO does not estimate the denominator $\sum_{j=1}^N \mathrm{sim}(Q_i, K_j)$ and their final output is obtained through post-hoc $\ell_2$ normalization, which does not correspond to the traditional normalization in attention. Furthermore, since the hard LSH is non-differentiable, YOSO cannot backpropagate through the hashing operation. As noted in Section 3.3 of their paper, the true derivative of the collision probability is numerically unstable and diverges as the similarity score $Q_i^TK_j$ approaches $1$, so YOSO replaces it with a _surrogate lower-bound gradient_ that is estimated using additional Bernoulli hash samples. Thus, YOSO's estimator is not differentiable in the usual sense and relies on surrogate-gradient approximations for training, incurring a quadratic time in embedding dimension $d$.

In contrast, our work explicitly utilizes the above formulation which is a normalized kernel attention and uses RACE sketches to approximate _both_ the numerator and denominator in strictly linear time. Our novel soft-bucketization strategy provides a smooth and fully differentiable relaxation of hyperplane hashing, enabling stable end-to-end training without surrogate gradients. Averaging across $ \mathrm{~L} $ independent RACE tables yields a low-variance estimator, and Theorem $2$ provides an explicit bias-variance guarantee for the resulting attention output. Finally, unlike YOSO, we show strong accuracy with bidirectional attention up to 64K sequence lengths, and our construction naturally supports a causal variant on which we show strong performance on the standard WikiText-103 benchmark.

In the revised version, we now explicitly describe how YOSO serves as an important precursor within the same kernel family and position our contribution as introducing a differentiable, provably accurate RACE-based estimator for normalized attention with long-sequence trainability and extreme-length scalability. We thank the reviewer for prompting us to clarify this relationship and hope this addresses the concern.


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
