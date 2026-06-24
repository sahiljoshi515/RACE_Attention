# Official Review - Reviewer_W9FA

Rating: 2  |  Soundness: 2  |  Presentation: 3  |  Contribution: 2  |  Confidence: 4

## Summary

This paper introduces a novel linear-time attention mechanism. The approach replaces the exponential softmax kernel with a monomial of cosine similarity raised to a power, enabling approximation through randomized projections. By leveraging angular similarity, Locality-Sensitive Hashing, the authors propose an efficient that enables outrageously large context windows  up to 75 million tokens on CPUs and 12 million on GPUs.

## Strengths

1. This method enables linear-time and memory-efficient attention that scales to tens of millions of tokens on standard hardware, which is impressive. 
2. The algorithm is simple, differentiable, and can serve as a drop-in replacement for softmax attention.

## Weaknesses

1. **This paper is very similar to YOSO [1] (for example, the finding the similarity between equation (1) and (2) in the text, the use of LSH in estimating the similarity function, the algorithm of estimating attention outputs via hashtables), but this paper does not discuss and contrast with [1].**
2. The experiments only show model accuracy on short sequence lengths (< 8K). What about longer sequences? 
3. The efficiency results in Figure 3 are not very meaningful as any linear attentions can be extremely efficient by tuning their hyperparameters. For example, for $\phi(Q) \phi(K)^T$ type attention, by setting the output dimension of $\phi$ to be 1, its efficiency can beat any other methods. To show efficiency, the runtime and memory results should be coupled with the corresponding accuracy results. 
4. Figure 5 has the same issue, what about the accuracy? 

**If the authors can address my concerns, I am willing to raise my score.**

[1] Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling. ICML 2021.

## Questions

see weakness section.


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
