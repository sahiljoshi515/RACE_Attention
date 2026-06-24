# Official Review - Reviewer_eQBU

Rating: 4  |  Soundness: 3  |  Presentation: 2  |  Contribution: 2  |  Confidence: 3

## Summary

This paper introduces RACE Attention, a method to address the quadratic time and memory complexity of standard softmax attention. The authors propose replacing the exponential softmax kernel with a high-degree monomial of an angular (cosine) similarity kernel. This specific kernel choice allows them to leverage Locality Sensitive Hashing (LSH) and Repeated Arrays-of-Count Estimators (RACE) sketches to compute the attention output in linear time and space complexity.

## Strengths

1.  The primary contribution and strength of this paper are the scaling results. Figure 5 shows that RACE on a CPU can outperform FlashAttention on a high-end GPU at massive sequence lengths, is a compelling demonstration of the algorithm's effect over hardware acceleration.

2. The paper is well-written and easy to follow.

3. The theoretical result also provides a nice bias-variance trade-off of their approach.

## Weaknesses

1. The paper seems to be lacking some important baselines. The authors compare their result to FlashAttention, however, at the moment FlashAttn 2 and 3 are also available that performs much faster and are not included in the comparison. Moreover, the paper focuses on alternatives to softmax and is for example lacking a comparison to Sigmoid Attention which also provides a simple kernel implementation.

2. The paper is a bit vague and ambiguous in their main algorithm. The authors argue that they use cosine kernel to prevent the exponential of softmax and be able to use RACE sketch. However, it seems that Algorithm 1 is still trying to implement softmax. Am I misunderstanding this? Technically, it seems that the connection between the features $\phi$ and the angular attention is never clearly made.

## Questions

1. Can authors elaborate on how to choose $\gamma$? Would it be through a hyperparameter search or is there a principled way of approximating a good value for it?

2. Once more question on $\gamma$, could authors provide any sensitivity analysis of how the final result changes with respect to the small changes in $\gamma$? Perhaps another useful figure would be to use the data from Fig 2 and plot the distribution of the attention distances between softmax and the angular attention to see how it varies as $\gamma$ is changed.


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
