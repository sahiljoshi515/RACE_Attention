# Official Review - Reviewer_if3U

Rating: 6  |  Soundness: 3  |  Presentation: 2  |  Contribution: 3  |  Confidence: 4

## Summary

This paper describes RACE attention as a linear-time alternative to softmax attention for very long contexts. The main idea is to replace softmax with powers of angular similarity, and then approximate this term using RACE sketches. To do this, the algorithm uses soft LSH so that its differentiable. This achieves far reduced complexity versus quadratic for standard attention, as is common in most methods for self-attention approximation. What is nice is that the experiments are broad and cover language modeling, masked LM, and classification. In this context, scaling experiments show processing of tens of millions of tokens on CPU and GPU for a single attention layer's forward-backward pass. This will be the main highlight of this work for most readers.

## Strengths

1. The scaling experiments are quite impressive. Regardless of my other comments below, this is a good practical contribution. Also, it is interesting that CPU-based RACE is viable and in some regimes can do better than FlashAttention. This point about algorithmic efficiency versus hardware acceleration could really be a main message of the paper (more on this below). In any case, reaching 50M/75M tokens is definitely a strength (but in the current version of the paper, this comes with some disclaimer).

2. The experimental breadth is very good. Both CPU and GPU kernels with OpenMP are mentioned. This is a strong engineering effort and if code is provided, it can benefit many groups working in this area. 

3. Experimental verification of how increasing degree can mimic exponential behavior in this setting is useful. Some analysis is included for the bias-variance to guide the choices in the sketching component. This is all good.

## Weaknesses

1. I am a bit confused by the numerous instances of "stress test" and therefore it unclear what the scaling experiments actually show. When stress testing 1 forward-backward pass with the multi-head attention layer, is this timing a single layer, not end-to-end model training? If so, the 75M token claim is for one attention operation, not training the full model? Is this paper only describing benchmarking the primitive or does any model work at these lengths? The reason for this question is the title "outrageously large context windows" -- is this only for the stress tests? The most reasonable reading of the title suggests full model capability.

2. I am having trouble understanding the tables on page 8. Is angular expected to be better than RACE? 

3. The paper https://proceedings.mlr.press/v139/zeng21a.html uses related ideas and also seems motivated by similar upstream papers. Another one is https://aclanthology.org/2022.iwslt-1.4.pdf. The positioning of this work on page 4/5 should at least describe how they differ.

## Questions

1. Minor: Is adapting the analysis to causal masking relatively easy (but hasn't been worked out yet) or does one run into problems?
2. check some of the references above. There may be others.


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
