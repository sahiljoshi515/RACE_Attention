# Meta-Review (Area Chair)

## summary

Initial reviews for the paper are somewhat mixed (2,4,6).

Reviewer W9FA notes that the work has similarities to existing work and notes that experiments have only been done on relatively short sequences, when the main pitch of the work is to enable very long context length.  Additionally, there are questions about the accuracy/computation trade-off based on how the embedding dimension is selected.

Reviewer eQBU raises questions about whether the most recent implementations of FlashAttention were using in the benchmark comparisons and also asks about alternative kernels (such as Sigmoid attention) which might allow for simple kernel implementations.  There are also questions to clarify certain aspects of the presentation.

Reviewer if3U also notes similarities to existing work and questions whether some of the claims are somewhat oversold in that the maximum computable context length reported in the abstract (75 million tokens) refers to an attention primitive and not a full model.

## reviewer_concerns

The majority of the concerns appear largely addressed in the rebuttal, with perhaps the exception of the framing of the work with respect to prior work as discussed below.  Reviewer eQBU explicitly notes all of their concerns being addressed and raising their score.

With regards to connections with prior work, the authors have provided fairly detailed responses about how their work differs from this prior work and how it results in practical benefits for scaling in the embedding dimension and making the overall operator differentiable.  Likewise, they have made corresponding modifications to the manuscript to delineate the differences between the two works.  In my view, the authors have largely addressed differentiating the current work from prior work.

## reviewer_scores

The review of this paper would have benefited significantly from a more substantial discussion period.

Reviewer eQBU notes that their concerns have been addressed and raised their score, while reviewer if3U notes that their support of the paper decreased after it was clarified that the 75M token context length referred to a single attention primitive and not a full model along with concerns of over-selling the work and it's novelty to prior work.  However, in response to this the authors have provided a more substantial discussion of how their work differs from this prior work along with corresponding modifications to the manuscript, yet reviewer if3U was not able to respond to this comment before the discussion was closed.

Whether reviewers eQBU and if3U would be satisfied with the authors' final response and paper modifications I cannot say, but in my view the authors have provided a strong discussion of the differences with the noted prior work and how this results in several practical advantages for the ultimate algorithm.  This, combined with several other positive aspects of the work noted by the reviewers regarding the systems work and experimental demonstration, leave me inclined to give the authors the benefit of the doubt and recommend acceptance for the paper.  Nevertheless, I would encourage the authors to be very judicious how they frame their work.  For example, even in the current abstract it is still somewhat ambiguous that the 75M token capacity refers to an attention primitive and not a full model.


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
