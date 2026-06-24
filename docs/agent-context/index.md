# RACE Attention docs - index

Queryable corpus fusing the paper, the ICLR 2026 peer review, and the codebase. Query with
`/rdocs <slug>` (or `bash docs/agent-context/rdocs-helper.sh <slug>`). Full-text: `/rdocs search <term>`.

## paper/ - arXiv:2510.04008v5 manuscript
- `paper/00-overview` - title, authors, abstract, contributions, code link, key caveats
- `paper/01-introduction-related-work` - problem, baselines (Linear/Performer/Linformer/YOSO/sparsity), key idea (Eqs 1-2)
- `paper/02-background` - LSH and the RACE/ACE sketch (the linear-time estimator)
- `paper/03-algorithm-noncausal` - Algorithm 1: soft bucketization → bucket aggregation → normalization
- `paper/04-complexity` - $\mathcal{O}(LNRd)$ time / $\mathcal{O}(L(NR+Rd))$ space vs FlashAttention $\mathcal{O}(N^2 d)$
- `paper/05-angular-kernel` - sharpened angular similarity, $\gamma\approx 8$, and the RACE-vs-YOSO contrast
- `paper/06-theory` - Theorem 1 (bias $P/\beta$ + variance $\mathcal{O}\!\left(\sqrt{\frac{\log(N/\delta)}{L}}\right)$); causal case is open
- `paper/07-experiments` - accuracy Tables 1-5 (Arxiv, CIFAR/QNLI/TinyStories, Food-101, WikiText/PTB)
- `paper/08-scaling` - 12M GPU / 75M CPU; speedups; "primitive not full model"
- `paper/09-causal-algorithm` - Algorithm 2 (causal running-prefix scan; OpenMP/CUDA)
- `paper/10-appendix-proofs` - proof structure (Lemmas 1-4, Theorem 2 kernel deviation)
- `paper/figures` - figure index map (figs/*.png)

## reviews/ - OpenReview forum RR8Lh8RHgA (Submission 22728; Accept Poster) [regenerable]
- `reviews/00-overview` - ratings table (2/4/6), decision, page map
- `reviews/reviewer-w9fa` - review, rating 2 (YOSO, short-seq, efficiency-without-accuracy)
- `reviews/reviewer-eqbu` - review, rating 4 (FA3/Sigmoid baselines, Alg 1 clarity, $\gamma$)
- `reviews/reviewer-if3u` - review, rating 6 (75M framing, novelty vs prior work, causal theory)
- `reviews/meta-review` - Area Chair summary
- `reviews/decision` - Program Chair decision
- `reviews/rebuttal-w9fa` / `rebuttal-eqbu` / `rebuttal-if3u` - threaded author responses + follow-ups
- `reviews/author-summaries` - revision summary, AC summary, response-to-all-reviewers

## codebase/ - implementation (github.com/sahiljoshi515/RACE_Attention) [00-overview regenerable]
- `codebase/00-overview` - auto-generated dir map + public-symbol index + install/usage
- `codebase/cpu-kernels` - kernels/cpu (OpenMP `race_prefix_mean_flat`)
- `codebase/gpu-kernels` - kernels/gpu (3-phase forward scan, two-pass backward)
- `codebase/python-api` - misc/race.py (RACEAttention, BatchedACE, RACEBlock; K/L/M ↔ P/L)
- `codebase/scaling-module` - scaling/ (soft_hash_probs, RaceCausalCuda, race_prefix_ref)
- `codebase/tests-benchmarks` - scaling/test_kernels.py, benchmark_time.py
- `codebase/training-scripts` - misc/*.py + notebooks/
- `codebase/vllm-backend` - feat/vllm-race-attention-backend status + integration seams

## crosswalk/ - synthesis (the why-this-corpus-exists layer)
- `crosswalk/paper-to-code` - each paper construct/equation → exact code symbol:file:line
- `crosswalk/concerns-tracker` - every reviewer concern → status → where addressed (paper/rebuttal/code)
- `crosswalk/open-questions` - risks/unknowns carried into the vLLM backend (causal proof gap, framing, decode kernel, FA parity)
