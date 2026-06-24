# Rebuttal thread - Reviewer_W9FA

Threaded discussion between the authors and Reviewer_W9FA (chronological).

## Response (1/2) to Reviewer W9FA  — Authors

> Q1. YOSO Attention [1] discussion.

A1. We thank the reviewer for highlighting the connection to YOSO. We discuss and outline the key differences between RACE and YOSO in the revised paper (lines 51-59, 263-272). While both approaches employ LSH to approximate attention, the underlying mechanisms and theoretical foundations differ substantially.

**1.) Kernel and Estimator:** YOSO relies on _Bernoulli sampling_ based on hard LSH collisions to approximate the angular kernel. RACE Attention instead uses a _smooth, differentiable relaxation_ of RACE hashing ([2], [3]) to approximate the angular kernel. Note that YOSO’s training cost scales quadratically in $d$, whereas RACE’s training cost is linear in $d$.

**2.) Theoretical guarantees:** YOSO provides no formal analysis of approximation error. In contrast, RACE comes with _explicit bias and variance guarantees_ (Theorem 2) and proves convergence to the corresponding $P$-powered angular kernel, giving a principled understanding of estimator quality.

**3.) Applicability and scalability:** YOSO reports results only up to 4K sequence length and only for bidirectional attention. On the other hand, RACE supports full and causal attention, demonstrating bidirectional attention up to 64K tokens, and evaluating perplexity on the widely used WikiText-103 benchmark for causal language modeling.

**For an in-depth comparison with YOSO please refer to answer to Q6 of Reviewer if3U.**

**[1] Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling. ICML 2021.**

**[2] Benjamin Coleman, Anshumali Shrivastava. Sub-linear RACE Sketches for Approximate Kernel Density Estimation on Streaming Data. WWW 2020.**

**[3] Benjamin Coleman, Richard G Baraniuk, Anshumali Shrivastava. Sub-linear Memory Sketches for Near Neighbor Search on Streaming Data. ICML 2020.**

> Q2. Reporting additional experiments.

A2. We appreciate the reviewer for the emphasis on longer-context evaluation. In our initial submission, we limited experiments to sequences up to 8K tokens because, based on our theoretical analysis and earlier pilot studies, we did not anticipate any qualitative change in RACE Attention’s behavior at longer contexts, as the underlying hashing and aggregation mechanism is expected to behave consistently as sequence length grows. However, to more comprehensively validate this and to strengthen the empirical performance as reviewer W9FA suggested, we now extend our experiments to substantially longer sequence lengths. Specifically, in the revised paper, we include accuracy results for experiments at **16K**, **32K**, and **64K** sequence lengths on the ArXiv classification task in Table 7. In addition, we provide long-context image classification experiment on Food-101 dataset using Vision Transformers at **16K**  sequence length in Table 9. These new results (summarized below) confirm our original expectation: **RACE Attention maintains stable accuracy and continues to scale efficiently even as context length increases by an order of magnitude.**

**Table:** Long-context ArXiv text classification performance on a 40GB A100 GPU. Train/Test denote per-epoch runtimes in seconds, and Acc. denotes test accuracy.
| **Method**               |      |    **16K**      |          |    |      |      **32K**     |          |    |       |    **64K**      |          |
|--------------------------|------------|----------|----------|----|-------------|----------|----------|----|-------------|----------|----------|
|                          | **Train ↓**| **Test ↓** | **Acc. ↑** | │  | **Train ↓** | **Test ↓** | **Acc. ↑** | │  | **Train ↓** | **Test ↓** | **Acc. ↑** |
| RACE (P=2,L=2)           | **80.5s**  | 3.9s     | 70.3%     | │  | **282s**    | 15s      | 89.4%     | │  | **561s**    | 22s      | 97.14%    |
| RACE (P=3,L=3)           | 82.4s      | 4.0s     | **71.3%**     | │  | 289s        | 15.6s    | 90.6%     | │  | 584s        | 22.5s    | **97.92%**    |
| RACE (P=4,L=4)           | 84.7s      | 4.1s     | 70.8%     | │  | 305s        | 16s      | **91.1%**     | │  | 594s        | 22.9s    | 97.45%    |
| Linear                   | 83.8s      | 4.0s     | 67.9%     | │  | 286s        | 15.9s    | 87.3%     | │  | 591s        | 22.8s    | 96.35%    |
| Linformer-128            | 86s        | **3.2s** | 64.1%     | │  | 296s        | **10.7s**| 87.5%     | │  | 616s        | **15.2s**| 97.4%     |
| Performer-256            | 128s       | 5.8s     | 68.9%     | │  | 449s        | 24.6s    | 86.5%     | │  | 952s        | 35s      | 96.61%    |
| FlashAttention2          | 95.7s      | 3.7s     | 69.8%     | │  | 471s        | 20s      | 89.7%     | │  | 1645s       | 47s      | 97%       |

## Response (2/2) to Reviewer W9FA  — Authors

> Q2. Reporting additional experiments (continued...)

**Table**: Long-context (16K) image classification (ViT) performance on a 40GB A100 GPU with Food-101 dataset. Train/Test denote per-epoch runtimes in seconds, and Acc. denotes test accuracy. *Linear Attention, Linformer, and Performer use batch size = 1 due to OOM at batch size = 8. RACE and FlashAttention2 remain memory-efficient and use batch size = 8.*

| **Method**            | **Train ↓** | **Test ↓** | **Acc. ↑** |
|-----------------------|-------------|------------|------------|
| RACE (P=2, L=2)   | **891s**    | **37s**    | 42.4%      |
| RACE (P=3, L=3)       | 950s        | 40s        | **43.5%**  |
| RACE (P=4, L=4)       | 1042s       | 42s        | 40.3%      |
| Linear                | 1166s       | 44s        | 41.4%      |
| Linformer-128         | 1250s       | 49s        | 20.2%      |
| Performer-256         | 2546s       | 105s       | 42.4%      |
| FlashAttention2       | 2600s       | 95s        | 42.1%      |

We add the aforementioned table in the updated paper as Table 9 in Appendix.

> Q3. For fair comparison, do efficiency results need accuracy context and vice-versa?

A3. We thank the reviewer for raising this point. Our efficiency experiments follow well-established practice in prior work on scalable attention mechanisms, including HyperAttention [5] (see fig. 4 in [5]) and Performer [6] (see fig. 3 in [6]), and we take care to avoid hyperparameter tuning that would artificially favor runtime or memory measurements.

In our accuracy evaluations, for RACE Attention, we fix the hyperparameters across context lengths 128-64K, i.e., $P,L \in$ {$2,3,4,5$}. The base configuration is identical in all methods per task (see Table 8). In the context of scaling experiments, for RACE we use the same $P,L \in$ {$2,3,4,5$}, and use identical configuration across all methods (e.g., embedding dimension, number of heads, batch size). We do _not_ re-tune any method to optimize speed or memory.

While linear attentions can appear arbitrarily efficient by collapsing feature dimensions (e.g., setting the feature dimension to 1), such configurations severely degrade accuracy and are not used in meaningful applications. To illustrate this, we trained the Linear Attention baseline on CIFAR-10 dataset with feature dimension 10 and observed a drop in accuracy from $\sim 60$% to $\sim 50$%. These degenerate settings are exactly what our fixed-hyperparameter protocol is designed to avoid. Moreover, RACE Attention scales linearly with embedding dimension $d$, whereas Linear Attention scales quadratically in $d$, further justifying the controlled comparison.

Under these accuracy-preserving settings, RACE Attention performs comparably to linear attentions for context lengths up to 64K. The meaningful difference in efficiency emerges beyond 128K tokens, where RACE continues to scale stably while many existing methods become infeasible. Training full models at 2-4M token context lengths is prohibitive due to memory and runtime limits, and our goal is to highlight this ultra-long-context regime, where RACE remains tractable _without any hyperparameter tuning_.

Finally, due to limited compute resources, training full models at extremely long contexts (128K+) is challenging universally. Nevertheless, our aim is twofold: (1) to rethink attention mechanisms for extreme long-context training, and (2) to demonstrate that RACE can operate in a regime where other attention mechanisms cannot. For these reasons, we believe the efficiency numbers in figs. 4-6 reliably reflect the practical scalability of RACE. We appreciate the opportunity to clarify this point. We explicitly note the hyperparameters of RACE in captions of figs. 4–6, which are same across all accuracy evaluations.


**[5] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruff, Amir Zandieh. HyperAttention: Long-Context Attention in Near-Linear Time. ICLR 2024.**

**[6] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller. Rethinking Attention with Performers. ICLR 2021.**


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
