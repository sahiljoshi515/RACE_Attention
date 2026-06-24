# Author cross-cutting posts

Author comments addressed to all reviewers / the Area Chair (not a single reviewer thread).

## Response to all Reviewers

First of all, we would like to thank all the Reviewers for carefully reading our paper and for their insightful comments. We have updated the paper to incorporate the following revisions according to Reviewers' feedback:

- **Discussion about YOSO Attention:** We outline the key differences and provide an in-depth comparison between the estimators of RACE and YOSO [1] on lines 51-59 and 263-272. They both use LSH, but their mechanisms differ substantially. While YOSO uses hard Bernoulli sampling from collisions, RACE uses a novel, differentiable estimator of a $P$-powered angular kernel ([2], [3]). Secondly, YOSO provides no formal error bounds. In contrast, RACE offers explicit bias/variance guarantees and convergence analysis (Theorem 2). Finally, YOSO's experiments are limited to  bidirectional attention up to 4K tokens, while RACE provides accuracy up to 64K tokens, supports full/causal attention, and evaluates causal language modeling task on WikiText-103 (standard benchmark). Thus, RACE offers a distinct kernel framework, stronger theory, and broader applicability. For a detailed explanation, we refer the reader to our response to Q6 of Reviewer if3U.

- **Additional long-context text classification and image classification experiments:** We conduct extensive experiments on substantially longer contexts to re-validate the efficiency and performance of RACE Attention. Specifically, we report results at **16K**, **32K**, and **64K** sequence lengths on the ArXiv classification task, including both training and inference time per epoch on a single A100 GPU. We additionally evaluate long-context image classification on Food-101 dataset using Vision Transformers at **16K** sequence length. Across all settings, RACE Attention matches or outperforms the baselines while delivering faster runtime. Please refer to Tables 7, 9.

- **Scaling experiments with Sigmoid Attention and FlashAttention3 on GH200 (96 GB):** We add two new baselines to our scaling experiments as suggested by Reviewer eQBU. The efficiency of RACE is clear: **CPU-RACE is up to 20x faster, and GPU-RACE is up to 2500x faster than FlashAttention-3 at 4M context length.** Please refer to Tables 5, 6, 7, and 8.

- **Significance of the Scaling experiments:** To clarify unambiguously, our scaling experiments stress-test only the attention mechanism itself, not end-to-end training of a full Transformer. This follows well-established practice in prior scalable-attention works such as HyperAttention [4] and Performer [5]. This decomposition is intentional: at extreme context lengths, attention dominates both memory and compute, and many alternative mechanisms fail well before reaching this regime. Demonstrating that attention alone can scale to tens of millions of tokens is therefore a necessary prerequisite for training full models at such lengths. Note that our stress tests benchmark a single forward–backward pass of a multi-head attention layer under fixed hyperparameters **(identical to those used in our accuracy evaluations)**, ensuring no hyperparameter re-tuning is done to favor speed or memory.

- **Polishing the paper:** We clarify how the feature maps $\phi(Q_i)$, $\phi(K_j)$ relate to Angular Attention and explain the role of the softmax operation in Algorithm 1 (lines 300-312). Additionally, we include fig. 2 to illustrate how a modest increase in the sharpening parameter $\gamma$ in Angular Attention can closely mimic Softmax Attention. Finally, we add a concluding paragraph outlining future directions for RACE Attention, and following Reviewer if3U's suggestion, we have revised the paper’s title to **"RACE Attention: A Linear-Time Attention Mechanism for Long-Sequence Training with Extreme-Length Attention-Layer Scaling"** which reflects the main contributions.

Please refer to the individual reviewer responses for detailed explanations. Each reviewer response has its own set of references.

**References:**

**[1] Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Fung, Vikas Singh. You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling. ICML 2021.**

**[2] Benjamin Coleman, Anshumali Shrivastava. Sub-linear RACE Sketches for Approximate Kernel Density Estimation on Streaming Data. WWW 2020.**

**[3] Benjamin Coleman, Richard G Baraniuk, Anshumali Shrivastava. Sub-linear Memory Sketches for Near Neighbor Search on Streaming Data. ICML 2020.**

**[4] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruff, Amir Zandieh. HyperAttention: Long-Context Attention in Near-Linear Time. ICLR 2024.**

**[5] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller. Rethinking Attention with Performers. ICLR 2021.**

## Summary for Area Chair(s): Reviewers' Concerns and Our Clarifications

Given the recent OpenReview incident and the interruption of the discussion phase, we would like to provide a concise summary of the review process for our submission. Our intention is to offer a clear overview of the reviewers’ concerns and how we addressed all of them during the rebuttal period. Finally, we highlight the novelty of our work, emphasizing its contributions to scalable and accurate long-context attention.

### **Reviewer W9FA:**

**1. Comparison with YOSO**: The reviewer noted that our original submission did not sufficiently explain how RACE Attention differs from YOSO, despite shared motivation around LSH-based angular kernels. In response, we substantially expanded the discussion (lines 51–59 and 263–272), clarifying that YOSO relies on non-differentiable hard LSH, surrogate gradients, and estimates only the unnormalized kernel numerator via Bernoulli collisions and then performs post-hoc normalization. RACE instead uses a smooth, differentiable relaxation of RACE hashing to approximate both numerator and denominator with formal bias–variance guarantees. We also highlighted that RACE supports both causal and bidirectional attention and is evaluated up to 64K tokens, whereas YOSO reports results only up to 4K, positioning RACE as a more stable and scalable refinement.

**2. Experiments Beyond 8K Contexts:** The reviewer requested evaluation at longer contexts. We extended experiments to 16K, 32K, and 64K variants on ArXiv classification (Table 7) and added 16K long-context ViT experiments (Table 9). All methods were run with matched hyperparameters to ensure clean scaling comparisons. These results demonstrate that RACE maintains accuracy while scaling strictly linearly.

**3. Accuracy–Efficiency Tradeoffs:** The reviewer emphasized that efficiency plots can be misleading if methods use degenerate hyperparameters. We clarified that all scaling experiments follow accuracy-preserving configurations consistent with existing long-context literature (e.g., HyperAttention, Performer). Hyperparameters were held fixed across all methods and context lengths, and we explicitly documented these settings in captions. Under these controlled settings, RACE matches baselines up to 64K and remains scalable at extreme lengths where others run out of memory.

### **Reviewer eQBU:**

**1. Additional Baselines:** The reviewer requested inclusion of FlashAttention3 and Sigmoid Attention. We added both baselines to the accuracy and scaling comparisons, validating that RACE Attention maintains stable accuracy while continuing to scale efficiently even as the context length increases.

**2. Softmax in Algorithm 1:** The reviewer was concerned that Algorithm 1 appeared to reintroduce softmax. We clarified that the operation is not the Transformer softmax but a local smoothing machinery ensuring differentiable LSH bucketization. 

**3. Connection Between Feature Maps and Angular Attention:** We added a complete derivation showing how our soft feature map arises from the $P$-powered angular kernel via LSH and RACE sketches, including collision probabilities and propagation to the output in lines 300-313.

**4. Understanding the Sensitivity of the Sharpening Parameter $\gamma$:** We clarified that $\gamma$ is not tunable but fixed by the number of hyperplanes ($\gamma=P$). We also added a Frobenius-norm analysis showing that approximation error $||angular–softmax||_F$ decreases as $\gamma$ increases, and that accurate approximations require only modest polynomial degrees in Fig. 2.

### **Reviewer if3U:**

**1. Title Framing and Interpretation of Scaling Experiments:** The reviewer sought clarity on whether the 75M token results reflect full-model capability or single-layer stress tests. We emphasized that these results reflect one attention layer’s forward-backward pass. To avoid misinterpretation, we revised the title to more precisely reflect the work’s scope: a linear-time attention mechanism scalable to extreme-length sequences, not full-model context windows.

**2. Causal Masking:** This was a minor clarification from the reviewer regarding whether causal masking is straightforward to incorporate. We explained that while we were able to implement the causal RACE Algorithm (Algorithm 2), the full theoretical analysis is non-trivial and outside the scope of this paper. For completeness, we refer the AC(s) to our detailed response to Q4 for reviewer if3U.

**3. YOSO Comparison:** We already outline this above for reviewer W9FA.

### **Novelty of Our Work** 
RACE Attention introduces the first differentiable sketch for angular-kernel attention, enabling end-to-end training while scaling to unprecedented context lengths (75M tokens on CPU, 12M on GPU) under accuracy-preserving settings. Beyond this, RACE offers formal approximation guarantees, efficient CPU/GPU kernels for causal and non-causal training, and extensive ablations that demonstrate stable accuracy alongside linear runtime and memory complexity.

## Paper Revision Summary

We thank all the reviewers and the area chair(s) for their time and constructive feedback in helping us improve the paper. We are encouraged that reviewers found the problem both important and timely (R **W9FA**, R **eQBU**, R **if3U**), appreciated the strength of our scaling experiments, the systems-level implementation, and the rigor of our theoretical analysis (R **eQBU**, R **if3U**). We are also grateful that reviewers highlighted the algorithm’s broad applicability, simple implementation, and its potential usefulness to the community (R **W9FA**, R **if3U**).

We have updated the paper to address all reviewers' concerns. The major revisions are summarized below:

**1. [W9FA, if3U]:** clarified the technical and theoretical novelty of RACE relative to YOSO in lines 51–59 and 263–272.   
**2. [W9FA]:** expanded the evaluation to sequence lengths $8\times$ larger than the original experiments, as shown in Tables 7 and 9 for text and image classification.   
**3. [W9FA]:** explained the correlation between efficiency and accuracy by stating that the scaling experiments employ exactly the same hyperparameters used in the accuracy experiments, as detailed in Figures 4, 5, and 6.  
**4. [eQBU]:** added FlashAttention-3 and Sigmoid Attention as additional baselines in both the accuracy and scaling tables.   
**5. [eQBU]:** expanded the explanation of the softmax operation in Algorithm 1 and articulated the linkage between soft feature maps $(\phi(Q_i), \phi(K_j))$ and angular attention in lines 300–313.   
**6. [eQBU]:** included a new plot demonstrating that the Frobenius error between the angular and softmax formulations decreases with increasing $\gamma$ (Figure 2).   
**7. [if3U]:** revised the title to better capture the core contributions of this work: strictly linear complexity in sequence length $N$ and dimension $d$, and the ability to scale the attention primitive to extreme sequence lengths.   
**8. [if3U]:** added the causal masking variant of RACE (Algorithm 2) to the Appendix to illustrate its implementation.   

**For a detailed summary of the full discussion, please refer to the next response.**


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
