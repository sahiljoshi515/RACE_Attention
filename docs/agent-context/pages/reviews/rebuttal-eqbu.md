# Rebuttal thread - Reviewer_eQBU

Threaded discussion between the authors and Reviewer_eQBU (chronological).

## Response (1/2) to Reviewer eQBU  — Authors

> Q1. Comparison with additional baselines.

A1. Thank you for raising this important point. We appreciate the reviewer for highlighting the ambiguity in our FlashAttention baselines. First, we clarify that all our experiments use **FlashAttention-2** as the GPU baseline. In the revised paper, we additionally include results for **FlashAttention-3** and **Sigmoid Attention** in the scaling comparisons (figs. 5–6). For the accuracy evaluations (tables 1, 2, 4), we report Sigmoid Attention, and omit FlashAttention-3 since it achieves the same accuracy as FlashAttention-2. We also report how our RACE Attention achieves significant speedups on CPU and GPU over these baselines in figs. 7 and 8 in the Appendix, following the evaluation style of [1] (see fig. 4 in [1]).

Although FlashAttention-2/3 and Sigmoid Attention are highly optimized kernels, they still scale quadratically with sequence length and thus become impractical in the long-context regime. We omit training Sigmoid Attention at extremely long lengths (8K-64K), as it is quadratic regardless of implementation, and FlashAttention-2 already serves as a representative quadratic baseline. In contrast, RACE Attention continues to scale efficiently far beyond the maximum sequence lengths reachable by these methods.

**[1] Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David P. Woodruff, Amir Zandieh. HyperAttention: Long-Context Attention in Near-Linear Time. ICLR 2024.**  

> Q2. The use of Softmax in Algorithm 1.

A2. Thank you for raising this important clarification. Our use of softmax in Algorithm 1 (see step 4) is not the softmax attention mechanism used in standard Transformers. In our Soft RACE, the softmax operation appears only as a local smoothing machinery to make the LSH bucket assignments differentiable. It replaces the hard {-1,1} indicator of a bucket with a continuous distribution over the $R=2^P$ corners, capturing how a token’s mass is shared among the $R$ buckets. Therefore, we emphasize that the softmax step in Algorithm 1 has nothing to do with the exponential weighting of queries and keys in softmax attention. The actual similarity function we approximate is the powered angular kernel. The connection to RACE arises because classical RACE sketches ([2], [3]) estimate powers of LSH collision kernels, and Soft RACE preserves this structure while making the assignments differentiable through a `softmax+tanh` operation. Thus, the "softmax" in Algorithm 1 serves only to provide soft bucketization, not softmax attention per se. We appreciate the opportunity to clarify this distinction and have added a paragraph explaining this concept on lines 290-298.

**[2] Benjamin Coleman, Anshumali Shrivastava. Sub-linear RACE Sketches for Approximate Kernel Density Estimation on Streaming Data. WWW 2020.**

**[3] Benjamin Coleman, Richard G Baraniuk, Anshumali Shrivastava. Sub-linear Memory Sketches for Near Neighbor Search on Streaming Data. ICML 2020.**

## Response (2/2) to Reviewer eQBU  — Authors

> Q3. The connection between feature maps and Angular Attention.

Thank you for this insightful question. Here we clarify how our soft feature map $\phi$ connects to the $P$-powered angular kernel. A classical RACE sketch draws $P$ random hyperplanes in $\mathbb{R}^d$. Each hyperplane has two sides, so together they create a sign pattern in {$\\pm 1$}$^P$ that describes on which side of each hyperplane a point lies. Stacking the hyperplanes into $ W \in \mathbb{R}^{P \times d}$ and for any $x\in\mathbb{R}^{d}$, the simple "hard" feature map is defined by 
$ \phi_{\text {hard }}(x):=\operatorname{sign}(W x) \in \\{\pm1\\}^P.
$ 

Two points $Q_i$ and $K_j$ share the same feature-map output when they fall on the same side of every hashing hyperplane; for angular LSH (SimHash), this occurs with probability

$$
\Pr\left[\phi_{\text{hard}}(Q_i)=\phi_{\text{hard}}(K_j)\right]
= S_{ij}
:= \left(1 - \frac{\cos^{-1}\left(\frac{Q_i^\top K_j}{\lVert Q_i\rVert\,\lVert K_j\rVert}\right)}{\pi}\right)^P,
$$

which is exactly the $P$-powered angular kernel. This gives the ideal $N \times N$ similarity matrix $S$ with $S_{ij}$ defined above. Soft RACE keeps this geometric structure intact, but avoids making a hard $\pm 1$ assignment. Instead, we compute a soft sign vector $\tanh (W x) \in[-1,1]^P$, which tells us how strongly $x$ lies on each side of each hyperplane. We then compare this soft sign vector to all $2^P$ possible {$\\pm 1$}$^P$ patterns and convert the similarities into a probability distribution (via softmax). This produces the soft feature map $\phi(x) \in \mathbb{R}^{R},$ which spreads weight over the regions that best match the smoothed signs of $x$. Consequently,
$\phi(Q_i)^{\top} \phi(K_j)$
acts as a soft analogous to the collision event $\phi_{\text {hard }}(Q_i)=\phi_{\text {hard }}(K_j)$, structurally aligned with the same powered angular kernel. These soft similarities define 

$$ \hat{S}_{i j}^{(\ell)}=\phi^{(\ell)}\left(Q_i\right)^{\top} \phi^{(\ell)}\left(K_j\right),$$

which are then averaged across $L$ tables to obtain $\widehat S =\tfrac{1}{L}\sum_{\ell=1}^L \widehat S^{(\ell)}$.
Theorem 2 then shows how this approximate kernel $\hat{S}$ propagates to the final attention output $\hat{O}$. We included this discussion on lines 296-303 and 312-323.

> Q4. Details on how to choose $\gamma$ in Angular vs. RACE Attention.

A4. We thank the reviewer for this important question. In Angular Attention, the sharpening exponent $\gamma$ is an independent hyperparameter; one typically selects $\gamma$ by inspecting attention heatmaps (e.g., $\gamma = 8$-10 already yields softmax-level sharpness; see updated figs. 2, 3). In RACE Attention, however, $\gamma$ is not a free hyperparameter, instead RACE Attention has $P$ (number of hyperplanes) and $L$ (number of hash tables) as the hyperparameters.

A key contribution of our method is the construction of learnable _Soft RACE_, a smooth, differentiable relaxation of RACE hashing ([1], [2]) that enables a linear-time approximation to a powered angular kernel. In a classical RACE sketch, concatenating $P$ random hyperplanes yields collision probabilities proportional to the $P$-th power of the angular kernel. **Soft RACE preserves this property**, so the effective sharpening exponent emerges directly from the hash construction: $\gamma = P$.

Thus, RACE Attention offers a principled and significantly lower-dimensional design space. In all of our experiments, we restrict $P \in$ {$2,3,4,5$}, which consistently provides strong accuracy and induces an exponential-like sharpening on $[0,1]$, closely matching softmax behavior. Larger $P$ values further sharpen the kernel but are rarely necessary. Meanwhile, the number of hash tables $L$ independently controls variance, as formalized in Theorem 2. Therefore, $\gamma$ in the context of RACE Attention does not require a hyperparameter search. It is fully determined by the choice of $P$.

> Q5. Sensitivity analysis with respect to $\gamma$.

A5. We thank the reviewer for the helpful suggestion. Visualizing the sensitivity of $\gamma$ is indeed valuable, so we now plot the Frobenius error between Softmax and Angular Attention as a function of $\gamma$ in fig. 2 of the updated paper. We can see that the error sharply decreases as $\gamma$ increases, demonstrating that softmax-level sharpness can be achieved with modest polynomial degree (_e.g.,_ $\gamma = 8$).

## Acknowledging Authors' Rebuttal  — Reviewer_eQBU

Thank you for the detailed response and clarification. Most of my questions are resolved now, I raised my score.


---
Source: OpenReview forum RR8Lh8RHgA (ICLR 2026, Submission 22728). Regenerate: scripts/build_reviews.sh
