# Extreme-length scaling study

All scaling plots: a **single forward-backward pass of a single attention layer**, batch size 1,
4 heads, d = 128, log-log axes, RACE (P,L) chosen to match FlashAttention2 accuracy. **These are
attention-primitive limits, not full-model context lengths** (see `crosswalk/open-questions`).

## CPU (Intel Xeon Gold 5220R)

- RACE scales to **75M tokens** in one forward-backward pass.
- FlashAttention becomes prohibitively slow at ~2M tokens (quadratic; does not OOM on CPU DRAM).
  CPU "FlashAttention" baseline = PyTorch `F.scaled_dot_product_attention`.
- At ~33M tokens RACE is **>10,000$\times$** faster; RACE finishes <10s there while FlashAttention takes
  ~$10^5$ s. At **75M tokens RACE finishes in ~100s**.
- Linear-attention baselines run ~10$\times$ slower than RACE and OOM around ~33M tokens.

## GPU (NVIDIA GH200, 96 GB)

- RACE scales to **12M tokens**; FlashAttention-2/3 and Sigmoid become impractical around ~4M.
- At ~4M tokens RACE takes **~0.1s** vs FlashAttention2 ~550s $\to$ **~5500$\times$** faster; ~5000$\times$ vs
  Sigmoid; ~2600$\times$ vs FlashAttention3.
- Memory: exact attention stores $\mathcal{O}(B\cdot H\cdot N\cdot d)$ activations+grads; RACE compresses to
  $\mathcal{O}(B\cdot H\cdot L\cdot(NR+Rd))$, giving **~3.5$\times$ longer** contexts than FlashAttention-2/3 / Sigmoid.

## Algorithm beats hardware

Comparing RACE on a **single CPU** against FlashAttention-2/3 / Sigmoid on a **GH200 GPU**:

- For $N \lesssim 131\text{K}$, GPU parallelism wins (FlashAttention faster).
- Beyond ~131K, the quadratic dependence dominates: at ~4M tokens **RACE-on-CPU** is ~**40$\times$**
  faster than FlashAttention2/Sigmoid-on-GPU and ~**20$\times$** faster than FlashAttention3-on-GPU.

Takeaway: in the long-context regime, algorithmic efficiency beats hardware acceleration. The
conclusion points to KV-cache and optimized causal CUDA kernels as the inference follow-up - which
is exactly the `feat/vllm-race-attention-backend` direction (`codebase/vllm-backend`).

---
Source: arXiv-2510.04008v5/main.tex §Experiments scaling L1570-1612 (CPU L1573-1574; GPU L1601-1602; algorithm-vs-hardware L1605-1612).
