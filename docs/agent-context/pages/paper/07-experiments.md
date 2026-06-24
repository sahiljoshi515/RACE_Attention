# Experiments (accuracy)

**Setup:** standard efficient-attention suites. Text classification (QNLI, SST-2, IMDB, Yahoo,
Arxiv); autoregressive LM (WikiText-103, PTB); masked LM (Tiny Stories); image classification
(CIFAR-10, FashionMNIST, Food-101 with ViT); long-range reasoning (LRA: ListOps, Text Retrieval).
$\beta$ is **trained**. Same backbone, optimizer, schedule, batch size across methods; accuracy runs
on a single NVIDIA A100. Baselines: FlashAttention2, Linear, Performer, Linformer, Sigmoid, YOSO.
"FlashAttention-2/3" = exact fused softmax (used interchangeably for accuracy/runtime).

## Table 1 - Arxiv long-document classification (A100 40GB; Train/Test = per-epoch seconds)

| Method | 16K Acc | 32K Acc | 64K Acc | 64K Train | 64K Test |
| --- | --- | --- | --- | --- | --- |
| RACE (P=2,L=2) | 70.3% | 89.4% | 97.14% | **561s** | 22s |
| RACE (P=3,L=3) | **71.3%** | 90.6% | **97.92%** | 584s | 22.5s |
| RACE (P=4,L=4) | 70.8% | **91.1%** | 97.4% | 594s | 22.9s |
| Linear | 67.9% | 87.3% | 96.35% | 591s | 22.8s |
| Linformer-128 | 64.1% | 87.5% | 97.4% | 616s | **15.2s** |
| Performer-256 | 68.9% | 86.5% | 96.61% | 952s | 35s |
| FlashAttention2 | 69.8% | 89.7% | 97% | 1645s | 47s |

RACE leads accuracy at 16K/32K/64K while training ~3$\times$ faster than FlashAttention2 at 64K.

## Table 2 - Diverse tasks

| Method | CIFAR-10 @1024 Acc | QNLI @2048 Acc | Tiny Stories @1024 PPL |
| --- | --- | --- | --- |
| RACE (P=4,L=4) | **65.9** | 61.1 | 2.6 |
| FlashAttention2 | 61.44 | 61.1 | 2.7 |
| Linear | 60.0 | 60.7 | 7.0 |
| Performer-256 | 64.9 | 61.0 | 10.0 |
| Linformer-128 | 63.7 | 60.6 | 3.7 |
| Sigmoid | 57.2 | 61.1 | 3.7 |
| Angular ($\gamma=8$) | 61.69 | **61.7** | **2.5** |

(Angular $\gamma=8$ is the exact powered-angular kernel without sketching - an upper bound on what RACE
approximates.)

## Table 3 - Food-101 @ 16K (ViT image classification)

| Method | Train | Test | Acc |
| --- | --- | --- | --- |
| RACE (P=2,L=2) | **891s** | **37s** | 42.4% |
| RACE (P=3,L=3) | 950s | 40s | **43.5%** |
| RACE (P=4,L=4) | 1042s | 42s | 40.3% |
| Linear | 1166s | 44s | 41.4% |
| Linformer-128 | 1250s | 49s | 20.2% |
| Performer-256 | 2546s | 105s | 42.4% |
| FlashAttention2 | 2600s | 95s | 42.1% |

Note: Linear/Linformer/Performer ran at batch size 1 (OOM at bs=8); RACE and FlashAttention2 at bs=8.

## Language modeling

- **WikiText-103 @1024 (PPL):** RACE(P=4,L=4) = **20.9**, ties FlashAttention2 = 20.9; Angular($\gamma=8$) = 19.0.
- **PTB (PPL):** RACE matches/improves on softmax (Table tab:ptb; RACE(P=4,L=4) outperforms FlashAttention2).

Both LM tasks use **causal** RACE (Algorithm 2). Additional appendix results: FashionMNIST, Tiny
Stories @512, LRA, Yahoo/IMDB/SST-2 (see `paper/10-appendix-proofs` page note and Tables A*).

**Reviewer caveat:** accuracy benchmarks are mostly **short context (<8K besides Arxiv)** while
the pitch is very long context (W9FA weakness #2). See `crosswalk/concerns-tracker`.

---
Source: arXiv-2510.04008v5/main.tex §Experiments L1340-1612 (Table 1 L1344-1393; Table 2 L1470-1491; Table 3 L1502-1514; WikiText L1529-1541).
