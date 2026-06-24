# Figure index (arXiv-2510.04008v5/figs/)

| File | Shows | Paper ref |
| --- | --- | --- |
| `Comparing_Softmax_and_RACE_Attention.png` | Linear RACE vs quadratic softmax; how $o_5$ is computed (full column vs LSH buckets) | Fig 1, L1113 |
| `angular_softmax.png` | Frobenius error between Angular and Softmax Attention vs $\gamma$ | Fig (frobenius), L1226 |
| `Heatmaps.png` | Softmax vs Angular kernels at increasing $\gamma$ (flat $\to$ sharp) | Fig (heatmap), L1263 |
| `RACE_GPU_5/6/7.png` | GPU scaling for RACE (P=2,L=2 / 3,3 / 4,4) | Fig scaling a-c, L1406-1421 |
| `RACE_GPU_1.png` | GPU scaling (RACE vs baselines) | Fig scaling d, L1425 |
| `FLASH_RACE_1.png` | CPU-RACE vs GPU baselines (algorithm-beats-hardware) | Fig flash-race, L1435 |
| `RACE_CPU_4.png` | CPU scaling to 75M tokens | Fig cpu4, L1441 |
| `Speedup_GPU.png` / `Speedup_CPU.png` | RACE speedup vs GPU baselines (RACE on GPU / CPU) | Fig speedup a/b, L1581-1588 |
| `yoso_vs_race.png` | YOSO CUDA kernel fails > 32K tokens; RACE keeps scaling | Fig yoso, L1593 |
| `Race_Attention_Flowchart.png` | Full RACE pipeline $Q,K,V \to$ soft-hash $\to$ bucket summaries $\to O$ | Fig complete-figure, L2677 |
| `Embedding.png`, `Query_attending_to_keys.png` | Intuition diagrams (soft bucketization, query-key mixing) | Appendix intuition figs |
| `RACE_CPU_*.png`, `RACE_GPU_*.png` (others) | Additional per-config scaling curves | Appendix |

These are images; the helper's text search won't index them. Open the PNGs directly under
`arXiv-2510.04008v5/figs/` when a visual is needed.

---
Source: arXiv-2510.04008v5/main.tex figure environments; file list from arXiv-2510.04008v5/figs/.
