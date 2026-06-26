# Full Llama-3.2-3B vs RACE-hybrid — forward latency with genuine FlashAttention-3

**Question.** With **genuine FlashAttention-3** as the softmax baseline, how much faster is a
RACE-hybrid at long context, where does it pay off, and **how does the speedup scale with the fraction
of attention layers linearized** (14/28 vs 21/28)?

**Hardware/software.** 1× NVIDIA **H200** (141 GB), SLURM `commons`. `swa_env` = **torch 2.8.0+cu128,
transformers 4.57.0**, genuine **FlashAttention-3** (`flash_attn_3` 3.0.0, Dao Hopper kernel). RACE
CUDA kernel JIT-built against torch 2.8 (`.torch_ext_fa3`). Jobs `144436` (alt), `144441` (srrr).

**What is timed.** The decoder stack `model.model(input_ids)` only (lm_head excluded — identical for
all and 269 GB at 1M), **forward-only / prefill** (`no_grad`, `use_cache=False`), **B=1, bf16**,
median over a few iters (adaptive: warm2/it3 ≤128K, warm1/it2 at 256K–512K, **warm1/it1 at 1M**).

- **full** — all 28 attention layers = FA3.
- **Layout A — `alt`** (`run_fwd_fa3.sbatch`): RACE on the 14 **odd** layers (1,3,…,27); the 14 even
  layers stay FA3. → **14/28 linearized.**
- **Layout B — `srrr`** (`run_fwd_fa3_srrr.sbatch`): **(Softmax, RACE, RACE, RACE)** repeating —
  softmax kept at layers **0,4,8,12,16,20,24** (7 FA3), RACE on the other **21**. → **21/28 linearized.**
- Both layouts: replaced layers run `RaceLlamaAttention` (custom CUDA `RaceCausalFn`); S=8 → (L,K)=(2,2),
  S=24 → (L,K)=(3,3). Toggle with `bench_forward.py --pattern {alt,srrr}`.

---

## Layout A — alternating (14/28 RACE): results (ms; speedup = full ÷ hybrid)

| T | full FA3 | hybrid_S8 | **S8 ×** | hybrid_S24 | **S24 ×** |
|---|---:|---:|:--:|---:|:--:|
| 4K   | 59.2     | 91.1     | 0.65 | 92.1     | 0.64 |
| 8K   | 120.9    | 130.9    | 0.92 | 146.8    | 0.82 |
| 16K  | 272.2    | 263.3    | **1.03** | 295.4    | 0.92 |
| 32K  | 681.7    | 595.0    | 1.15 | 651.3    | **1.05** |
| 64K  | 1 910.0  | 1 443.4  | 1.32 | 1 554.6  | 1.23 |
| 128K | 6 002.0  | 3 976.4  | 1.51 | 4 223.4  | 1.42 |
| 256K | 20 810.4 | 12 312.8 | 1.69 | 12 815.1 | 1.62 |
| 512K | 76 600.0 | 41 612.7 | 1.84 | 42 966.4 | 1.78 |
| **1M** | **291 457.9** | **151 619.4** | **1.92** | **154 033.9** | **1.89** |

Raw: `results/fwd_latency_fa3.csv`. Plot: `results/fwd_latency_fa3.png`.
(FA3 also nearly **halves the full-model time vs sdpa**: 512K full = 76.6 s FA3 vs 158.6 s sdpa.)

**Crossover (hybrid first beats full):** **16K** for S8, **32K** for S24.

---

## Layout B — (Softmax, RACE, RACE, RACE), 21/28 RACE: results

| T | full FA3 | srrr_S8 | **S8 ×** | srrr_S24 | **S24 ×** |
|---|---:|---:|:--:|---:|:--:|
| 4K   | 59.2     | 108.0    | 0.55 | 109.4    | 0.54 |
| 8K   | 121.4    | 137.6    | 0.88 | 163.4    | 0.74 |
| 16K  | 275.5    | 259.6    | **1.06** | 310.4    | 0.89 |
| 32K  | 681.5    | 558.4    | 1.22 | 638.2    | **1.07** |
| 64K  | 1 902.2  | 1 224.4  | 1.55 | 1 397.2  | 1.36 |
| 128K | 5 995.8  | 3 009.2  | 1.99 | 3 387.3  | 1.77 |
| 256K | 20 786.8 | 8 251.9  | 2.52 | 9 097.4  | 2.29 |
| 512K | 76 621.5 | 25 172.6 | 3.04 | 27 150.2 | 2.82 |
| **1M** | **291 387.1** | **84 118.7** | **3.46** | **88 631.1** | **3.29** |

Raw: `results/fwd_latency_fa3_srrr.csv`. Plots: `results/fwd_latency_fa3_srrr.png`,
head-to-head `results/fwd_latency_fa3_compare.png`. The `full` baseline reproduces Layout A's to
**≤1.2 %** (same code path), so the cross-layout comparison is fair. **Crossover: 16K (S8) / 32K (S24).**

### srrr vs alt — how much the extra 7 RACE layers buy (alt ÷ srrr)

| T | 64K | 128K | 256K | 512K | **1M** |
|---|--:|--:|--:|--:|--:|
| **S8**  | 1.18× | 1.32× | 1.49× | 1.65× | **1.80×** |
| **S24** | 1.11× | 1.25× | 1.41× | 1.58× | **1.74×** |

srrr is *slower* than alt below ~16K (S8) / ~32K (S24) — the extra RACE layers add overhead the
quadratic savings don't yet cover.

---

## What the numbers mean (and don't)

1. **The hybrid is a long-context win only.** Below the crossover it is *slower* than full FA3 —
   ~0.65× at 4K (≈54 % slower), and S24 is still 0.92× at 16K. The RACE path's per-layer overhead
   (two soft-hash softmaxes over S buckets, an fp32 prefix-scan kernel) isn't repaid until attention
   dominates total runtime. **Quote 1.9× only with its context: it is a ≥256K-token prefill figure.**

2. **Both layouts give a bounded constant-factor speedup — NOT linear / not sub-quadratic.** The
   softmax layers that remain (14 for alt, **7 for srrr**) keep the model **O(T²)**. Quadratic-coefficient
   fits (`ms = a·T² + b·T`, long T) give `a_full / a_hybrid ≈` **2.03 (alt)** and **≈4.0 (srrr)** — i.e.
   the speedup ceiling is `28 / (FA3 layers kept)` = **2× (alt) / 4× (srrr)**. Large-T log-log slopes are
   all super-linear (full ≈1.93; srrr ≈1.6 — still >1, not linear). At 1M, observed speedups are
   **96 % (alt 1.92/2)** and **87 % (srrr 3.46/4)** of their ceilings, approached *from below* and
   beginning to saturate. The ceilings are nominal upper bounds — a RACE layer still costs ~7 % of an FA3
   layer at 1M, so the true asymptote is slightly under 2×/4×. **A *fully* RACE model (0 FA3 layers) is
   what would give true linear scaling.**

3. **Speedup scales with the linearized fraction, as expected.** Going from 14/28 → 21/28 RACE layers
   lifts the 1M speedup from 1.92× → **3.46×** (S8) — the srrr-vs-alt ratio (1.80× at 1M, S8) tracks the
   `14/7 = 2×` layer-count prediction (90 % of it). The trade: srrr is a touch *slower* than alt at short
   T and loads more RACE modules.

4. **Memory is not a differentiator here.** Peak memory is byte-identical across full/hybrid at every T
   (≈86 GB at 1M, ~39 % headroom on the H200) — all are O(T) memory (FA3 never materializes the T²
   score matrix). Do not read the `mem_gb` column as a memory saving; srrr's only memory cost is a static
   **+0.7 GB** for loading 21 vs 14 RACE modules.

---

## Verification (adversarial multi-agent audit)

Independent auditor agents recomputed everything from the raw CSV/log/code for **both** layouts (4
agents for alt, 2 for srrr — all PASS on integrity):

- **Genuine FA3 confirmed (no silent fallback).** Both runs: preflight asserts `is_flash_attn_3_available`
  + a real `flash_attn_func` forward; `config._attn_implementation=flash_attention_3`; transformers' FA3
  dispatch path has no try/except fallback; O(T) memory + O(T²) latency = unmistakable flash-kernel
  signature (eager would OOM). alt agrees with the independent 3-point FA3 smoke to <1.6 %; srrr's `full`
  baseline reproduces alt's to ≤1.2 %.
- **Layout integrity confirmed.** Replaced layers → `RaceCausalFn` → `race_fused_fwd` (true O(T) causal
  prefix scan, no T² matrix, no sequence-axis softmax); `race_cuda.so` built/loaded. The .out logs the
  exact partition: alt = 14 odd; **srrr = 21 RACE, softmax kept at [0,4,8,12,16,20,24]** (matches
  `srrr_layers(i)=i%4!=0`). S=8/S=24 labels correct.
- **Numbers reproduce exactly** — alt 1M 1.92×/1.89×; **srrr 1M 3.46×/3.29× vs full, 1.80×/1.74× vs alt**;
  quad-coef ratios 2.03 (alt) / 4.0 (srrr); 27/27 rows each, strictly monotone, no NaN; crossovers 16K/32K.

All "concern" verdicts were **framing/caveats, not arithmetic errors** (points 1–4 above). One precision
fix from the audit: the "~1.8× vs alt" headline is **S8-specific** (S24 = 1.74×).

**Honest caveats to carry forward:** prefill-only (no KV-cache/decode); the **1M point is single-shot**
(it=1) — within 3.4 % of an O(T^1.9) extrapolation and robust to ±10 % jitter, but n=1; 256K/512K
"median of 2" reports the slower iter (conservative); RACE core runs **fp32** internally while FA3 runs
bf16 (conservative for the speedup); single GPU, B=1, one run, no error bars.

---

## Reproduce

```bash
cd /scratch/sj157/RACE_Attention
sbatch distill/run_fwd_fa3.sbatch          # Layout A (alt, 14/28), genuine FA3, exps 12..20
sbatch distill/run_fwd_fa3_srrr.sbatch     # Layout B (srrr, 21/28)
# or directly: bench_forward.py --attn-impl flash_attention_3 --pattern {alt,srrr}
PY=/scratch/sj157/race_vit_env/bin/python
$PY distill/plot_forward.py  distill/results/fwd_latency_fa3_srrr.csv   # per-layout plot
$PY distill/plot_patterns.py                                            # head-to-head + tables
```

**Bottom line.** With genuine FA3 as the baseline, the forward speedup scales with how many attention
layers you linearize. **Alternating (14/28 RACE): ~1.9× at 1M** (bounded ~2× ceiling). **(Softmax,RACE,
RACE,RACE) (21/28 RACE): 3.46× (S8) / 3.29× (S24) at 1M** — 1.80× faster than alternating, bounded by a
~4× ceiling (28/7). Both cross over at **16K/32K** and are slower below that, and both remain **O(T²)**
(the kept softmax layers dominate at long T) — a constant-factor win, not a complexity change. True
linear scaling needs *all* attention layers replaced.
