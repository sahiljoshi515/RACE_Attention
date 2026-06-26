# RACE Attention — progress overview

_Status snapshot (updated 2026-06-26). For the published method see `README.md`; this file tracks
the long-context kernel + Llama-hybrid **distillation** work layered on top. **New here? Read the
"Current state — how to resume" section directly below — it is the authoritative handoff; the
sections after it are the historical record of how we got here.**_

**RACE Attention** (ICLR 2026) is a strictly **linear-time** attention: it replaces the quadratic
softmax(QKᵀ)V core with randomized soft-hash similarity aggregation (angular LSH buckets + a causal
prefix scan), as a drop-in for softmax attention.

---

## Repository map

| Path | What it is |
|---|---|
| `notebooks/` | Quick-start examples (classification, LM, MLM, ViT) — published artifact |
| `misc/` | Paper training scripts (gpt, lm, mlm, vit, classification, food-101, arxiv_64K) |
| `kernels/cpu/` , `kernels/gpu/` | RACE attention kernels — C++/CUDA (forward/backward + **decode** kernel) |
| `scaling/` | **Long-context kernel benchmarks** — causal RACE CUDA vs FlashAttention-3 (kernel level) |
| `distill/` | **Llama-3.2-3B → RACE-hybrid** — the staged distillation pipeline + eval + latency (main work) |

---

## ⭐ Current state — how to resume (2026-06-26)

**Goal:** distill `meta-llama/Llama-3.2-3B-Instruct` into a **RACE-hybrid** (some softmax-attention
layers replaced by linear-time RACE soft-LSH attention) that keeps accuracy while being faster for
life. Locked config: **AR/S24** = alternating, **14 of 28** attention layers → RACE with **L=3, K=3
(S=24 buckets)**. Constraints: configs ∈ AR(50%)/ARRR(75%) × {S8,S24}; **no learnable hash**.

### Best model so far → `distill/checkpoints/best/race_hybrid_AR_S24_p1f_unfreeze_step1340.pt`
Stage F (MLP-unfrozen). **Headline result: unfreezing the MLPs recovered BOTH general knowledge AND
long-context retrieval — the frozen base, not RACE's bucket readout, was the dominant limiter.**

| metric | frozen base (Stage C) | **Stage F (best)** | teacher |
|---|--:|--:|--:|
| MMLU (5-shot) | 24.7 | **34.5** | 62 |
| HellaSwag | 36.5 | **55.4** | 70.5 |
| Winogrande | 52.1 | **55.1** | 69.9 |
| RULER-16K | 22.9 | **54.6** | 80.5 |
| RULER-32K | 15.2 | **42.7** | 80.2 |
| ppl (teacher-forced) | ~21 | **6.2** | 15.2 |

`niah_single_1` 0→100@16K, `variable_tracking` 0→24, `niah_multivalue` 27→74; only `cwe` stays 0. All
curves were still rising at 351M tokens. The Stage-F `.pt` carries `base_state` (trained MLPs, ~5.6 GB);
`eval_ruler.build_model` / `eval_choice` load it automatically when present.

### The staged RADLADS-style recipe (each stage `--load-race-checkpoint`s the previous)
`A align` (FineWeb, hidden-MSE, lr 1e-4 — **not** 1e-3, that diverges since we replace attention) →
`B KD` (mixed FineWeb+synthetic-RULER, `0.1·hidden + 1·KL + 0.5·CE`, de-collapses: NIAH-4K 0→1.0, ppl
22.5) → `C ctx-ext` (4K→16K, RULER-16K 22.9) → **`F unfreeze`** (continue from C with `--unfreeze mlp`,
**base-LR 5e-5**, ~351M tok, plain mixed @4K — recovers both axes, table above). `G` (in flight) adds a
4K→16K→32K length **curriculum** on top of unfreeze (~600M tok). Loss objective lives in
`global_utils.py` (hidden MSE + `T²`-scaled `KL(teacher‖student)` + next-token CE); it is **logit/KD
distillation with dark knowledge**, T currently 1.

### How to run (cluster: H200, SLURM partition `commons`, account `as143`, `gres gpu:h200`)
| Action | Command (from `distill/`) | Env |
|---|---|---|
| **Train** (DDP) | `sbatch run_distill_ddp.sbatch` (prefix-env + `--export=ALL` for comma'd `--curriculum`) | `race_vit_env` (`source env.sh`), ext `.torch_ext_distill` |
| **Eval RULER** | `CKPT=<path> sbatch run_ruler_ckpt.sbatch` / `run_ruler_multi.sbatch` (16/32/64K) | `race_vit_env` |
| **Eval MMLU/HS/WG** | `CKPT=<path> sbatch run_eval_choice_ckpt.sbatch` (`--num-fewshot 5` for MMLU) | `race_vit_env` |
| **Eval LongBench** | `sbatch run_longbench.sbatch` (uses `refs/lb_*.json` + `refs/lb_metrics.py`) | `race_vit_env` |
| **Prefill/decode latency** | `sbatch run_bench_pd.sbatch` / `run_bench_pd_b8.sbatch` (FA3 softmax + RACE decode kernel) | **`swa_env`** (`source env_fa3.sh`), ext `.torch_ext_fa3` |
| **Regression tests** | `sbatch run_tests_{loss,train_mechanics,race_kernel,ckpt_rollout,distill_local}.sbatch` | `race_vit_env` |

Latency summary: hybrid is **faster at decode everywhere** (up to 1.86× @64K B=8) with lower memory;
prefill crossover ≈32K. Decode uses the fused **RACE decode kernel** (`kernels/gpu/decode_kernel.cu`,
`race_decode_build.py`, wrapped by `scaling/race_decode_cuda.py`).

### Gotchas (these will bite a fresh run)
- **FA3 only runs under `swa_env`** (torch 2.8). `race_vit_env`'s `flash_attn_3` `.so` is torch-2.8 ABI
  and crashes under torch 2.10. Latency benches must use `env_fa3.sh`; training uses `env.sh`.
- **Checkpoints/data/results are NOT in git** (see `.gitignore`) — they live on the cluster FS:
  `distill/checkpoints/` (238 GB), `data/fineweb` (staged parquet), `distill/results/`. A fresh clone
  has the **code only**; point evals at the existing `checkpoints/best/*.pt` on disk.
- `pg19` dataset is dead (datasets 4.x dropped script datasets) → use the **`fineweb_local`** source
  in `data_long.py` reading pre-staged parquet (`stage_fineweb.py` stages it).
- `sbatch --export=ALL,VAR=...` splits on commas inside `--curriculum`/`--task-weights` → use
  **prefix-env assignment** + bare `--export=ALL`.
- 32K teacher-KD OOMs without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **DDP must all-reduce base grads** when `--unfreeze` is on (fixed) — else MLPs diverge per-rank.
- `--unfreeze mlp` keeps embeddings + kept-softmax q/k/v/o **frozen** (preserves retrieval heads);
  ~2.5B/3.3B trainable, not a literal full unfreeze. Checkpoints then carry `base_state`.

### Code health
Distillation scripts (global + local) + the RACE CUDA kernel were **verified faithful & correct** by a
5-agent test workflow — zero critical/major bugs; custom backward matches fp64 to ~1e-6 (the earlier
fp16 reverse-subtraction backward bug is fixed). Reusable suite: `distill/tests_*.py` + `run_tests_*.sbatch`.

### In flight
- **Stage G** (`p1g_curriculum`, unfreeze + 4K→16K→32K curriculum, ~600M tok) — running; on completion
  run the full eval suite vs Stage F to see if the curriculum lifts long-context numbers.
- Open: `cwe` (frequency aggregation) still 0; push more tokens + retrieval-head placement; 4-config
  sweep only after AR/S24 clears.

Full writeup: `RACE2_DISTILLATION_REPORT.md`. Per-stage detail: `distill/checkpoints/best/MANIFEST.md`
(not in git — on disk) + the `distill/REPORT_*.md` files.

---

## Work done so far

### 1. `scaling/` — causal RACE CUDA kernels: correctness, a real bug fix, optimization, FA3 crossover
_Single attention core, B=1/H=8/D=128, bf16, causal, params (M,L,K) ∈ {(1,2,2),(1,3,3),(1,4,4)} → S=8/24/64. See `scaling/REPORT.md`._

- **Backward bug found & fixed.** Shipped v1 backward reconstructed prefix states by reverse-subtraction
  from fp16 finals → catastrophic cancellation; **Q/K grads wrong past ~1K tokens** (relerr >1 by 16K).
  New **forward-scan backward (v2)** is exact (~1e-7 at all T). gradV and forward were always fine.
- **Both kernels optimized** with a chunked parallel scan: forward **2.6–20× faster** at 1M; full
  fwd+bwd (v3) **2.4–7.2× faster** at 1M. Both remain exact — **734-check** suite passes.
- **Crossover vs FA3** (RACE O(T) vs FA3 O(T²)): inference 32K–64K, training 64K–256K; at 1M RACE is
  **13–90× (fwd) / 7–28× (fwd+bwd)** faster than FA3.

### 2. `distill/` — Llama-3.2-3B-Instruct hybrid (14 odd attention layers → causal RACE)
_28 layers, 24 Q / 8 KV heads, head_dim 128, GQA + llama3 RoPE. `RaceLlamaAttention` keeps q/k/v/o + RoPE + repeat_kv, swaps only softmax → RACE._

- **Distillation pilot (proof-of-concept, succeeds).** Teacher-forced local distillation of the RACE
  layers vs frozen Llama. Over 100 steps, per-layer attn/hidden MSE ↓ and cosine ↑ monotonically
  (hidden cosine 0.59–0.68 → **0.92–0.94**). Custom `race_backward` drives the q/k/v grads.
  See `distill/REPORT.md` (`results/metrics_S{8,24}.jsonl`).

- **Full-model forward latency with genuine FlashAttention-3** ✅ _(latest — `distill/REPORT_fwd_fa3.md`)_
  Full Llama (all attn = FA3) vs two RACE-hybrid **layouts**, T = **4K→1M**, decoder-stack forward-only,
  H200. `bench_forward.py --pattern {alt,srrr}`. Adversarially re-verified by auditor agents (genuine FA3
  + layout integrity confirmed; numbers reproduce exactly).

  | layout (RACE/28) | 64K | 256K | **1M speedup vs full** | ceiling (28 ÷ FA3-kept) |
  |---|:--:|:--:|:--:|:--:|
  | **alt** — odd layers (14) | 1.32× | 1.69× | **1.92× / 1.89×** (S8/S24) | 28/14 = 2× |
  | **srrr** — (S,R,R,R)×7 (21) | 1.55× | 2.52× | **3.46× / 3.29×** (S8/S24) | 28/7 = 4× |

  srrr (1 softmax + 3 RACE per block; softmax kept at 0,4,…,24) is **1.80× faster than alt at 1M (S8)**.
  **Both cross over at 16K (S8) / 32K (S24) and are slower below that.** The speedup is a **bounded
  constant factor** (≈96 % of the 2× / 87 % of the 4× ceiling at 1M), **not linear** — the kept softmax
  layers keep the model O(T²). A *fully* RACE model would be needed for true linear scaling.

- **Global hybrid-student distillation — ARRR (75% RACE)** ✅ _(latest — `distill/REPORT_global.md`)_
  From *local* teacher-forcing to a **global end-to-end rollout** (student runs on its own hidden states;
  RACE outputs feed later layers). `distill_global.py`, two model instances (frozen teacher + hybrid
  student), `--pattern` (`ARRR`/`AR`/`AAR`), loss = `1·hidden_MSE + 0.5·KL`, train RACE-only, B=2 +
  gradient checkpointing, 200 steps. 5-agent code review + 10-check GPU smoke (no-teacher-forcing,
  ckpt-grad parity) + 2-agent result audit — all pass.

  | run | hidden cos | KL | student ppl (teacher 22) | criteria |
  |---|:--:|:--:|:--:|:--:|
  | **ARRR (21/28 RACE)** | 0.077 → **0.324** | 6.35 → 3.53 | 10152 → **612** | 7/7 ✓ |
  | AR (14/28, comparison) | 0.106 → **0.523** | 6.55 → 1.90 | 13215 → **120** | 7/7 ✓ |

  **The 75% hybrid is globally trainable & stable** (no NaN, frozen base bit-identical, grads flow). It is
  *harder* than 50% and **not converged** (still rising at 200 steps; ppl far above teacher; below the
  *local* pilot, which is an easier objective). Per-layer cosine is a **valley** — front/lower-middle RACE
  layers worst (~0.17 @ L6–9), recovering toward the output (compounding through dense RACE blocks).

- **ARRR 1000-step + CE — LM-quality readiness gate** ✅ _(latest — `distill/REPORT_arrr_1k_ce.md`)_
  Longer ARRR run with **CE added** (`loss = 1·hidden + 0.5·KL + 0.1·CE`), 1000 steps, B=2, checkpoints
  @250/500/750/1000. Adds `--save-every`/`--grad-clip`, per-step ppl, eval min-layer-cosine, 5 plots incl.
  `compare_previous`. 2-agent code review + CE/ckpt smoke + result audit. **All 8 criteria met:**

  | metric | 200-step | **1000-step + CE** | teacher |
  |---|:--:|:--:|:--:|
  | student ppl | 612 | **146** (−76%) | 22 |
  | KL | 3.53 | **2.09** | — |
  | final-norm cos | 0.500 | **0.641** | — |
  | mean RACE-layer cos | 0.324 | **0.478** | — |
  | min-layer cos | ~0.17 | **0.33 @L6** | — |

  **Substantial recovery, not yet teacher-grade.** ppl still ~6.6× teacher; strong targets (ppl<100,
  KL<1.5) not reached; curves **still descending at 1000 steps** (budget-limited). Verdict: RULER/LongBench
  are *closer to* meaningful but **not clearly ready** — needs more steps and/or a learnable hash.

---

## Environments (H200, SLURM `commons`, account `as143`)

| Env | torch / transformers | Use | Ext cache |
|---|---|---|---|
| `race_vit_env` | 2.10 / 5.5.0 | RACE kernel + distillation | `.torch_ext_distill` |
| `swa_env` | 2.8 / 4.57.0 | **genuine FA3** runs (`flash_attn_3` 3.0.0) | `.torch_ext_fa3` |

⚠️ **FA3 only works under `swa_env`** (`source distill/env_fa3.sh`). race_vit_env's `flash_attn_3`
`.so` is torch-2.8 ABI and crashes under torch 2.10 (`undefined symbol: _ZN3c10…cuda…`); transformers'
availability check passes from metadata alone, so FA3 jobs include a fail-fast preflight.

---

### 3. RULER long-context evaluation — teacher vs ARRR (eval-only) ✅ _(latest — `distill/REPORT_ruler.md`)_
Eval-only harness `distill/eval_ruler.py` (no training). Llama-3.2-3B-Instruct (teacher) vs
the ARRR hybrid on RULER 32K & 64K (6 tasks, 20/task) from `tonychenxyz/ruler-full`,
query-agnostic split + Llama chat re-template. Teacher = KV-cached decode; hybrid = full
recompute (no RACE KV cache, `use_cache=False`). Build faithfulness confirmed via
`ppl_probe.py` (teacher ppl 22.0, ARRR 147.4 — both match the distillation eval; bf16 cast lossless).

| Model | Ctx | Avg score | Decode tok/s | Peak mem |
|---|:--:|:--:|:--:|:--:|
| teacher | 32K / 64K | 80.2 / 80.5 | 65.5 / 68.1 | 12.6 / 18.8 GB |
| arrr | 32K / 64K | 0.0 / 0.0 | 1.63 / 0.68 | 8.9 / 11.3 GB |

ARRR scores 0 (degenerate repetition) at **both** 32K/64K **and in-distribution at 4096** —
genuine: teacher-forced ppl 146 does not yield coherent free autoregressive generation (the
harness build is verified faithful). Teacher: niah/vt strong; cwe=0 for both (format-correct
but wrong words). Outputs: `results/ruler_{teacher,arrr}_{32k,64k}.json` + 4 PNGs + `ruler_summary.md`.

### 4. Hybrid incremental decode — KV cache (softmax) + RACE running state ✅ _(latest — `distill/REPORT_decode_cache.md`)_
Made hybrid decode incremental: softmax layers use a normal HF KV cache; RACE layers carry two
fixed-size prefix sums (A=Σ probsK, B=Σ probsK⊗v) advanced one token at a time — O(1) in context
length, pure torch (no kernel change). `race_llama_attention.py` (enable/reset_decode_cache, T==1
branch + prefill capture, batched Q/K soft-hash); `eval_ruler.py --decode-mode cache|recompute`
(default cache; asserts layer-0 softmax). Training byte-identical (gated, state not in state_dict).
Verified 3 ways: exact math (0.0 fp64 diff), training-safety/fp32 numerics, and **token-identical**
greedy output vs recompute. Decode latency flat ~18.7 ms/tok 8K→524K; teacher flat to 64K then
rises (17→28 ms) and OOMs at 524K → **crossover ~131K, hybrid 1.5× faster at 262K**, memory always
lower. At ≤64K hybrid is 0.73× (decode is weight-bandwidth-bound there, not attention-bound).
RULER deliverable re-run in cache mode: quality identical, ARRR decode 1.6→52.7 tok/s.

## In flight / next

- **Incremental decode (KV cache): done & verified** — flat-in-T decode, beats teacher >~131K.
- **RULER eval (teacher vs ARRR, 32K/64K): done & verified** — harness faithful; ARRR collapses
  under free generation (0 RULER) while teacher-forced ppl is 146. A learnable-hash / longer
  schedule that lifts generation quality is the prerequisite before RULER is meaningful for ARRR.
- Full-model **forward** latency (FA3): **done & verified**.
- **Global ARRR (75%) distillation: done & verified** — stable global rollout confirmed.
- **ARRR 1000-step + CE (LM-quality gate): done & verified** — ppl 612→146; all 8 criteria met but not
  yet teacher-grade (still descending; RULER/LongBench not clearly ready yet).
- Natural follow-ups (not done): **even longer schedule** (curves still descending at 1000) + **unfreeze/
  learn the RACE hash geometry** to push ppl toward the teacher (the clear next lever); curriculum to curb
  early-layer compounding; THEN RULER/LongBench; full-model **fwd+bwd / training** latency with FA3;
  replacing *all* attention layers (true linear hybrid); M-ensembling; bf16 RACE scan core.
