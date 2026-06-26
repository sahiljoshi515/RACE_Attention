# RACE 2.0 — Distilling Llama-3.2-3B into a Linear-Time RACE Hybrid: Progress Report

_Status: 2026-06-26. Companion to `RACE2_PAPER_STORY.md` (positioning/novelty) and the per-stage
metrics in `distill/results/`. This report explains the method, the architecture, every training
stage, the benchmark results, and the prefill/decode latencies (B=1 and B=8)._

---

## 0. TL;DR

We take a **pretrained, quadratic** transformer (Llama-3.2-3B-Instruct) and **distil** it into a
**RACE hybrid**: half of its attention layers are replaced by a strictly **linear-time** RACE
attention, the rest stay softmax. The conversion is a **one-time** training cost (~0.7–1.0B tokens);
the result is **faster at long context for life**, while we work to keep teacher-level accuracy.

- **Speed (genuine FlashAttention-3 vs FA3+RACE-kernel hybrid):** the hybrid is faster at **decode for
  every context length** (1.10× @4K → **1.68× @64K**, B=1) and faster at **prefill beyond ~32K**
  (1.19× @64K), with lower memory throughout. The advantage **grows with context length**.
- **Quality:** the staged distillation de-collapsed the model; then **unfreezing the MLPs (Stage F)
  recovered BOTH general capability AND long-context retrieval** — the frozen base was the dominant
  limiter for both. vs the frozen-base best: **MMLU 24.7→34.5, HellaSwag 36→55, RULER-16K 22.9→54.6,
  RULER-32K 15.2→42.7**, with previously-stuck tasks recovered (distractor-needle `niah_single_1`
  **0→100**, variable-tracking 0→24). All curves were **still climbing** at 351M tokens.
- **Honest gaps:** not yet teacher-grade (MMLU 34.5 vs 62; RULER 54.6 vs 80 @16K) but on a clear
  rising trajectory — the path is more KD tokens at the proper-dosage unfreeze + a long-context
  curriculum on top of it. `cwe` (aggregation) is the one task still at 0.

---

## 1. What is distillation (here)?

**Knowledge distillation** = train a *student* model to imitate a fixed *teacher* model. We run both
on the **same input tokens** and push the student's internal + output behavior toward the teacher's:

```
input tokens ──► TEACHER (frozen Llama, all softmax)  ──► teacher hidden states + teacher logits
       │                                                          │ (targets, no-grad)
       └────────► STUDENT (hybrid: 14 softmax + 14 RACE) ──► student hidden states + student logits
   loss = 0.1·MSE(student_h, teacher_h  @ RACE layers)      # match per-layer representations
        + 1.0·KL(student_logits ‖ teacher_logits)           # match the full next-token distribution
        + 0.5·CE(student_logits, next_token)                # standard language-model loss
   loss.backward()  →  update ONLY the RACE layers (+ optionally the MLPs in Stage E/F)
```

It is **not** plain fine-tuning: the teacher's per-layer hidden states and full-vocab output
distribution are the supervision. Two model instances are loaded (frozen teacher + trainable
student). The student runs **end-to-end on its own hidden states** ("global" distillation —
`distill_global.py`): a RACE layer's approximate output feeds the next layer, so errors must be
absorbed by the network, not hidden by teacher-forcing the hidden states.

We do **not** distil the softmax *attention map* — RACE uses a different mechanism, so forcing its
attention matrix to equal softmax would be wrong. We only constrain the **outputs** (hidden states +
logits), leaving RACE free to approximate them its own way.

---

## 2. The architecture

**Teacher:** Llama-3.2-3B-Instruct — 28 decoder layers, 24 query / 8 KV heads (GQA), head_dim 128,
RoPE, 128K native context.

**RACE attention** (the linear-time replacement, arXiv 2510.04008): replaces the quadratic
`softmax(QKᵀ)V` with **angular soft-LSH bucketing + a causal prefix scan**. Per token, the query/key
are soft-hashed into `S = L·2^K` buckets (a softmax over R buckets within each of L groups). The
causal state is two per-bucket running sums — `A[s]=Σ probsK[t,s]` (count), `B[s]=Σ probsK[t,s]·v_t`
(value-sum) — and the readout is the query-weighted **bucket mean** `out_t = Σ_s probsQ[t,s]·B[s]/A[s]`.
This is **O(T)** in sequence length. We use **S=24** (L=3, K=3).

**The hybrid (drop-in):** we replace the **14 odd-indexed** attention layers with RACE (the "AR"
pattern = Alternating softmax/RACE = **50% RACE**), keeping the other 14 as softmax. Only the RACE
modules' q/k/v/o + a temperature scalar train; the rest of Llama is, by default, **frozen** (the
model stays bit-identical except the swapped attention). The kept-softmax layers are the **structural
edge over pure-linear conversions** (RADLADS etc.): they preserve exact retrieval that linear
attention loses.

**Inference / decode:** softmax layers use a normal KV cache; RACE layers carry the two fixed-size
running sums (`A`, `B`) advanced one token at a time — **O(1) in context length** — via a fused CUDA
**decode kernel** (`kernels/gpu/decode_kernel.cu`). Prefill uses FlashAttention-3 for the softmax
layers + the RACE causal-scan kernel.

---

## 3. How we distilled it — the staged recipe

Modeled on **RADLADS** (arXiv 2505.03005): distil at short context first, then extend; align → KD →
context-extend. Data: **FineWeb** (web text, pre-staged local parquet) + **on-the-fly synthetic
RULER-style retrieval** (needle-in-haystack, multi-key, variable-tracking, etc.), mixed ~70/30.

| Stage | What trains | Data | Ctx | Loss (h/kl/ce) | Tokens | Outcome |
|---|---|---|--:|---|--:|---|
| **A** align | RACE only | FineWeb | 4096 | 1.0 / 0 / 0 | ~130M | hidden-cos 0.04→**0.64** |
| **B** KD | RACE only | mixed | 4096 | 0.1 / 1.0 / 0.5 | ~350M | **de-collapse**; NIAH-4K 0→**1.0**; ppl 81→**22.5** |
| **C** ctx-extend | RACE only | mixed | 4096→16384 | 0 / 0.3 / 1.0 | ~196M | **RULER 22.9@16K / 15.2@32K** (best frozen-base) |
| **D** ruler-push | RACE only | mixed+aug | →32768 | 0.1 / 1.0 / 0.5 | ~140M | **regressed** RULER (21.5/12.3) → abandoned |
| **E** partial unfreeze | RACE + **MLP** | mixed+aug | →16384 | 0.1 / 1.0 / 0.5 | ~100M | under-dosed (base-LR 1e-5, 400 steps) → MMLU flat |
| **F** full-KD unfreeze | RACE + **MLP** | mixed | 4096 | 0.1 / 1.0 / 0.5 | ~351M | **proper dosage** (base-LR 5e-5): ppl→teacher, HS 36→50 |

Key methodological findings along the way:
- **The original ~16M-token run collapsed** (low teacher-forced ppl but RULER=0, repetition in free
  generation): caused by token-level teacher-forcing + an ~off CE term + too few tokens. Stages A–C
  (RADLADS-style KD on mixed data, ~680M tokens) fixed it.
- **More KD has diminishing returns on RULER** (Stage D regressed): the hard retrieval tasks are
  *capability*-bound, not token-bound.
- **MMLU needs the base unfrozen.** Frozen-base hybrids sit at **chance** on MMLU even 5-shot — the
  lossy RACE attention corrupts the signal the frozen MLPs expect, and they can't adapt. Stage E
  (a 400-step, base-LR 1e-5 tail) was too small to move 2.1B MLP params; **Stage F** (base-LR 5e-5,
  ~351M tokens) is the proper-dosage unfreeze and is recovering capability fast.

**Partial unfreeze (Stage E/F):** `--unfreeze mlp` trains the **MLP blocks + layer norms across all
28 layers (~2.1B params)** plus the RACE layers (~0.35B), while keeping the **softmax attention
(retrieval heads) and embeddings frozen** — so ~2.5B of 3.3B params train, but it is *not* a literal
full unfreeze (the retrieval heads are deliberately preserved).

---

## 4. Evaluation results (vs the teacher)

All student numbers are **held-out**; MMLU is 5-shot (standard), HellaSwag acc_norm, Winogrande acc;
RULER + LongBench stream the real benchmarks (not our synthetic training data).

| Model | MMLU | HellaSwag | Winogrande | RULER-16K | RULER-32K | RULER-64K | LongBench |
|---|--:|--:|--:|--:|--:|--:|--:|
| **teacher** | 62.0 | 70.5 | 69.9 | 80.5 | 80.2 | 80.5 | 47.2 |
| Stage C (frozen base) | 24.7 | 36.5 | 52.1 | 22.9 | 15.2 | — | — |
| Stage E (under-dosed unfreeze) | 23.9 | 38.8 | 51.5 | 21.5 | 12.3 | — | — |
| **Stage F @1340 (full MLP-unfreeze)** | **34.5** | **55.4** | **55.1** | **54.6** | **42.7** | **21.0** | _(pending)_ |

**The MLP-unfreeze recovered BOTH axes — the frozen base was the dominant limiter for retrieval too,
not just MMLU.** vs the frozen-base Stage C:
- **MMLU** 24.7 (chance) → 26.6 @335 → **34.5 @1340**, still climbing at 351M tokens; genuine
  per-subject knowledge (college_chemistry 83, us_foreign_policy 80, medical_genetics 78). HellaSwag
  36→**55**, Winogrande 52→55, ppl 6.2 (teacher 15.2), 74% teacher top1.
- **RULER 16K 22.9 → 54.6 (2.4×); 32K 15.2 → 42.7 (2.8×); 64K 21.0** — and the tasks that were
  **stuck at 0** are recovered: `niah_single_1` (distractor needle) **0→100 @16K / 95 @32K**,
  `variable_tracking` **0→24**, `niah_multivalue` 27→74. Only `cwe` (aggregation) stays 0.
- Crucially, **Stage F used plain mixed data (no distractor augmentation) and trained at 4K only** —
  so this is the *unfreeze alone*, extrapolating to 16–64K. (The earlier "bucket-readout recall
  ceiling" hypothesis was wrong: with the MLPs adapted, the kept-softmax layers + MLPs route
  retrieval effectively. 64K drops more because Stage F never trained beyond 4K — a Stage-C-style
  long-context curriculum *on top of* the unfreeze should lift 32K/64K further.)

Stage F is now the best checkpoint on every axis.

- **Frozen-base hybrid (Stage C):** recovered single-needle retrieval (RULER `niah_single_2` 95% @16K;
  a 3B hybrid beating RADLADS's pure-linear 7B at ~16.5) but the **distractor (`niah_single_1`),
  multi-hop (`vt`), aggregation (`cwe`) tasks sat at 0**, and MMLU was at chance.
- **Unfreezing the MLPs (Stage F) lifted both** — the frozen base, not the RACE readout, was the
  ceiling: **RULER-16K 22.9→54.6, 32K 15.2→42.7**, with `niah_single_1` **0→100**, `vt` 0→24,
  `multivalue` 27→74; and **MMLU 24.7→34.5, HellaSwag 36→55** (ppl 6.2 vs teacher 15.2, 74% top1).
  Only `cwe` (frequency aggregation) remains 0. All curves still rising at 351M tokens.

---

## 5. Prefill + Decode latency (the speed story)

Setup: **genuine FlashAttention-3** for all softmax layers (full model + hybrid's kept-softmax) and
the **custom fused RACE decode kernel** for RACE layers (verified engaged on all 14 RACE modules);
bf16, H200, decode = 64 new tokens, warmup excluded. `distill/bench_prefill_decode.py` under
`distill/env_fa3.sh` (swa_env / torch 2.8 / FA3). Plot: `distill/results/prefill_decode_latency.png`.

### B=1

| T | Prefill (ms) full → hybrid | Decode (ms/tok) full → hybrid | Peak mem full → hybrid |
|--:|--:|--:|--:|
| 4K | 61 → 98 (0.62×) | 20.3 → 18.4 (1.10×) | 7.2 → 7.0 GB |
| 8K | 124 → 159 (0.78×) | 20.4 → 18.4 (1.11×) | 8.0 → 7.5 GB |
| 16K | 281 → 316 (0.89×) | 20.4 → 18.4 (1.11×) | 9.6 → 8.6 GB |
| 32K | 685 → 686 (1.00×) | 22.8 → 19.3 (1.18×) | 12.7 → 10.8 GB |
| 64K | 1947 → 1640 (**1.19×**) | 36.4 → 21.7 (**1.68×**) | 18.8 → 15.1 GB |

### B=8 (decode ms/tok is per decode-step serving all 8 sequences)

| T | Prefill (ms) full → hybrid | Decode (ms/tok) full → hybrid | Peak mem full → hybrid |
|--:|--:|--:|--:|
| 4K | 441 → 567 (0.78×) | 18.5 → 17.6 (1.06×) | 12.6 → 10.8 GB |
| 8K | 961 → 1130 (0.85×) | 18.7 → 17.6 (1.06×) | 18.8 → 15.1 GB |
| 16K | 2197 → 2405 (0.91×) | 28.3 → 17.8 (**1.59×**) | 31.2 → 23.7 GB |
| 32K | 5508 → 5358 (1.03×) | 50.4 → 28.7 (**1.76×**) | 55.9 → 40.9 GB |
| 64K | 15338 → 12911 (**1.19×**) | 94.5 → 50.7 (**1.86×**) | 105.3 → **75.3 GB** |

**Batching amplifies the hybrid's advantage.** At B=8 the KV cache is 8× larger, so the full model's
decode degrades much harder with context (18.5→**94.5** ms/tok) while the hybrid stays far flatter
(17.6→50.7) — the decode speedup grows to **1.86× @64K** (vs 1.68× at B=1). Prefill crossover is again
~32K (1.19× @64K). Critically, **memory:** the full model hits **105 GB @64K** (near the 141 GB H200
limit), the hybrid only **75 GB** — so the hybrid serves larger batches / longer context where the
full model OOMs. (No OOM occurred here; both fit at 64K×B8.)

**How to read it:**
- **Decode:** hybrid wins at **every** length and the gap **widens with T** — RACE layers carry O(1)
  state (flat ~18 ms/tok) while the full model's KV cache makes per-token cost grow. Hybrid also uses
  less memory.
- **Prefill:** below ~32K the hybrid is *slower* (RACE per-layer launch overhead dominates at short
  T); it **crosses over at ~32K** and pulls ahead at 64K. The larger 2–3× prefill gains appear at
  256K–1M (kernel-level RACE-vs-FA3 is 13–90× at 1M). The hybrid speedup is a **bounded constant
  factor** ≈ `depth / (softmax layers kept)` (= 2× for the 50%-RACE AR layout, 4× for 75%-RACE srrr).

---

## 6. Honest assessment

**Works / publishable now:** the conversion recipe (de-collapse + single-needle retrieval recovery),
the **decode + long-context prefill speed/memory wins** (growing with context), and a 3B hybrid
beating a pure-linear 7B on RULER@16K. The kept-softmax layers are the demonstrated edge.

**Open:** (1) **MMLU recovery** — directionally working under the full-dosage unfreeze (HellaSwag +
ppl prove capability transfers), final number pending; (2) **full RULER parity** — multi-hop /
aggregation / distractor tasks remain hard (a linear-attention recall limit; the per-bucket
**delta-rule** variant we prototyped + verified did **not** beat the bucket-mean — the limiter is
bucket *addressing*, not the readout, so that lever is off the table); (3) the speedup is a bounded
constant (kept-softmax layers keep the model O(T²)) — true linear scaling needs a fully-RACE model.

---

## 7. Infrastructure built this effort
- `distill/distill_global.py` — staged distillation trainer; `--unfreeze mlp` partial-unfreeze (+
  `base_state` checkpoints), in-loop free-gen/NIAH probe, `--data mixed` + task-aug knobs.
- `distill/eval_choice.py` — MMLU(5-shot)/HellaSwag/Winogrande loglikelihood harness (collapse-immune).
- `distill/eval_longbench.py` — faithful LongBench (official prompts/metrics).
- `distill/eval_ruler.py` — RULER at any context length + hybrid decode (FA3 + RACE kernel).
- `distill/bench_prefill_decode.py` — the FA3-vs-FA3+RACE prefill/decode latency bench (B configurable).
- Env: `distill/env.sh` (race_vit_env, torch 2.10, training) and `distill/env_fa3.sh` (swa_env, torch
  2.8, genuine FA3 — the only env where FA3 works).

## 8. Next steps
1. Finish Stage F → **final MMLU** verdict (does proper-dosage MLP-unfreeze recover it?).
2. If MMLU still lags: a **true full unfreeze** (also softmax attention + embeddings) or more KD tokens.
3. **RULER hard tasks**: retrieval-head-aware **layer placement** (keep the teacher's retrieval-head
   layers softmax) — the productive lever now that delta-RACE is ruled out.
4. Multi-model-family (Qwen/Mistral) + the faster-for-life amortization curves for the paper.
