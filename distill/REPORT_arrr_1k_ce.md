# ARRR 1000-step global distillation + CE — LM-quality readiness gate

**Question.** Can the **75% ARRR** RACE-hybrid recover **usable** perplexity/logits — enough that a
later RULER/LongBench evaluation would be *meaningful*? (No long-context eval in this task.) The prior
200-step ARRR run was stable but far from usable (ppl ~612 vs teacher ~22). This run trains **1000 steps**
and adds a **CE term** to the objective.

**Setup.** Same verified pipeline (`distill/distill_global.py`, two instances, end-to-end rollout, no
teacher forcing, RACE-only training, frozen base). ARRR = 21/28 RACE (softmax kept at 0,4,…,24), S=8
(L=2,K=2). **Loss = 1.0·hidden_MSE + 0.5·KL(T=1) + 0.1·CE.** B=2, T=4096, bf16, gradient checkpointing,
chunked KL, AdamW lr 5e-5, warmup 50, clip 1.0, **1000 steps**, eval@25, checkpoints@250/500/750/1000.
Job 144459, 1×H200, ~21 min. `results/metrics_global_arrr_1k_ce.jsonl`; independently re-verified from the
raw jsonl by an auditor agent.

---

## Results — final held-out eval (step 975) vs the 200-step ARRR baseline

| metric | 200-step ARRR | **1000-step + CE** | required | strong target |
|---|---|---|---|---|
| student perplexity | 612 | **146** (−76%) | <612 ✓ | <100 — *not yet* |
| logits KL (T=1) | 3.53 | **2.09** | <3.53 ✓ | <1.5 — *not yet* |
| final-norm hidden cosine | 0.500 | **0.641** | >0.50 ✓ | >0.65 — *≈there* |
| mean RACE-layer cosine | 0.324 | **0.478** | >0.324 ✓ | >0.50 — *close* |
| min-layer cosine | ~0.17 @L6–9 | **0.33 @L6** | reduced ✓ | — |
| top-1 / top-5 logit agreement | 0.168 / — | **0.30 / 0.59** | — | — |
| teacher perplexity (ref, flat) | 22.0 | 22.0 | — | — |

Trajectory (eval): ppl **11644 → 146**, KL 6.48 → 2.09, mean cos 0.076 → 0.478, final-norm cos 0.234 →
0.641, group cos early/mid/late **0.37 / 0.46 / 0.60** (was 0.21 / 0.31 / 0.46). Train loss 5.49 → 1.78,
CE 9.43 → 4.79. Stable throughout: **no NaN/Inf**, grad-norm bounded (≤150), **base_grad_count == 0 every
step**, `race_grad_nonzero_frac` = 1.0, peak **50.2 GB**, **frozen base bit-identical** (snapshot drift 0.0).
Checkpoints: `checkpoints/arrr_L2_K2_arrr_1k_ce_step{250,500,750,1000}.pt` (race params, ~2.1 GB each).
Plots: `results/arrr_1k_ce_{eval_trends,loss_breakdown,per_layer_cosine,layer_groups,compare_previous}.png`.

## All 8 success criteria met
1. Stable ✓ · 2. ppl ≪ 612 (146) ✓ · 3. KL < 3.53 (2.09) ✓ · 4. final-norm cos > 0.50 (0.641) ✓ ·
5. mean cos > 0.324 (0.478) ✓ · 6. early/mid valley reduced (min-layer cos 0.17→0.33; early group
0.21→0.37) ✓ · 7. frozen base unchanged ✓ · 8. RACE grads nonzero ✓.

## Honest readiness verdict

**Substantial recovery, but not yet teacher-grade.** Adding CE + 5× more steps cut ARRR perplexity ~4.2×
(612 → 146) and roughly doubled every fidelity metric. The model now produces meaningfully aligned logits
(top-1 agreement 30%, top-5 59%). **However, ppl 146 is still ~6.6× the teacher's 22**, and the two
"strong" gates (ppl<100, KL<1.5) were *not* reached. Crucially, **the curves are still descending at step
1000** (last-5 evals: ppl 153→146, KL 2.13→2.09, cos still rising) — this is training-budget-limited, not a
plateau. **Read:** RULER/LongBench are now *closer to* meaningful but **not clearly ready** — a long-context
score at ppl 146 would still be confounded by basic LM error. Recommended before long-context eval: more
steps (the trend predicts continued gains) and/or **unfreezing the RACE hash geometry** + a stronger CE/KD
schedule to push ppl toward the teacher. The numbers here are hidden-state/logit-match metrics, **not** a
downstream task score.

## Reproduce
```bash
source distill/env.sh
sbatch distill/run_arrr_1k_ce.sbatch        # ARRR 75%, 1000 steps, CE 0.1, ckpt@250/500/750/1000
# plots (auto-run by the sbatch):
$PYBIN distill/plot_global.py --tag arrr_1k_ce --out-prefix arrr_1k_ce --compare arrr ar
```
Prior runs (`metrics_global_{arrr,ar}.jsonl`, local pilot) untouched.
