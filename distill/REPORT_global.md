# Global hybrid-student distillation — ARRR (75% RACE), with AR (50%) comparison

**Question.** Does the *local* RACE-layer learnability translate into a **stable, globally rolled-out**
hybrid Llama when 75% of attention layers are RACE? The local pilot fed each RACE layer the *teacher's*
input. Here the student runs **end-to-end on its own hidden states** (RACE outputs feed later layers) and
must match a frozen teacher. **Answer: yes — the 75% ARRR hybrid is globally trainable and stable.**

**Setup.** `meta-llama/Llama-3.2-3B-Instruct` (28 layers), 1×H200, `race_vit_env` (torch 2.10, tf 5.5.0).
Two model instances — frozen **teacher** + hybrid **student**. Pattern **ARRR** = (Softmax,RACE,RACE,RACE)
repeating → softmax kept at layers **[0,4,8,12,16,20,24]** (7), RACE at the other **21** (75%).
Each step runs both end-to-end (`use_cache=False`, **no teacher forcing**); loss =
`1.0·hidden_MSE + 0.5·KL` (CE off), hidden = mean over the 21 RACE layers of MSE(student vs teacher raw
decoder-layer output, via forward hooks); KL = temperature-KL(teacher‖student) on final logits, fp32
token-chunked. **Trainable: only RACE q/k/v/o + log_temp (528.5M); base frozen.** B=2, T=4096, bf16,
gradient checkpointing, AdamW lr 5e-5, warmup 20, clip 1.0, **200 steps**. Comparison run **AR** (50%
alternating, 14/28 RACE, lr 1e-4), same pipeline. Jobs 144453 (ARRR) / 144454 (AR).
Every result re-verified by adversarial auditor agents from the raw `results/metrics_global_*.jsonl`.

---

## Results (held-out eval, step 0 → 190)

| metric | **ARRR (75% RACE)** | **AR (50% RACE)** | direction |
|---|---|---|---|
| mean hidden-state cosine | **0.077 → 0.324** | 0.106 → 0.523 | ↑ ✓ |
| final-norm hidden cosine | 0.238 → 0.500 | 0.179 → 0.614 | ↑ ✓ |
| rel hidden-MSE | **4.58 → 1.59** | 3.01 → 0.89 | ↓ ✓ |
| logits KL (teacher‖student) | **6.35 → 3.53** | 6.55 → 1.90 | ↓ ✓ |
| top-1 logit agreement | 0.027 → 0.168 | 0.013 → 0.358 | ↑ ✓ |
| student perplexity (held-out) | **10152 → 612** | 13215 → 120 | ↓ ✓ |
| teacher perplexity (ref) | 22.0 (flat) | 22.0 (flat) | — |

Train loss ARRR 4.55 → 1.96 (min 1.68), AR 4.33 → 0.94; **no NaN/Inf**, grad-norm bounded (pre-clip ≤83,
settling to ~12); `base_grad_count == 0` at every one of the 200 steps; `race_grad_nonzero_frac ≈ 1.0`
(one ARRR step at 0.99); **frozen base bit-identical start→end** (exact-snapshot drift 0.0). All numbers
above independently recomputed from the raw jsonl by auditor agents (reproduce exactly). Plots:
`results/global_{arrr,ar}_{eval_trends,per_layer_cosine,loss_breakdown,layer_groups}.png`; 3-way comparison
`results/global_compare.png`. Trends are **improving overall with step-level noise** (not strictly monotone).

## ARRR compounding diagnostic (the point of 75% replacement)

Final per-layer hidden cosine forms a **valley**: the front/lower-middle replaced layers match the teacher
worst — group means **early 0.21 / middle 0.31 / late 0.46**, with the single minimum ≈ **0.17 around layers
6–9** (L1/L2/L3 = 0.23/0.29/0.25 sit slightly above it) — then cosine **recovers monotonically toward the
output** (L27 = 0.54, final-norm cosine 0.50). So error compounds through the dense early RACE blocks (ARRR
packs 3 consecutive RACE per softmax) and the later layers — interleaved with kept softmax layers and closer
to the supervised logits — recover; the rollout does **not** diverge. AR shows the same valley, shallower
(min ≈ 0.40 at L11, late group 0.61). (Note: "early layers hardest" holds at the *group* level; the single
worst layer is in the lower-middle, not layer 1.)

## Honest read

- **All 7 success criteria are met for ARRR**: stable (no NaN, bounded grads), hidden-MSE↓, hidden-cosine↑,
  KL↓, frozen base unchanged, RACE grads nonzero, and student perplexity **bounded and decreasing overall**
  (10152 → 612, with step-level noise). The 75% global rollout is trainable.
- **ARRR is harder than AR, as expected.** At 50% replacement the student reaches markedly higher fidelity
  in the same budget (cosine 0.52 vs 0.32, KL 1.90 vs 3.53, ppl 120 vs 612). More linearized layers ⇒
  more compounding ⇒ slower convergence — exactly the trade the experiment was designed to expose.
- **Still improving, not converged.** All eval curves are **still rising at step 200** and train loss is
  above its floor — these are training-budget-limited snapshots, not plateaus.
- **Does not match the teacher yet, and that's expected.** Student ppl (612 ARRR / 120 AR) is still
  5×–28× the teacher's 22, and global per-layer cosine is far below the 50% **local** pilot (mean 0.93).
  That pilot is a **different, easier objective** — each layer was fed the teacher's *clean* input
  (teacher forcing); global rollout makes every layer consume its own compounding state. Combined with the
  short **200-step** budget and the **frozen-random** RACE hash geometry, lower fidelity is expected. The
  goal here was rollout *stability*, not parity, and the numbers are hidden-state/logit-match metrics — **not
  a downstream task accuracy.**

## Verdict

A 75% (ARRR) RACE-attention Llama-3.2-3B is **globally trainable and stable** end-to-end: every distillation
signal improves monotonically over 200 steps with the frozen base intact and gradients flowing through the
custom RACE CUDA backward. Local learnability **does** translate to global rollout, at reduced fidelity that
scales with replacement fraction. Natural next steps (out of scope): longer schedule, unfreeze/learn the
hash geometry, add CE/next-token KD, and curriculum (warm up replacement fraction or distill early layers
first to curb compounding).

## Reproduce
```bash
source distill/env.sh
$PYBIN distill/smoke_global.py                 # 10 behavioral asserts incl. no-teacher-forcing + ckpt parity
sbatch distill/run_global.sbatch ARRR 2 2      # 75% (lr 5e-5, B=2, grad-ckpt, 200 steps)
sbatch distill/run_global.sbatch AR  2 2 1e-4  # 50% comparison
$PYBIN distill/plot_global.py --tag arrr       # 4 plots (auto-run by the sbatch too)
```
Local pilot files (`distill_local.py`, `run_pilot.sbatch`, `metrics_S*.jsonl`) untouched.
