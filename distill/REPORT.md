# Llama-3.2-3B → hybrid RACE-attention distillation — pilot report

**Question:** can RACE attention layers learn to imitate the original Llama attention with a little distillation? **Answer (pilot): yes.** Across both tested configs, per-layer attention-output and hidden-state MSE **decrease monotonically** and cosine similarity **increases monotonically** over 100 teacher-forced steps, training is stable, and the custom RACE CUDA kernels drive the q/k/v gradients.

## Setup

- **Model:** `meta-llama/Llama-3.2-3B-Instruct` (28 layers, 24 Q / 8 KV heads, head_dim 128, GQA, llama3 RoPE), bf16, frozen = **teacher**.
- **Hybrid:** odd layers (1,3,…,27 → **14 layers, 50%**) replaced by `RaceLlamaAttention` — a drop-in for `LlamaAttention` that keeps Llama's q/k/v/o + RoPE + `repeat_kv` and swaps only softmax→**causal RACE soft-hash** (custom CUDA `RaceCausalFn`: `race_fused_fwd` + `race_backward`).
- **Trainable:** RACE q/k/v/o (copied from Llama) + a scalar `log_temp` (**352.3M** params across 14 layers). **Frozen:** embeddings, RMSNorms, MLPs, lm_head, even-layer attention, and the RACE hyperplanes/prototypes (fixed random buffers).
- **Distillation:** teacher-forced **local** — each RACE layer is fed the *teacher's* input hidden state; loss = per-layer `MSE(attn_out)` + `MSE(layer_out)` (reported separately, plus cosine + scale-invariant relative-MSE).
- **Data:** `HuggingFaceFW/fineweb` streamed, packed to **T=4096**, **B=8**; fixed disjoint held-out eval batch.
- **Optim:** AdamW, **lr 1e-4**, linear warmup 10, grad-clip 1.0; **100 steps**. **1× H200**, ~85–87 GB peak, ~16–21K tok/s.

## Results (held-out eval, step 0 → 90)

| metric | S=8 (L=2,K=2) | S=24 (L=3,K=3) | criterion |
|---|---|---|---|
| relative attn-MSE | **11.89 → 1.29** | **25.78 → 2.03** | ↓ ✓ |
| relative hidden-MSE | **0.437 → 0.057** | **0.766 → 0.078** | ↓ ✓ |
| attention-output cosine | **0.124 → 0.370** | **0.124 → 0.317** | ↑ ✓ |
| hidden-state cosine | **0.682 → 0.937** | **0.587 → 0.920** | ↑ ✓ |

All four are monotone (only sub-0.001 single-step wiggles). Final per-layer hidden cosine ranges ~0.79 (layer 1) → **0.96** (layer 27) for S=8 — later layers match best; earlier replaced layers are harder. Plots: `results/eval_trends.png`, `results/final_per_layer_cosine.png`. Raw logs: `results/metrics_S{8,24}.jsonl`.

**Interpretation:** with frozen random hash geometry and only q/k/v/o + temperature training, the RACE layers cannot perfectly reproduce softmax attention in 100 steps (attention-output relative error stays ~1.3–2.0), but they clearly **learn**: directions align (cosine up) and the **hidden states that actually propagate are matched well** (cosine 0.92–0.94, ~6–8% relative error). S=8 edges out S=24 here (easier to fit fewer fixed buckets in a short pilot).

## Verification (multi-agent audit — found & fixed real bugs)

Three independent auditor agents reviewed the code and recomputed the logs. They **confirmed**: the model is a faithful softmax→RACE drop-in (soft-hash verified bit-for-bit vs the canonical reference; no head-mixing; correct GQA/RoPE/causality), the teacher is fully frozen (0 trainable base params, asserted), teacher-forcing is correct (each layer fed the teacher's input; grads flow through the frozen norm/MLP to the RACE params), the held-out eval is genuinely disjoint and fixed, and the **custom `race_backward` kernel is exercised** (q/k/v grads can only arrive through it; the smoke test asserts nonzero RACE grads + zero base grads).

They also found **two real issues, both now fixed and re-run**:
1. **BUG (layer 27 hidden target):** transformers 5.5.0 overwrites `hidden_states[-1]` with the *post-final-RMSNorm* tensor (`tie_last_hidden_states=True`), so the last replaced layer was matched against the wrong target (this caused the 12× MSE spikes seen in the first unstable run). **Fix:** capture each replaced layer's **raw** input/output via decoder-layer forward hooks instead of `hidden_states[i+1]`. After the fix, layer 27's hidden cosine is a clean 0.96/0.95.
2. **CONCERN (latent `M>1`):** the ensemble count `M` was accepted but silently ignored (no ensemble axis). **Fix:** assert `M==1` so it fails loudly; M-ensembling is left as a follow-up.

**Honest nuance (from the audit):** grad-clipping at 1.0 **never actually fired** (grad norms stayed 0.03–0.9) — the real stabilizer over the earlier lr=5e-4 run (which oscillated, train loss spiking to ~13) was the **5× lower lr + warmup**. Clipping is kept as cheap insurance.

## Deliverables

`distill/`: `race_llama_attention.py` (drop-in), `hybrid.py` (replace/freeze), `data_fineweb.py`, `distill_local.py` (training+metrics+eval), `smoke_distill.py` (build+grad-flow checks), `plot_distill.py`, `env.sh`, `run_pilot.sbatch`, `README.md`. Results + plots in `distill/results/` (current `metrics_S{8,24}.jsonl` are the layer-27-fixed runs; `*_l27bug.jsonl` and `*_lr5e4.jsonl` retained for comparison).

Run: `source distill/env.sh` then `sbatch distill/run_pilot.sbatch 2 2` (S=8) / `… 3 3` (S=24); smoke via `distill/smoke_distill.py`.

## Status vs success criteria

| criterion | result |
|---|---|
| Training stable | ✓ (lr 1e-4 + warmup; finite loss/grads throughout) |
| Loss decreases in 50–100 steps | ✓ (attn & hidden MSE both ↓, monotone) |
| Hidden-state similarity improves | ✓ (cosine 0.59–0.68 → 0.92–0.94) |
| Attention-output similarity improves | ✓ (cosine 0.12 → 0.32–0.37) |

**Proof-of-concept succeeds.** Natural next steps (not in scope): unfreeze/learn the hyperplanes, add CE/next-token KD, longer schedule, global (full-hybrid) distillation, and M-ensembling.
