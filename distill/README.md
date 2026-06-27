# Llama-3.2-3B → hybrid RACE-attention distillation (prototype)

Proof-of-concept: replace the **odd** attention layers (1,3,…,27 → 14 layers, 50%) of
`meta-llama/Llama-3.2-3B-Instruct` with **causal RACE attention** and check whether
those layers can *learn to imitate* the original Llama attention via a short
teacher-forced distillation. Uses the merged custom CUDA RACE kernels
(`RaceCausalFn`: `race_fused_fwd` + `race_backward`).

## What it does
- **Teacher** = frozen Llama (bf16). **Student** = the RACE modules (q/k/v/o copied
  from Llama + a learnable temperature; hyperplanes/prototypes frozen-random).
- **Teacher-forced local**: each RACE layer is fed the teacher's input hidden state
  at that layer; we minimize, per replaced layer, `MSE(attn_out)` + `MSE(layer_out)`.
- **Trainable**: only RACE q/k/v/o + `log_temp` (~350M params across 14 layers). The
  custom `race_backward` kernel produces the q/k/v gradients; o_proj/temperature via
  autograd. Everything else (embeddings, RMSNorms, MLPs, lm_head, even-layer
  attention) is frozen.

## Files
- `race_llama_attention.py` — `RaceLlamaAttention`, a drop-in for `LlamaAttention`
  (tf 5.5.0): Llama q/k/v/o + RoPE + GQA, softmax→RACE soft-hash, causal RACE core.
- `hybrid.py` — build/copy RACE modules, `convert_to_hybrid` (in-place swap for
  inference/global), freeze + trainable-param helpers.
- `data_fineweb.py` — stream `HuggingFaceFW/fineweb`, pack to 4096, disjoint eval.
- `distill_local.py` — the training loop + metrics + eval@10 (writes
  `results/metrics_S{S}.jsonl`).
- `smoke_distill.py` — kernel build + one step + grad-flow / frozen assertions +
  B=8 memory probe.
- `env.sh`, `run_pilot.sbatch` — environment + SLURM launcher.

## Run (H200)
```bash
# interactive smoke
srun --partition=commons --account=as143 --gres=gpu:h200:1 --cpus-per-task=8 \
     --mem=128G --time=00:40:00 bash -lc \
     'cd /scratch/sj157/RACE_Attention && source distill/env.sh && $PYBIN distill/smoke_distill.py'

# pilots (two configs)
sbatch distill/run_pilot.sbatch 2 2     # S=8
sbatch distill/run_pilot.sbatch 3 3     # S=24
```
Outputs: `distill/results/metrics_S8.jsonl`, `metrics_S24.jsonl`, plots, `pilot_*.out`.

## Notes
- Env: `race_vit_env` (transformers 5.5.0, torch 2.10). The CUDA kernel JIT-builds in
  a torch-2.10-specific cache (`.torch_ext_distill`); first build ~1–2 min.
- `B=8, T=4096`, AdamW lr=5e-4, 100 steps. Drop `--batch-size 4` if memory-tight.
- Success = per-layer attn/hidden MSE decrease and cosine similarity increase over
  the run (see eval@10 lines).

---

# Global rollout (ARRR) — `distill_global.py`

The **next** experiment: move from local teacher-forced layer distillation to a
**global student rollout** with a more aggressive pattern. The student runs
**end-to-end on its own hidden states** (RACE outputs feed later layers); we match a
frozen teacher. Tests whether a high-replacement hybrid stays trainable.

- **Pattern** (`--pattern`, via `hybrid.pattern_pred`): `A`=keep Llama softmax attn,
  `R`=RACE. `AR`=alternating (14/28), `AAR`=~9/28, **`ARRR`=(Softmax,RACE,RACE,RACE)
  repeating = 21/28 RACE** (softmax kept at layers 0,4,8,12,16,20,24).
- **Two model instances**: frozen teacher + hybrid student. Each step runs both
  end-to-end (`use_cache=False`); **no teacher forcing**.
- **Loss** = `1.0*hidden + 0.5*KL` (CE off by default). `hidden` = mean over the
  RACE layers of MSE(student raw layer-output, teacher raw layer-output) captured by
  decoder-layer forward hooks (avoids the layer-27 final-norm overwrite). `KL` is the
  temperature-KL of the final logits, computed in fp32 token-chunks (`--kl-chunk`).
- **Trainable**: only RACE q/k/v/o + `log_temp`. Base frozen (asserted + fingerprinted).
- **Memory**: global backprop runs through the full student, so use **B=2 with
  gradient checkpointing** (`--grad-checkpoint`) + chunked KL. Fallback ladder if OOM:
  `--kl-chunk 1024/512` → `--seq-len 2048` → `--batch-size 1`.

## Run (H200)
```bash
source distill/env.sh                       # race_vit_env (torch 2.10), NOT env_fa3.sh

# smoke (build + grad-flow + no-teacher-forcing + ckpt-parity + mem probe)
$PYBIN distill/smoke_global.py

# ARRR main (B=2, grad-checkpoint, lr 5e-5, 200 steps)
sbatch distill/run_global.sbatch ARRR 2 2

# AR (50% alternating) comparison, same pipeline, lr 1e-4
sbatch distill/run_global.sbatch AR 2 2 1e-4

# direct (no SLURM):
$PYBIN distill/distill_global.py --pattern ARRR --race-l 2 --race-k 2 \
    --seq-len 4096 --batch-size 2 --max-steps 200 --eval-every 10 \
    --lr 5e-5 --warmup-steps 20 --grad-checkpoint --kl-chunk 2048 --tag arrr
$PYBIN distill/plot_global.py --tag arrr
```
Outputs: `results/metrics_global_<tag>.jsonl`, `results/global_<tag>.out`, and 4 plots
`results/global_arrr_{eval_trends,per_layer_cosine,loss_breakdown,layer_groups}.png`.

## Success (over 200 steps)
Stable (no NaN/explosion), hidden MSE↓ / hidden cosine↑ / KL↓, frozen base unchanged,
RACE grads nonzero, student perplexity bounded. Because ARRR is aggressive, matching
the 50% model is **not** required — the goal is global rollout stability; the
`layer_groups` plot characterizes early-layer error compounding. The local pilot files
(`distill_local.py`, `run_pilot.sbatch`, `metrics_S*.jsonl`) are untouched.

## Continuation flags: CE, checkpoints, trainable hash geometry
`distill_global.py` supports continuing/extending a run:
- `--ce-weight W` — add next-token CE to the objective (`loss = hidden + kl_w·KL + W·CE`);
  per-step `train_ppl_est` is logged. Default loss-weight flags: `--hidden-weight/--kl-weight/--ce-weight`.
- `--save-every N` — checkpoint the trainable RACE params (`race.state_dict()`, ~2.1 GB) to
  `distill/checkpoints/<pattern>_L<l>_K<k>_<tag>_step<N>.pt`.
- `--load-race-checkpoint PATH` — **continue** from a prior RACE checkpoint (loads q/k/v/o + log_temp +
  hash buffers; does NOT reset the base) instead of from teacher-copied projections.
- `--train-hash-geometry` — convert the RACE soft-hash geometry `planes_T`/`protos_T` from frozen random
  **buffers into trainable `nn.Parameter`s** (via `RaceLlamaAttention.make_hash_trainable`). Gradients
  reach them through `_soft_hash → probsK/probsQ → RaceCausalFn` (autograd handles probs→planes). They get
  their **own optimizer group** at `--hash-lr` (default 1e-5), separate from q/k/v/o/log_temp at `--lr`.
  Logged: per-group lr, `proj_grad_norm`/`hash_grad_norm` per step, and `hash_drift` (L2 from init) per eval.

### Trainable-hash run (continue from step-1000, learn the LSH geometry)
```bash
source distill/env.sh
sbatch distill/run_arrr_trainable_hash.sbatch     # loads step1000, --train-hash-geometry, ce 0.3, hash-lr 1e-5
# direct:
$PYBIN distill/distill_global.py --pattern ARRR --race-l 2 --race-k 2 \
    --seq-len 4096 --batch-size 2 --max-steps 1000 --eval-every 25 --save-every 250 \
    --load-race-checkpoint checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt --train-hash-geometry \
    --lr 5e-5 --hash-lr 1e-5 --warmup-steps 50 --grad-clip 1.0 \
    --hidden-weight 1.0 --kl-weight 0.5 --ce-weight 0.3 --grad-checkpoint --tag arrr_trainhash
$PYBIN distill/plot_global.py --tag arrr_trainhash --out-prefix arrr_trainhash --compare arrr_1k_ce arrr
```
6 plots: `results/arrr_trainhash_{eval_trends,loss_breakdown,per_layer_cosine,layer_groups,compare_previous,grad_norms}.png`.
Tests whether **learning the hash geometry** beats the frozen-random run (target: ppl<100, KL<1.5). Prior
runs/checkpoints are not overwritten (new tag).
