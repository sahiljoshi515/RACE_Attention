# Reproducing the RACE-hybrid from scratch (Llama-3.2-3B-Instruct → AR/S24, current best)

This is the **exact, validated path** that took stock `meta-llama/Llama-3.2-3B-Instruct` to the current
best checkpoint `checkpoints/best/race_hybrid_AR_S24_p1g_curriculum_step1050.pt`. Five stages, each
continuing the previous via `--load-race-checkpoint`. All runs: **1 node × 8 H200**, DDP, bf16, grad
checkpointing, via `run_distill_ddp.sbatch` (offline `env.sh` / `race_vit_env` / `.torch_ext_distill`).

## Architecture (fixed for every stage)
AR/S24 = the 14 **odd** decoder layers (1,3,…,27) of the 28-layer Llama have their softmax attention
replaced by **RACE soft-LSH attention**, `--race-l 3 --race-k 3` → **S = L·2^K = 24 buckets**. The 14
even layers keep softmax (retrieval heads). No learnable hash. Embeddings + kept-softmax q/k/v/o are
**always frozen**; in the unfreeze stages the **MLPs + norms** also train.

## Objective (`global_utils.py`)
`loss = hidden_weight·MSE(per-layer hidden) + kl_weight·T²·KL(softmax(teacher/T) ‖ softmax(student/T)) + ce_weight·CE(next-token)`
— full end-to-end **global rollout** (student runs on its own hidden states), teacher detached, T=1.
This is logit/KD distillation with dark knowledge + a hidden-state alignment term + a hard-label CE term.

## The five stages

| # | tag | loads | steps | seq-len (curriculum) | h / kl / ce | LR race / base | data (`--lm-frac`) | ≈ tokens | wall-clock (8×H200) |
|---|---|---|--:|---|---|---|---|--:|--:|
| **A** align | `p1a_align_v2` | stock Llama | 1000 | 4096 | **1 / 0 / 0** | 1e-4 / frozen | FineWeb | ~0.13B | 0h36 |
| **B** KD | `p1b_kd` | A | 1340 | 4096 | **0.1 / 1 / 0.5** | 1e-4 / frozen | mixed (0.7) | ~0.35B | 1h35 |
| **C** ctx-ext | `p1c_ext` | B | 450 | 4096→16384 | **0 / 0.3 / 1** | 5e-5 / frozen | mixed (0.7) | ~0.35B | 0h56 |
| **F** unfreeze | `p1f_unfreeze` | C | 1340 | 4096 | **0.1 / 1 / 0.5** | 5e-5 / **5e-5 (mlp)** | mixed (0.7) | ~0.35B | 1h48 |
| **G** curriculum | `p1g_curriculum` | F | 1050 | 4096→16384→32768 | **0.1 / 1 / 0.5** | 5e-5 / **5e-5 (mlp)** | mixed (0.6) + task-aug | ~1.2B | 5h34 |

**Totals: ≈ 2.4B training tokens, ≈ 10.8 h wall-clock on one 8×H200 node ≈ ~87 H200-GPU-hours.**
(Token counts = micro-batch 2 × grad-accum × seq-len × 8 GPUs × steps, summed over curriculum phases;
"mixed" = FineWeb LM + synthetic-RULER retrieval, `--lm-frac` is the LM fraction.)

Note the path **skips** two abandoned branches: `p1d_ruler` (explicit-32K push that *regressed* RULER)
and `p1e_unfreeze` (under-dosed base-LR 1e-5). F loads directly from **C**.

## Exact commands

`--curriculum` / `--task-weights` contain commas, which `sbatch --export=ALL,VAR=...` splits on — so set
those vars as a **prefix env assignment** with a bare `--export=ALL`. Run from repo root.

```bash
cd /scratch/sj157/RACE_Attention/distill

# ---- Stage A: hidden-state alignment (RACE-only, lr 1e-4 — NOT 1e-3, which diverges) ----
sbatch --export=ALL,PATTERN=AR,TAG=p1a_align_v2,ARGS="--race-l 3 --race-k 3 --seq-len 4096 \
--batch-size 2 --grad-accum 2 --max-steps 1000 --lr 1e-4 --lr-schedule cosine --total-steps 1000 \
--warmup-steps 60 --min-lr-ratio 0.05 --grad-clip 1.0 --hidden-weight 1.0 --kl-weight 0.0 \
--ce-weight 0.0 --kl-chunk 2048 --grad-checkpoint --data fineweb --eval-every 80 --probe-every 500 \
--probe-niah 8 --save-every 1000 --save-resume-every 250" run_distill_ddp.sbatch

# ---- Stage B: logit KD + CE (de-collapse) ----
sbatch --export=ALL,PATTERN=AR,TAG=p1b_kd,ARGS="--race-l 3 --race-k 3 --seq-len 4096 --batch-size 2 \
--grad-accum 4 --max-steps 1340 --lr 1e-4 --lr-schedule cosine --total-steps 1340 --warmup-steps 100 \
--min-lr-ratio 0.1 --grad-clip 1.0 --hidden-weight 0.1 --kl-weight 1.0 --ce-weight 0.5 --kl-temp 1.0 \
--kl-chunk 2048 --grad-checkpoint --data mixed --lm-frac 0.7 --long-source fineweb_local \
--eval-every 100 --probe-every 100 --probe-niah 8 --save-every 335 --save-resume-every 200 \
--load-race-checkpoint checkpoints/ar_L3_K3_p1a_align_v2_step1000.pt" run_distill_ddp.sbatch

# ---- Stage C: context extension 4K->16K (CE-led) ----
CURR="4096:0,16384:150"
PATTERN=AR TAG=p1c_ext CURRICULUM="$CURR" \
ARGS="--race-l 3 --race-k 3 --seq-len 4096 --batch-size 2 --grad-accum 4 --max-steps 450 --lr 5e-5 \
--lr-schedule cosine --total-steps 450 --warmup-steps 30 --min-lr-ratio 0.1 --grad-clip 1.0 \
--hidden-weight 0.0 --kl-weight 0.3 --ce-weight 1.0 --kl-temp 1.0 --kl-chunk 2048 --grad-checkpoint \
--data mixed --lm-frac 0.7 --long-source fineweb_local --curriculum $CURR --eval-every 75 \
--eval-seqlen 4096 --probe-every 150 --probe-niah 8 --save-every 225 --save-resume-every 150 \
--load-race-checkpoint checkpoints/ar_L3_K3_p1b_kd_step1340.pt" \
  sbatch --export=ALL run_distill_ddp.sbatch

# ---- Stage F: MLP unfreeze (recovers MMLU + long-ctx) — base-lr 5e-5 is the key knob ----
PATTERN=AR TAG=p1f_unfreeze \
ARGS="--race-l 3 --race-k 3 --seq-len 4096 --batch-size 2 --grad-accum 4 --max-steps 1340 --lr 5e-5 \
--base-lr 5e-5 --unfreeze mlp --lr-schedule cosine --total-steps 1340 --warmup-steps 100 \
--min-lr-ratio 0.1 --grad-clip 1.0 --hidden-weight 0.1 --kl-weight 1.0 --ce-weight 0.5 --kl-temp 1.0 \
--kl-chunk 2048 --grad-checkpoint --data mixed --lm-frac 0.7 --long-source fineweb_local \
--eval-every 100 --eval-seqlen 4096 --probe-every 200 --probe-niah 8 --save-every 335 \
--save-resume-every 100 \
--load-race-checkpoint checkpoints/best/race_hybrid_AR_S24_p1c_ext_step450.pt" \
  sbatch --export=ALL run_distill_ddp.sbatch

# ---- Stage G: unfreeze + 4K->16K->32K length curriculum (doubles RULER-64K) ----
CURR="4096:0,16384:200,32768:800"
PATTERN=AR TAG=p1g_curriculum \
ARGS="--race-l 3 --race-k 3 --seq-len 4096 --batch-size 2 --grad-accum 4 --max-steps 1050 --lr 5e-5 \
--base-lr 5e-5 --unfreeze mlp --lr-schedule cosine --total-steps 1050 --warmup-steps 80 \
--min-lr-ratio 0.1 --grad-clip 1.0 --hidden-weight 0.1 --kl-weight 1.0 --ce-weight 0.5 --kl-temp 1.0 \
--kl-chunk 1024 --grad-checkpoint --data mixed --lm-frac 0.6 --long-source fineweb_local \
--task-weights niah_single:2,variable_tracking:2,common_words:2 --n-distractors 16 \
--curriculum $CURR --eval-every 100 --eval-seqlen 4096 --probe-every 200 --probe-niah 8 \
--save-every 350 --save-resume-every 100 \
--load-race-checkpoint checkpoints/best/race_hybrid_AR_S24_p1f_unfreeze_step1340.pt" \
  sbatch --export=ALL run_distill_ddp.sbatch
```

Prerequisites: stage the FineWeb parquet once (`python stage_fineweb.py`, writes `data/fineweb`) so the
`fineweb_local` source works (the old `pg19` source is dead). For any explicit-32K step keep
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set in `env.sh`) to avoid the 32K-transition OOM.

## Final numbers (current best, Stage G)
MMLU 35.7 / HellaSwag 56.1 / Winogrande 54.0 (5-shot, 1000 ex) · RULER 55.3@16K / 47.4@32K / 45.0@64K
(20/task) · LongBench 19.9. Teacher: 62 / 70.5 / 69.9 · 80.5/80.2/80.5 · 47.2.
