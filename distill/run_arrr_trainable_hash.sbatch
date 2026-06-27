#!/bin/bash
#SBATCH --job-name=arrr_trainhash
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/arrr_trainhash_%j.out

# Continue ARRR (75% RACE) global distillation from the step-1000 checkpoint, now with
# TRAINABLE hash geometry (planes_T/protos_T -> nn.Parameter) on a separate, smaller
# lr group, and a stronger CE (0.3). 1000 more steps. race_vit_env (torch 2.10) +
# .torch_ext_distill (NOT env_fa3.sh). Readiness gate — no RULER/LongBench.
set -e
cd /scratch/sj157/RACE_Attention
source distill/env.sh

$PYBIN distill/distill_global.py --pattern ARRR --race-l 2 --race-k 2 --M 1 \
    --seq-len 4096 --batch-size 2 --max-steps 1000 --eval-every 25 --save-every 250 \
    --load-race-checkpoint checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt \
    --train-hash-geometry \
    --lr 5e-5 --hash-lr 1e-5 --warmup-steps 50 --grad-clip 1.0 \
    --hidden-weight 1.0 --kl-weight 0.5 --kl-temp 1.0 --ce-weight 0.3 \
    --grad-checkpoint --kl-chunk 2048 --seed 0 --tag arrr_trainhash

$PYBIN distill/plot_global.py --tag arrr_trainhash --out-prefix arrr_trainhash \
    --compare arrr_1k_ce arrr || echo "(plotting failed; rerun plot_global.py manually)"
echo DONE
