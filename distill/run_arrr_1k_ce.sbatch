#!/bin/bash
#SBATCH --job-name=arrr_1k_ce
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/arrr_1k_ce_%j.out

# Longer GLOBAL ARRR (75% RACE) distillation with a CE term, to recover basic LM
# quality (perplexity/logits) before any long-context eval. 1000 steps, B=2 +
# gradient checkpointing + chunked KL, checkpoints every 250 steps. race_vit_env
# (torch 2.10) + .torch_ext_distill (NOT env_fa3.sh).
set -e
cd /scratch/sj157/RACE_Attention
source distill/env.sh

$PYBIN distill/distill_global.py --pattern ARRR --race-l 2 --race-k 2 --M 1 \
    --seq-len 4096 --batch-size 2 --max-steps 1000 --eval-every 25 --save-every 250 \
    --lr 5e-5 --warmup-steps 50 --grad-clip 1.0 \
    --hidden-weight 1.0 --kl-weight 0.5 --kl-temp 1.0 --ce-weight 0.1 \
    --grad-checkpoint --kl-chunk 2048 --seed 0 --tag arrr_1k_ce

$PYBIN distill/plot_global.py --tag arrr_1k_ce --out-prefix arrr_1k_ce --compare arrr ar \
    || echo "(plotting failed; rerun plot_global.py manually)"
echo DONE
