#!/bin/bash
#SBATCH --job-name=smoke_ceonly
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
"$PYBIN" distill/distill_global.py --pattern AR --race-l 3 --race-k 3 \
    --seq-len 4096 --batch-size 2 --grad-accum 1 --max-steps 4 --eval-every 2 \
    --hidden-weight 0.0 --kl-weight 0.0 --ce-weight 1.0 --grad-checkpoint \
    --data mixed --lm-frac 0.7 --long-source fineweb_local --lr 5e-5 --warmup-steps 1 \
    --tag smoke_ceonly --load-race-checkpoint checkpoints/ar_L3_K3_p1c_ext_step450.pt
echo "SMOKE_CEONLY_DONE"
