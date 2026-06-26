#!/bin/bash
#SBATCH --job-name=smoke_probe
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# P2 smoke: 6-step AR/S24 run loading the real (collapsed) ar_s24h ckpt, with the in-loop
# de-collapse probe ON. Validates: --load-race-checkpoint path, probe logs collapsed_frac/
# niah4k_acc (expect ~1.0 / ~0.0 on the collapsed ckpt), and training CONTINUES correctly
# afterward (per-step base-grad==0 + final assert_base_unchanged + finite-grad asserts).
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
"$PYBIN" distill/distill_global.py \
    --pattern AR --race-l 3 --race-k 3 \
    --seq-len 4096 --batch-size 2 --grad-accum 1 --max-steps 6 \
    --eval-every 3 --probe-every 3 --probe-niah 8 --probe-max-new 16 \
    --lr 1e-4 --warmup-steps 2 --hidden-weight 0.1 --kl-weight 1.0 --ce-weight 0.5 \
    --kl-chunk 2048 --grad-checkpoint --data fineweb --tag smoke_probe \
    --load-race-checkpoint checkpoints/ar_L3_K3_abl2_ar_s24h_step2000.pt
echo "SMOKE_PROBE_DONE"
