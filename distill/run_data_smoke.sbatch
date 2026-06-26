#!/bin/bash
#SBATCH --job-name=race_data_smoke
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# Verify the local-parquet + offline-HF data path produces training steps FAST on a compute
# node (the fix for the idle-reaper: no hub call at job start). Trains ~15 real steps.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
# Force offline: data is local (data/fineweb/*.parquet), tokenizer is cached -> no hub call
# can hang on the HF 504 outage.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "local fineweb shards:"; ls -la distill/../data/fineweb/**/*.parquet 2>/dev/null || ls -R data/fineweb 2>/dev/null | head
echo "=== train 15 steps (expect step logs within ~1 min) ==="
CUDA_VISIBLE_DEVICES=0 "$PYBIN" distill/distill_global.py --pattern ARRR --race-l 2 --race-k 2 \
    --seq-len 4096 --batch-size 2 --max-steps 15 --eval-every 10 --warmup-steps 2 \
    --lr 5e-5 --ce-weight 0.1 --kl-weight 0.5 --kl-chunk 2048 --grad-checkpoint \
    --data fineweb --tag data_smoke
echo "=== DATA_SMOKE_DONE ==="
