#!/bin/bash
#SBATCH --job-name=mqar_delta
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# THE KEY TEST: delta-RACE vs mean-RACE on synthetic MQAR associative recall.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh

echo "===== MQAR: delta-RACE vs mean-RACE (S=24, L=6 K=2, 2 layers, d=128) ====="
CUDA_VISIBLE_DEVICES=0 "$PYBIN" distill/mqar_test.py \
    --steps 4000 --batch 64 --eval-batch 512 --lr 3e-3 --log-every 250 \
    --hidden 128 --heads 4 --kv-heads 4 --layers 2 --mlp-hidden 256 \
    --race-l 6 --race-k 2 \
    --kv-pairs 8 --queries 8 --num-keys 64 --num-values 64

echo "===== MQAR_DONE ====="
