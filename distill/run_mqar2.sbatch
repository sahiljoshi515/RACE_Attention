#!/bin/bash
#SBATCH --job-name=mqar_delta2
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# KEY TEST round 2: remove the shared frozen-hash bottleneck and probe a few regimes
# to characterize delta-vs-mean on MQAR recall.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh

run () { echo; echo "##### CONFIG: $* #####"; CUDA_VISIBLE_DEVICES=0 "$PYBIN" distill/mqar_test.py "$@"; }

# (A) trainable hash, S=24, original task (8 pairs / 64 values)
run --trainable-hash --steps 4000 --batch 64 --eval-batch 512 --lr 3e-3 --log-every 500 \
    --hidden 128 --heads 4 --kv-heads 4 --layers 2 --mlp-hidden 256 \
    --race-l 6 --race-k 2 --kv-pairs 8 --queries 8 --num-keys 64 --num-values 64

# (B) trainable hash, BIGGER S=96 (L=24,K=2) -> more buckets, less collision
run --trainable-hash --steps 4000 --batch 64 --eval-batch 512 --lr 3e-3 --log-every 500 \
    --hidden 128 --heads 4 --kv-heads 4 --layers 2 --mlp-hidden 256 \
    --race-l 24 --race-k 2 --kv-pairs 8 --queries 8 --num-keys 64 --num-values 64

# (C) trainable hash, S=24, EASIER task (4 pairs / 16 values) -> recall feasible for tiny model
run --trainable-hash --steps 4000 --batch 64 --eval-batch 512 --lr 3e-3 --log-every 500 \
    --hidden 128 --heads 4 --kv-heads 4 --layers 2 --mlp-hidden 256 \
    --race-l 6 --race-k 2 --kv-pairs 4 --queries 4 --num-keys 16 --num-values 16

echo "===== MQAR2_DONE ====="
