#!/bin/bash
#SBATCH --job-name=race_ablation
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# Phase-0 architecture ablation: ~6 cheap 4K cells run in PARALLEL, one per GPU (single-GPU
# each, NO DDP). Picks the architecture (pattern x S x hash) before the big curriculum run.
# Each cell saves a numbered eval-compatible checkpoint at the end for the RULER-4K probe.
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh

run_cell () {   # $1=gpu  $2=tag  $3...=extra distill args
  local gpu=$1; shift; local tag=$1; shift
  CUDA_VISIBLE_DEVICES=$gpu "$PYBIN" distill/distill_global.py \
      --seq-len 4096 --batch-size 2 --max-steps 400 --eval-every 50 --save-every 400 \
      --lr 5e-5 --warmup-steps 30 --lr-schedule cosine --total-steps 400 \
      --hidden-weight 1.0 --kl-weight 0.5 --ce-weight 0.1 --kl-chunk 2048 \
      --grad-checkpoint --seed 0 --tag "$tag" "$@" \
      > distill/results/ablation_${tag}.log 2>&1 &
}

# baseline (current failing arch) + capacity sweep. S = L * 2^K: S8=(2,2) S24=(3,3) S64=(4,4).
run_cell 0 abl_arrr_s8    --pattern ARRR    --race-l 2 --race-k 2
run_cell 1 abl_arrr_s64   --pattern ARRR    --race-l 4 --race-k 4
run_cell 2 abl_arrr_s64h  --pattern ARRR    --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3
run_cell 3 abl_ar_s24h    --pattern AR      --race-l 3 --race-k 3 --train-hash-geometry --hash-lr 1e-3
run_cell 4 abl_ar_s64h    --pattern AR      --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3
run_cell 5 abl_edges_s64h --pattern edges:4 --race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3
wait
echo "########## ablation cells done; per-cell logs in results/ablation_*.log ##########"
grep -H "eval@350\|eval@400" distill/results/ablation_*.log || true
