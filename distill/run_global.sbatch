#!/bin/bash
#SBATCH --job-name=race_global
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/global_%j.out

# GLOBAL hybrid-student distillation (end-to-end rollout). Two model instances:
# frozen teacher + hybrid student (pattern-selected RACE layers). B=2 + gradient
# checkpointing + chunked KL. Uses race_vit_env (torch 2.10) + .torch_ext_distill
# (NOT env_fa3.sh -- that is the torch-2.8 FA3 path).
#
# Usage: sbatch distill/run_global.sbatch <PATTERN> <L> <K> [lr] [batch] [steps]
#   ARRR main:      sbatch distill/run_global.sbatch ARRR 2 2
#   AR comparison:  sbatch distill/run_global.sbatch AR   2 2 1e-4
set -e
cd /scratch/sj157/RACE_Attention
source distill/env.sh

PATTERN=${1:-ARRR}; L=${2:-2}; K=${3:-2}; LR=${4:-5e-5}; BSZ=${5:-2}; STEPS=${6:-200}
TAG=$(echo "$PATTERN" | tr '[:upper:]' '[:lower:]')
echo "==== global distill pattern=$PATTERN L=$L K=$K lr=$LR B=$BSZ steps=$STEPS tag=$TAG ===="

$PYBIN distill/distill_global.py --pattern "$PATTERN" --race-l "$L" --race-k "$K" --M 1 \
    --seq-len 4096 --batch-size "$BSZ" --max-steps "$STEPS" --eval-every 10 \
    --lr "$LR" --warmup-steps 20 --clip 1.0 \
    --hidden-weight 1.0 --kl-weight 0.5 --kl-temp 1.0 --ce-weight 0.0 \
    --grad-checkpoint --kl-chunk 2048 --seed 0 --tag "$TAG"

$PYBIN distill/plot_global.py --tag "$TAG" || echo "(plotting failed; rerun plot_global.py manually)"
echo DONE
