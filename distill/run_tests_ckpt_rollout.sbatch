#!/bin/bash
#SBATCH --job-name=tests_ckpt_rollout
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/tests_ckpt_rollout_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/tests_ckpt_rollout_%j.err

set -e
cd /scratch/sj157/RACE_Attention
source distill/env.sh
echo "[job] node=$(hostname) gpu:"; nvidia-smi -L
$PYBIN distill/tests_ckpt_rollout.py
echo "[job] exit rc=$?"
