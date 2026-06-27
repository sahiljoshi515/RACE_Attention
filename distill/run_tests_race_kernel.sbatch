#!/bin/bash
#SBATCH --job-name=race_kernel_tests
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
echo "===== RACE custom CUDA kernel correctness tests ====="
CUDA_VISIBLE_DEVICES=0 "$PYBIN" distill/tests_race_kernel.py
echo "EXIT_CODE=$?"
