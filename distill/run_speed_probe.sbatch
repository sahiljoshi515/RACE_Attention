#!/bin/bash
#SBATCH --job-name=speed_probe
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -euo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
"$PYBIN" speed_probe.py --context-lens 32768 65536 --n-steps 40 --warmup 8
