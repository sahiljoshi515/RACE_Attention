#!/bin/bash
#SBATCH --job-name=decode_scaling
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -euo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
echo "########## token-identical re-validation (post soft-hash batching) ##########"
"$PYBIN" validate_decode_cache.py
echo "########## decode scaling: teacher vs hybrid ##########"
"$PYBIN" synthetic_decode_probe.py --lens 8192 32768 65536 131072 262144 524288 --teacher-max 262144
