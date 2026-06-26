#!/bin/bash
#SBATCH --job-name=smoke_longbench
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
"$PYBIN" eval_longbench.py --model teacher --tasks qasper multifieldqa_en triviaqa \
    --max-examples 5 --out-tag teacher_smoke
echo "SMOKE_LONGBENCH_DONE"
