#!/bin/bash
#SBATCH --job-name=choice_p1f335
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
"$PYBIN" eval_choice.py --model ar --checkpoint checkpoints/ar_L3_K3_p1f_unfreeze_step335.pt \
    --out-tag p1f335_5s --tasks mmlu hellaswag winogrande --max-examples 1000 --num-fewshot 5
"$PYBIN" eval_choice.py --summary
echo "CHOICE_P1F335_DONE"
