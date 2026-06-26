#!/bin/bash
#SBATCH --job-name=eval_choice_ckpt
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# MMLU/HellaSwag/Winogrande loglik eval of one hybrid checkpoint (env CKPT). Backs up the
# prior choice_ar.json (collapsed baseline) so the recovery is comparable.
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
: "${CKPT:?set CKPT to the checkpoint path}"
[ -f results/choice_ar.json ] && cp -f results/choice_ar.json results/choice_ar_baseline.json
"$PYBIN" eval_choice.py --model ar --checkpoint "$CKPT" \
    --tasks mmlu hellaswag winogrande --max-examples 200
"$PYBIN" eval_choice.py --summary
cat results/choice_summary.md
echo "EVAL_CHOICE_CKPT_DONE"
