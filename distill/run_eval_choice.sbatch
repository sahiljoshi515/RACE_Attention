#!/bin/bash
#SBATCH --job-name=eval_choice
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# MMLU / HellaSwag / Winogrande loglikelihood baseline: teacher (ceiling) + the two
# best collapsed v2 checkpoints (AR/S24, ARRR/S8). Loglik scoring bypasses the free-gen
# collapse, so this both delivers the benchmark harness AND tells us whether the collapsed
# models still retain choice-discrimination (expected: MMLU partially retained).
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60          # model+tok cached; the 3 MC datasets stream from hub
COMMON="--tasks mmlu hellaswag winogrande --max-examples 200"

CUDA_VISIBLE_DEVICES=0 "$PYBIN" eval_choice.py --model teacher $COMMON \
    > results/eval_choice_teacher.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PYBIN" eval_choice.py --model ar \
    --checkpoint checkpoints/ar_L3_K3_abl2_ar_s24h_step2000.pt $COMMON \
    > results/eval_choice_ar.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 "$PYBIN" eval_choice.py --model arrr \
    --checkpoint checkpoints/arrr_L2_K2_abl2_arrr_s8_step2000.pt $COMMON \
    > results/eval_choice_arrr.log 2>&1 &
wait

echo "########## per-model tails ##########"
for f in results/eval_choice_teacher.log results/eval_choice_ar.log results/eval_choice_arrr.log; do
  echo "=== $f ==="; tail -12 "$f" || true
done
echo "########## summary ##########"
"$PYBIN" eval_choice.py --summary || true
cat results/choice_summary.md || true
echo "EVAL_CHOICE_DONE"
