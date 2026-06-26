#!/bin/bash
#SBATCH --job-name=bench_suite
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:3
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# Reliable MMLU(5-shot)/HellaSwag/Winogrande on teacher + best hybrids (Stage C, Stage B).
# 1000 examples/task, per-subject MMLU breakdown. Datasets stream from hub (not offline).
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
COMMON="--tasks mmlu hellaswag winogrande --max-examples 1000 --num-fewshot 5"

CUDA_VISIBLE_DEVICES=0 "$PYBIN" eval_choice.py --model teacher --out-tag teacher_5s $COMMON \
    > results/bench_teacher.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PYBIN" eval_choice.py --model ar \
    --checkpoint checkpoints/best/race_hybrid_AR_S24_p1c_ext_step450.pt --out-tag p1c_5s $COMMON \
    > results/bench_p1c.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 "$PYBIN" eval_choice.py --model ar \
    --checkpoint checkpoints/best/race_hybrid_AR_S24_p1b_kd_step1340.pt --out-tag p1b_5s $COMMON \
    > results/bench_p1b.log 2>&1 &
wait
echo "########## per-model tails ##########"
for f in results/bench_teacher.log results/bench_p1c.log results/bench_p1b.log; do
  echo "=== $f ==="; tail -8 "$f" || true
done
"$PYBIN" eval_choice.py --summary || true
cat results/choice_summary.md || true
echo "BENCH_SUITE_DONE"
