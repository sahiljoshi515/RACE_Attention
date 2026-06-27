#!/bin/bash
#SBATCH --job-name=longbench
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# LongBench (English subset, default tasks, 50/task) on teacher + one hybrid (env CKPT).
set -uo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export HF_HUB_DOWNLOAD_TIMEOUT=60
: "${CKPT:?set CKPT to the hybrid checkpoint path}"
COMMON="--max-examples 50"
CUDA_VISIBLE_DEVICES=0 "$PYBIN" eval_longbench.py --model teacher --out-tag teacher $COMMON \
    > results/lb_teacher.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 "$PYBIN" eval_longbench.py --model ar --checkpoint "$CKPT" --out-tag stageE $COMMON \
    > results/lb_stageE.log 2>&1 &
wait
echo "########## LongBench tails ##########"
for f in results/lb_teacher.log results/lb_stageE.log; do echo "=== $f ==="; tail -6 "$f"; done
"$PYBIN" eval_longbench.py --summary || true
echo "LONGBENCH_DONE"
