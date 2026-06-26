#!/bin/bash
#SBATCH --job-name=race_distill
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/pilot_%j.out

# Usage: sbatch distill/run_pilot.sbatch <L> <Kbits>   e.g. sbatch ... 2 2   or   3 3
set -e
cd /scratch/sj157/RACE_Attention
source distill/env.sh
L=${1:-2}; K=${2:-2}
echo "==== pilot L=$L K=$K ===="
$PYBIN distill/distill_local.py --L $L --Kbits $K --M 1 \
    --batch-size 8 --seq-len 4096 --steps 100 --eval-every 10 \
    --lr 1e-4 --clip 1.0 --warmup 10
echo DONE
