#!/bin/bash
#SBATCH --job-name=test_distill_local
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/test_distill_local_%j.out

set -e
cd /scratch/sj157/RACE_Attention
source distill/env.sh
echo "==== tests_distill_local ===="
$PYBIN distill/tests_distill_local.py
echo DONE
