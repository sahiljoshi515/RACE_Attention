#!/bin/bash
#SBATCH --job-name=ruler_full_65536
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -euo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
COMMON="--context-len 65536 --max-examples 20"
echo "########## TEACHER 65536 ##########"
"$PYBIN" eval_ruler.py --model teacher $COMMON
echo "########## ARRR 65536 ##########"
"$PYBIN" eval_ruler.py --model arrr --checkpoint checkpoints/arrr_L2_K2_arrr_1k_ce_step1000.pt $COMMON
echo "########## DONE 65536 ##########"
