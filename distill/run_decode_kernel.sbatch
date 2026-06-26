#!/bin/bash
#SBATCH --job-name=decode_kernel
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
echo "########## build race_decode ext ##########"
"$PYBIN" /scratch/sj157/RACE_Attention/kernels/gpu/race_decode_build.py
echo "########## unit + sequence correctness ##########"
"$PYBIN" /scratch/sj157/RACE_Attention/scaling/correctness_decode_step.py
echo "########## token-identity (recompute vs cache vs cache+kernel) ##########"
"$PYBIN" validate_decode_cache.py
echo "########## decode ms/tok: kernel OFF ##########"
"$PYBIN" synthetic_decode_probe.py --lens 8192 32768 131072 262144 --decode-kernel off
echo "########## decode ms/tok: kernel ON ##########"
"$PYBIN" synthetic_decode_probe.py --lens 8192 32768 131072 262144 --decode-kernel on
