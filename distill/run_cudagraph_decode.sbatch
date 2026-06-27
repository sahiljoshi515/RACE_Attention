#!/bin/bash
#SBATCH --job-name=cudagraph_decode
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
set -euo pipefail
cd /scratch/sj157/RACE_Attention/distill
source /scratch/sj157/RACE_Attention/distill/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "########## build race_decode ext (with current-stream launch fix) ##########"
"$PYBIN" /scratch/sj157/RACE_Attention/kernels/gpu/race_decode_build.py

echo "########## CUDAGRAPH decode, batch=1, sdpa=efficient (fair attn) ##########"
"$PYBIN" cudagraph_decode_probe.py --lens 8192 32768 65536 131072 --batch 1 --sdpa efficient

echo "########## CUDAGRAPH decode, batch=8, sdpa=efficient ##########"
"$PYBIN" cudagraph_decode_probe.py --lens 8192 32768 --batch 8 --sdpa efficient
