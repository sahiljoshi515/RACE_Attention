#!/bin/bash
#SBATCH --job-name=race_smoke
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
# End-to-end smoke for the continued-distillation infra (compute node = has internet).
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh
rm -f distill/checkpoints/*smoke* distill/results/metrics_global_smoke*   # fresh resume test
P="$PYBIN distill/distill_global.py --grad-checkpoint --warmup-steps 1 --kl-chunk 1024"

echo "===== 1. data_long self-smoke ====="
"$PYBIN" distill/data_long.py || echo "DATA_LONG SMOKE FAILED"

echo "===== 2. trainer fineweb 3-step (1 GPU, default path) ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 2 --race-k 2 --seq-len 1024 \
    --batch-size 1 --max-steps 3 --eval-every 2 --tag smoke_fw

echo "===== 3. trainer mixed 3-step (data_long) ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 4 --race-k 4 --data mixed --seq-len 1024 \
    --batch-size 1 --max-steps 3 --eval-every 2 --tag smoke_mix

echo "===== 4. curriculum boundary 4-step (1024 -> 2048 at step 2) ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 4 --race-k 4 --curriculum "1024:0,2048:2" \
    --batch-size 1 --max-steps 4 --eval-every 10 --eval-seqlen 1024 --tag smoke_cur

echo "===== 5a. grad-accum=2 + cosine 4-step ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 2 --race-k 2 --seq-len 1024 --batch-size 1 \
    --grad-accum 2 --lr-schedule cosine --total-steps 4 --max-steps 4 --eval-every 10 \
    --save-resume-every 2 --tag smoke_res
echo "===== 5b. resume auto -> continue to step 6 ====="
CUDA_VISIBLE_DEVICES=0 $P --pattern AR --race-l 2 --race-k 2 --seq-len 1024 --batch-size 1 \
    --grad-accum 2 --lr-schedule cosine --total-steps 4 --max-steps 6 --eval-every 10 \
    --save-resume-every 2 --resume auto --tag smoke_res

echo "===== 6. DDP 2-GPU 3-step (manual allreduce) ====="
"$PYBIN" -m torch.distributed.run --standalone --nproc_per_node=2 \
    distill/distill_global.py --grad-checkpoint --warmup-steps 1 --kl-chunk 1024 --ddp \
    --pattern AR --race-l 2 --race-k 2 --seq-len 1024 --batch-size 1 --max-steps 3 \
    --eval-every 10 --tag smoke_ddp

echo "===== 7. teacher-answers-needle sanity ====="
CUDA_VISIBLE_DEVICES=0 "$PYBIN" distill/smoke_needle.py || echo "NEEDLE SMOKE FAILED"

echo "===== SMOKE_DONE ====="
