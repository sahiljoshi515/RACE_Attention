#!/bin/bash
#SBATCH --job-name=race_distill
#SBATCH --partition=commons
#SBATCH --account=as143
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --output=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#SBATCH --error=/scratch/sj157/RACE_Attention/distill/results/%x_%j.out
#
# Generic 8xH200 DDP distillation launcher with full-state auto-resume + self-resubmit so a
# multi-day run survives the 24h cap. Set these env vars when submitting:
#   ARGS    : distill_global.py args EXCEPT --pattern/--tag/--ddp/--resume (added here)
#   PATTERN : layer pattern, e.g. ARRR / AR / edges:4
#   TAG     : run tag (also names the resume ckpt + .done sentinel)
# Example (Phase 1, de-collapse at 4K, ~1B tok = 64K tok/step x ~16k steps):
#   sbatch --export=ALL,PATTERN=AR,TAG=p1_decollapse,\
#ARGS="--race-l 4 --race-k 4 --train-hash-geometry --hash-lr 1e-3 --seq-len 4096 \
#--batch-size 2 --grad-accum 1 --max-steps 16000 --lr 5e-5 --warmup-steps 200 \
#--lr-schedule cosine --total-steps 16000 --ce-weight 0.1 --kl-weight 0.5 --kl-chunk 2048 \
#--grad-checkpoint --save-every 4000 --save-resume-every 100 --eval-every 100" \
#     distill/run_distill_ddp.sbatch
set -uo pipefail
cd /scratch/sj157/RACE_Attention
source distill/env.sh

: "${ARGS:?set ARGS to the distill_global.py argument string}"
: "${PATTERN:?set PATTERN (layer pattern, e.g. ARRR/AR/edges:4)}"
: "${TAG:?set TAG}"
sani=$(echo "$PATTERN" | tr -d ':/' | tr 'A-Z' 'a-z')
SENT="distill/checkpoints/${sani}_${TAG}.done"

if [ -f "$SENT" ]; then echo "already COMPLETE ($SENT); exiting"; exit 0; fi

# Best-effort: forward SLURM's pre-timeout USR1 to the torchrun child so the trainer can
# flush a resume checkpoint. (Periodic --save-resume-every is the primary safety net.)
forward_usr1 () { echo "[sbatch] caught USR1 -> forwarding to $PID"; kill -USR1 "$PID" 2>/dev/null || true; }
trap forward_usr1 USR1

echo "[sbatch] torchrun 8xH200 | PATTERN=$PATTERN TAG=$TAG | ARGS=$ARGS"
"$PYBIN" -m torch.distributed.run --standalone --nproc_per_node=8 \
    distill/distill_global.py $ARGS --ddp --resume auto --pattern "$PATTERN" --tag "$TAG" &
PID=$!
wait "$PID"
rc=$?
echo "[sbatch] torchrun exit $rc"

if [ -f "$SENT" ]; then
    echo "[sbatch] training COMPLETE ($SENT) -- no requeue"
else
    echo "[sbatch] incomplete -> resubmitting to continue from resume ckpt"
    sbatch --export=ALL,ARGS="$ARGS",PATTERN="$PATTERN",TAG="$TAG" distill/run_distill_ddp.sbatch
fi
