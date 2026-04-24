#!/bin/bash
#SBATCH --job-name=ksl-both200
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/train_both200_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/train_both200_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT_DIR=$WORKDIR/checkpoints_gpt_both_200
LOG_DIR=$WORKDIR/logs
TOTAL_EPOCHS=200

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Auto-chain: queue next job immediately ─────────────────────────────────────
NEXT_JOB=$(sbatch --dependency=afterany:$SLURM_JOB_ID "$WORKDIR/train_gpt_both_200_job.sh" | awk '{print $4}')
echo "Next job queued: $NEXT_JOB (will start after this job ends)"

# ── Environment ────────────────────────────────────────────────────────────────
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/katoch.aa/.conda/envs/motion-s

PYTHON=/home/katoch.aa/.conda/envs/motion-s/bin/python

echo "Python : $($PYTHON --version)"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ── Auto-resume from latest checkpoint ────────────────────────────────────────
LATEST_CKPT=$(ls -t "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Resuming from: $LATEST_CKPT"
    RESUME_FLAG="--resume $LATEST_CKPT"
else
    echo "No checkpoint found, starting fresh."
    RESUME_FLAG=""
fi

# ── Launch training ────────────────────────────────────────────────────────────
cd "$WORKDIR"

$PYTHON train_gpt.py \
    --csv        "$DATA_DIR/train.csv" \
    --ckpt_dir   "$CKPT_DIR" \
    --epochs     $TOTAL_EPOCHS \
    --batch_size 8 \
    --lr         3e-4 \
    --t5         t5-base \
    --max_frames 128 \
    --warmup     500 \
    --seed       42 \
    --use_both \
    $RESUME_FLAG

# ── Cancel next job if training is complete ────────────────────────────────────
LAST_CKPT=$(ls -t "$CKPT_DIR"/checkpoint_epoch_*.pth 2>/dev/null | head -1)
LAST_EPOCH=$(echo "$LAST_CKPT" | grep -o '[0-9]\+' | tail -1)
if [ -n "$LAST_EPOCH" ] && [ "$LAST_EPOCH" -ge "$TOTAL_EPOCHS" ]; then
    echo "Training complete at epoch $LAST_EPOCH. Cancelling queued job $NEXT_JOB."
    scancel $NEXT_JOB
else
    echo "Job finished. Epoch $LAST_EPOCH/$TOTAL_EPOCHS done. Next job $NEXT_JOB will continue."
fi
