#!/bin/bash
#SBATCH --job-name=ksl-momask
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/train_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/train_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT_DIR=$WORKDIR/checkpoints
LOG_DIR=$WORKDIR/logs

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Environment ────────────────────────────────────────────────────────────────
module load anaconda3/2024.06

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/katoch.aa/.conda/envs/motion-s

PYTHON=/home/katoch.aa/.conda/envs/motion-s/bin/python

echo "Python  : $($PYTHON --version)"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
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

$PYTHON train.py \
    --csv        "$DATA_DIR/train.csv" \
    --ckpt_dir   "$CKPT_DIR" \
    --epochs     100 \
    --batch_size 8 \
    --lr         3e-5 \
    --t5         t5-base \
    --seed       42 \
    $RESUME_FLAG

echo "Job finished at $(date)"
