#!/bin/bash
#SBATCH --job-name=ksl-momask
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/train_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/train_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT_DIR=$WORKDIR/checkpoints
LOG_DIR=$WORKDIR/logs

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# ── Environment ────────────────────────────────────────────────────────────────
module load cuda/12.3.0

# Use the existing conda environment or fall back to the system Python
if command -v conda &>/dev/null && conda env list | grep -q "^vit-cd "; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate vit-cd
    PYTHON=$(which python)
else
    module load python/3.13.5
    PYTHON=$(which python3)
fi

echo "Python: $PYTHON"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Install missing Python deps (first run only) ───────────────────────────────
$PYTHON -m pip install --quiet --user \
    torch torchvision \
    transformers sentencepiece \
    pandas numpy tqdm

# ── Launch training ────────────────────────────────────────────────────────────
cd "$WORKDIR"

$PYTHON train.py \
    --csv       "$DATA_DIR/train.csv" \
    --ckpt_dir  "$CKPT_DIR" \
    --epochs    100 \
    --batch_size 32 \
    --lr        1e-4 \
    --t5        t5-base \
    --seed      42

echo "Job finished at $(date)"
