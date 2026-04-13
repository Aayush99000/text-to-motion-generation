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
# momask-env venv (Python 3.13 + torch 2.5.1+cu121) lives in $HOME
# python/3.13.5 module provides the required libpython3.13.so.1.0 runtime
module load python/3.13.5 cuda/12.1.1

PYTHON=/home/katoch.aa/momask-env/bin/python

echo "Python  : $($PYTHON --version)"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ── Launch training ────────────────────────────────────────────────────────────
cd "$WORKDIR"

$PYTHON train.py \
    --csv        "$DATA_DIR/train.csv" \
    --ckpt_dir   "$CKPT_DIR" \
    --epochs     100 \
    --batch_size 32 \
    --lr         1e-4 \
    --t5         t5-base \
    --seed       42

echo "Job finished at $(date)"
