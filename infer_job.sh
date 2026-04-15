#!/bin/bash
#SBATCH --job-name=ksl-infer-gloss
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/infer_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/infer_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT_DIR=$WORKDIR/checkpoints
LOG_DIR=$WORKDIR/logs

mkdir -p "$LOG_DIR"

# ── Environment ────────────────────────────────────────────────────────────────
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/katoch.aa/.conda/envs/motion-s

PYTHON=/home/katoch.aa/.conda/envs/motion-s/bin/python

echo "Python  : $($PYTHON --version)"
echo "GPU     : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

CHECKPOINT="$WORKDIR/checkpoints_gloss/checkpoint_epoch_100.pth"
LENGTH_EST="$WORKDIR/length_estimator_gloss.pth"
SUBMISSION="$WORKDIR/submission_gloss_v2.csv"

cd "$WORKDIR"

# ── Step 1: Train length estimator ────────────────────────────────────────────
echo ""
echo "===== Step 1: Train Length Estimator ====="
$PYTHON train_length_estimator.py \
    --checkpoint "$CHECKPOINT" \
    --csv        "$DATA_DIR/train.csv" \
    --output     "$LENGTH_EST" \
    --epochs     30 \
    --batch_size 128 \
    --lr         1e-3 \
    --use_gloss

echo ""
echo "===== Step 2: Run Inference ====="
$PYTHON inference.py \
    --checkpoint       "$CHECKPOINT" \
    --length_estimator "$LENGTH_EST" \
    --test_csv         "$DATA_DIR/test.csv" \
    --output           "$SUBMISSION" \
    --batch_size       32 \
    --num_iter         25 \
    --temperature      0.7 \
    --top_k            256 \
    --use_gloss

echo ""
echo "===== Done ====="
echo "Submission saved → $SUBMISSION"
echo "Row count: $(wc -l < $SUBMISSION)"
