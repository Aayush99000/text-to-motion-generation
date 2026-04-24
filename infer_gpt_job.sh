#!/bin/bash
#SBATCH --job-name=ksl-gpt-infer
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/infer_gpt_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/infer_gpt_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT_DIR=$WORKDIR/checkpoints_gpt
LOG_DIR=$WORKDIR/logs

mkdir -p "$LOG_DIR"

# ── Environment ────────────────────────────────────────────────────────────────
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/katoch.aa/.conda/envs/motion-s

PYTHON=/home/katoch.aa/.conda/envs/motion-s/bin/python

echo "Python : $($PYTHON --version)"
echo "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
$PYTHON -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

cd "$WORKDIR"

echo ""
echo "===== Running T2M-GPT Inference ====="
$PYTHON inference_gpt.py \
    --checkpoint  "$CKPT_DIR/checkpoint_epoch_100.pth" \
    --test_csv    "$DATA_DIR/test.csv" \
    --output      "$WORKDIR/submission_gpt.csv" \
    --temperature 0.8 \
    --top_k       256 \
    --batch_size  32

echo ""
echo "===== Done ====="
echo "Row count: $(wc -l < $WORKDIR/submission_gpt.csv)"
