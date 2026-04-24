#!/bin/bash
#SBATCH --job-name=ksl-infer-new
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/infer_new_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/infer_new_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
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

# ── Inference 1: Sentence-only 200ep ──────────────────────────────────────────
echo ""
echo "===== T2M-GPT sentence-only ep200 t=0.6 ====="
$PYTHON inference_gpt.py \
    --checkpoint  "$WORKDIR/checkpoints_gpt_200/checkpoint_epoch_200.pth" \
    --test_csv    "$DATA_DIR/test.csv" \
    --output      "$WORKDIR/submission_gpt200_sent_t0.6.csv" \
    --temperature 0.6 \
    --top_k       256 \
    --batch_size  32
echo "Row count: $(wc -l < $WORKDIR/submission_gpt200_sent_t0.6.csv)"
ls -lh "$WORKDIR/submission_gpt200_sent_t0.6.csv"

# ── Inference 2: Gloss+sentence 200ep ─────────────────────────────────────────
echo ""
echo "===== T2M-GPT gloss+sentence ep200 t=0.6 ====="
$PYTHON inference_gpt.py \
    --checkpoint  "$WORKDIR/checkpoints_gpt_both_200/checkpoint_epoch_200.pth" \
    --test_csv    "$DATA_DIR/test.csv" \
    --output      "$WORKDIR/submission_gpt200_both_t0.6.csv" \
    --temperature 0.6 \
    --top_k       256 \
    --batch_size  32 \
    --use_both
echo "Row count: $(wc -l < $WORKDIR/submission_gpt200_both_t0.6.csv)"
ls -lh "$WORKDIR/submission_gpt200_both_t0.6.csv"

echo ""
echo "===== All inference done ====="
