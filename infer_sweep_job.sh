#!/bin/bash
#SBATCH --job-name=ksl-gpt-sweep
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/sweep_gpt_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/sweep_gpt_%j.err

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT=$WORKDIR/checkpoints_gpt/checkpoint_epoch_100.pth
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

for TEMP in 0.3 0.5 0.6 0.7; do
    OUT="$WORKDIR/submission_gpt_t${TEMP}.csv"
    echo ""
    echo "===== Temperature $TEMP ====="
    $PYTHON inference_gpt.py \
        --checkpoint  "$CKPT" \
        --test_csv    "$DATA_DIR/test.csv" \
        --output      "$OUT" \
        --temperature $TEMP \
        --top_k       256 \
        --batch_size  32
    echo "Rows: $(wc -l < $OUT)"
done

echo ""
echo "===== Sweep complete ====="
ls -lh "$WORKDIR"/submission_gpt_t*.csv
