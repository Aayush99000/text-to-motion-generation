#!/bin/bash
#SBATCH --job-name=ksl-ep-sweep
#SBATCH --partition=sharing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/infer_ep_sweep_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/infer_ep_sweep_%j.err

WORKDIR=/scratch/katoch.aa/text-to-motion
DATA_DIR=$WORKDIR/dataset
CKPT_DIR=$WORKDIR/checkpoints_gpt_200

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/katoch.aa/.conda/envs/motion-s
PYTHON=/home/katoch.aa/.conda/envs/motion-s/bin/python

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
cd "$WORKDIR"

for EP in 050 075 100; do
    echo ""
    echo "===== Sentence ep${EP} t=0.6 ====="
    $PYTHON inference_gpt.py \
        --checkpoint  "$CKPT_DIR/checkpoint_epoch_${EP}.pth" \
        --test_csv    "$DATA_DIR/test.csv" \
        --output      "$WORKDIR/submission_gpt200_sent_ep${EP}_t0.6.csv" \
        --temperature 0.6 \
        --top_k       256 \
        --batch_size  32
    echo "Rows: $(wc -l < $WORKDIR/submission_gpt200_sent_ep${EP}_t0.6.csv)"
    ls -lh "$WORKDIR/submission_gpt200_sent_ep${EP}_t0.6.csv"
done

echo ""
echo "===== All done ====="
