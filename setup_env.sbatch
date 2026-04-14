#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/katoch.aa/text-to-motion/logs/setup_env_%j.out
#SBATCH --error=/scratch/katoch.aa/text-to-motion/logs/setup_env_%j.err

set -e

ENV_NAME="motion-s"
ENV_DIR="/home/katoch.aa/.conda/envs/${ENV_NAME}"
LOG_DIR="/scratch/katoch.aa/text-to-motion/logs"
mkdir -p "$LOG_DIR"

echo "=============================="
echo " Setting up ${ENV_NAME} env"
echo " Node   : $(hostname)"
echo " GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=============================="

# ── Load modules ──────────────────────────────────────────────────────────────
module load anaconda3/2024.06
module load cuda/12.1.1

source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Remove old env if exists ──────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} \|${ENV_DIR}"; then
    echo "[setup] Removing existing '${ENV_NAME}' env..."
    conda env remove -p "$ENV_DIR" -y
fi

# ── Create fresh env with Python 3.11 ────────────────────────────────────────
echo "[setup] Creating '${ENV_NAME}' with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y

conda activate "$ENV_DIR"
PYTHON="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

echo "[setup] Python: $($PYTHON --version)"

# ── Install PyTorch cu121 (Python 3.11, fully supported) ─────────────────────
echo "[setup] Installing PyTorch cu121..."
$PIP install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ── Install remaining dependencies ───────────────────────────────────────────
echo "[setup] Installing transformers, sentencepiece, pandas, numpy, tqdm..."
$PIP install transformers sentencepiece pandas numpy tqdm

# ── Verify everything ─────────────────────────────────────────────────────────
echo ""
echo "====== Verification ======"
$PYTHON -c "
import torch, transformers, sentencepiece, pandas, numpy, tqdm

print('torch         :', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version  :', torch.version.cuda)
print('GPU           :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('transformers  :', transformers.__version__)
print('sentencepiece :', sentencepiece.__version__)
print('pandas        :', pandas.__version__)
print('numpy         :', numpy.__version__)

# Quick tensor test on GPU
if torch.cuda.is_available():
    x = torch.randn(3, 4).cuda()
    print('GPU tensor test: PASSED', x.shape)
else:
    print('GPU tensor test: FAILED - CUDA not available')
"
echo "=========================="
echo "Done. Activate with: conda activate ${ENV_DIR}"
