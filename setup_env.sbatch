#!/bin/bash
#SBATCH --job-name=setup-env
#SBATCH --partition=sharing
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

# ── Load anaconda module only (PyTorch pip wheel bundles its own CUDA runtime) ──
module load anaconda3/2024.06

source "$(conda info --base)/etc/profile.d/conda.sh"

# ── Remove old env if exists ──────────────────────────────────────────────────
if conda env list | grep -q "${ENV_DIR}"; then
    echo "[setup] Removing existing '${ENV_NAME}' env..."
    conda env remove -p "$ENV_DIR" -y
fi

# ── Create fresh env with Python 3.11 ────────────────────────────────────────
echo "[setup] Creating '${ENV_NAME}' with Python 3.11..."
conda create -p "$ENV_DIR" python=3.11 -y

PYTHON="$ENV_DIR/bin/python"
PIP="$ENV_DIR/bin/pip"

echo "[setup] Python: $($PYTHON --version)"

# ── Install all non-torch deps via conda-forge (prebuilt, no compiler needed) ─
echo "[setup] Installing deps via conda-forge..."
conda install -p "$ENV_DIR" -c conda-forge \
    numpy pandas tqdm sentencepiece transformers -y

$PIP install --upgrade pip

# ── Install PyTorch cu121 via pip (conda torch lacks proper CUDA build) ──────
echo "[setup] Installing PyTorch + torchvision (cu121)..."
$PIP install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

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

if torch.cuda.is_available():
    x = torch.randn(3, 4).cuda()
    print('GPU tensor test: PASSED', x.shape)
else:
    print('GPU tensor test: FAILED - CUDA not available')
"
echo "=========================="
echo "Done. Activate with: conda activate ${ENV_DIR}"
