#!/bin/bash
# setup_env.sh — Create the momask conda environment on the Explorer HPC cluster
# Run once interactively:  bash /scratch/katoch.aa/text-to-motion/setup_env.sh

set -e

ENV_NAME="momask"
PYTHON_VERSION="3.11"

# ── Init conda ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"

# Remove old env if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup] Removing existing '${ENV_NAME}' env …"
    conda env remove -n "$ENV_NAME" -y
fi

# ── Create fresh env ──────────────────────────────────────────────────────────
echo "[setup] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION} …"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

conda activate "$ENV_NAME"

# ── PyTorch with CUDA 12.1 (Python 3.11, fully supported) ────────────────────
echo "[setup] Installing PyTorch cu121 …"
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# ── Other dependencies ────────────────────────────────────────────────────────
echo "[setup] Installing remaining dependencies …"
pip install transformers sentencepiece pandas numpy tqdm

# ── Verify CUDA ───────────────────────────────────────────────────────────────
echo "[setup] Verifying CUDA availability …"
python -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version  : {torch.version.cuda}')
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
"

echo "[setup] Done. Activate with: conda activate ${ENV_NAME}"
