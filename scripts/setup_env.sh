#!/bin/bash
# Environment setup for Audio8_TTS Basque fine-tuning on the L40.
# Run with: nohup bash setup_env.sh > setup.log 2>&1 &
set -euo pipefail
export PATH=/root/.local/bin:$PATH
cd /root/work/Audio8_TTS

echo "[$(date)] Creating venv (python 3.12)..."
uv venv --python 3.12 .venv
VENV=/root/work/Audio8_TTS/.venv

echo "[$(date)] Installing training requirements (torch cu124)..."
uv pip install --python "$VENV/bin/python" --torch-backend=cu124 -r requirements-train.txt

echo "[$(date)] Installing dataset/manifest deps (datasets, pandas, pyarrow, librosa)..."
uv pip install --python "$VENV/bin/python" datasets pandas pyarrow librosa

echo "[$(date)] Verifying torch + CUDA..."
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "[$(date)] Verifying transformers + deepspeed..."
"$VENV/bin/python" -c "import transformers, deepspeed, accelerate; print('transformers', transformers.__version__, 'deepspeed', deepspeed.__version__, 'accelerate', accelerate.__version__)"

echo "[$(date)] INSTALL_DONE"
