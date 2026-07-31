#!/bin/bash
# Launch Audio8_TTS SFT fine-tuning for Basque on the L40.
# Uses the SLR76 prepared manifest (7136 utterances, no reference audio).
set -euo pipefail
export PATH=/root/.local/bin:$PATH

cd /root/work/Audio8_TTS

PYTHON=/root/work/Audio8_TTS/.venv/bin/python \
MODEL=/root/work/models/Audio8-TTS-Preview-0.6b \
TRAIN_JSONL=/root/work/data/slr76/train_slr76_prepared.jsonl \
OUTPUT_DIR=/root/work/outputs/audio8_tts_sft_basque \
EXPORT_DIR=/root/work/outputs/audio8_tts_sft_basque/export \
NPROC_PER_NODE=1 \
BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=4 \
LEARNING_RATE=1e-5 \
NUM_TRAIN_EPOCHS=3 \
MAX_LENGTH=2048 \
BF16=true \
GRADIENT_CHECKPOINTING=false \
SAVE_STEPS=100 \
LOGGING_STEPS=10 \
SAVE_TOTAL_LIMIT=3 \
DATALOADER_NUM_WORKERS=2 \
REPORT_TO=tensorboard \
bash audio8_tts_sft.sh
