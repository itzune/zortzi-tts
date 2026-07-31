#!/usr/bin/env bash
# Phase 1 (base/phonotactics): train on Common Voice 26.0 Basque.
# Multi-speaker, no reference audio → learns the text→semantic (phonotactics)
# mapping broadly.
#
# Per senior ML review: STOP AT STEP 6000. Phonotactics converge by ~6k-8k;
# training further entrenches the flat CV read-speech prior (prosodic
# hysteresis), which the conservative Phase 2 LR cannot fully reverse.
# Save all checkpoints (save_total_limit=10) so we can pick the best base.
#
# Prerequisites:
#   - CV 26.0 downloaded + extracted to /root/work/data/cv26/
#   - Manifest built: /root/work/data/cv26/train_cv26_prepared.jsonl
#   - wandb installed + logged in (WANDB_TOKEN in /root/work/.env)
set -euo pipefail

# --- wandb ---
set -a
source /root/work/.env   # WANDB_TOKEN
set +a
export WANDB_PROJECT="zortzi-tts"
export WANDB_ENTITY="itzune"
export WANDB_MODE="online"

# --- checkpoint audio sampling (generate_samples.py at every save) ---
export SAMPLE_CALLBACK=1
export SAMPLE_PROCESSOR_MODEL="/root/work/models/Audio8-TTS-Preview-0.6b/"
# Neutral reference clips (verified: pitch range/median < 0.5)
# NEU_00030 was too expressive (ratio 1.00) and bled style into probes.
# Maider NEU_05850: ratio=0.35  "Aurrelaria prest dago jokatzeko."
export SAMPLE_VOICE_MAIDER="/root/work/data/hitz/maider/NEU_05850.wav|Aurrelaria prest dago jokatzeko."
export SAMPLE_OUTPUT_DIR="/root/work/outputs/audio8_tts_sft_basque_cv26/samples"
export SAMPLE_TEMPERATURE=0.8
export SAMPLE_TOP_P=0.95
export SAMPLE_SEED=42

cd /root/work/Audio8_TTS

PYTHON=/root/work/Audio8_TTS/.venv/bin/python \
MODEL=/root/work/models/Audio8-TTS-Preview-0.6b/ \
TRAIN_JSONL=/root/work/data/cv26/train_cv26_prepared.jsonl \
OUTPUT_DIR=/root/work/outputs/audio8_tts_sft_basque_cv26/ \
BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_TRAIN_EPOCHS=2 \
MAX_STEPS=6000 \
LEARNING_RATE=1e-5 \
LR_SCHEDULER_TYPE=cosine \
SAVE_STEPS=1000 \
SAVE_TOTAL_LIMIT=10 \
LOGGING_STEPS=10 \
WARMUP_RATIO=0.01 \
GRADIENT_CHECKPOINTING=false \
REPORT_TO=wandb \
  bash audio8_tts_sft.sh
