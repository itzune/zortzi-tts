#!/usr/bin/env bash
# Phase 2 (merged): Voice Anchor + Prosody Polish
#
# Per senior ML review: instead of running a flat Phase 2 (27k clips) then a
# separate upsampled Phase 3, we run the UPSAMPLED manifest directly off the
# Phase 1 (CV base) checkpoint. This anchors voice identity AND balances
# prosody in a single pass, avoiding solidifying the neutral-reading prior
# only to push it back out.
#
# Manifest: 41,300 clips (declarative ×1, interrogative ×3, exclamative ×3)
#   - Interrogative mix is healthy: 50% wh-questions, 25% yes/no (al/ote),
#     25% other — so "?" won't collapse to only final-pitch-rise prosody.
#   - References: cross-clip self-conditioning (ref = different clip, same
#     speaker) — teaches the zero-shot voice-cloning path.
#
# Prerequisites:
#   - Phase 1 complete: /root/work/outputs/audio8_tts_sft_basque_cv26/
#   - Upsampled manifest: /root/work/data/hitz/train_hitz_prosody_prepared.jsonl
#   - Codec codes on disk: /root/work/data/hitz/codes/  (from original prep)
set -euo pipefail

# --- wandb ---
set -a
source /root/work/.env   # WANDB_TOKEN
set +a
export WANDB_PROJECT="zortzi-tts"
export WANDB_ENTITY="itzune"
export WANDB_MODE="online"

# --- checkpoint audio callback (both voices, fixed reference clips) ---
export SAMPLE_CALLBACK=1
export SAMPLE_PROCESSOR_MODEL="/root/work/models/Audio8-TTS-Preview-0.6b/"
# Neutral reference clips (verified: pitch range/median < 0.5)
# NEU_00030 was too expressive (ratio 1.00 Maider / 0.82 Antton) and bled
# style into probes. These are 3x/2x more neutral per pyin analysis.
# Maider NEU_05850: ratio=0.35  "Aurrelaria prest dago jokatzeko."
# Antton NEU_11782: ratio=0.47  "Inguru hura gerrillarien esku izan da denbora luzean."
export SAMPLE_VOICE_MAIDER="/root/work/data/hitz/maider/NEU_05850.wav|Aurrelaria prest dago jokatzeko."
export SAMPLE_VOICE_ANTTON="/root/work/data/hitz/antton/NEU_11782.wav|Inguru hura gerrillarien esku izan da denbora luzean."
export SAMPLE_OUTPUT_DIR="/root/work/outputs/audio8_tts_sft_basque_final/samples"
export SAMPLE_TEMPERATURE=0.8
export SAMPLE_TOP_P=0.95
export SAMPLE_SEED=42

cd /root/work/Audio8_TTS

PYTHON=/root/work/Audio8_TTS/.venv/bin/python \
MODEL=/root/work/outputs/audio8_tts_sft_basque_cv26/ \
TRAIN_JSONL=/root/work/data/hitz/train_hitz_prosody_prepared.jsonl \
OUTPUT_DIR=/root/work/outputs/audio8_tts_sft_basque_final/ \
BATCH_SIZE=4 \
GRADIENT_ACCUMULATION_STEPS=4 \
NUM_TRAIN_EPOCHS=3 \
LEARNING_RATE=5e-6 \
LR_SCHEDULER_TYPE=linear \
WARMUP_RATIO=0.03 \
SAVE_STEPS=500 \
LOGGING_STEPS=10 \
SAVE_TOTAL_LIMIT=2 \
GRADIENT_CHECKPOINTING=false \
REPORT_TO=wandb \
  bash audio8_tts_sft.sh
