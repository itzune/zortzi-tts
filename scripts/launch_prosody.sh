#!/usr/bin/env bash
# Phase 3: Prosody Polish
# Fine-tune from the anchored model on HiTZ data with interrogatives and
# exclamatives upsampled 3×. Very low LR to avoid forgetting phonotactics.
#
# Run AFTER launch_anchor.sh has completed.
set -euo pipefail

cd /root/work/Audio8_TTS

# ── Paths ──────────────────────────────────────────────────────────────
export MODEL="/root/work/outputs/audio8_tts_sft_basque_anchored/"
export TRAIN_JSONL="/root/work/data/hitz/train_hitz_prosody_prepared.jsonl"
export OUTPUT_DIR="/root/work/outputs/audio8_tts_sft_basque_prosody/"

# ── Original model (for processor/tokenizer in checkpoint callback) ────
export SAMPLE_PROCESSOR_MODEL="/root/work/models/Audio8-TTS-Preview-0.6b/"

# ── Training hyperparameters ───────────────────────────────────────────
export BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=4
export NUM_TRAIN_EPOCHS=2
export LEARNING_RATE=2e-6        # very low: polish, don't forget
export MAX_LENGTH=2048
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=3
export GRADIENT_CHECKPOINTING=false
export SEED=42
export REPORT_TO=wandb
export WANDB_PROJECT=zortzi-tts
export WANDB_ENTITY=itzune

# ── Checkpoint audio callback (both voices) ────────────────────────────
export SAMPLE_CALLBACK=1
export SAMPLE_OUTPUT_DIR="/root/work/probes/prosody"
export SAMPLE_TEMPERATURE=0.8
export SAMPLE_TOP_P=0.95
export SAMPLE_SEED=42
export SAMPLE_VOICE_MAIDER="/root/work/data/hitz/maider/NEU_00030.wav|Besteak beste, soldata berriro negoziatzea eskatzen dute langileek."
export SAMPLE_VOICE_ANTTON="/root/work/data/hitz/antton/NEU_00030.wav|Besteak beste, soldata berriro negoziatzea eskatzen dute langileek."

# ── DeepSpeed ──────────────────────────────────────────────────────────
export DEEPSPEED_CONFIG="/root/work/zortzi-tts/scripts/deepspeed_zero2.json"

# ── Run ────────────────────────────────────────────────────────────────
export NPROC_PER_NODE=1
bash audio8_tts_sft.sh
