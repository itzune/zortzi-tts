#!/usr/bin/env bash
# Example end-to-end manifest build for Basque Audio8_TTS fine-tuning.
#
# Adjust the paths to your downloaded datasets, then:
#   bash scripts/build_train.sh
#
# Datasets (see RESEARCH.md §3):
#   - Common Voice 25.0 eu  (CC0)        https://datacollective.mozillafoundation.org/datasets/cmn2hwe0d01n8mm07wug9r5he
#   - OpenSLR SLR76         (CC BY-SA 4) https://openslr.org/76/
#   - HiTZ Aholab Maider/Antton (CC BY 4) — studio quality, best reference voices
set -euo pipefail

CV_TSV="${CV_TSV:-/data/cv-corpus-25.0-eu/validated.tsv}"
CV_CLIPS="${CV_CLIPS:-/data/cv-corpus-25.0-eu/clips}"
SLR_ROOT="${SLR_ROOT:-/data/slr76}"          # expects line_index.tsv + wav/
HITZ_ROOT="${HITZ_ROOT:-/data/hitz}"         # adjust to your Maider/Antton layout
OUT_DIR="${OUT_DIR:-data}"

mkdir -p "$OUT_DIR"

# 1. Common Voice — validated, >=2 upvotes, drop clips > 20s (EOS-stability guard)
uv run zortzi-manifest from-common-voice \
  --tsv "$CV_TSV" --clips "$CV_CLIPS" \
  --min-upvotes 2 --max-duration-sec 20 \
  --output "$OUT_DIR/cv_eu.jsonl"

# 2. OpenSLR SLR76 — generic transcript loader (verify column layout)
uv run zortzi-manifest from-transcripts \
  --transcripts "$SLR_ROOT/line_index.tsv" \
  --audio-dir "$SLR_ROOT/wav" --audio-from-id --audio-ext wav \
  --max-duration-sec 20 \
  --output "$OUT_DIR/slr76.jsonl"

# 3. HiTZ Aholab (Maider/Antton) — studio quality. Already spoken-word text,
#    so normalization is safe/idempotent. Use one clip as a fixed zero-shot
#    reference voice for the studio subset.
uv run zortzi-manifest from-transcripts \
  --transcripts "$HITZ_ROOT/line_index.tsv" \
  --audio-dir "$HITZ_ROOT/wav" --audio-from-id --audio-ext wav \
  --fixed-reference-audio "$HITZ_ROOT/ref/maider_ref.wav" \
  --fixed-reference-text "erreferentzia hau da." \
  --output "$OUT_DIR/hitz.jsonl"

# 4. Merge into one training manifest
uv run zortzi-manifest merge \
  "$OUT_DIR/cv_eu.jsonl" "$OUT_DIR/slr76.jsonl" "$OUT_DIR/hitz.jsonl" \
  --output "$OUT_DIR/train.jsonl"

# 5. Inspect
uv run zortzi-manifest stats "$OUT_DIR/train.jsonl"

echo ""
echo "Next: run audio8_tts_prepare.py on $OUT_DIR/train.jsonl, then audio8_tts_sft.sh"
