#!/usr/bin/env bash
# Generate diverse Basque samples with the final Phase 2 model.
# Uses PyTorch inference (GPU, BF16) for highest fidelity.
#
# Run on the GPU server:
#   bash /root/work/zortzi-tts/scripts/generate_final_samples.sh
set -euo pipefail

MODEL="/root/work/outputs/audio8_tts_sft_basque_final"
PYTHON="/root/work/Audio8_TTS/.venv/bin/python"
OUT="/root/work/final_samples"

# Neutral reference clips
MAIDER_REF="/root/work/data/hitz/maider/NEU_05850.wav"
MAIDER_REF_TEXT="Aurrelaria prest dago jokatzeko."
ANTTON_REF="/root/work/data/hitz/antton/NEU_11782.wav"
ANTTON_REF_TEXT="Inguru hura gerrillarien esku izan da denbora luzean."

mkdir -p "$OUT"

# ── Test sentences (diverse sentence types) ────────────────────────────
# Format: id|voice|text
SENTENCES=(
  # 1. Greetings & introductions (declarative)
  "greeting_maider|maider|Kaixo mundua! Nire izena Maider da."
  "greeting_antton|antton|Kaixo mundua! Nire izena Antton da."

  # 2. Wh-questions (test focus prosody — pitch peak on galde-hitza)
  "whq_nondik_maider|maider|Nondik atera duzu hainbeste diru auto berri hori erosteko?"
  "whq_nondik_antton|antton|Nondik atera duzu hainbeste diru auto berri hori erosteko?"
  "whq_zer_antton|antton|Zer egin duzu asteburuan?"
  "whq_nora_maider|maider|Nora joango zara oporretan?"

  # 3. Yes/no questions (test rising intonation)
  "ynq_badago_maider|maider|Ba al dago taberna bat hurbil?"

  # 4. Exclamations (test expressive prosody)
  "excl_zein_maider|maider|Zein polita den gaur eguneko eguraldia!"
  "excl_ai_antton|antton|Ai, zenbat lan egin behar dut bihar arte!"

  # 5. Longer declarative (narrative)
  "narrative_maider|maider|Euskal Herriko historia aberatsa da, milaka urteko tradizioak mantentzen baitira gaur egunera arte."
  "narrative_antton|antton|Euskara Europako hizkuntzarik zaharrenetako bat da, eta gaur egun milioi bat pertsonak hitz egiten dute."

  # 6. Short practical phrases
  "short_esker_antton|antton|Eskerrik asko zure laguntzagatik."
  "short_barkatu_maider|maider|Barkatu, ez dut ondo ulertu zer esan nahi duzun."

  # 7. Numbers/symbols (NOTE: not normalized — will show raw pronunciation)
  "numbers_maider|maider|2026an 3 liburu erosi nituen."

  # 8. Mixed sentence (declarative + question)
  "mixed_antton|antton|Atzo Bilbora joan nintzen, eta gaur Donostian nago. Zu non bizi zara?"
)

echo "Generating ${#SENTENCES[@]} samples from $MODEL"
echo "Output: $OUT"
echo ""

cd /root/work/Audio8_TTS

i=0
for entry in "${SENTENCES[@]}"; do
  i=$((i+1))
  IFS='|' read -r id voice text <<< "$entry"

  if [ "$voice" = "maider" ]; then
    ref_audio="$MAIDER_REF"
    ref_text="$MAIDER_REF_TEXT"
  else
    ref_audio="$ANTTON_REF"
    ref_text="$ANTTON_REF_TEXT"
  fi

  echo "[$i/${#SENTENCES[@]}] $voice: $text"

  $PYTHON audio8_tts_infer.py \
    --model "$MODEL" \
    --text "$text" \
    --reference-audio "$ref_audio" \
    --reference-text "$ref_text" \
    --output "$OUT/${id}.wav" \
    --temperature 0.8 --top-p 0.95 \
    2>&1 | grep -E "^(saved|Error|Traceback)" || true
done

echo ""
echo "=== Done. Files: ==="
ls -lh "$OUT"/*.wav
