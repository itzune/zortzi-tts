#!/usr/bin/env bash
# Generate the same 16 Basque samples via ONNX INT4 runtime.
# Mirrors generate_final_samples.sh but uses the smallest (INT4) ONNX model.
set -euo pipefail

MODEL_DIR="/root/work/outputs/onnx_p2_final"
VOICES_DIR="/root/work/voices_final"
PYTHON="/root/work/Audio8_TTS/onnx_runtime/.venv/bin/python"
OUT="/root/work/onnx_final_samples"

mkdir -p "$OUT"

# Same sentences as generate_final_samples.sh
SENTENCES=(
  "greeting_maider|maider|Kaixo mundua! Nire izena Maider da."
  "greeting_antton|antton|Kaixo mundua! Nire izena Antton da."
  "whq_nondik_maider|maider|Nondik atera duzu hainbeste diru auto berri hori erosteko?"
  "whq_nondik_antton|antton|Nondik atera duzu hainbeste diru auto berri hori erosteko?"
  "whq_zer_antton|antton|Zer egin duzu asteburuan?"
  "whq_nora_maider|maider|Nora joango zara oporretan?"
  "ynq_badago_maider|maider|Ba al dago taberna bat hurbil?"
  "excl_zein_maider|maider|Zein polita den gaur eguneko eguraldia!"
  "excl_ai_antton|antton|Ai, zenbat lan egin behar dut bihar arte!"
  "narrative_maider|maider|Euskal Herriko historia aberatsa da, milaka urteko tradizioak mantentzen baitira gaur egunera arte."
  "narrative_antton|antton|Euskara Europako hizkuntzarik zaharrenetako bat da, eta gaur egun milioi bat pertsonak hitz egiten dute."
  "short_esker_antton|antton|Eskerrik asko zure laguntzagatik."
  "short_barkatu_maider|maider|Barkatu, ez dut ondo ulertu zer esan nahi duzun."
  "numbers_maider|maider|2026an 3 liburu erosi nituen."
  "mixed_antton|antton|Atzo Bilbora joan nintzen, eta gaur Donostian nago. Zu non bizi zara?"
)

cd /root/work/Audio8_TTS/onnx_runtime

echo "Generating ${#SENTENCES[@]} INT4 ONNX samples from $MODEL_DIR"
echo "Output: $OUT"
echo ""

i=0
for entry in "${SENTENCES[@]}"; do
  i=$((i+1))
  IFS='|' read -r id voice text <<< "$entry"
  echo "[$i/${#SENTENCES[@]}] $voice: $text"
  $PYTHON -m arktts_runtime.cli \
    --model-dir "$MODEL_DIR" \
    --voices-dir "$VOICES_DIR" \
    --text "$text" \
    --voice "$voice" \
    --output "$OUT/${id}.wav" \
    --temperature 0.8 --top-p 0.95 \
    --max-new-tokens 512 \
    2>&1 | grep -E "^(saved|Error|Traceback)" || true
done

echo ""
echo "=== Done. Files: ==="
ls -lh "$OUT"/*.wav
