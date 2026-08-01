#!/usr/bin/env python3
"""Generate demo samples with the ONNX runtime (FP16 + INT4, CPU).

Loads each precision once, then synthesises 6 sentences × 2 voices = 12 clips.
Records per-sample inference time, audio duration, and RTF.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "/root/work/Audio8_TTS/onnx_runtime")
from arktts_runtime.runtime import ArkTtsRuntime  # noqa: E402

MODEL_DIR = "/root/work/outputs/onnx_p2_final"
VOICES_DIR = "/root/work/voices_final"
OUT = Path("/root/work/demo_samples")
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    ("s1", "Kaixo, ona goiza denoi."),
    ("s2", "Nondik zatoz zu?"),
    ("s3", "Ba al dakizu euskaraz hitz egiten?"),
    ("s4", "Zein polita dago gaur eguzkia!"),
    ("s5", "Euskara Europako hizkuntzarik zaharrenetako bat da, eta milaka urteko historia du."),
    ("s6", "Atzo Bilbora joan nintzen, eta zu non bizi zara?"),
]

SEED = 42
TEMP = 0.8
TOP_P = 0.95
MAX_TOKENS = 1024


def run_precision(precision: str) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  ONNX {precision.upper()}  (CPU)")
    print(f"{'=' * 60}")

    t0 = time.time()
    runtime = ArkTtsRuntime(MODEL_DIR, VOICES_DIR, precision=precision, threads=5)
    load_time = time.time() - t0
    sample_rate = int(runtime.manifest["sample_rate"])
    print(f"  loaded in {load_time:.2f}s  (sample_rate={sample_rate})")

    timing: dict = {"model": f"onnx_{precision}", "load_time_s": round(load_time, 2), "samples": []}

    for voice_name in ["maider", "antton"]:
        for sid, text in SENTENCES:
            out_path = OUT / f"onnx_{precision}_{voice_name}_{sid}.wav"
            print(f"  onnx_{precision}_{voice_name}_{sid}: {text[:55]}")

            t1 = time.time()
            audio, _codes = runtime.synthesize(
                text=text,
                voice=voice_name,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMP,
                top_p=TOP_P,
                seed=SEED,
            )
            t2 = time.time()

            sf.write(str(out_path), audio, sample_rate)
            audio_dur = len(audio) / sample_rate
            infer_time = t2 - t1
            rtf = infer_time / audio_dur if audio_dur > 0 else float("inf")

            timing["samples"].append({
                "voice": voice_name,
                "sentence_id": sid,
                "text": text,
                "output": str(out_path),
                "inference_time_s": round(infer_time, 3),
                "audio_duration_s": round(audio_dur, 3),
                "rtf": round(rtf, 3),
                "sample_rate": sample_rate,
            })
            print(f"    {infer_time:.2f}s → {audio_dur:.2f}s  RTF={rtf:.2f}")

    return timing


def main() -> None:
    results = []
    for precision in ["fp16", "int4"]:
        results.append(run_precision(precision))

    with open(OUT / "timing_onnx.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nTiming → {OUT / 'timing_onnx.json'}")


if __name__ == "__main__":
    main()
