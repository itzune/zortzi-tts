#!/usr/bin/env python3
"""Generate demo samples with Piper TTS (CPU).

Loads each voice model once via the Python API, then synthesises
6 sentences × 2 voices = 12 clips.
Records per-sample inference time, audio duration, and RTF.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import soundfile as sf

PIPER_DIR = Path(__file__).resolve().parent.parent / "piper_models"
OUT = Path(__file__).resolve().parent.parent / "probes" / "demo_piper"
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    ("s1", "Kaixo, ona goiza denoi."),
    ("s2", "Nondik zatoz zu?"),
    ("s3", "Ba al dakizu euskaraz hitz egiten?"),
    ("s4", "Zein polita dago gaur eguzkia!"),
    ("s5", "Euskara Europako hizkuntzarik zaharrenetako bat da, eta milaka urteko historia du."),
    ("s6", "Atzo Bilbora joan nintzen, eta zu non bizi zara?"),
]

MODELS = {
    "maider": str(PIPER_DIR / "eu-maider-medium.onnx"),
    "antton": str(PIPER_DIR / "eu-antton-medium.onnx"),
}


def main() -> None:
    from piper import PiperVoice

    timing: dict = {"model": "piper", "load_time_s": 0, "samples": []}

    for voice_name, model_path in MODELS.items():
        print(f"\n{'=' * 60}")
        print(f"  Piper  voice={voice_name}")
        print(f"{'=' * 60}")

        t0 = time.time()
        voice = PiperVoice.load(model_path)
        load_time = time.time() - t0
        print(f"  loaded in {load_time:.2f}s")

        # Warmup (first call initialises ONNX session internals)
        with wave.open("/tmp/piper_warmup.wav", "wb") as wf:
            voice.synthesize_wav("warmup", wf)

        for sid, text in SENTENCES:
            out_path = OUT / f"piper_{voice_name}_{sid}.wav"
            print(f"  piper_{voice_name}_{sid}: {text[:55]}")

            t1 = time.time()
            with wave.open(str(out_path), "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            t2 = time.time()

            info = sf.info(str(out_path))
            audio_dur = info.duration
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
                "sample_rate": info.samplerate,
            })
            print(f"    {infer_time:.3f}s → {audio_dur:.2f}s  RTF={rtf:.3f}")

    with open(OUT / "timing_piper.json", "w") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    print(f"\nTiming → {OUT / 'timing_piper.json'}")


if __name__ == "__main__":
    main()
