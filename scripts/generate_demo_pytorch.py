#!/usr/bin/env python3
"""Generate demo samples with the fine-tuned PyTorch model (GPU, BF16).

Loads the model once, then synthesises 6 sentences × 2 voices = 12 clips.
Records per-sample inference time, audio duration, and RTF.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

MODEL = "/root/work/outputs/audio8_tts_sft_basque_final"
OUT = Path("/root/work/demo_samples")
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = [
    ("s1", "Kaixo, egun on guztioi."),
    ("s2", "Nondik zatoz zu?"),
    ("s3", "Ba al dakizu euskaraz hitz egiten?"),
    ("s4", "Zein polita dagoen gaur eguzkia!"),
    ("s5", "Euskara Europako hizkuntzarik zaharrenetako bat da, eta milaka urteko historia du."),
    ("s6", "Ni Bilbon bizi naiz, eta zu non bizi zara?"),
]

# Only regenerate changed sentences (empty = all)
ONLY = []
if ONLY:
    SENTENCES = [(s, t) for s, t in SENTENCES if s in ONLY]

VOICES = {
    "maider": {
        "ref_audio": "/root/work/data/hitz/maider/NEU_05850.wav",
        "ref_text": "Aurrelaria prest dago jokatzeko.",
    },
    "antton": {
        "ref_audio": "/root/work/data/hitz/antton/NEU_11782.wav",
        "ref_text": "Inguru hura gerrillarien esku izan da denbora luzean.",
    },
}

SEED = 42
TEMP = 0.8
TOP_P = 0.95
MAX_TOKENS = 1024


def main() -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Loading model from {MODEL} …")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=dtype).eval().to(device)
    sample_rate = int(model.config.codec_sample_rate)
    load_time = time.time() - t0
    print(f"  loaded in {load_time:.2f}s  (sample_rate={sample_rate})")

    timing: dict = {"model": "pytorch", "load_time_s": round(load_time, 2), "samples": []}

    for voice_name, ref in VOICES.items():
        for sid, text in SENTENCES:
            out_path = OUT / f"pytorch_{voice_name}_{sid}.wav"
            print(f"  pytorch_{voice_name}_{sid}: {text[:55]}")

            inputs = processor(
                text=[text],
                reference_audio=[ref["ref_audio"]],
                reference_text=[ref["ref_text"]],
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generator = torch.Generator(device=device).manual_seed(SEED)

            t1 = time.time()
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMP,
                    top_p=TOP_P,
                    do_sample=True,
                    generator=generator,
                    return_dict_in_generate=True,
                )
                waveforms, waveform_lengths = model.decode_audio(output.codes)
            t2 = time.time()

            wav_len = int(waveform_lengths[0])
            waveform = waveforms[0, :wav_len].float().cpu().numpy()
            sf.write(str(out_path), waveform, sample_rate)

            audio_dur = wav_len / sample_rate
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

    with open(OUT / "timing_pytorch.json", "w") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    print(f"\nTiming → {OUT / 'timing_pytorch.json'}")


if __name__ == "__main__":
    main()
