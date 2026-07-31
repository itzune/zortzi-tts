#!/usr/bin/env python3
"""Register Maider and Antton voices for the ONNX runtime.

Uses the ONNX codec encoder to convert reference audio clips to codec codes,
then saves them in the VoiceStore format (codes.npy + meta.json).
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

# Voice definitions: name → (audio_path, reference_text)
VOICES = {
    "maider": (
        "/root/work/data/hitz/maider/NEU_05850.wav",
        "Aurrelaria prest dago jokatzeko.",
    ),
    "antton": (
        "/root/work/data/hitz/antton/NEU_11782.wav",
        "Inguru hura gerrillarien esku izan da denbora luzean.",
    ),
}

MODEL_DIR = Path("/root/work/outputs/onnx_p2_3k")
VOICES_DIR = Path("/root/work/voices")
ENCODER_PATH = MODEL_DIR / "registration" / "codec_encoder_fp16.onnx"
REG_MANIFEST_PATH = MODEL_DIR / "registration" / "registration_manifest.json"
RUNTIME_MANIFEST_PATH = MODEL_DIR / "runtime_manifest.json"


def decode_audio(path: str, target_rate: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if int(sample_rate) != int(target_rate):
        factor = gcd(int(sample_rate), int(target_rate))
        audio = resample_poly(audio, target_rate // factor, int(sample_rate) // factor).astype(
            np.float32
        )
    padding = (-audio.size) % 2048
    if padding:
        audio = np.pad(audio, (0, padding))
    return np.ascontiguousarray(audio.reshape(1, 1, -1)), int(sample_rate)


def register_voice(name: str, audio_path: str, reference_text: str, fingerprint: str):
    reg_manifest = json.loads(REG_MANIFEST_PATH.read_text())
    target_rate = int(reg_manifest["sample_rate"])
    num_codebooks = int(reg_manifest["num_codebooks"])

    print(f"Registering voice '{name}' from {audio_path}...", flush=True)
    audio, source_rate = decode_audio(audio_path, target_rate)
    print(f"  audio: {audio.shape}, source_rate={source_rate}, target_rate={target_rate}", flush=True)

    # Load ONNX encoder
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(ENCODER_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_type = session.get_inputs()[0].type
    values = audio.astype(np.float16 if input_type == "tensor(float16)" else np.float32)
    codes = np.asarray(session.run(None, {"audio": values})[0], dtype=np.int64)
    del session

    if codes.ndim == 3:
        codes = codes[0]
    print(f"  codes shape: {codes.shape} (expected [{num_codebooks}, T])", flush=True)
    if codes.ndim != 2 or codes.shape[0] != num_codebooks:
        raise ValueError(f"invalid codes shape: {codes.shape}")

    # Save voice
    voice_dir = VOICES_DIR / name
    voice_dir.mkdir(parents=True, exist_ok=True)
    np.save(voice_dir / "codes.npy", codes.astype(np.uint16))

    with open(audio_path, "rb") as f:
        source_sha256 = hashlib.sha256(f.read()).hexdigest()

    meta = {
        "name": name,
        "reference_text": " ".join(reference_text.strip().split()),
        "shape": list(codes.shape),
        "dtype": "uint16",
        "sample_rate": target_rate,
        "source_audio": Path(audio_path).name,
        "source_sample_rate": source_rate,
        "source_sha256": source_sha256,
        "model_fingerprint": reg_manifest["model_fingerprint"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_kind": "script_registration",
    }
    (voice_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(f"  ✓ saved {voice_dir}/codes.npy and meta.json", flush=True)


def main():
    runtime_manifest = json.loads(RUNTIME_MANIFEST_PATH.read_text())
    fingerprint = runtime_manifest["model_fingerprint"]

    # Verify fingerprint matches registration manifest
    reg_manifest = json.loads(REG_MANIFEST_PATH.read_text())
    if reg_manifest.get("model_fingerprint") != fingerprint:
        print("ERROR: fingerprint mismatch between runtime and registration manifests!", flush=True)
        sys.exit(1)

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    for name, (audio_path, reference_text) in VOICES.items():
        register_voice(name, audio_path, reference_text, fingerprint)

    print(f"\n✓ Registered {len(VOICES)} voices in {VOICES_DIR}", flush=True)
    print(f"  Voices: {', '.join(VOICES.keys())}", flush=True)


if __name__ == "__main__":
    main()
