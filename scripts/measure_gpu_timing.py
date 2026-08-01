#!/usr/bin/env python3
"""Measure GPU inference timing to assess real-time streaming feasibility."""
import time
import sys
import torch
from pathlib import Path

sys.path.insert(0, "/root/work/Audio8_TTS")
from transformers import AutoConfig, AutoModel, AutoProcessor
import soundfile as sf
import numpy as np

MODEL = "/root/work/outputs/audio8_tts_sft_basque_final"
REF_AUDIO = "/root/work/data/hitz/maider/NEU_05850.wav"
REF_TEXT = "Aurrelaria prest dago jokatzeko."
TEXT = "Kaixo mundua! Nire izena Maider da eta gaur eguraldi ederra dugu."

device = torch.device("cuda")
dtype = torch.bfloat16

print("Loading model...")
t0 = time.time()
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, config=config, trust_remote_code=True)
model = model.to(device=device, dtype=dtype).eval()
model.load_codec(device=device, dtype=dtype)
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
print(f"  loaded in {time.time()-t0:.1f}s")

# Prepare inputs via processor (like audio8_tts_infer.py)
ref_audio, sr = sf.read(REF_AUDIO, dtype="float32")
if ref_audio.ndim > 1:
    ref_audio = ref_audio.mean(axis=1)

inputs = processor(text=[TEXT], reference_audio=[(ref_audio, sr)], reference_text=[REF_TEXT], return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"\nGenerating: {TEXT}")
torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        return_dict_in_generate=True,
    )
torch.cuda.synchronize()
t_gen = time.time() - t0
n_frames = output.codes.shape[2]
audio_dur = n_frames * 2048 / 44100
print(f"  generated {n_frames} frames in {t_gen:.2f}s")
print(f"  audio duration: {audio_dur:.2f}s")
print(f"  RTF (generation only): {t_gen/audio_dur:.2f}x")
print(f"  per-frame: {t_gen/n_frames*1000:.1f}ms (need <46ms for real-time)")

# Decode
torch.cuda.synchronize()
t0 = time.time()
with torch.inference_mode():
    waveforms, _ = model.decode_audio(output.codes)
torch.cuda.synchronize()
t_dec = time.time() - t0
print(f"  codec decode: {t_dec:.2f}s")
print(f"  total: {t_gen+t_dec:.2f}s, overall RTF: {(t_gen+t_dec)/audio_dur:.2f}x")
print(f"  GPU mem: {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
