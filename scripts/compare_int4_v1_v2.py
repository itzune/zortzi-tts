#!/usr/bin/env python3
"""Compare INT4 v1 (no emb quant) vs v2 (with emb quant) prosody."""
import soundfile as sf
import numpy as np
import librosa
import os

v1_dir = "/root/work/onnx_final_samples"
v2_dir = "/root/work/onnx_v2_samples"


def prosody(path, focus_end_sec=None):
    y, sr = librosa.load(path, sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    vf = f0[voiced]
    if len(vf) < 3:
        return None
    mean = np.nanmean(vf)
    std = np.nanstd(vf)
    dur = len(y) / sr
    if focus_end_sec:
        n = int(focus_end_sec * sr / 512)
        focus_f0 = f0[:n]
        focus_voiced = focus_f0[~np.isnan(focus_f0)]
        focus_peak = np.nanmax(focus_voiced) if len(focus_voiced) > 0 else mean
    else:
        focus_peak = mean
    boundary = np.nanmean(vf[-5:]) - mean
    return {
        "dur_s": round(dur, 1),
        "f0_std": round(float(std), 1),
        "focus_peak": round(float(focus_peak), 1),
        "boundary": round(float(boundary), 1),
    }


files = sorted(f for f in os.listdir(v1_dir) if f.endswith(".wav"))
print(f"{'File':<32} | {'--INT4 v1 (no emb)--':^26} | {'--INT4 v2 (emb quant)--':^26}")
print(f"{'':32} | {'dur  std   fcs   bnd':^26} | {'dur  std   fcs   bnd':^26}")
print("-" * 91)
for f in files:
    focus_end = 1.2 if "whq" in f else None
    v1 = prosody(f"{v1_dir}/{f}", focus_end_sec=focus_end)
    v2_path = f"{v2_dir}/{f}"
    if not os.path.exists(v2_path):
        continue
    v2 = prosody(v2_path, focus_end_sec=focus_end)
    if v1 and v2:
        v1_str = f"{v1['dur_s']:>3}s {v1['f0_std']:>5.1f} {v1['focus_peak']:>5.1f} {v1['boundary']:+6.1f}"
        v2_str = f"{v2['dur_s']:>3}s {v2['f0_std']:>5.1f} {v2['focus_peak']:>5.1f} {v2['boundary']:+6.1f}"
        print(f"{f:<32} | {v1_str:^26} | {v2_str:^26}")
