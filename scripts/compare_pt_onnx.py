#!/usr/bin/env python3
"""Compare PyTorch vs ONNX INT4 prosody on the same sentences."""
import soundfile as sf
import numpy as np
import librosa
import os

pt_dir = "/root/work/final_samples"
onnx_dir = "/root/work/onnx_final_samples"


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
        "f0_mean": round(float(mean), 1),
        "f0_std": round(float(std), 1),
        "focus_peak": round(float(focus_peak), 1),
        "boundary": round(float(boundary), 1),
    }


files = sorted(f for f in os.listdir(pt_dir) if f.endswith(".wav"))
print(f"{'File':<32} | {'--PyTorch (GPU)--':^28} | {'--ONNX INT4 (CPU)--':^28}")
print(f"{'':32} | {'dur  std   focus  bndry':^28} | {'dur  std   focus  bndry':^28}")
print("-" * 95)
for f in files:
    focus_end = 1.2 if "whq" in f else None
    pt = prosody(f"{pt_dir}/{f}", focus_end_sec=focus_end)
    onnx_path = f"{onnx_dir}/{f}"
    if not os.path.exists(onnx_path):
        continue
    on = prosody(onnx_path, focus_end_sec=focus_end)
    if pt and on:
        pt_str = f"{pt['dur_s']:>3}s {pt['f0_std']:>5.1f} {pt['focus_peak']:>6.1f} {pt['boundary']:+6.1f}"
        on_str = f"{on['dur_s']:>3}s {on['f0_std']:>5.1f} {on['focus_peak']:>6.1f} {on['boundary']:+6.1f}"
        print(f"{f:<32} | {pt_str:^28} | {on_str:^28}")
    elif pt:
        print(f"{f:<32} | (missing ONNX)")
