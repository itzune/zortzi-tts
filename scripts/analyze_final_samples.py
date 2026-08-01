#!/usr/bin/env python3
"""Prosody analysis of final model samples."""
import soundfile as sf
import numpy as np
import librosa
import os

samples_dir = "/root/work/final_samples"


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
        "voiced": int(len(vf)),
    }


files = sorted(os.listdir(samples_dir))
header = f"{'File':<35} {'dur':>4} {'f0_mean':>7} {'f0_std':>7} {'focus':>7} {'bndry':>7}"
print(header)
print("-" * 70)
for f in files:
    if not f.endswith(".wav"):
        continue
    focus_end = 1.2 if "whq" in f else None
    m = prosody(f"{samples_dir}/{f}", focus_end_sec=focus_end)
    if m:
        print(
            f"{f:<35} {m['dur_s']:>4}s {m['f0_mean']:>7.1f} {m['f0_std']:>7.1f} "
            f"{m['focus_peak']:>7.1f} {m['boundary']:+7.1f}"
        )
    else:
        print(f"{f:<35}  (no voiced frames)")
