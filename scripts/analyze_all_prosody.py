#!/usr/bin/env python3
"""Analyze prosody of all callback-generated probe audio files."""
import os
import numpy as np
import librosa
from pathlib import Path

SAMPLES_DIR = "/root/work/outputs/audio8_tts_sft_basque_cv26/samples"
FOCUS_END_SEC = 1.2  # wh-word "Nondik" focus window for s2

def analyze(path, focus_end_sec=None):
    try:
        y, sr = librosa.load(path, sr=16000)
        f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
        times = librosa.times_like(f0, sr=sr)
        mask = voiced & ~np.isnan(f0)
        f0v = f0[mask]
        times_v = times[mask]
        if len(f0v) < 5:
            return None
        f0_mean = float(np.mean(f0v))
        result = {
            "f0_std": float(np.std(f0v)),
            "f0_mean": f0_mean,
            "boundary_delta": float(f0v[-1] - f0_mean),
            "dur": round(len(y) / sr, 1),
        }
        if focus_end_sec is not None:
            focus_f0 = f0v[times_v <= focus_end_sec]
            if len(focus_f0) > 0:
                result["focus_peak"] = float(np.max(focus_f0))
                result["focus_delta"] = float(np.max(focus_f0) - f0_mean)
        return result
    except Exception as e:
        return {"error": str(e)}

steps = [1000, 2000, 3000, 4000, 5000, 6000]

print("=" * 85)
print("PROSODY ANALYSIS — Phase 1 CV base (callback audio, 1 sample per step)")
print("  s1 = declarative (sibilant stress test)")
print("  s2 = wh-question  (focus on 'Nondik', should peak early then fall)")
print("  f0_std: flat CV ~15-25 Hz, expressive >35 Hz")
print("  focus_delta: focus_peak - mean (positive = pitch emphasis on wh-word)")
print("  boundary_delta: end_f0 - mean (negative = falling = correct for wh-Q)")
print("=" * 85)

for voice in ["maider", "default"]:
    print("\n--- %s ---" % voice)
    print("%-6s %6s %8s %8s %10s %10s %10s" %
          ("step", "dur", "f0_mean", "f0_std", "focus_pk", "foc_dlt", "bndry_dlt"))
    for step in steps:
        for sidx in [1, 2]:
            fname = "%s_s%d_step%d.wav" % (voice, sidx, step)
            path = os.path.join(SAMPLES_DIR, fname)
            if not os.path.exists(path):
                continue
            focus_end = FOCUS_END_SEC if sidx == 2 else None
            m = analyze(path, focus_end)
            if m is None or "error" in m:
                print("%-6s %6s %8s %8s %10s %10s %10s" %
                      ("s%d/%d" % (sidx, step), "-", "-", "-", "-", "-", "-"))
                continue
            fp = m.get("focus_peak", 0)
            fd = m.get("focus_delta", 0)
            print("%-6s %5.1fs %7.1f %7.1f %9.1f %9.1f %9.1f" %
                  ("s%d/%d" % (sidx, step), m["dur"], m["f0_mean"], m["f0_std"],
                   fp if fp else 0, fd if fd else 0, m["boundary_delta"]))

# Download key files for local listening
print("\n=== key files for download ===")
for step in [3000, 6000]:
    for sidx in [1, 2]:
        fname = "maider_s%d_step%d.wav" % (sidx, step)
        print("  %s/%s" % (SAMPLES_DIR, fname))
