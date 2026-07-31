#!/usr/bin/env python3
"""Scan HiTZ clips for prosodically neutral reference clips.

Per senior ML review: the fixed eval reference must be strictly neutral
(low pitch variance) so probe prosody is driven by target text, not style
transfer from the reference audio.

NEU_00030.wav FAILED this check (range/median=1.00, should be <0.3).
"""
import json
import os
import random
import numpy as np
import librosa

random.seed(42)
base = "/root/work/data/hitz"

# Collect declarative clips per speaker from the UNPREPARED manifest
# (prepared manifest lacks the target 'audio' field)
clips = {"maider": [], "antton": []}
for line in open("/root/work/data/hitz/train_hitz_paired.jsonl"):
    r = json.loads(line)
    t = r["text"].strip()
    if not t.endswith("."):
        continue
    path = r["audio"]  # absolute path to target wav
    if os.path.exists(path):
        spk = "maider" if "/maider/" in path else "antton" if "/antton/" in path else None
        if spk:
            clips[spk].append((path, t[:60], r["text"]))

print("Declarative clips: maider=%d, antton=%d" % (len(clips["maider"]), len(clips["antton"])))

# Sample 120 per speaker, analyze pitch neutrality
results = {}
for spk, lst in clips.items():
    sample = random.sample(lst, min(120, len(lst)))
    scored = []
    for path, txt, full_txt in sample:
        try:
            y, sr = librosa.load(path, sr=16000)
            f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr)
            f0v = f0[voiced & ~np.isnan(f0)]
            if len(f0v) < 20:
                continue
            med = np.median(f0v)
            ratio = np.ptp(f0v) / med
            std = np.std(f0v)
            scored.append((ratio, std, med, len(f0v), path, full_txt))
        except Exception:
            continue
    scored.sort()  # lowest ratio = most neutral
    results[spk] = scored
    print("\n=== %s: %d analyzed, top 5 most neutral ===" % (spk, len(scored)))
    for i, (ratio, std, med, n, path, txt) in enumerate(scored[:5]):
        name = os.path.basename(path)
        print("  %d. %s ratio=%.2f std=%.1f f0=%.0fHz | %s" % (i+1, name, ratio, std, med, txt[:60]))

print("\n=== RECOMMENDED neutral references ===")
for spk in ["maider", "antton"]:
    if results[spk]:
        ratio, std, med, n, path, txt = results[spk][0]
        print("%s: %s" % (spk, path))
        print("  ratio=%.2f (target <0.3) f0=%.0fHz std=%.1f" % (ratio, std, med))
        print("  text: %s" % txt)
