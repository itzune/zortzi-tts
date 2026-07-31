#!/usr/bin/env python3
"""Compare prosody metrics across checkpoints with multiple seeds.

Single-sample prosody metrics are noisy (sampling at temp 0.8). This script
generates probes with N seeds per checkpoint and reports mean ± std, giving
a statistically meaningful comparison for the Phase 2 starting-point decision.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

PROBE_SENTENCES = [
    ("Atzo goizean, Joanes artzainak bere txakur txikiari zurezko makila "
     "bota zion sasi artean; baina, ustekabean, xagu beltz bat izututa "
     "atera zen, eta baserri zaharrerantz ihes egin zuen.", None),
    ("\u201cNondik atera duzu hainbeste diru auto berri hori erosteko?\u201d", 1.2),
]


def analyze_prosody(audio_path, sr=16000, focus_end_sec=None):
    if not _HAS_LIBROSA:
        return None
    try:
        y, sr = librosa.load(audio_path, sr=sr)
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
            "boundary_delta": float(f0v[-1] - f0_mean),
        }
        if focus_end_sec is not None:
            focus_f0 = f0v[times_v <= focus_end_sec]
            if len(focus_f0) > 0:
                result["focus_peak"] = float(np.max(focus_f0))
        return result
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="name:/path/to/checkpoint pairs")
    ap.add_argument("--voice", required=True,
                    help="name|/path/to/ref.wav|reference text")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--processor", default="/root/work/models/Audio8-TTS-Preview-0.6b/")
    ap.add_argument("--output-dir", default="/root/work/probes/compare")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse checkpoints
    ckpts = []
    for spec in args.checkpoints:
        name, path = spec.split(":", 1)
        ckpts.append((name, path))

    # Parse voice
    vparts = args.voice.split("|", 2)
    voice_name, ref_audio, ref_text = vparts

    # Map checkpoint names to integer step values for generate_samples --step
    step_map = {"3k": 3000, "5k": 5000, "6k": 6000, "0": 0}

    all_results = {}
    for ck_name, ck_path in ckpts:
        print("\n=== %s (%s) ===" % (ck_name, ck_path), file=sys.stderr)
        base_step = step_map.get(ck_name, 0)
        for sent_idx, (sentence, focus_end) in enumerate(PROBE_SENTENCES, 1):
            metrics_list = []
            for seed in range(args.seeds):
                # Use a unique output dir per seed to avoid filename collisions
                seed_dir = out_dir / ("%s_seed%d" % (ck_name, seed))
                seed_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable, "/root/work/Audio8_TTS/generate_samples.py",
                    "--model", ck_path,
                    "--processor", args.processor,
                    "--output-dir", str(seed_dir),
                    "--step", str(base_step),
                    "--seed", str(seed),
                    "--voice", args.voice,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if proc.returncode != 0:
                    print("  FAILED %s_s%d_seed%d: %s" % (ck_name, sent_idx, seed, proc.stderr[-200:]), file=sys.stderr)
                    continue
                # generate_samples creates: {voice}_s{idx}_step{step}.wav
                wav_path = seed_dir / ("%s_s%d_step%d.wav" % (voice_name, sent_idx, base_step))
                if wav_path.exists():
                    m = analyze_prosody(str(wav_path), focus_end_sec=focus_end)
                    if m:
                        metrics_list.append(m)
                        print("  %s_s%d_seed%d: f0_std=%.1f focus_peak=%s bndry=%.1f" %
                              (ck_name, sent_idx, seed, m["f0_std"],
                               ("%.1f" % m["focus_peak"]) if "focus_peak" in m else "-",
                               m["boundary_delta"]), file=sys.stderr)
            key = "%s_s%d" % (ck_name, sent_idx)
            all_results[key] = metrics_list

    # Summary table
    print("\n" + "=" * 70)
    print("PROSODY COMPARISON (mean ± std over %d seeds)" % args.seeds)
    print("=" * 70)
    print("%-20s %12s %12s %12s" % ("checkpoint/sent", "f0_std(Hz)", "focus_peak", "bndry_delta"))
    print("-" * 70)
    for key in sorted(all_results.keys()):
        metrics = all_results[key]
        if not metrics:
            print("%-20s %12s" % (key, "NO DATA"))
            continue
        f0stds = [m["f0_std"] for m in metrics]
        bdeltas = [m["boundary_delta"] for m in metrics]
        peaks = [m["focus_peak"] for m in metrics if "focus_peak" in m]
        f0_str = "%.1f ± %.1f" % (statistics.mean(f0stds), statistics.stdev(f0stds) if len(f0stds) > 1 else 0)
        peak_str = ("%.1f ± %.1f" % (statistics.mean(peaks), statistics.stdev(peaks) if len(peaks) > 1 else 0)) if peaks else "-"
        bd_str = "%.1f ± %.1f" % (statistics.mean(bdeltas), statistics.stdev(bdeltas) if len(bdeltas) > 1 else 0)
        print("%-20s %12s %12s %12s" % (key, f0_str, peak_str, bd_str))


if __name__ == "__main__":
    main()
