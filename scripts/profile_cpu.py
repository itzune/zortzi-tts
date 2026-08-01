#!/usr/bin/env python3
"""Profile CPU inference: where does time go? Slow AR vs Fast AR vs decode.
Also test thread counts, and measure memory bandwidth vs compute bound."""
import time
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/root/work/Audio8_TTS/onnx_runtime")
from arktts_runtime.runtime import ArkTtsRuntime

MODEL = "/root/work/outputs/onnx_p2_final"
VOICES = "/root/work/voices_final"
TEXT = "Kaixo mundua! Nire izena Maider da."

import os
import onnxruntime as ort

for threads in [5, 8]:
    print(f"\n{'='*60}")
    print(f"Threads: {threads}")
    print(f"{'='*60}")
    os.environ["OMP_NUM_THREADS"] = str(threads)

    rt = ArkTtsRuntime(Path(MODEL), Path(VOICES), precision="int4", threads=threads)
    hop = int(rt.manifest["codec_hop_length"])
    sr = int(rt.manifest["sample_rate"])
    ms_per_frame = hop / sr * 1000

    # Profile individual components
    reference_codes, meta = rt.voices.load("maider")
    prompt = rt.prompt_builder.build(TEXT, meta["reference_text"], reference_codes)
    prompt_len = int(prompt.shape[2])

    slow_caches = rt._empty_slow_caches()
    positions = np.arange(prompt_len, dtype=np.int64)

    # --- Slow AR: prompt processing (T=prompt_len) ---
    t0 = time.perf_counter()
    for _ in range(3):
        slow_caches = rt._empty_slow_caches()
        logits, hidden = rt._slow_step(prompt, positions, slow_caches)
    t_slow_prompt = (time.perf_counter() - t0) / 3

    # --- Slow AR: single token step (T=1) ---
    col = np.concatenate([[100], [0]*10]).reshape(1, -1, 1)
    pos = np.asarray([prompt_len], dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(20):
        rt._slow_step(col, pos, slow_caches)
    t_slow_step = (time.perf_counter() - t0) / 20

    # --- Fast AR: 10 sequential steps ---
    fast_caches = rt._empty_fast_caches()
    rt._fast_step(hidden, 0, True, 0, fast_caches)
    t0 = time.perf_counter()
    for _ in range(10):
        fc = rt._empty_fast_caches()
        rt._fast_step(hidden, 0, True, 0, fc)
        for p in range(1, 10):
            rt._fast_step(hidden, 0, False, p, fc)
    t_fast_all = (time.perf_counter() - t0) / 10

    # --- Codec decode: 12 frames ---
    test_codes = np.zeros((1, 10, 12), dtype=np.int64)
    t0 = time.perf_counter()
    for _ in range(5):
        rt.decode_codes(test_codes)
    t_decode = (time.perf_counter() - t0) / 5

    per_frame_total = t_slow_step + t_fast_all
    rtf = per_frame_total / (ms_per_frame / 1000)

    print(f"  Slow AR prompt ({prompt_len} tok): {t_slow_prompt*1000:.0f}ms")
    print(f"  Slow AR per-token step:            {t_slow_step*1000:.1f}ms")
    print(f"  Fast AR (10 codebook steps):       {t_fast_all*1000:.1f}ms")
    print(f"  Codec decode (12 frames):          {t_decode*1000:.1f}ms")
    print(f"  ────────────────────────────────────────")
    print(f"  Per-frame total (1 slow + 10 fast): {per_frame_total*1000:.1f}ms")
    print(f"  Audio per frame:                   {ms_per_frame:.1f}ms")
    print(f"  Real-time budget:                  {ms_per_frame:.1f}ms")
    print(f"  RTF:                               {rtf:.1f}x")
    print(f"  Slow AR share:                     {t_slow_step/per_frame_total*100:.0f}%")
    print(f"  Fast AR share:                     {t_fast_all/per_frame_total*100:.0f}%")

    # Memory bandwidth estimate
    slow_size = 517e6  # INT4 bytes
    fast_size = 39e6
    print(f"\n  --- Bandwidth analysis (slow AR) ---")
    print(f"  Slow AR weights: {slow_size/1e6:.0f} MB")
    print(f"  Per-token weight read: {slow_size/1e6:.0f} MB in {t_slow_step*1000:.1f}ms")
    bw = slow_size / t_slow_step / 1e9
    print(f"  Effective bandwidth: {bw:.1f} GB/s")
