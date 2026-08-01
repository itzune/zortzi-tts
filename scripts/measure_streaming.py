#!/usr/bin/env python3
"""Measure streaming latency: time-to-first-chunk and per-chunk timing."""
import time
import sys
from pathlib import Path

# ONNX CPU streaming
sys.path.insert(0, "/root/work/Audio8_TTS/onnx_runtime")
from arktts_runtime.runtime import ArkTtsRuntime

MODEL = "/root/work/outputs/onnx_p2_final"
VOICES = "/root/work/voices_final"
TEXT = "Kaixo mundua! Nire izena Maider da eta gaur eguraldi ederra dugu."

rt = ArkTtsRuntime(Path(MODEL), Path(VOICES), precision="int4", threads=5)
hop = int(rt.manifest["codec_hop_length"])
sr = int(rt.manifest["sample_rate"])
ms_per_frame = hop / sr * 1000
chunk_ms = ms_per_frame * 12
print(f"Model: INT4 ONNX (CPU, 5 threads)")
print(f"Audio per chunk (12 frames): {chunk_ms:.0f} ms")
print(f"Text: {TEXT}")
print()

t0 = time.time()
t_first = None
chunk_times = []
audio_dur_total = 0

for event in rt.stream(
    text=TEXT, voice="maider", max_new_tokens=512,
    temperature=0.8, top_p=0.95, seed=42, chunk_frames=12,
):
    now = time.time()
    if event["type"] == "audio_chunk":
        if t_first is None:
            t_first = now - t0
        chunk_gen = now - t0 if not chunk_times else now - (t0 + sum(chunk_times))
        # audio duration in this chunk
        dur_ms = len(event["audio"]) / sr * 1000
        audio_dur_total += dur_ms
        elapsed = now - t0
        rtf_chunk = chunk_gen / dur_ms * 1000 if chunk_gen > 0 else 0
        print(f"  chunk {event['seq']:>2}: {dur_ms:>6.0f}ms audio, "
              f"gen so far {elapsed:>5.2f}s, frames={event['frame_count']}")
        chunk_times.append(elapsed)
    elif event["type"] == "complete":
        total = time.time() - t0
        print()
        print(f"Time to first chunk: {t_first:.2f}s")
        print(f"Total generation:    {total:.2f}s")
        print(f"Total audio:         {audio_dur_total/1000:.2f}s")
        print(f"Overall RTF:         {total/(audio_dur_total/1000):.2f}x "
              f"({'real-time' if total < audio_dur_total/1000 else 'SLOWER than real-time'})")
        # Per-chunk delta
        if len(chunk_times) > 1:
            deltas = [chunk_times[i]-chunk_times[i-1] for i in range(1, len(chunk_times))]
            avg_delta = sum(deltas)/len(deltas)
            print(f"Avg per-chunk gen:   {avg_delta:.2f}s (need <{chunk_ms/1000:.2f}s for real-time)")
