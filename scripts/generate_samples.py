#!/usr/bin/env python3
"""Generate Basque test-sentence audio from a checkpoint.

Standalone script invoked by ``CheckpointSamplesCallback`` (or manually) to
synthesise a fixed set of probe sentences and save WAVs to disk.  The output
is a JSON array on stdout so the callback can pick up the file list and log
the audio to Weights & Biases.

The two probe sentences stress-test Basque sibilants/affricates (tx, tz, z, s,
x) and question intonation — exactly the phonemes where under-training shows
up (k→t, s→z).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

from audio8_tts_data import clean_text

# --------------------------------------------------------------------------- #
# prosody analysis (optional — soft dep on librosa)
# --------------------------------------------------------------------------- #

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def analyze_prosody(audio_path: str, sr: int = 16000,
                     focus_end_sec: float | None = None) -> dict | None:
    """Extract pitch-based prosodic metrics from a generated probe audio.

    Per senior ML review, these metrics track prosodic recovery from the flat
    Common Voice read-speech prior:

    - ``f0_std_hz``: pitch variance.  Flat CV read-speech ~15-25 Hz,
      expressive speech > 35 Hz.
    - ``focus_peak_hz``: max F0 in the first ``focus_end_sec`` seconds.
      For the Basque wh-question probe, this should land on "Nondik" (the
      galde-hitza / focus word), not at the sentence end.
    - ``boundary_delta_hz``: last-voiced-frame F0 minus utterance mean.
      Wh-questions should be negative (falling); yes/no questions positive.
    """
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
        f0_std = float(np.std(f0v))
        end_f0 = float(f0v[-1])

        result = {
            "f0_mean_hz": round(f0_mean, 2),
            "f0_std_hz": round(f0_std, 2),
            "end_f0_hz": round(end_f0, 2),
            "boundary_delta_hz": round(end_f0 - f0_mean, 2),
            "voiced_frames": int(len(f0v)),
        }

        if focus_end_sec is not None:
            focus_mask = times_v <= focus_end_sec
            focus_f0 = f0v[focus_mask]
            if len(focus_f0) > 0:
                result["focus_peak_hz"] = round(float(np.max(focus_f0)), 2)
                result["focus_mean_hz"] = round(float(np.mean(focus_f0)), 2)

        return result
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# probe sentences
# --------------------------------------------------------------------------- #

# Each probe is (text, focus_end_sec) where focus_end_sec delimits the
# wh-word region for focus-peak-pitch extraction (None = no focus metric).
PROBE_SENTENCES: list[tuple[str, float | None]] = [
    # 1 — sibilant/affricate stress test: txakur txiki, xagu, sasi, baserri
    #     zaharrerantz, zurezko, ustekabean, izututa, artzainak, makila...
    ("Atzo goizean, Joanes artzainak bere txakur txikiari zurezko makila "
     "bota zion sasi artean; baina, ustekabean, xagu beltz bat izututa "
     "atera zen, eta baserri zaharrerantz ihes egin zuen.", None),
    # 2 — wh-question intonation: "Nondik" is the galde-hitza (focus word).
    #     Correct Basque prosody: pitch peak on "Nondik", then falling.
    #     English/Spanish leakage: flat body + final rise on "diru?".
    ("\u201cNondik atera duzu hainbeste diru auto berri hori erosteko?\u201d", 1.2),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return getattr(torch, value)


def synthesize(
    model,
    processor,
    *,
    text: str,
    device: torch.device,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Return (waveform, sample_rate, finished)."""
    proc_kwargs: dict = {"text": [text], "return_tensors": "pt"}
    if reference_audio is not None:
        proc_kwargs.update(
            reference_audio=[reference_audio],
            reference_text=[reference_text],
        )
    inputs = processor(**proc_kwargs)
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,          # sampling — greedy loops forever (see RESEARCH.md)
            generator=generator,
            return_dict_in_generate=True,
        )
        waveforms, waveform_lengths = model.decode_audio(output.codes)

    waveform = waveforms[0, : int(waveform_lengths[0])].float().cpu().numpy()
    finished = bool(output.finished[0])
    sample_rate = int(model.config.codec_sample_rate)
    return waveform, sample_rate, finished


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Basque probe-sentence audio from a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="checkpoint or model dir (weights + config)")
    parser.add_argument("--processor", default=None,
                        help="dir for processor/tokenizer (defaults to --model; "
                             "use the original model dir when loading a DeepSpeed "
                             "checkpoint that lacks processor files)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0, help="training step (for filenames)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    # Reference voices — each --voice is "name|audio_path|reference_text"
    # (pipe-separated to avoid argparse multi-value headaches).
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help='reference voice as "name|/path/to.wav|reference text" '
             "(can be repeated; a built-in no-reference voice is always added)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Always include a no-reference ("default") voice.
    voices: list[tuple[str, Path | None, str | None]] = [("default", None, None)]
    for spec in args.voice:
        parts = spec.split("|", 2)
        if len(parts) != 3:
            print(f"WARNING: skipping malformed --voice '{spec}' "
                  f"(expected name|path|text)", file=sys.stderr)
            continue
        name, audio_path, ref_text = parts
        voices.append((name, Path(audio_path), ref_text))

    processor_src = args.processor or args.model
    # DeepSpeed intermediate checkpoints lack processor/tokenizer files.
    # Fall back to the original model dir if the checkpoint doesn't have them.
    if not (Path(processor_src) / "processing_arktts.py").exists():
        fallback = os.environ.get("SAMPLE_PROCESSOR_MODEL", "")
        if fallback and (Path(fallback) / "processing_arktts.py").exists():
            processor_src = fallback
        elif (Path("/root/work/models/Audio8-TTS-Preview-0.6b/") / "processing_arktts.py").exists():
            processor_src = "/root/work/models/Audio8-TTS-Preview-0.6b/"
    print(f"[generate_samples] model={args.model} processor={processor_src} "
          f"device={device} voices={len(voices)} sentences={len(PROBE_SENTENCES)}",
          file=sys.stderr)

    processor = AutoProcessor.from_pretrained(processor_src, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).eval().to(device)

    results: list[dict] = []
    for voice_name, ref_audio, ref_text in voices:
        for sent_idx, (sentence, focus_end_sec) in enumerate(PROBE_SENTENCES, start=1):
            tag = f"{voice_name}_s{sent_idx}_step{args.step}"
            wav_path = output_dir / f"{tag}.wav"
            try:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                waveform, sample_rate, finished = synthesize(
                    model,
                    processor,
                    text=clean_text(sentence),
                    device=device,
                    generator=generator,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    reference_audio=ref_audio,
                    reference_text=ref_text,
                )
                sf.write(wav_path, waveform, sample_rate)
                duration = round(len(waveform) / sample_rate, 2)
                prosody = analyze_prosody(str(wav_path), focus_end_sec=focus_end_sec)
                entry = {
                    "voice": voice_name,
                    "sentence_idx": sent_idx,
                    "text": sentence,
                    "path": str(wav_path),
                    "sample_rate": sample_rate,
                    "duration_sec": duration,
                    "finished": finished,
                    "step": args.step,
                }
                if prosody:
                    entry["prosody"] = prosody
                    print(f"  {tag}: {duration:.1f}s "
                          f"{'OK' if finished else 'NO_EOS'} "
                          f"f0_std={prosody.get('f0_std_hz','?')}",
                          file=sys.stderr)
                else:
                    print(f"  {tag}: {duration:.1f}s "
                          f"{'OK' if finished else 'NO_EOS'}", file=sys.stderr)
                results.append(entry)
            except Exception as exc:
                print(f"  {tag}: FAILED {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                results.append({
                    "voice": voice_name,
                    "sentence_idx": sent_idx,
                    "text": sentence,
                    "path": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "step": args.step,
                })

    # JSON to stdout for the callback to parse
    json.dump(results, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
