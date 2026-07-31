#!/usr/bin/env python3
"""Trainer callback: generate probe-sentence audio at every checkpoint.

Hooked into ``audio8_tts_sft.py`` via ``trainer.add_callback(...)``.  When the
Trainer saves a checkpoint (``on_save``), this callback spawns
``generate_samples.py`` as a subprocess against the freshly-saved checkpoint
directory, collects the resulting WAV files, and logs them to Weights & Biases
as ``wandb.Audio`` artefacts.

Design choices
--------------
* **Subprocess, not in-process** — loading the model from the saved checkpoint
  proves the checkpoint is actually loadable (catches DeepSpeed / codec issues
  early) and avoids touching training state (no eval/train mode flipping, no
  codec loading into the live model).
* **Never crashes training** — every exception is caught and logged.
* **Sampling, not greedy** — greedy decoding loops forever on this model
  (see RESEARCH.md).  Hard-coded ``do_sample=True``.
* **Configurable via environment variables** so the same patched
  ``audio8_tts_sft.py`` works for both the no-reference base phase and the
  voice-anchor phase:

  ============================ ==============================================
  ``SAMPLE_CALLBACK``          ``1`` to enable (default: off)
  ``SAMPLE_VOICE_MAIDER``      ``audio|text``  (pipe-separated) for Maider ref
  ``SAMPLE_VOICE_ANTTON``      ``audio|text``  for Antton ref
  ``SAMPLE_OUTPUT_DIR``        where to save WAVs (default: ``{output}/samples``)
  ``SAMPLE_TEMPERATURE``       default ``0.8``
  ``SAMPLE_TOP_P``             default ``0.95``
  ``SAMPLE_SEED``              default ``42``
  ``SAMPLE_MAX_NEW_TOKENS``    default ``1024``
  ============================ ==============================================
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from transformers import TrainerCallback

LOGGER = logging.getLogger("zortzi.samples")

# generate_samples.py lives next to this file in the Audio8_TTS repo root.
_SCRIPT = Path(__file__).resolve().parent / "generate_samples.py"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _parse_voice(spec: str, fallback_name: str = "ref") -> str:
    """Turn ``audio_path|reference_text`` into ``name|audio_path|reference_text``."""
    parts = spec.split("|", 1)
    if len(parts) != 2:
        return ""
    audio, text = parts
    return f"{fallback_name}|{audio.strip()}|{text.strip()}"


class CheckpointSamplesCallback(TrainerCallback):
    """Generate + log probe audio at each checkpoint save."""

    def __init__(self) -> None:
        self.python = sys.executable
        self.temperature = float(_env("SAMPLE_TEMPERATURE", "0.8"))
        self.top_p = float(_env("SAMPLE_TOP_P", "0.95"))
        self.seed = int(_env("SAMPLE_SEED", "42"))
        self.max_new_tokens = int(_env("SAMPLE_MAX_NEW_TOKENS", "1024"))

        # Build --voice flags from env vars.
        # Format: SAMPLE_VOICE_MAIDER="/path/to.wav|reference text"
        # The voice name is derived from the env-var suffix (maider, antton, ...).
        self.voices: list[str] = []
        for key in ("SAMPLE_VOICE_MAIDER", "SAMPLE_VOICE_ANTTON"):
            spec = _env(key)
            if spec:
                voice_name = key.replace("SAMPLE_VOICE_", "").lower()
                parsed = _parse_voice(spec, fallback_name=voice_name)
                if parsed:
                    self.voices.append(parsed)
                    LOGGER.info("samples: voice %s: %s", voice_name, parsed)

        self.output_root = _env("SAMPLE_OUTPUT_DIR")  # resolved per-run below
        self.processor_model = _env("SAMPLE_PROCESSOR_MODEL")  # original model dir for processor/tokenizer
        self._wandb = None
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            LOGGER.warning("samples: wandb not available; files saved to disk only")

    # ------------------------------------------------------------------ #
    # wandb helpers
    # ------------------------------------------------------------------ #

    def _log_wandb(self, results: list[dict], step: int) -> None:
        if self._wandb is None:
            return
        try:
            audio_logs: dict = {}
            prosody_logs: dict = {}
            for r in results:
                if not r.get("path"):
                    continue
                voice = r["voice"]
                sent_idx = r["sentence_idx"]
                text = r["text"]
                duration = r.get("duration_sec", "?")
                key = f"samples/{voice}/sentence_{sent_idx}"
                caption = (
                    f"step {step} · {duration}s · "
                    f"{'OK' if r.get('finished') else 'NO_EOS'}\n"
                    f"{text[:100]}"
                )
                audio_logs[key] = self._wandb.Audio(
                    r["path"],
                    sample_rate=r.get("sample_rate", 44100),
                    caption=caption,
                )
                # Log prosody metrics as scalars (per voice + sentence)
                prosody = r.get("prosody")
                if prosody:
                    prefix = f"prosody/{voice}/s{sent_idx}"
                    for metric, value in prosody.items():
                        prosody_logs[f"{prefix}/{metric}"] = value
            if audio_logs:
                self._wandb.log(audio_logs, step=step, commit=False)
            if prosody_logs:
                self._wandb.log(prosody_logs, step=step, commit=False)
            logged = len(audio_logs) + len(prosody_logs)
            LOGGER.info("samples: logged %d audio + %d prosody metrics to wandb at step %d",
                        len(audio_logs), len(prosody_logs), step)
        except Exception as exc:
            LOGGER.warning("samples: wandb logging failed: %s", exc)

    # ------------------------------------------------------------------ #
    # TrainerCallback
    # ------------------------------------------------------------------ #

    def on_save(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if _env("SAMPLE_CALLBACK") != "1":
            return control

        step = state.global_step
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if not ckpt_dir.is_dir():
            # Some setups save to different paths; try best_model_checkpoint.
            ckpt_dir = Path(state.best_model_checkpoint or "")
        if not ckpt_dir.is_dir():
            LOGGER.warning("samples: checkpoint dir not found for step %d, skipping", step)
            return control

        out_dir = Path(self.output_root or (Path(args.output_dir) / "samples"))
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.python,
            str(_SCRIPT),
            "--model", str(ckpt_dir),
            "--output-dir", str(out_dir),
            "--step", str(step),
            "--temperature", str(self.temperature),
            "--top-p", str(self.top_p),
            "--seed", str(self.seed),
            "--max-new-tokens", str(self.max_new_tokens),
        ]
        if self.processor_model:
            cmd.extend(["--processor", self.processor_model])
        for voice in self.voices:
            cmd.extend(["--voice", voice])

        LOGGER.info("samples: generating probe audio for step %d from %s", step, ckpt_dir)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,   # 10 min hard cap
            )
            if proc.returncode != 0:
                LOGGER.warning(
                    "samples: generate_samples.py exited %d: %s",
                    proc.returncode,
                    proc.stderr[-500:],
                )
                return control

            results = json.loads(proc.stdout) if proc.stdout.strip() else []
            if not results:
                LOGGER.warning("samples: no results from generate_samples.py")
                return control

            self._log_wandb(results, step)
            ok = sum(1 for r in results if r.get("path"))
            with_prosody = sum(1 for r in results if r.get("prosody"))
            LOGGER.info("samples: step %d — %d/%d audio generated, %d with prosody metrics",
                        step, ok, len(results), with_prosody)
        except subprocess.TimeoutExpired:
            LOGGER.warning("samples: generate_samples.py timed out for step %d", step)
        except Exception as exc:
            LOGGER.warning("samples: callback error for step %d: %s", step, exc)

        return control
