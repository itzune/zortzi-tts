"""Optional audio duration probing via ffprobe.

Used for pre-filtering long clips before the (relatively expensive) codec
encoding step in audio8_tts_prepare.py. ffmpeg/ffprobe must be on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class DurationError(RuntimeError):
    pass


def ffprobe_available() -> bool:
    return shutil.which("ffprobe") is not None


def audio_duration_seconds(path: Path) -> float:
    """Return media duration in seconds, or raise :class:`DurationError`."""
    if not path.is_file():
        raise DurationError(f"audio file not found: {path}")
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise DurationError(
            f"ffprobe failed on {path}: {proc.stderr.strip() or 'no stderr'}"
        )
    try:
        data = json.loads(proc.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        raise DurationError(f"could not parse duration for {path}: {exc}") from exc
