"""Shared data structures and I/O for manifest building."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Matches the upstream validate_sample_id rule: filename-safe, no path sep.
_ID_RE = re.compile(r"[\w.-]+", flags=re.UNICODE)


@dataclass(frozen=True)
class Record:
    """One utterance ready to become a manifest line.

    ``audio`` is an absolute path on disk. ``reference_audio`` /
    ``reference_text`` are optional and must appear together (the zero-shot
    voice-cloning path in audio8_tts_prepare.py).
    """

    id: str
    text: str
    audio: Path
    reference_audio: Path | None = None
    reference_text: str | None = None


class Source(ABC):
    """A dataset that yields :class:`Record` objects."""

    name: str = "source"

    @abstractmethod
    def iter_records(self) -> Iterator[Record]:
        """Yield records. Audio paths must be absolute and may not yet exist."""
        raise NotImplementedError


def is_valid_id(sample_id: str) -> bool:
    """Mirror upstream validate_sample_id without raising."""
    sid = sample_id.strip()
    if not sid or sid in {".", ".."}:
        return False
    if Path(sid).name != sid:
        return False
    return bool(_ID_RE.fullmatch(sid))


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (line_number, obj) for non-empty JSON object lines, 1-indexed."""
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            yield line_number, json.loads(raw)


def manifest_line(record: Record, *, path_mode: str) -> str:
    """Serialize a record to one JSONL line matching audio8_tts_prepare.py."""
    row: dict[str, Any] = {
        "id": record.id,
        "text": record.text,
        "audio": _format_path(record.audio, path_mode),
    }
    if record.reference_audio is not None and record.reference_text is not None:
        row["reference_audio"] = _format_path(record.reference_audio, path_mode)
        row["reference_text"] = record.reference_text
    return json.dumps(row, ensure_ascii=False) + "\n"


def _format_path(path: Path, path_mode: str) -> str:
    if path_mode == "absolute":
        return str(path.resolve())
    # relative: as stored; resolved elsewhere by the manifest dir
    return str(path)
