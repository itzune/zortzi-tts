"""Mozilla Common Voice (eu) loader.

A Common Voice corpus export has this layout::

    cv-corpus-25.0-eu/
      validated.tsv   (and train.tsv / dev.tsv / test.tsv / invalid.tsv)
      clips/
        common_voice_eu_12345.mp3
        ...

The TSV columns are: client_id, path, sentence, up_votes, down_votes, age,
gender, accent, locale, segment. We use ``path`` (relative to ``clips/``) for
the audio, ``sentence`` for the text, and the ``path`` stem as the manifest id
(which is already filename-safe and unique).
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from basque_manifest.records import Record, Source


class CommonVoiceSource(Source):
    name = "common_voice"

    def __init__(
        self,
        tsv: Path,
        clips: Path,
        *,
        min_upvotes: int = 0,
        max_downvotes: int | None = None,
    ) -> None:
        self.tsv = Path(tsv)
        self.clips = Path(clips)
        self.min_upvotes = int(min_upvotes)
        self.max_downvotes = (
            None if max_downvotes is None else int(max_downvotes)
        )
        if not self.tsv.is_file():
            raise FileNotFoundError(f"Common Voice TSV not found: {self.tsv}")
        if not self.clips.is_dir():
            raise FileNotFoundError(f"clips dir not found: {self.clips}")

    def iter_records(self) -> Iterator[Record]:
        with self.tsv.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                sentence = (row.get("sentence") or "").strip()
                if not sentence:
                    continue
                try:
                    up = int(row.get("up_votes") or 0)
                    down = int(row.get("down_votes") or 0)
                except ValueError:
                    continue
                if up < self.min_upvotes:
                    continue
                if self.max_downvotes is not None and down > self.max_downvotes:
                    continue
                clip_path = row.get("path")
                if not clip_path:
                    continue
                audio = (self.clips / clip_path).resolve()
                yield Record(
                    id=Path(clip_path).stem,
                    text=sentence,
                    audio=audio,
                )
