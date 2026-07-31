"""Generic transcript loader (TSV/CSV) for OpenSLR SLR76, HiTZ, and similar.

Many small Basque speech corpora ship as a simple mapping of utterance id ->
transcript, with audio in a sibling directory. Rather than hard-code each
layout, this loader takes a transcripts file plus column indices/headers and
yields records. Point it at the right columns for your dataset:

OpenSLR SLR76 (Basque)
    Typically a ``line_index.tsv`` with ``<utt_id>\\t<transcript>`` and a
    ``wav`` directory (verify the exact names against your download). Example::

        zortzi-manifest from-transcripts \\
          --transcripts /data/slr76/line_index.tsv \\
          --audio-dir /data/slr76/wav --audio-ext wav \\
          --id-col 0 --text-col 1 --audio-from-id

HiTZ Aholab (Maider / Antton)
    Studio corpora behind the aHoTTS voices; layout varies by release. Once
    you have a transcript file with id+text and the wav dir, the same command
    applies. These are already orthographic spoken-word transcripts, so they
    are the safest ``text`` source and the best zero-shot reference voices.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from basque_manifest.records import Record, Source


class TranscriptSource(Source):
    name = "transcripts"

    def __init__(
        self,
        transcripts: Path,
        audio_dir: Path,
        *,
        delimiter: str = "\t",
        id_col: int | str = 0,
        text_col: int | str = 1,
        audio_col: int | str | None = None,
        audio_ext: str = "wav",
        audio_from_id: bool = False,
        has_header: bool | None = None,
        split_first: bool = False,
        id_prefix: str = "",
    ) -> None:
        self.transcripts = Path(transcripts)
        self.audio_dir = Path(audio_dir)
        self.delimiter = delimiter
        self.id_col = id_col
        self.text_col = text_col
        self.audio_col = audio_col
        self.audio_ext = audio_ext.lstrip(".")
        self.audio_from_id = audio_from_id
        self.has_header = has_header
        self.split_first = split_first
        self.id_prefix = id_prefix
        if not self.transcripts.is_file():
            raise FileNotFoundError(f"transcripts file not found: {self.transcripts}")
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"audio dir not found: {self.audio_dir}")
        if self.split_first:
            # split_first: each line is "<id><delim><text...>"; audio uses the
            # raw (unprefixed) id, so audio_from_id / audio_col are irrelevant.
            if not delimiter:
                raise ValueError("--delimiter must be set with --split-first")
        elif not self.audio_from_id and self.audio_col is None:
            raise ValueError(
                "either --audio-from-id or --audio-col must be set "
                "(or use --split-first)"
            )

    def iter_records(self) -> Iterator[Record]:
        if self.split_first:
            yield from self._iter_split_first()
            return
        with self.transcripts.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=self.delimiter,
                                quoting=csv.QUOTE_NONE)
            rows = list(reader)
        header: dict[str, int] | None = None
        start = 0
        if self.has_header is True:
            header = {name: i for i, name in enumerate(rows[0])}
            start = 1
        elif self.has_header is None:
            # auto-detect: if the configured id_col looks like a header label,
            # treat the first row as a header.
            if rows and isinstance(self.id_col, str) and self.id_col in rows[0]:
                header = {name: i for i, name in enumerate(rows[0])}
                start = 1

        def col(row: list[str], key: int | str) -> str:
            if isinstance(key, str):
                if header is None or key not in header:
                    raise KeyError(
                        f"column header {key!r} not found in {self.transcripts}"
                    )
                return row[header[key]]
            return row[key]

        for row in rows[start:]:
            if not row:
                continue
            raw_id = col(row, self.id_col).strip()
            text = col(row, self.text_col).strip()
            if not raw_id or not text:
                continue
            sample_id = f"{self.id_prefix}{raw_id}" if self.id_prefix else raw_id
            if self.audio_from_id:
                audio = (self.audio_dir / f"{raw_id}.{self.audio_ext}").resolve()
            else:
                assert self.audio_col is not None
                audio_name = col(row, self.audio_col).strip()
                audio = (self.audio_dir / audio_name).resolve()
            yield Record(id=sample_id, text=text, audio=audio)

    def _iter_split_first(self) -> Iterator[Record]:
        """Yield records from a ``<id><delim><text with delimiter>`` file.

        Used by HiTZ ``corpus.txt`` (``NEU_00001 <sentence>``): the id is the
        first token and the text is everything after the first delimiter, so
        a plain ``csv`` split would shatter the text. Audio filenames always
        use the raw (unprefixed) id.
        """
        with self.transcripts.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(self.delimiter, 1)
                if len(parts) < 2:
                    continue
                raw_id = parts[0].strip()
                text = parts[1].strip()
                if not raw_id or not text:
                    continue
                sample_id = (
                    f"{self.id_prefix}{raw_id}" if self.id_prefix else raw_id
                )
                audio = (self.audio_dir / f"{raw_id}.{self.audio_ext}").resolve()
                yield Record(id=sample_id, text=text, audio=audio)
