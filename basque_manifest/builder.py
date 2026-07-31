"""Orchestrate: source records -> normalize -> filter -> write manifest JSONL."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from basque_manifest.audio import DurationError, audio_duration_seconds, ffprobe_available
from basque_manifest.normalize import normalize_text
from basque_manifest.records import Record, Source, is_valid_id, manifest_line

LOGGER = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    output_jsonl: Path
    normalize: bool = True
    lowercase: bool = False
    expand_ordinals: bool = False
    min_chars: int = 1
    max_chars: int = 2000
    max_duration_sec: float | None = None
    path_mode: str = "absolute"  # "absolute" | "relative"
    fixed_reference_audio: Path | None = None
    fixed_reference_text: str | None = None
    require_audio_exists: bool = True


@dataclass
class BuildStats:
    total: int = 0
    written: int = 0
    skipped_empty: int = 0
    skipped_long: int = 0
    skipped_id: int = 0
    skipped_audio: int = 0
    skipped_duration: int = 0
    duplicates: int = 0
    duplicate_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        d = {k: v for k, v in vars(self).items() if k != "duplicate_ids"}
        d["duplicate_ids_count"] = len(self.duplicate_ids)
        return d


def build_manifest(sources: list[Source], config: BuildConfig) -> BuildStats:
    """Run all sources through normalization/filtering into one JSONL."""
    if config.fixed_reference_audio is not None:
        if config.fixed_reference_text is None:
            raise ValueError(
                "--fixed-reference-text is required with --fixed-reference-audio"
            )
        if not config.fixed_reference_audio.is_file():
            raise FileNotFoundError(
                f"fixed reference audio not found: {config.fixed_reference_audio}"
            )

    stats = BuildStats()
    seen: set[str] = set()
    use_duration = config.max_duration_sec is not None
    if use_duration and not ffprobe_available():
        LOGGER.warning("ffprobe not found; --max-duration-sec will be ignored")
        use_duration = False
    probe_duration = use_duration

    ref_audio = (
        config.fixed_reference_audio.resolve()
        if config.fixed_reference_audio is not None
        else None
    )
    ref_text = config.fixed_reference_text

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("w", encoding="utf-8") as out:
        for source in sources:
            LOGGER.info("scanning source: %s", source.name)
            for record in tqdm(
                source.iter_records(), desc=source.name, unit="utt"
            ):
                stats.total += 1

                if not is_valid_id(record.id):
                    stats.skipped_id += 1
                    continue
                if record.id in seen:
                    stats.duplicates += 1
                    if len(stats.duplicate_ids) < 10:
                        stats.duplicate_ids.append(record.id)
                    continue

                text = record.text
                if config.normalize:
                    text = normalize_text(
                        text,
                        lowercase=config.lowercase,
                        expand_ordinals=config.expand_ordinals,
                    )
                if len(text) < config.min_chars:
                    stats.skipped_empty += 1
                    continue
                if len(text) > config.max_chars:
                    stats.skipped_long += 1
                    continue

                if config.require_audio_exists and not record.audio.is_file():
                    stats.skipped_audio += 1
                    continue

                if probe_duration:
                    try:
                        if audio_duration_seconds(record.audio) > config.max_duration_sec:  # type: ignore[operator]
                            stats.skipped_duration += 1
                            continue
                    except DurationError as exc:
                        LOGGER.debug("duration probe failed: %s", exc)

                seen.add(record.id)
                final = Record(
                    id=record.id,
                    text=text,
                    audio=record.audio,
                    reference_audio=ref_audio,
                    reference_text=ref_text,
                )
                out.write(manifest_line(final, path_mode=config.path_mode))
                stats.written += 1

    LOGGER.info(
        "wrote %d/%d records to %s (skipped: empty=%d long=%d id=%d "
        "audio=%d duration=%d dup=%d)",
        stats.written, stats.total, config.output_jsonl,
        stats.skipped_empty, stats.skipped_long, stats.skipped_id,
        stats.skipped_audio, stats.skipped_duration, stats.duplicates,
    )
    return stats


def merge_manifests(
    inputs: list[Path], output: Path, *, path_mode: str = "absolute"
) -> int:
    """Concatenate manifests, dropping duplicate ids (first wins)."""
    seen: set[str] = set()
    written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for path in inputs:
            from basque_manifest.records import read_jsonl

            for _, row in read_jsonl(path):
                rid = row.get("id")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                out.write(
                    __import__("json").dumps(row, ensure_ascii=False) + "\n"
                )
                written += 1
    LOGGER.info("merged %d records into %s", written, output)
    return written


def pair_references(
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    seed: int = 42,
) -> int:
    """Attach a cross-clip reference to every record (voice-anchor pairing).

    For zero-shot voice-cloning fine-tuning, the reference codes sit in the
    model's *unsupervised* context while the target codes are supervised. If
    the reference were the same clip as the target the model could trivially
    copy it, so each record is paired with a *different* record from the same
    manifest (same speaker). A fixed ``seed`` makes the pairing reproducible.

    Run this per-speaker (one manifest = one speaker) so references always
    match the target voice, then :func:`merge_manifests` the results.
    """
    from basque_manifest.records import read_jsonl

    records = [row for _, row in read_jsonl(input_jsonl)]
    n = len(records)
    if n < 2:
        raise ValueError(f"need >=2 records to pair, got {n}")
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_jsonl.open("w", encoding="utf-8") as out:
        for k in range(n):
            target = shuffled[k]
            ref = shuffled[(k + 1) % n]
            row = {
                "id": target["id"],
                "text": target["text"],
                "audio": target["audio"],
                "reference_audio": ref["audio"],
                "reference_text": ref["text"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    LOGGER.info("paired %d records (seed=%d) -> %s", written, seed, output_jsonl)
    return written


def normalize_manifest(
    path: Path, *, lowercase: bool = False, expand_ordinals: bool = False
) -> int:
    """Re-normalize the ``text``/``reference_text`` of an existing manifest."""
    import json

    from basque_manifest.records import read_jsonl

    rows: list[dict] = []
    for _, row in read_jsonl(path):
        if "text" in row:
            row["text"] = normalize_text(
                row["text"], lowercase=lowercase, expand_ordinals=expand_ordinals
            )
        if "reference_text" in row:
            row["reference_text"] = normalize_text(
                row["reference_text"],
                lowercase=lowercase,
                expand_ordinals=expand_ordinals,
            )
        rows.append(row)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)
