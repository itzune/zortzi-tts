"""End-to-end tests for the manifest builder and CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from basque_manifest.builder import (
    BuildConfig,
    build_manifest,
    merge_manifests,
    pair_references,
)
from basque_manifest.cli import main as cli_main
from basque_manifest.sources.common_voice import CommonVoiceSource
from basque_manifest.sources.transcripts import TranscriptSource


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")  # builder only checks existence (no decode here)
    return path


def _make_cv(tmp: Path) -> tuple[Path, Path]:
    clips = tmp / "clips"
    for name in ("common_voice_eu_1.mp3", "common_voice_eu_2.mp3", "common_voice_eu_3.mp3"):
        _touch(clips / name)
    tsv = tmp / "validated.tsv"
    tsv.write_text(
        "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment\n"
        "c1\tcommon_voice_eu_1.mp3\t2026an etorriko da.\t2\t0\t\t\t\teu\t\n"
        "c2\tcommon_voice_eu_2.mp3\tKaixo mundua!\t5\t0\t\t\t\teu\t\n"
        # low upvotes -> filtered when min_upvotes=2
        "c3\tcommon_voice_eu_3.mp3\tBost euro.\t1\t0\t\t\t\teu\t\n",
        encoding="utf-8",
    )
    return tsv, clips


def test_common_voice_build_normalizes_and_filters(tmp_path: Path) -> None:
    tsv, clips = _make_cv(tmp_path)
    source = CommonVoiceSource(tsv, clips, min_upvotes=2)
    out = tmp_path / "cv.jsonl"
    config = BuildConfig(output_jsonl=out, path_mode="absolute")
    stats = build_manifest([source], config)

    # Source pre-filters by min_upvotes=2, so the 1-upvote row is dropped
    # before the builder sees it: 2 yielded, 2 written.
    assert stats.written == 2
    assert stats.total == 2

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2

    by_id = {r["id"]: r for r in rows}
    # number was normalized; suffix preserved
    assert by_id["common_voice_eu_1"]["text"] == "bi mila eta hogeita seian etorriko da."
    # plain text untouched
    assert by_id["common_voice_eu_2"]["text"] == "Kaixo mundua!"
    # audio path is absolute and exists
    assert Path(by_id["common_voice_eu_2"]["audio"]).is_absolute()
    assert Path(by_id["common_voice_eu_2"]["audio"]).is_file()
    # no reference fields when none configured
    assert "reference_audio" not in by_id["common_voice_eu_1"]


def test_fixed_reference_attached(tmp_path: Path) -> None:
    tsv, clips = _make_cv(tmp_path)
    ref_audio = _touch(tmp_path / "ref.wav")
    source = CommonVoiceSource(tsv, clips, min_upvotes=2)
    config = BuildConfig(
        output_jsonl=tmp_path / "ref.jsonl",
        fixed_reference_audio=ref_audio,
        fixed_reference_text="erreferentzia hau da.",
    )
    build_manifest([source], config)
    rows = [json.loads(line) for line in (tmp_path / "ref.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in rows:
        assert r["reference_audio"] == str(ref_audio.resolve())
        assert r["reference_text"] == "erreferentzia hau da."


def test_transcript_source_audio_from_id(tmp_path: Path) -> None:
    audio_dir = tmp_path / "wav"
    for i in (1, 2):
        _touch(audio_dir / f"utt{i:03d}.wav")
    tsv = tmp_path / "line_index.tsv"
    tsv.write_text(
        "utt001\tHamar hitz daude hemen.\n"
        "utt002\t%50 egin du.\n",
        encoding="utf-8",
    )
    source = TranscriptSource(
        tsv, audio_dir, id_col=0, text_col=1, audio_from_id=True, audio_ext="wav"
    )
    out = tmp_path / "slr.jsonl"
    build_manifest([source], BuildConfig(output_jsonl=out))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "Hamar hitz daude hemen."
    assert rows[1]["text"] == "ehuneko berrogeita hamar egin du."
    assert rows[0]["audio"].endswith("utt001.wav")


def test_duplicate_ids_dropped_across_merge(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps({"id": "x", "text": "bat", "audio": "/a/x.wav"}) + "\n"
        + json.dumps({"id": "y", "text": "bi", "audio": "/a/y.wav"}) + "\n",
        encoding="utf-8",
    )
    b.write_text(
        json.dumps({"id": "y", "text": "bi berriz", "audio": "/b/y.wav"}) + "\n"  # dup
        + json.dumps({"id": "z", "text": "hiru", "audio": "/b/z.wav"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "merged.jsonl"
    n = merge_manifests([a, b], out)
    assert n == 3  # x, y (first wins), z
    ids = [json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert ids == ["x", "y", "z"]


def test_missing_audio_skipped(tmp_path: Path) -> None:
    tsv = tmp_path / "t.tsv"
    tsv.write_text("u1\tKaixo.\nu2\tAgur.\n", encoding="utf-8")
    audio_dir = tmp_path / "wav"
    _touch(audio_dir / "u1.wav")
    # u2.wav deliberately not created
    source = TranscriptSource(tsv, audio_dir, audio_from_id=True, audio_ext="wav")
    stats = build_manifest([source], BuildConfig(output_jsonl=tmp_path / "o.jsonl"))
    assert stats.written == 1
    assert stats.skipped_audio == 1


def test_cli_from_common_voice(tmp_path: Path) -> None:
    tsv, clips = _make_cv(tmp_path)
    out = tmp_path / "cli.jsonl"
    rc = cli_main([
        "from-common-voice", "--tsv", str(tsv), "--clips", str(clips),
        "--min-upvotes", "2", "--output", str(out),
    ])
    assert rc == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all({"id", "text", "audio"} <= set(r) for r in rows)


def test_cli_stats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tsv, clips = _make_cv(tmp_path)
    out = tmp_path / "cli.jsonl"
    cli_main(["from-common-voice", "--tsv", str(tsv), "--clips", str(clips),
              "--min-upvotes", "0", "--output", str(out)])
    rc = cli_main(["stats", str(out)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "records:" in captured.out


# --------------------------------------------------------------------------- #
# HiTZ corpus format: split_first + id_prefix
# --------------------------------------------------------------------------- #

def _make_hitz(tmp: Path, speaker: str, n: int = 3) -> tuple[Path, Path]:
    """Mirror the Zenodo HiTZ corpus.txt layout: '<id> <text with spaces>'."""
    audio_dir = tmp / speaker
    for i in range(1, n + 1):
        _touch(audio_dir / f"NEU_{i:05d}.wav")
    corpus = tmp / f"{speaker}_corpus.txt"
    lines = [
        f"NEU_{i:05d} Gaur eguerdi onak izango ditu eta bihar ere bai. {i}"
        for i in range(1, n + 1)
    ]
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return corpus, audio_dir


def test_split_first_parses_hitz_format(tmp_path: Path) -> None:
    corpus, audio_dir = _make_hitz(tmp_path, "maider")
    source = TranscriptSource(
        corpus, audio_dir, delimiter=" ", split_first=True, audio_ext="wav",
    )
    records = list(source.iter_records())
    assert len(records) == 3
    # id is the first token only
    assert records[0].id == "NEU_00001"
    # text is everything after the first space (not just the second word)
    assert records[0].text == "Gaur eguerdi onak izango ditu eta bihar ere bai. 1"
    # audio uses the raw (unprefixed) id
    assert records[0].audio.name == "NEU_00001.wav"


def test_split_first_with_id_prefix(tmp_path: Path) -> None:
    corpus, audio_dir = _make_hitz(tmp_path, "maider")
    source = TranscriptSource(
        corpus, audio_dir, delimiter=" ", split_first=True,
        id_prefix="maider_", audio_ext="wav",
    )
    records = list(source.iter_records())
    assert records[0].id == "maider_NEU_00001"  # prefixed id
    assert records[0].audio.name == "NEU_00001.wav"  # raw audio filename


def test_split_first_build_normalizes(tmp_path: Path) -> None:
    corpus, audio_dir = _make_hitz(tmp_path, "maider")
    source = TranscriptSource(
        corpus, audio_dir, delimiter=" ", split_first=True, id_prefix="maider_",
    )
    out = tmp_path / "maider.jsonl"
    build_manifest([source], BuildConfig(output_jsonl=out))
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    # trailing ' 1' number normalized by the Basque normalizer
    assert rows[0]["text"].endswith(" bat")
    assert rows[0]["id"] == "maider_NEU_00001"


# --------------------------------------------------------------------------- #
# pair-references: cross-clip voice-anchor pairing
# --------------------------------------------------------------------------- #

def test_pair_references_cross_clip_no_self(tmp_path: Path) -> None:
    manifest = tmp_path / "spk.jsonl"
    rows = [
        {"id": f"u{i}", "text": f"esaldi {i}", "audio": f"/d/u{i}.wav"}
        for i in range(5)
    ]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = tmp_path / "paired.jsonl"
    n = pair_references(manifest, out, seed=42)
    assert n == 5
    paired = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    targets = {r["id"] for r in paired}
    assert targets == {"u0", "u1", "u2", "u3", "u4"}  # every record is a target once
    for r in paired:
        # reference is a different record (no degenerate same-clip copy)
        assert r["reference_audio"] != r["audio"]
        assert r["reference_text"] != r["text"]
        assert "reference_audio" in r and "reference_text" in r


def test_pair_references_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "spk.jsonl"
    rows = [
        {"id": f"u{i}", "text": f"esaldi {i}", "audio": f"/d/u{i}.wav"}
        for i in range(8)
    ]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    pair_references(manifest, a, seed=7)
    pair_references(manifest, b, seed=7)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


def test_pair_references_needs_two_records(tmp_path: Path) -> None:
    manifest = tmp_path / "one.jsonl"
    manifest.write_text(
        json.dumps({"id": "u0", "text": "bakarra", "audio": "/d/u0.wav"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=">=2"):
        pair_references(manifest, tmp_path / "out.jsonl", seed=1)


def test_cli_pair_references(tmp_path: Path) -> None:
    manifest = tmp_path / "spk.jsonl"
    rows = [
        {"id": f"u{i}", "text": f"esaldi {i}", "audio": f"/d/u{i}.wav"}
        for i in range(4)
    ]
    manifest.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    out = tmp_path / "paired.jsonl"
    rc = cli_main(["pair-references", str(manifest), "--output", str(out), "--seed", "3"])
    assert rc == 0
    paired = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(paired) == 4
    assert all(r["reference_audio"] != r["audio"] for r in paired)
