"""Command-line interface for the Basque manifest builder.

Subcommands:
  from-common-voice   build a manifest from a Common Voice eu export
  from-transcripts    build a manifest from a generic TSV/CSV transcript file
  merge               concatenate manifests, dropping duplicate ids
  normalize           re-normalize text in an existing manifest (in place)
  stats               print summary stats about a manifest
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from basque_manifest.builder import (
    BuildConfig,
    build_manifest,
    merge_manifests,
    normalize_manifest,
    pair_references,
)
from basque_manifest.records import read_jsonl
from basque_manifest.sources.common_voice import CommonVoiceSource
from basque_manifest.sources.transcripts import TranscriptSource

LOGGER = logging.getLogger("zortzi")


# --------------------------------------------------------------------------- #
# shared option helpers
# --------------------------------------------------------------------------- #

def _add_normalize_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-normalize", action="store_true",
                   help="skip Basque text normalization (data already clean)")
    p.add_argument("--lowercase", action="store_true", help="lowercase text")
    p.add_argument("--expand-ordinals", action="store_true",
                   help="rewrite 'N.' ordinals (ambiguous with sentence-final "
                        "numbers; off by default)")
    p.add_argument("--min-chars", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=2000)
    p.add_argument("--max-duration-sec", type=float, default=None,
                   help="drop clips longer than this (uses ffprobe)")
    p.add_argument("--path-mode", choices=("absolute", "relative"),
                   default="absolute")
    p.add_argument("--fixed-reference-audio", type=Path, default=None,
                   help="attach this reference clip to every record")
    p.add_argument("--fixed-reference-text", default=None,
                   help="transcript for --fixed-reference-audio (required with it)")


def _config_from_args(args: argparse.Namespace, output: Path) -> BuildConfig:
    return BuildConfig(
        output_jsonl=output,
        normalize=not args.no_normalize,
        lowercase=args.lowercase,
        expand_ordinals=args.expand_ordinals,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_duration_sec=args.max_duration_sec,
        path_mode=args.path_mode,
        fixed_reference_audio=args.fixed_reference_audio,
        fixed_reference_text=args.fixed_reference_text,
    )


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #

def cmd_from_common_voice(args: argparse.Namespace) -> int:
    source = CommonVoiceSource(
        tsv=args.tsv,
        clips=args.clips,
        min_upvotes=args.min_upvotes,
        max_downvotes=args.max_downvotes,
    )
    config = _config_from_args(args, args.output)
    stats = build_manifest([source], config)
    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_from_transcripts(args: argparse.Namespace) -> int:
    source = TranscriptSource(
        transcripts=args.transcripts,
        audio_dir=args.audio_dir,
        delimiter=args.delimiter,
        id_col=args.id_col,
        text_col=args.text_col,
        audio_col=args.audio_col,
        audio_ext=args.audio_ext,
        audio_from_id=args.audio_from_id,
        has_header=args.has_header,
        split_first=args.split_first,
        id_prefix=args.id_prefix,
    )
    config = _config_from_args(args, args.output)
    stats = build_manifest([source], config)
    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    written = merge_manifests(args.inputs, args.output, path_mode=args.path_mode)
    print(f"merged {written} records into {args.output}")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    n = normalize_manifest(
        args.manifest,
        lowercase=args.lowercase,
        expand_ordinals=args.expand_ordinals,
    )
    print(f"normalized {n} records in {args.manifest}")
    return 0


def cmd_pair_references(args: argparse.Namespace) -> int:
    written = pair_references(args.input, args.output, seed=args.seed)
    print(f"paired {written} records into {args.output}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    lengths: list[int] = []
    ids: list[str] = []
    have_ref = 0
    for _, row in read_jsonl(args.manifest):
        lengths.append(len(str(row.get("text", ""))))
        if row.get("id"):
            ids.append(str(row["id"]))
        if row.get("reference_audio"):
            have_ref += 1
    if not lengths:
        print("empty manifest")
        return 0
    lengths.sort()
    n = len(lengths)
    # crude speaker guess: id prefix before the first digit run
    speakers = Counter()
    for rid in ids:
        prefix = rid.rstrip("0123456789")
        speakers[prefix] += 1
    print(f"records:        {n}")
    print(f"with reference: {have_ref}")
    print(f"text chars:     min={lengths[0]} median={lengths[n // 2]} "
          f"mean={sum(lengths) / n:.1f} max={lengths[-1]}")
    print(f"unique ids:     {len(set(ids))}")
    top = ", ".join(f"{k}={v}" for k, v in speakers.most_common(5))
    print(f"id prefixes:    {top}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zortzi-manifest",
        description="Basque manifest builder for Audio8_TTS fine-tuning.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    cv = sub.add_parser("from-common-voice", help="Common Voice eu export")
    cv.add_argument("--tsv", type=Path, required=True,
                    help="validated.tsv (or train.tsv) path")
    cv.add_argument("--clips", type=Path, required=True, help="clips/ dir")
    cv.add_argument("--min-upvotes", type=int, default=0)
    cv.add_argument("--max-downvotes", type=int, default=None)
    cv.add_argument("--output", "-o", type=Path, required=True)
    _add_normalize_flags(cv)
    cv.set_defaults(func=cmd_from_common_voice)

    tr = sub.add_parser("from-transcripts", help="generic TSV/CSV transcripts")
    tr.add_argument("--transcripts", type=Path, required=True)
    tr.add_argument("--audio-dir", type=Path, required=True)
    tr.add_argument("--delimiter", default="\t")
    tr.add_argument("--id-col", default=0,
                    help="column index or header name for the id")
    tr.add_argument("--text-col", default=1,
                    help="column index or header name for the text")
    tr.add_argument("--audio-col", default=None,
                    help="column index/name for the audio filename")
    tr.add_argument("--audio-ext", default="wav")
    tr.add_argument("--audio-from-id", action="store_true",
                    help="audio filename = '{id}.{audio-ext}'")
    tr.add_argument("--split-first", action="store_true",
                    help="split each line on the first delimiter only "
                         "(id=<part0>, text=<part1>); for 'NEU_00001 <text>' "
                         "formats where the text contains the delimiter")
    tr.add_argument("--id-prefix", default="",
                    help="prefix raw ids (e.g. 'maider_'); audio filenames "
                         "still use the raw id")
    tr.add_argument("--has-header", dest="has_header", default=None,
                    action="store_true", help="first row is a header")
    tr.add_argument("--no-header", dest="has_header",
                    action="store_false", help="first row is data")
    tr.add_argument("--output", "-o", type=Path, required=True)
    _add_normalize_flags(tr)
    tr.set_defaults(func=cmd_from_transcripts)

    mg = sub.add_parser("merge", help="concatenate manifests, drop dup ids")
    mg.add_argument("inputs", type=Path, nargs="+")
    mg.add_argument("--output", "-o", type=Path, required=True)
    mg.add_argument("--path-mode", choices=("absolute", "relative"),
                    default="absolute")
    mg.set_defaults(func=cmd_merge)

    nm = sub.add_parser("normalize", help="re-normalize text in a manifest")
    nm.add_argument("manifest", type=Path)
    nm.add_argument("--in-place", action="store_true",
                    help="rewrite the file (default; output goes back to it)")
    nm.add_argument("--lowercase", action="store_true")
    nm.add_argument("--expand-ordinals", action="store_true")
    nm.set_defaults(func=cmd_normalize)

    pr = sub.add_parser(
        "pair-references",
        help="attach cross-clip references for voice-anchor fine-tuning",
    )
    pr.add_argument("input", type=Path, help="single-speaker manifest")
    pr.add_argument("--output", "-o", type=Path, required=True)
    pr.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the pairing shuffle")
    pr.set_defaults(func=cmd_pair_references)

    st = sub.add_parser("stats", help="summary stats about a manifest")
    st.add_argument("manifest", type=Path)
    st.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
