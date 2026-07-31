#!/usr/bin/env python3
"""Upsample interrogative/exclamative sentences in a prepared manifest.

Reads a prepared JSONL manifest (audio codes already on disk) and writes a new
one where sentences ending in '?' or '!' are duplicated N times. This biases
training toward expressive prosody without needing to re-encode any audio.

Usage:
    python upsample_expressive.py \\
        --input  /root/work/data/hitz/train_hitz_paired_prepared.jsonl \\
        --output /root/work/data/hitz/train_hitz_prosody_prepared.jsonl \\
        --interrogative-factor 3 \\
        --exclamative-factor 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def classify(text: str) -> str:
    """Classify a sentence by its terminal punctuation."""
    t = text.strip()
    if t.endswith("?"):
        return "interrogative"
    if t.endswith("!"):
        return "exclamative"
    return "declarative"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--interrogative-factor", type=int, default=3,
                    help="How many times to duplicate interrogatives (default: 3)")
    ap.add_argument("--exclamative-factor", type=int, default=3,
                    help="How many times to duplicate exclamatives (default: 3)")
    ap.add_argument("--declarative-factor", type=int, default=1,
                    help="How many times to duplicate declaratives (default: 1)")
    args = ap.parse_args()

    factors = {
        "interrogative": args.interrogative_factor,
        "exclamative": args.exclamative_factor,
        "declarative": args.declarative_factor,
    }

    counts = {"interrogative": 0, "exclamative": 0, "declarative": 0}
    written = {"interrogative": 0, "exclamative": 0, "declarative": 0}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open() as fin, args.output.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            kind = classify(text)
            counts[kind] += 1
            factor = factors[kind]
            for _ in range(factor):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written[kind] += 1

    total_in = sum(counts.values())
    total_out = sum(written.values())
    print(f"Input:  {total_in} records", file=sys.stderr)
    for k in ["declarative", "interrogative", "exclamative"]:
        orig_pct = 100 * counts[k] / total_in if total_in else 0
        new_pct = 100 * written[k] / total_out if total_out else 0
        print(f"  {k:14s}: {counts[k]:6d} ({orig_pct:5.1f}%) -> "
              f"{written[k]:6d} ({new_pct:5.1f}%)  [×{factors[k]}]",
              file=sys.stderr)
    print(f"Output: {total_out} records", file=sys.stderr)


if __name__ == "__main__":
    main()
