"""Basque text normalization for Audio8_TTS manifests.

WHY THIS EXISTS
---------------
Audio8_TTS performs **no** language-specific text normalization of its own.
This was verified by reading the upstream training scripts end-to-end:

  - audio8_tts_data.py      -> clean_text() only collapses whitespace
  - audio8_tts_prepare.py   -> passes `text` straight through
  - audio8_tts_sft.py       -> build_sft_example() calls clean_text() only
  - processing_arktts.py    -> (HF processor) _clean_text, whitespace only

There is no G2P/phonemizer, no number-to-words, no abbreviation expansion,
no case folding, no language-ID token. Text is BPE-tokenized verbatim over a
Qwen2-style ~151.6K vocab. Therefore **everything** that should be spoken as
words rather than symbols must be normalized HERE, in the manifest `text`
field, before `audio8_tts_prepare.py` encodes the audio.

The most error-prone piece is number-to-words, because Basque uses a
vigesimal (base-20) system. The tables and the "eta" connector rule follow
the standard (batua) tables from Euskaltzaindia / Omniglot, cross-checked
against a native reference. The exact lexical forms of 17-19 have dialectal
variants (see TODO) and the multi-group "eta" placement for numbers like
1122 should be audited by a native speaker before a production run.
"""

from __future__ import annotations

import re

__all__ = ["normalize_text", "number_to_basque"]


# --------------------------------------------------------------------------- #
# Number tables (standard batua)
# --------------------------------------------------------------------------- #

_ONES = [
    "",            # 0
    "bat",         # 1
    "bi",          # 2
    "hiru",        # 3
    "lau",         # 4
    "bost",        # 5
    "sei",         # 6
    "zazpi",       # 7
    "zortzi",      # 8
    "bederatzi",   # 9
    "hamar",       # 10
    "hamaika",     # 11
    "hamabi",      # 12
    "hamahiru",    # 13
    "hamalau",     # 14
    "hamabost",    # 15
    "hamasei",     # 16
    "hamazazpi",   # 17
    "hemezortzi",  # 18
    "hemeretzi",   # 19
]

# Vigesimal tens: only 20/40/60/80 are atomic; the rest are compounds.
_TENS = {
    20: "hogei",
    40: "berrogei",
    60: "hirurogei",
    80: "laurogei",
}

_HUNDREDS = {
    100: "ehun",
    200: "berrehun",
    300: "hirurehun",
    400: "laurehun",
    500: "bostehun",
    600: "seiehun",
    700: "zazpiehun",
    800: "zortziehun",
    900: "bederatziehun",
}


# --------------------------------------------------------------------------- #
# Integer -> Basque words
# --------------------------------------------------------------------------- #

def _sub100(n: int) -> str:
    """Convert an integer in 1..99 to Basque words (vigesimal)."""
    if n < 20:
        return _ONES[n]
    if n in _TENS:
        return _TENS[n]
    ten = max(t for t in _TENS if t <= n)   # the largest vigesimal ten <= n
    rem = n - ten                            # remainder in 1..19
    return f"{_TENS[ten]}ta {_ONES[rem]}"


def _int_to_eu(n: int) -> str:
    """Convert a non-negative integer to Basque words."""
    if n == 0:
        return "zero"
    parts: list[str] = []
    # millions (1e6) and billions (1e9) -> "milioi" / "miliar"
    if n >= 1_000_000_000:
        b = n // 1_000_000_000
        parts.append("miliar bat" if b == 1 else f"{_int_to_eu(b)} miliar")
        n %= 1_000_000_000
    if n >= 1_000_000:
        m = n // 1_000_000
        parts.append("milioi bat" if m == 1 else f"{_int_to_eu(m)} milioi")
        n %= 1_000_000
    if n >= 1000:
        t = n // 1000
        parts.append("mila" if t == 1 else f"{_int_to_eu(t)} mila")
        n %= 1000
    if n >= 100:
        parts.append(_HUNDREDS[(n // 100) * 100])
        n %= 100
    if n > 0:
        parts.append(_sub100(n))
    # "eta" (and) connects the final non-empty group to the preceding ones.
    #   101  -> ehun eta bat
    #   122  -> ehun eta hogeita bi
    #   1100 -> mila eta ehun
    #   1001 -> mila eta bat
    # TODO: confirm whether 1122 is "mila ehun eta hogeita bi" (single eta,
    #       as implemented) vs "mila eta ehun eta hogeita bi" with a native
    #       speaker. The single-eta rule is the simplest consistent with all
    #       attested examples.
    if len(parts) == 1:
        return parts[0]
    return " ".join(parts[:-1]) + " eta " + parts[-1]


def number_to_basque(n: int) -> str:
    """Public integer -> Basque words, including negatives."""
    if n < 0:
        return "minus " + _int_to_eu(-n)
    return _int_to_eu(n)


def _digits_to_eu(s: str) -> str:
    """Read a digit string one digit at a time (for decimal fractions)."""
    return " ".join(_ONES[int(d)] for d in s)


# --------------------------------------------------------------------------- #
# Thousands-separator stripping
# --------------------------------------------------------------------------- #
# Basque/European convention: ',' = decimal, '.' = thousands. English: the
# reverse. We use the 3-digit-grouping heuristic to disambiguate: a ',' or '.'
# flanked by digits and followed by exactly 3 digits (then a boundary) is
# treated as a thousands separator and removed. Anything else is left for the
# decimal/integer pass. This handles 1.000, 1.000.000, 1,500 as thousands and
# 3,14 / 3.14 as decimals.

_THOUSANDS_SEP_RE = re.compile(r"(?<=[0-9])([.,])(?=[0-9]{3}(?:\D|$))")


def _strip_thousands_separators(text: str) -> str:
    return _THOUSANDS_SEP_RE.sub("", text)


# --------------------------------------------------------------------------- #
# Symbol / currency / unit expansion
# --------------------------------------------------------------------------- #
# These are rewritten BEFORE number expansion so the surrounding digits get
# verbalized by the number pass.

# Spoken Basque puts "ehuneko" before the number, so rewrite both orders.
# "N%" -> "ehuneko N"  and  "%N" -> "ehuneko N"
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_PERCENT_PREFIX_RE = re.compile(r"%\s*(\d+(?:[.,]\d+)?)")

# "N€" / "N$" / "N£" -> "N euro" / "N dolar" / "N libera"
_CURRENCY_SUFFIX_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([€$£])")
# "€N" / "$N" / "£N" -> "euro N" / "dolar N" / "libera N"
_CURRENCY_PREFIX_RE = re.compile(r"([€$£])\s*(\d+(?:[.,]\d+)?)")
_CURRENCY_WORDS = {"€": "euro", "$": "dolar", "£": "libera"}

# "N°" -> "N gradu"
_DEGREE_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*°")

# Misc standalone symbols with an unambiguous Basque reading. Deliberately
# small: '+', '/', '@', '#' are context-dependent and left alone (see TODO).
_SYMBOL_WORDS = {
    "&": "eta",
}


def _expand_symbols(text: str) -> str:
    text = _PERCENT_RE.sub(r"ehuneko \1", text)
    text = _PERCENT_PREFIX_RE.sub(r"ehuneko \1", text)
    text = _CURRENCY_SUFFIX_RE.sub(
        lambda m: f"{m.group(1)} {_CURRENCY_WORDS[m.group(2)]}", text
    )
    text = _CURRENCY_PREFIX_RE.sub(
        lambda m: f"{_CURRENCY_WORDS[m.group(1)]} {m.group(2)}", text
    )
    text = _DEGREE_RE.sub(r"\1 gradu", text)
    for sym, word in _SYMBOL_WORDS.items():
        text = text.replace(sym, f" {word} ")
    return text


# --------------------------------------------------------------------------- #
# Ordinals (opt-in, because "1." is ambiguous with a sentence-final number)
# --------------------------------------------------------------------------- #
# Basque ordinals: 1st=lehena, 2nd=bigarrena, ... -garrena suffix. We only
# rewrite N. when followed by whitespace/end, and only when explicitly enabled.

_ORDINAL_MAP = {
    "1": "lehena",
    "2": "bigarrena",
    "3": "hirugarrena",
    "4": "laugarrena",
    "5": "bostgarrena",
    "6": "seigarrena",
    "7": "zazpigarrena",
    "8": "zortzigarrena",
    "9": "bederatzigarrena",
    "10": "hamargarrena",
}
_ORDINAL_RE = re.compile(r"(?<![\w.])(\d+)\.(?=\s|$)")


def _expand_ordinals(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        return _ORDINAL_MAP.get(m.group(1)) or (
            number_to_basque(int(m.group(1))) + "garrena"
        )

    return _ORDINAL_RE.sub(repl, text)


# --------------------------------------------------------------------------- #
# Number expansion (integers + decimals)
# --------------------------------------------------------------------------- #
# Matches a run of digits optionally containing one ',' or '.' decimal mark.
# Surrounding Basque suffix letters (e.g. the "-an" in "2026an") are preserved
# because they are not part of the match.

_NUMBER_RUN_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _expand_one_number(match: re.Match[str]) -> str:
    raw = match.group(0)
    if "," in raw or "." in raw:
        sep = "," if "," in raw else "."
        int_part, frac_part = raw.split(sep, 1)
        int_words = _int_to_eu(int(int_part)) if int_part else ""
        frac_words = _digits_to_eu(frac_part)
        return f"{int_words} koma {frac_words}".strip()
    return _int_to_eu(int(raw))


def _expand_numbers(text: str) -> str:
    return _NUMBER_RUN_RE.sub(_expand_one_number, text)


# --------------------------------------------------------------------------- #
# Whitespace
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")


def _clean_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def normalize_text(
    text: str,
    *,
    lowercase: bool = False,
    expand_ordinals: bool = False,
) -> str:
    """Normalize Basque text for an Audio8_TTS manifest ``text`` field.

    Pipeline (order matters):

      1. strip thousands separators (1.000 -> 1000) via 3-digit grouping
      2. expand %, currency, degree, '&'
      3. expand ordinals (1. -> lehena)  [opt-in, default off]
      4. expand decimal + integer digit runs to Basque words
         (preserves attached Basque suffixes, e.g. "2026an" -> "...seian")
      5. collapse whitespace
      6. optional lowercase

    Intentionally NOT done here:
      - G2P / phonemization  (Audio8_TTS is raw-BPE, no phoneme layer)
      - abbreviation expansion beyond the small symbol table
      - sentence splitting / punctuation removal (prosody lives in punctuation)
      - '+' '/' '@' '#' verbalization (context-dependent; see TODOs)

    Add an abbreviation table if your data needs it.
    """
    if text is None:
        return ""
    text = str(text)
    text = _strip_thousands_separators(text)
    text = _expand_symbols(text)
    if expand_ordinals:
        text = _expand_ordinals(text)
    text = _expand_numbers(text)
    text = _clean_whitespace(text)
    if lowercase:
        text = text.lower()
    return text
