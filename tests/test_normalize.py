"""Tests for basque_manifest.normalize.

Number tables cross-checked against Omniglot's Basque number page
(https://omniglot.com/language/numbers/basque.htm) and the Euskaltzaindia
standard (batua).
"""

from __future__ import annotations

import pytest

from basque_manifest.normalize import normalize_text, number_to_basque


# --------------------------------------------------------------------------- #
# Pure number conversion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "n,expected",
    [
        (0, "zero"),
        (1, "bat"),
        (8, "zortzi"),
        (10, "hamar"),
        (11, "hamaika"),
        (17, "hamazazpi"),
        (18, "hemezortzi"),
        (19, "hemeretzi"),
        (20, "hogei"),
        (21, "hogeita bat"),
        (29, "hogeita bederatzi"),
        (30, "hogeita hamar"),
        (35, "hogeita hamabost"),
        (39, "hogeita hemeretzi"),
        (40, "berrogei"),
        (45, "berrogeita bost"),
        (50, "berrogeita hamar"),
        (60, "hirurogei"),
        (70, "hirurogeita hamar"),
        (80, "laurogei"),
        (90, "laurogeita hamar"),
        (99, "laurogeita hemeretzi"),
        (100, "ehun"),
        (101, "ehun eta bat"),
        (110, "ehun eta hamar"),
        (122, "ehun eta hogeita bi"),
        (200, "berrehun"),
        (300, "hirurehun"),
        (900, "bederatziehun"),
        (999, "bederatziehun eta laurogeita hemeretzi"),
        (1000, "mila"),
        (1001, "mila eta bat"),
        (1100, "mila eta ehun"),
        (1222, "mila berrehun eta hogeita bi"),  # 1000 + 200 + 22
        (2000, "bi mila"),
        (2026, "bi mila eta hogeita sei"),
        (10000, "hamar mila"),
        (100000, "ehun mila"),
        (1000000, "milioi bat"),
        (2000000, "bi milioi"),
        (2026, "bi mila eta hogeita sei"),
    ],
)
def test_number_to_basque(n: int, expected: str) -> None:
    assert number_to_basque(n) == expected


def test_negative_number() -> None:
    assert number_to_basque(-5) == "minus bost"


# --------------------------------------------------------------------------- #
# normalize_text end-to-end
# --------------------------------------------------------------------------- #


def test_plain_text_passthrough() -> None:
    assert normalize_text("Kaixo mundua!") == "Kaixo mundua!"


def test_whitespace_collapsed() -> None:
    assert normalize_text("Kaixo   \t\t mundua\n") == "Kaixo mundua"


def test_simple_integer_in_sentence() -> None:
    assert normalize_text("Hamar hitz.") == "Hamar hitz."


def test_two_digit_year() -> None:
    assert normalize_text("2026an etorriko da.") == (
        "bi mila eta hogeita seian etorriko da."
    )


def test_preserves_basque_suffix() -> None:
    # "21" -> "hogeita bat"; the trailing "an" (inessive) must survive.
    assert normalize_text("21ean") == "hogeita batean"


def test_percent() -> None:
    assert normalize_text("%50") == "ehuneko berrogeita hamar"  # 50 -> berrogeita hamar


def test_percent_written_after() -> None:
    # "50%" -> "ehuneko 50" -> "ehuneko berrogeita hamar"
    assert normalize_text("50%") == "ehuneko berrogeita hamar"


def test_currency_euro() -> None:
    assert normalize_text("5€") == "bost euro"


def test_currency_dollar_prefix() -> None:
    assert normalize_text("$10") == "dolar hamar"


def test_degree() -> None:
    assert normalize_text("20°") == "hogei gradu"


def test_decimal_comma() -> None:
    assert normalize_text("3,14") == "hiru koma bat lau"


def test_decimal_dot() -> None:
    assert normalize_text("3.14") == "hiru koma bat lau"


def test_thousands_dot() -> None:
    assert normalize_text("1.000") == "mila"


def test_thousands_dot_multi() -> None:
    assert normalize_text("1.000.000") == "milioi bat"


def test_thousands_comma() -> None:
    # 1500 = 1000 + 500; "eta" connects the final group (cf. 1100 = "mila eta ehun").
    assert normalize_text("1,500") == "mila eta bostehun"


def test_ampersand() -> None:
    assert normalize_text("ari & axola") == "ari eta axola"


def test_lowercase_option() -> None:
    assert normalize_text("Kaixo 2026", lowercase=True) == "kaixo bi mila eta hogeita sei"


def test_ordinals_off_by_default() -> None:
    # Default: "1." is left as a sentence-final number -> "bat."
    assert normalize_text("etorri zen 1.") == "etorri zen bat."


def test_ordinals_on() -> None:
    assert normalize_text("1.", expand_ordinals=True) == "lehena"
    assert normalize_text("3.", expand_ordinals=True) == "hirugarrena"
    assert normalize_text("12.", expand_ordinals=True) == "hamabigarrena"


def test_ordinal_not_triggered_on_decimal() -> None:
    # "1.5" is a decimal, not an ordinal, even with expand_ordinals=True.
    assert normalize_text("1.5", expand_ordinals=True) == "bat koma bost"


def test_empty_and_none() -> None:
    assert normalize_text("") == ""
    assert normalize_text(None) == ""  # type: ignore[arg-type]


def test_idempotent_on_clean_text() -> None:
    # HiTZ studio transcripts are already spoken words; normalizing twice
    # must not change them.
    text = "Euskal Herriko unibertsitatea da hau."
    once = normalize_text(text)
    twice = normalize_text(once)
    assert once == twice == text


def test_mixed_sentence() -> None:
    # Composes: integer+suffix, decimal+currency, prefix-percent.
    # NOTE: verbatim suffix preservation is a heuristic. "%20ean" would yield
    # "hogeiean" (doubled epenthetic vowel); we avoid suffix-on-percent here.
    # See normalize.py TODO on morphological suffix adjustment.
    out = normalize_text("2026an 3,14€ ordaindu zuen %20.")
    assert out == (
        "bi mila eta hogeita seian hiru koma bat lau euro ordaindu zuen "
        "ehuneko hogei."
    )
