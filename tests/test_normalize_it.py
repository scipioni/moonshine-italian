"""Italian normalization fixtures: accents, apostrophes, numbers."""

import pytest

from moonshine_it.normalize_it import _int_to_italian, expand_numbers, normalize_text


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (0, "zero"),
        (1, "uno"),
        (7, "sette"),
        (13, "tredici"),
        (18, "diciotto"),
        (21, "ventuno"),
        (28, "ventotto"),
        (31, "trentuno"),
        (42, "quarantadue"),
        (58, "cinquantotto"),
        (69, "sessantanove"),
        (88, "ottantotto"),
        (99, "novantanove"),
        (100, "cento"),
        (101, "centuno"),
        (110, "centodieci"),
        (234, "duecentotrentaquattro"),
        (500, "cinquecento"),
        (999, "novecentonovantanove"),
        (1000, "mille"),
        (1001, "milleuno"),
        (2100, "duemilacento"),
        (34000, "trentaquattromila"),
        (100000, "centomila"),
        (250000, "duecentocinquantamila"),
    ],
)
def test_int_to_italian(n, expected):
    assert _int_to_italian(n) == expected


def test_decimal_number():
    assert expand_numbers("3,14") == "tre virgola uno quattro"
    assert expand_numbers("10.5") == "dieci virgola cinque"


def test_accents_preserved_and_lowercase():
    text = "Perché non andiamo à Roma? È bellìssima!"
    out = normalize_text(text)
    assert out == "perché non andiamo à roma è bellìssima"


def test_apostrophes_preserved():
    text = "L\u2019acqua dell\u2019amico, un po\u2019 di più"
    out = normalize_text(text)
    assert out == "l'acqua dell'amico un po' di più"


def test_numbers_expanded():
    out = normalize_text("Ho 21 gatti e 3 cani")
    assert out == "ho ventuno gatti e tre cani"


def test_numbers_not_expanded_when_disabled():
    out = normalize_text("Ho 21 gatti", expand_nums=False)
    assert out == "ho 21 gatti"


def test_case_preserved_when_lowercase_false():
    out = normalize_text("Perché É Così", lowercase=False)
    assert out == "Perché É Così"


def test_whitespace_collapsed():
    assert normalize_text("  ciao   mondo  ") == "ciao mondo"


def test_orphan_apostrophe_removed():
    assert normalize_text("ciao ' mondo") == "ciao mondo"
