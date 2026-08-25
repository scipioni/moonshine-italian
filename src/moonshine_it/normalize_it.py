"""Italian transcript normalization for ASR targets.

Accents are always preserved. Options: lowercasing and number-to-words
expansion (both configured in config.yaml under
preparation.transcript_normalization).
"""

from __future__ import annotations

import re

_ONES = [
    "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto",
    "nove", "dieci", "undici", "dodici", "tredici", "quattordici", "quindici",
    "sedici", "diciassette", "diciotto", "diciannove",
]
_TENS = {
    2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta", 6: "sessanta",
    7: "settanta", 8: "ottanta", 9: "novanta",
}
_HUNDREDS = ["", "cento", "duecento", "trecento", "quattrocento", "cinquecento",
             "seicento", "settecento", "ottocento", "novecento"]

_APOSTROPHE_MAP = {
    "\u2019": "'",  # right single quotation mark
    "\u02bc": "'",  # modifier letter apostrophe
    "`": "'",
    "\u2018": "'",
}

_TRATED = re.compile(r"\b(vent|trent|quarant|cinquant|sessant|settant|ottant|novant)i?(uen|otto)\b")
_ELISION = re.compile(r"[aeiou]'")  # keep vowel+' elisions (dell'acqua)


def _under_thousand(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        if o == 1 or o == 8:  # ventuno, ventotto (final-vowel truncation)
            return _TENS[t][:-1] + _ONES[o]
        if o == 0:
            return _TENS[t]
        return _TENS[t] + _ONES[o]
    h, rest = divmod(n, 100)
    hundreds = _HUNDREDS[h]
    if rest == 0:
        return hundreds
    # cento + uno/otto/ottanta.. → centuno, centotto, centottanta
    if rest == 1 or rest == 8 or 80 <= rest <= 89:
        hundreds = hundreds[:-1]
    return hundreds + _under_thousand(rest)


def _int_to_italian(n: int) -> str:
    if n < 0:
        return "meno " + _int_to_italian(-n)
    if n < 1000:
        return _under_thousand(n)
    thousands, rest = divmod(n, 1000)
    if thousands == 1:
        head = "mille"
    else:
        head = _under_thousand(thousands) + "mila"
    if rest == 0:
        return head
    return head + _under_thousand(rest)


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def expand_numbers(text: str) -> str:
    """Replace numerals with Italian words (integers and decimals)."""

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        if "," in raw or "." in raw:
            int_part, _, frac = raw.partition("," if "," in raw else ".")
            words = _int_to_italian(int(int_part or 0))
            if frac:
                digits = " ".join(_ONES[int(d)] for d in frac if d.isdigit())
                words += " virgola " + digits
            return words
        return _int_to_italian(int(raw))

    return _NUMBER_RE.sub(repl, text)


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
    expand_nums: bool = True,
) -> str:
    """Normalize an Italian transcript into an ASR target string.

    - maps typographic apostrophes to "'"
    - expands numbers to words (optional)
    - keeps accents and vowel elisions, drops other punctuation
    - collapses whitespace
    """
    for src, dst in _APOSTROPHE_MAP.items():
        text = text.replace(src, dst)
    if expand_nums:
        text = expand_numbers(text)
    # Drop punctuation except apostrophes (elision dell'acqua / truncation po').
    text = re.sub(r"[^\w\s'àèéìòùÀÈÉÌÒÙ]", " ", text)
    if lowercase:
        text = text.lower()
    # Apostrophes must attach to a letter on the left (l'acqua, dell'amico,
    # po', un'); anything else is a stray quote and gets dropped.
    text = re.sub(r"(?<!\w)'", "", text)
    # Re-join a spurious space after known elisions only: "l' acqua" ->
    # "l'acqua". Real truncations like "po' di" keep their word boundary.
    elisions = (
        "l|un|all|dall|dell|nell|sull|quest|quell|bell|quel|pur|senz|finor|ogn"
    )
    text = re.sub(rf"\b({elisions})' +", r"\1'", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
