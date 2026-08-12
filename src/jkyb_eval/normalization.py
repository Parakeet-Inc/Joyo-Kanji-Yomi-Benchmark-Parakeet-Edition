"""Canonical comparison forms for kana and ordinary Japanese text."""

from __future__ import annotations

import unicodedata

import jaconv

_SIMPLE_SUBSTITUTIONS = {
    "ヂ": "ジ",
    "ヅ": "ズ",
    "ヲ": "オ",
    "ヰ": "イ",
    "ヱ": "エ",
}

_KANA_VOWELS: dict[str, str] = {}
for _characters, _vowel in (
    ("アカサタナハマヤラワガザダバパァャヮ", "ア"),
    ("イキシチニヒミリギジヂビピィ", "イ"),
    ("ウクスツヌフムユルグズヅブプゥュヴ", "ウ"),
    ("エケセテネヘメレゲゼデベペェ", "エ"),
    ("オコソトノホモヨロヲゴゾドボポォョ", "オ"),
):
    for _character in _characters:
        _KANA_VOWELS[_character] = _vowel


def _is_hiragana(character: str) -> bool:
    return "\u3040" <= character <= "\u309f"


def _is_separator(character: str) -> bool:
    return unicodedata.category(character)[0] in {"P", "S", "Z"} or character.isspace()


def canonicalize_yomi(value: str, *, preserve_tags: bool = False) -> str:
    """Normalize comparison-only kana conventions and remove punctuation."""
    result: list[str] = []
    previous = "ア"
    for raw_character in unicodedata.normalize("NFKC", value):
        if preserve_tags and raw_character in {"<", ">"}:
            result.append(raw_character)
            continue
        if _is_separator(raw_character):
            continue

        character = raw_character
        if _is_hiragana(character):
            character = jaconv.hira2kata(character)
        character = _SIMPLE_SUBSTITUTIONS.get(character, character)

        if character == "ー":
            expanded = _KANA_VOWELS.get(previous, previous)
            result.append(expanded)
            previous = expanded
            continue

        previous_vowel = _KANA_VOWELS.get(previous)
        if character == "イ" and previous_vowel == "エ":
            result.append("エ")
            previous = "エ"
            continue
        if character == "ウ" and previous_vowel == "オ":
            result.append("オ")
            previous = "オ"
            continue

        result.append(character)
        previous = character
    return "".join(result)


def canonicalize_text(value: str) -> str:
    """Normalize text for ordinary Whisper CER."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value)
        if not _is_separator(character)
    )
