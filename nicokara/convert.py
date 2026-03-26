from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import Any

import pykakasi


def build_kakasi_converter():
    return pykakasi.kakasi()


def build_hiragana_converter():
    kakasi = build_kakasi_converter()

    def to_hiragana(text: str) -> str:
        converted = "".join(item["hira"] for item in kakasi.convert(text))
        return converted.replace("\u3000", " ")

    return to_hiragana


def strip_punctuation(text: str) -> str:
    return "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )


def convert_payload_to_hiragana(payload: dict[str, Any]) -> dict[str, Any]:
    to_hiragana = build_hiragana_converter()
    converted = deepcopy(payload)

    if isinstance(converted.get("text"), str):
        converted["text"] = to_hiragana(converted["text"])

    for segment in converted.get("segments", []):
        if isinstance(segment.get("text"), str):
            segment["text"] = to_hiragana(segment["text"])

        for word in segment.get("words", []):
            if isinstance(word.get("text"), str):
                word["text"] = to_hiragana(word["text"])

    return converted


def convert_text_to_hiragana(
    text: str,
    *,
    remove_blank_lines: bool = True,
    remove_punctuation: bool = False,
) -> str:
    to_hiragana = build_hiragana_converter()
    converted_lines = []
    for line in re.split(r"\r\n|\r|\n", text):
        if remove_blank_lines and not line.strip():
            continue
        converted_line = to_hiragana(line)
        if remove_punctuation:
            converted_line = strip_punctuation(converted_line)
        converted_lines.append(converted_line)
    return "\n".join(converted_lines) + ("\n" if converted_lines else "")
