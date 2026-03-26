from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .convert import build_hiragana_converter, build_kakasi_converter

SPECIAL_REPLACEMENTS = {
    "goodbye": "ぐっばい",
}

SMALL_KANA_MAP = str.maketrans(
    {
        "ぁ": "あ",
        "ぃ": "い",
        "ぅ": "う",
        "ぇ": "え",
        "ぉ": "お",
        "ゃ": "や",
        "ゅ": "ゆ",
        "ょ": "よ",
        "っ": "つ",
        "ゎ": "わ",
    }
)

VOICED_BASE_MAP = str.maketrans(
    {
        "が": "か",
        "ぎ": "き",
        "ぐ": "く",
        "げ": "け",
        "ご": "こ",
        "ざ": "さ",
        "じ": "し",
        "ず": "す",
        "ぜ": "せ",
        "ぞ": "そ",
        "だ": "た",
        "ぢ": "ち",
        "づ": "つ",
        "で": "て",
        "ど": "と",
        "ば": "は",
        "び": "ひ",
        "ぶ": "ふ",
        "べ": "へ",
        "ぼ": "ほ",
        "ぱ": "は",
        "ぴ": "ひ",
        "ぷ": "ふ",
        "ぺ": "へ",
        "ぽ": "ほ",
        "ゔ": "う",
    }
)

ROW_GROUPS = [
    "あいうえおわ",
    "かきくけこ",
    "さしすせそ",
    "たちつてと",
    "なにぬねの",
    "はひふへほ",
    "まみむめも",
    "やゆよ",
    "らりるれろ",
]

VOWEL_GROUPS = {
    "あかさたなはまやらわ": "a",
    "いきしちにひみり": "i",
    "うくすつぬふむゆる": "u",
    "えけせてねへめれ": "e",
    "おこそとのほもよろを": "o",
}

ROW_INDEX = {char: idx for idx, chars in enumerate(ROW_GROUPS) for char in chars}
VOWEL_INDEX = {char: vowel for chars, vowel in VOWEL_GROUPS.items() for char in chars}


@dataclass(frozen=True)
class AsrChar:
    char: str
    start: float
    end: float
    confidence: float
    seg_id: int
    word_index: int
    word_text: str


@dataclass(frozen=True)
class LyricWord:
    line_id: int
    word_id: int
    text: str
    normalized: str
    ruby_text: str


@dataclass(frozen=True)
class LyricLine:
    line_id: int
    text: str
    normalized: str
    words: list[LyricWord]


@dataclass(frozen=True)
class LyricChar:
    char: str
    line_id: int
    word_id: int
    line_char_index: int
    word_char_index: int


def normalize_for_alignment(text: str, to_hiragana) -> str:
    normalized = to_hiragana(text).lower().replace("\u3000", " ")
    for source, target in SPECIAL_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)

    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if char.isspace():
            continue
        if category.startswith("P"):
            continue
        chars.append(char)
    return "".join(chars)


def split_display_words(text: str, to_hiragana, kakasi) -> list[str]:
    explicit_parts = [part for part in re.split(r"[ \u3000]+", text.strip()) if part]
    if len(explicit_parts) > 1:
        return explicit_parts

    auto_parts = []
    for item in kakasi.convert(text.strip()):
        raw_part = str(item.get("orig", ""))
        if normalize_for_alignment(raw_part, to_hiragana):
            auto_parts.append(raw_part)

    if auto_parts:
        return auto_parts
    return explicit_parts


def has_kanji(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def build_ruby_text(text: str, kakasi) -> str:
    if not has_kanji(text):
        return ""

    ruby_parts = []
    for item in kakasi.convert(text):
        hira = str(item.get("hira", "")).replace("\u3000", " ").strip()
        if hira:
            ruby_parts.append(hira)
    return "".join(ruby_parts)


def canonical_kana(char: str) -> str:
    return char.translate(SMALL_KANA_MAP).translate(VOICED_BASE_MAP)


def substitution_cost(asr_char: str, lyric_char: str) -> float:
    if asr_char == lyric_char:
        return 0.0

    asr_base = canonical_kana(asr_char)
    lyric_base = canonical_kana(lyric_char)

    if asr_base == lyric_base:
        return 0.25

    if ROW_INDEX.get(asr_base) == ROW_INDEX.get(lyric_base):
        return 0.55

    if VOWEL_INDEX.get(asr_base) == VOWEL_INDEX.get(lyric_base):
        return 0.75

    return 1.0


def deletion_cost(char: AsrChar) -> float:
    return 0.3 + 0.7 * max(0.0, min(1.0, char.confidence))


def insertion_cost(_: LyricChar) -> float:
    return 0.9


def load_lyrics_lines(path: Path) -> tuple[list[LyricLine], list[LyricChar]]:
    to_hiragana = build_hiragana_converter()
    kakasi = build_kakasi_converter()
    lines: list[LyricLine] = []
    lyric_chars: list[LyricChar] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue

        line_id = len(lines)
        words: list[LyricWord] = []
        line_char_index = 0

        for word_id, word_text in enumerate(split_display_words(raw_line, to_hiragana, kakasi)):
            normalized_word = normalize_for_alignment(word_text, to_hiragana)
            words.append(
                LyricWord(
                    line_id=line_id,
                    word_id=word_id,
                    text=word_text,
                    normalized=normalized_word,
                    ruby_text=build_ruby_text(word_text, kakasi),
                )
            )
            for word_char_index, char in enumerate(normalized_word):
                lyric_chars.append(
                    LyricChar(
                        char=char,
                        line_id=line_id,
                        word_id=word_id,
                        line_char_index=line_char_index,
                        word_char_index=word_char_index,
                    )
                )
                line_char_index += 1

        normalized_line = "".join(word.normalized for word in words)
        lines.append(
            LyricLine(
                line_id=line_id,
                text=raw_line,
                normalized=normalized_line,
                words=words,
            )
        )

    return lines, lyric_chars


def load_asr_chars(path: Path) -> list[AsrChar]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Input JSON must contain a 'segments' list.")

    to_hiragana = build_hiragana_converter()
    asr_chars: list[AsrChar] = []
    word_index = 0

    for seg_id, segment in enumerate(segments):
        words = segment.get("words", [])
        if not isinstance(words, list):
            continue

        for word in words:
            raw_text = str(word.get("text", ""))
            normalized = normalize_for_alignment(raw_text, to_hiragana)
            if not normalized:
                word_index += 1
                continue

            start = float(word.get("start", 0.0))
            end = float(word.get("end", start))
            confidence = float(word.get("confidence", 0.0))
            duration = max(0.0, end - start)
            step = duration / len(normalized) if normalized else 0.0

            for char_offset, char in enumerate(normalized):
                char_start = start + step * char_offset
                char_end = end if char_offset == len(normalized) - 1 else start + step * (char_offset + 1)
                asr_chars.append(
                    AsrChar(
                        char=char,
                        start=char_start,
                        end=char_end,
                        confidence=confidence,
                        seg_id=seg_id,
                        word_index=word_index,
                        word_text=raw_text,
                    )
                )
            word_index += 1

    return asr_chars


def align_characters(asr_chars: list[AsrChar], lyric_chars: list[LyricChar]) -> list[int | None]:
    m = len(asr_chars)
    n = len(lyric_chars)

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + deletion_cost(asr_chars[i - 1])
        back[i][0] = "del"

    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + insertion_cost(lyric_chars[j - 1])
        back[0][j] = "ins"

    for i in range(1, m + 1):
        asr_char = asr_chars[i - 1]
        for j in range(1, n + 1):
            lyric_char = lyric_chars[j - 1]

            sub_cost = dp[i - 1][j - 1] + substitution_cost(asr_char.char, lyric_char.char)
            del_cost = dp[i - 1][j] + deletion_cost(asr_char)
            ins_cost = dp[i][j - 1] + insertion_cost(lyric_char)

            best_cost = sub_cost
            best_op = "sub"
            if del_cost < best_cost:
                best_cost = del_cost
                best_op = "del"
            if ins_cost < best_cost:
                best_cost = ins_cost
                best_op = "ins"

            dp[i][j] = best_cost
            back[i][j] = best_op

    lyric_to_asr: list[int | None] = [None] * n
    i = m
    j = n
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "sub":
            char_cost = substitution_cost(asr_chars[i - 1].char, lyric_chars[j - 1].char)
            if char_cost < 1.0:
                lyric_to_asr[j - 1] = i - 1
            i -= 1
            j -= 1
        elif op == "del":
            i -= 1
        elif op == "ins":
            j -= 1
        else:
            break

    return lyric_to_asr


def _aggregate_indices(indices: list[int], asr_chars: list[AsrChar]) -> dict[str, Any]:
    matched_chars = len(indices)
    if not indices:
        return {
            "start": None,
            "end": None,
            "segment_ids": [],
            "matched_chars": 0,
        }

    starts = [asr_chars[index].start for index in indices]
    ends = [asr_chars[index].end for index in indices]
    segment_ids = sorted({asr_chars[index].seg_id for index in indices})
    return {
        "start": min(starts),
        "end": max(ends),
        "segment_ids": segment_ids,
        "matched_chars": matched_chars,
    }


def _collect_matches(
    lines: list[LyricLine],
    lyric_chars: list[LyricChar],
    lyric_to_asr: list[int | None],
) -> tuple[dict[int, list[int]], dict[tuple[int, int], list[int]]]:
    line_matches = {line.line_id: [] for line in lines}
    word_matches = {
        (line.line_id, word.word_id): []
        for line in lines
        for word in line.words
    }

    for lyric_index, asr_index in enumerate(lyric_to_asr):
        if asr_index is None:
            continue
        lyric_char = lyric_chars[lyric_index]
        line_matches[lyric_char.line_id].append(asr_index)
        word_matches[(lyric_char.line_id, lyric_char.word_id)].append(asr_index)

    return line_matches, word_matches


def build_word_level_payload(json_path: str | Path, lyrics_path: str | Path) -> dict[str, Any]:
    json_source = Path(json_path)
    lyrics_source = Path(lyrics_path)

    lines, lyric_chars = load_lyrics_lines(lyrics_source)
    asr_chars = load_asr_chars(json_source)
    lyric_to_asr = align_characters(asr_chars, lyric_chars)
    line_matches, word_matches = _collect_matches(lines, lyric_chars, lyric_to_asr)

    output_lines = []
    flat_words = []
    global_word_id = 0

    for line in lines:
        line_stats = _aggregate_indices(line_matches[line.line_id], asr_chars)
        total_chars = len(line.normalized)

        output_words = []
        for word in line.words:
            word_key = (line.line_id, word.word_id)
            word_stats = _aggregate_indices(word_matches[word_key], asr_chars)
            word_payload = {
                "global_word_id": global_word_id,
                "line_id": line.line_id,
                "word_id": word.word_id,
                "text": word.text,
                "normalized_text": word.normalized,
                "ruby_text": word.ruby_text,
                "start": word_stats["start"],
                "end": word_stats["end"],
                "matched_chars": word_stats["matched_chars"],
                "total_chars": len(word.normalized),
                "coverage": word_stats["matched_chars"] / len(word.normalized) if word.normalized else 0.0,
                "segment_ids": word_stats["segment_ids"],
            }
            output_words.append(word_payload)
            flat_words.append(word_payload)
            global_word_id += 1

        output_lines.append(
            {
                "line_id": line.line_id,
                "text": line.text,
                "normalized_text": line.normalized,
                "start": line_stats["start"],
                "end": line_stats["end"],
                "matched_chars": line_stats["matched_chars"],
                "total_chars": total_chars,
                "coverage": line_stats["matched_chars"] / total_chars if total_chars else 0.0,
                "segment_ids": line_stats["segment_ids"],
                "words": output_words,
            }
        )

    matched_char_count = sum(1 for item in lyric_to_asr if item is not None)
    return {
        "version": 1,
        "source": {
            "asr_json": str(json_source),
            "lyrics_txt": str(lyrics_source),
        },
        "alignment": {
            "asr_char_count": len(asr_chars),
            "lyric_char_count": len(lyric_chars),
            "matched_char_count": matched_char_count,
            "coverage": matched_char_count / len(lyric_chars) if lyric_chars else 0.0,
            "line_count": len(output_lines),
            "word_count": len(flat_words),
        },
        "lines": output_lines,
        "words": flat_words,
    }


def build_line_level_payload(json_path: str | Path, lyrics_path: str | Path) -> dict[str, Any]:
    word_payload = build_word_level_payload(json_path, lyrics_path)
    line_entries = []
    for line in word_payload["lines"]:
        line_entries.append(
            {
                "line_id": line["line_id"],
                "text": line["text"],
                "normalized_text": line["normalized_text"],
                "start": line["start"],
                "end": line["end"],
                "matched_chars": line["matched_chars"],
                "total_chars": line["total_chars"],
                "coverage": line["coverage"],
                "segment_ids": line["segment_ids"],
            }
        )

    alignment = word_payload["alignment"]
    source = word_payload["source"]
    return {
        "json_path": source["asr_json"],
        "lyrics_path": source["lyrics_txt"],
        "asr_char_count": alignment["asr_char_count"],
        "lyric_char_count": alignment["lyric_char_count"],
        "lyrics": line_entries,
    }


def align_lyrics_to_asr(json_path: str | Path, lyrics_path: str | Path) -> dict[str, Any]:
    return build_word_level_payload(json_path, lyrics_path)
