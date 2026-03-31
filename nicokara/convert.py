from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
import urllib.request
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol


JMDICT_RELEASE_API_URL = "https://api.github.com/repos/Doublevil/JmdictFurigana/releases/latest"
JMDICT_ASSETS = {
    "JmdictFurigana.txt": "jmdict",
    "JmnedictFurigana.txt": "jmnedict",
}
CACHE_DIR = Path.home() / ".cache" / "nicokara"
DEFAULT_FURIGANA_DB_PATH = CACHE_DIR / "jmdict_furigana.sqlite3"
KANJI_LIKE_MARKS = {"々", "〆", "ヶ", "ヵ"}
SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょゎゕゖゔっ")
LEADING_DISCOURAGED_KANA = SMALL_KANA | {"ー"}


def build_kakasi_converter():
    try:
        import pykakasi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pykakasi is not installed. Install it with `pip install pykakasi`."
        ) from exc
    return pykakasi.kakasi()


def strip_punctuation(text: str) -> str:
    return "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )


def has_kanji(text: str) -> bool:
    return any(_is_kanji_like(char) for char in text)


def katakana_to_hiragana(text: str) -> str:
    chars = []
    for char in text:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def _is_kanji_like(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff" or char in KANJI_LIKE_MARKS


@dataclass(frozen=True)
class ReadingToken:
    surface: str
    ruby: str
    pronunciation: str


@dataclass(frozen=True)
class RubyPart:
    ruby: str
    rt: str | None = None


@dataclass(frozen=True)
class FuriganaEntry:
    reading: str
    parts: list[RubyPart]


@dataclass(frozen=True)
class WordReading:
    ruby_text: str
    pronunciation_text: str
    source: str
    ruby_parts: list[RubyPart]


@dataclass(frozen=True)
class JapaneseTextProcessorConfig:
    backend: str = "auto"
    split_mode: str = "C"
    furigana_resource_path: str | Path | None = None
    reading_overrides_path: str | Path | None = None


class ReadingBackend(Protocol):
    name: str

    def tokenize(self, text: str) -> list[ReadingToken]:
        ...


class KakasiBackend:
    name = "pykakasi"

    def __init__(self) -> None:
        self._kakasi = build_kakasi_converter()

    def tokenize(self, text: str) -> list[ReadingToken]:
        tokens = []
        for item in self._kakasi.convert(text):
            surface = str(item.get("orig", ""))
            reading = str(item.get("hira", "")).replace("\u3000", " ").strip()
            if not surface:
                continue
            if not reading:
                reading = katakana_to_hiragana(surface)
            tokens.append(
                ReadingToken(
                    surface=surface,
                    ruby=reading,
                    pronunciation=reading,
                )
            )
        return tokens


class FugashiBackend:
    name = "fugashi"

    def __init__(self) -> None:
        try:
            from fugashi import Tagger
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "fugashi is not installed. Install it with `pip install 'fugashi[unidic-lite]'`."
            ) from exc
        self._tagger = Tagger()

    def tokenize(self, text: str) -> list[ReadingToken]:
        tokens = []
        for token in self._tagger(text):
            surface = token.surface
            if not surface or surface.isspace():
                continue

            feature = token.feature
            ruby_kana = (
                getattr(feature, "kana", None)
                or getattr(feature, "pron", None)
                or getattr(feature, "kanaBase", None)
                or getattr(feature, "pronBase", None)
                or surface
            )
            pronunciation_kana = (
                getattr(feature, "pron", None)
                or getattr(feature, "kana", None)
                or getattr(feature, "pronBase", None)
                or getattr(feature, "kanaBase", None)
                or ruby_kana
            )

            ruby = katakana_to_hiragana(str(ruby_kana))
            pronunciation = katakana_to_hiragana(str(pronunciation_kana))
            if not ruby:
                ruby = katakana_to_hiragana(surface)
            if not pronunciation:
                pronunciation = ruby

            tokens.append(
                ReadingToken(
                    surface=surface,
                    ruby=ruby,
                    pronunciation=pronunciation,
                )
            )
        return tokens


class SudachiBackend:
    name = "sudachipy"

    def __init__(self, *, split_mode: str = "C") -> None:
        try:
            from sudachipy import dictionary, tokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SudachiPy is not installed. Install it with `pip install SudachiPy sudachidict_core`."
            ) from exc

        split_mode_name = str(split_mode or "C").upper()
        try:
            self._split_mode = getattr(tokenizer.Tokenizer.SplitMode, split_mode_name)
        except AttributeError as exc:
            raise ValueError(
                f"Unsupported Sudachi split mode `{split_mode}`. Use A, B, or C."
            ) from exc

        self._tokenizer = dictionary.Dictionary().create()

    def tokenize(self, text: str) -> list[ReadingToken]:
        tokens = []
        for morpheme in self._tokenizer.tokenize(text, self._split_mode):
            surface = morpheme.surface()
            if not surface or surface.isspace():
                continue

            ruby = katakana_to_hiragana(morpheme.reading_form())
            if not ruby:
                ruby = katakana_to_hiragana(surface)

            pronunciation = ruby
            pos = morpheme.part_of_speech()
            if pos and pos[0] == "助詞":
                pronunciation = {
                    "は": "わ",
                    "へ": "え",
                    "を": "お",
                }.get(surface, pronunciation)

            tokens.append(
                ReadingToken(
                    surface=surface,
                    ruby=ruby,
                    pronunciation=pronunciation,
                )
            )
        return tokens


class FuriganaLexicon:
    def __init__(
        self,
        *,
        exact_map: dict[str, FuriganaEntry] | None = None,
        keyed_map: dict[tuple[str, str], FuriganaEntry] | None = None,
    ) -> None:
        self._exact_map = exact_map or {}
        self._keyed_map = keyed_map or {}

    @classmethod
    def load(
        cls,
        path: str | Path | None,
    ) -> FuriganaLexicon | None:
        if path is None:
            return None

        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Furigana resource `{source}` does not exist.")

        text = _read_text_resource(source)
        lower_name = source.name.lower()
        if lower_name.endswith(".json") or lower_name.endswith(".json.gz"):
            payload = json.loads(text)
            exact_map, keyed_map = _parse_json_lexicon(payload)
            return cls(exact_map=exact_map, keyed_map=keyed_map)

        exact_map, keyed_map = _parse_text_lexicon(text)
        return cls(exact_map=exact_map, keyed_map=keyed_map)

    def lookup_entry(self, text: str, reading: str | None = None) -> FuriganaEntry | None:
        if text in self._exact_map:
            return self._exact_map[text]
        if reading is not None:
            return self._keyed_map.get((text, _normalize_reading_key(reading)))
        return None

    def lookup_unique_text(self, text: str) -> FuriganaEntry | None:
        if text in self._exact_map:
            return self._exact_map[text]
        matches = [
            entry
            for (candidate_text, _), entry in self._keyed_map.items()
            if candidate_text == text
        ]
        if len(matches) == 1:
            return matches[0]
        return None


class BuiltinFuriganaLexicon:
    _instance: BuiltinFuriganaLexicon | None = None

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    @classmethod
    def ensure_default(cls) -> BuiltinFuriganaLexicon | None:
        if os.environ.get("NICOKARA_DISABLE_AUTO_FURIGANA_DOWNLOAD") == "1":
            return None
        if cls._instance is not None:
            return cls._instance

        try:
            if not DEFAULT_FURIGANA_DB_PATH.exists():
                _build_builtin_furigana_db(DEFAULT_FURIGANA_DB_PATH)
        except Exception:
            return None

        if DEFAULT_FURIGANA_DB_PATH.exists():
            cls._instance = cls(DEFAULT_FURIGANA_DB_PATH)
        return cls._instance

    def lookup_entry(self, text: str, reading: str) -> FuriganaEntry | None:
        conn = self._ensure_connection()
        row = conn.execute(
            "SELECT reading, furigana_json FROM entries WHERE text = ? AND reading = ? LIMIT 1",
            (text, _normalize_reading_key(reading)),
        ).fetchone()
        if row is None:
            return None
        return FuriganaEntry(
            reading=row[0],
            parts=_deserialize_parts(row[1]),
        )

    def lookup_unique_text(self, text: str) -> FuriganaEntry | None:
        conn = self._ensure_connection()
        rows = conn.execute(
            "SELECT reading, furigana_json FROM entries WHERE text = ? LIMIT 2",
            (text,),
        ).fetchall()
        if len(rows) != 1:
            return None
        return FuriganaEntry(
            reading=rows[0][0],
            parts=_deserialize_parts(rows[0][1]),
        )

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
        return self._conn


class JapaneseTextProcessor:
    def __init__(
        self,
        *,
        backend: ReadingBackend,
        overrides: FuriganaLexicon | None = None,
        resource: FuriganaLexicon | None = None,
        builtin_resource: BuiltinFuriganaLexicon | None = None,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.name
        self._overrides = overrides
        self._resource = resource
        self._builtin_resource = builtin_resource
        self._fallback_reader: KakasiBackend | None = None
        if backend.name != "pykakasi":
            try:
                self._fallback_reader = KakasiBackend()
            except ModuleNotFoundError:
                self._fallback_reader = None
        self.backend_label = self.backend_name
        if self._builtin_resource is not None:
            self.backend_label = f"{self.backend_label}+JmdictFurigana"
        if self._fallback_reader is not None:
            self.backend_label = f"{self.backend_label}+pykakasi"

    def tokenize(self, text: str) -> list[ReadingToken]:
        tokens = []
        for token in self.backend.tokenize(text):
            token = self._refine_token_reading(token)
            entry, source = self._lookup_entry(token.surface, token.ruby)
            ruby = entry.reading if entry is not None else token.ruby
            pronunciation = (
                ruby if source in {"override", "resource", "builtin"} else token.pronunciation
            )
            tokens.append(
                ReadingToken(
                    surface=token.surface,
                    ruby=ruby,
                    pronunciation=pronunciation,
                )
            )
        return tokens

    def split_words(self, text: str) -> list[str]:
        explicit_parts = [part for part in re.split(r"[ \u3000]+", text.strip()) if part]
        if len(explicit_parts) > 1:
            return explicit_parts

        auto_parts = []
        for token in self.tokenize(text.strip()):
            if normalize_for_alignment(token.surface, self.to_alignment_hiragana):
                auto_parts.append(token.surface)

        if auto_parts:
            return auto_parts
        return explicit_parts

    def to_alignment_hiragana(self, text: str) -> str:
        if not text:
            return ""
        return "".join(token.pronunciation for token in self.tokenize(text)).replace(
            "\u3000", " "
        )

    def to_ruby_hiragana(self, text: str) -> str:
        if not text:
            return ""
        return "".join(token.ruby for token in self.tokenize(text)).replace(
            "\u3000", " "
        )

    def resolve_word_reading(self, text: str) -> WordReading:
        if not text:
            return WordReading(
                ruby_text="",
                pronunciation_text="",
                source="none",
                ruby_parts=[],
            )
        if not has_kanji(text):
            return WordReading(
                ruby_text="",
                pronunciation_text="",
                source="none",
                ruby_parts=[],
            )

        ruby_text = self.to_ruby_hiragana(text)
        pronunciation_text = self.to_alignment_hiragana(text)

        for candidate_reading in self._candidate_readings(text, ruby_text):
            direct_entry, direct_source = self._lookup_entry(text, candidate_reading)
            if direct_entry is not None:
                return WordReading(
                    ruby_text=direct_entry.reading,
                    pronunciation_text=pronunciation_text or direct_entry.reading,
                    source=direct_source,
                    ruby_parts=direct_entry.parts or _build_aligned_parts(
                        text,
                        direct_entry.reading,
                        self,
                    ),
                )

        part_sources: list[str] = []
        ruby_parts: list[RubyPart] = []
        for token in self.tokenize(text):
            entry, source = self._lookup_entry(token.surface, token.ruby)
            part_sources.append(source)
            if entry is not None and entry.parts:
                ruby_parts.extend(entry.parts)
            else:
                ruby_parts.extend(
                    _build_aligned_parts(
                        token.surface,
                        token.ruby,
                        self,
                    )
                )

        source = _resolve_source(part_sources)
        return WordReading(
            ruby_text=ruby_text,
            pronunciation_text=pronunciation_text or ruby_text,
            source=source,
            ruby_parts=ruby_parts,
        )

    def _lookup_entry(self, text: str, reading: str) -> tuple[FuriganaEntry | None, str]:
        if self._overrides is not None:
            entry = self._overrides.lookup_entry(text)
            if entry is not None:
                return entry, "override"

        if self._resource is not None:
            entry = self._resource.lookup_entry(text, reading)
            if entry is not None:
                return entry, "resource"
            unique_entry = self._resource.lookup_unique_text(text)
            if unique_entry is not None:
                return unique_entry, "resource"

        if self._builtin_resource is not None:
            entry = self._builtin_resource.lookup_entry(text, reading)
            if entry is not None:
                return entry, "builtin"
            unique_entry = self._builtin_resource.lookup_unique_text(text)
            if unique_entry is not None:
                return unique_entry, "builtin"

        return None, "backend"

    def _refine_token_reading(self, token: ReadingToken) -> ReadingToken:
        if self._fallback_reader is None or not has_kanji(token.surface):
            return token

        normalized_ruby = _normalize_reading_key(token.ruby)
        normalized_pronunciation = _normalize_reading_key(token.pronunciation)
        if _reading_looks_resolved(token.surface, normalized_ruby):
            return ReadingToken(
                surface=token.surface,
                ruby=normalized_ruby,
                pronunciation=normalized_pronunciation or normalized_ruby,
            )

        fallback_tokens = self._fallback_reader.tokenize(token.surface)
        fallback_ruby = _normalize_reading_key("".join(item.ruby for item in fallback_tokens))
        if not fallback_ruby:
            return token

        ruby = normalized_ruby or fallback_ruby
        pronunciation = normalized_pronunciation or fallback_ruby
        if not _reading_looks_resolved(token.surface, ruby):
            ruby = fallback_ruby
        if not _reading_looks_resolved(token.surface, pronunciation):
            pronunciation = fallback_ruby

        return ReadingToken(
            surface=token.surface,
            ruby=ruby,
            pronunciation=pronunciation,
        )

    def _candidate_readings(self, text: str, primary_reading: str) -> list[str]:
        candidates = [primary_reading]
        if self._fallback_reader is not None and has_kanji(text):
            fallback_tokens = self._fallback_reader.tokenize(text)
            fallback_reading = "".join(token.ruby for token in fallback_tokens).strip()
            if fallback_reading and fallback_reading not in candidates:
                candidates.append(fallback_reading)
        return candidates


def normalize_for_alignment(text: str, to_hiragana) -> str:
    normalized = to_hiragana(text).lower().replace("\u3000", " ")

    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if char.isspace():
            continue
        if category.startswith("P"):
            continue
        chars.append(char)
    return "".join(chars)


def build_text_processor(
    *,
    backend: str = "auto",
    split_mode: str = "C",
    furigana_resource_path: str | Path | None = None,
    reading_overrides_path: str | Path | None = None,
) -> JapaneseTextProcessor:
    backend_name = str(backend or "auto").lower()
    candidates = ["fugashi", "sudachi", "pykakasi"] if backend_name == "auto" else [backend_name]
    errors: list[str] = []

    selected_backend: ReadingBackend | None = None
    for candidate in candidates:
        try:
            if candidate == "fugashi":
                selected_backend = FugashiBackend()
            elif candidate == "sudachi":
                selected_backend = SudachiBackend(split_mode=split_mode)
            elif candidate == "pykakasi":
                selected_backend = KakasiBackend()
            else:
                raise ValueError(
                    f"Unsupported reading backend `{backend}`. Use auto, fugashi, sudachi, or pykakasi."
                )
            break
        except (ModuleNotFoundError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")

    if selected_backend is None:
        raise ModuleNotFoundError(
            "Could not initialize any Japanese reading backend. Tried: "
            + "; ".join(errors)
        )

    return JapaneseTextProcessor(
        backend=selected_backend,
        overrides=FuriganaLexicon.load(reading_overrides_path),
        resource=FuriganaLexicon.load(furigana_resource_path),
        builtin_resource=BuiltinFuriganaLexicon.ensure_default(),
    )


def build_hiragana_converter(
    *,
    backend: str = "auto",
    split_mode: str = "C",
    furigana_resource_path: str | Path | None = None,
    reading_overrides_path: str | Path | None = None,
):
    processor = build_text_processor(
        backend=backend,
        split_mode=split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )

    def to_hiragana(text: str) -> str:
        return processor.to_alignment_hiragana(text)

    return to_hiragana


def convert_payload_to_hiragana(
    payload: dict[str, Any],
    *,
    backend: str = "auto",
    split_mode: str = "C",
    furigana_resource_path: str | Path | None = None,
    reading_overrides_path: str | Path | None = None,
) -> dict[str, Any]:
    to_hiragana = build_hiragana_converter(
        backend=backend,
        split_mode=split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )
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
    backend: str = "auto",
    split_mode: str = "C",
    furigana_resource_path: str | Path | None = None,
    reading_overrides_path: str | Path | None = None,
) -> str:
    to_hiragana = build_hiragana_converter(
        backend=backend,
        split_mode=split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )
    converted_lines = []
    for line in re.split(r"\r\n|\r|\n", text):
        if remove_blank_lines and not line.strip():
            continue
        converted_line = to_hiragana(line)
        if remove_punctuation:
            converted_line = strip_punctuation(converted_line)
        converted_lines.append(converted_line)
    return "\n".join(converted_lines) + ("\n" if converted_lines else "")


def _read_text_resource(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


def _parse_json_lexicon(
    payload: Any,
) -> tuple[dict[str, FuriganaEntry], dict[tuple[str, str], FuriganaEntry]]:
    exact_map: dict[str, FuriganaEntry] = {}
    keyed_map: dict[tuple[str, str], FuriganaEntry] = {}

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str):
                exact_map[key] = FuriganaEntry(
                    reading=_normalize_reading_key(value),
                    parts=[],
                )
        return exact_map, keyed_map

    if not isinstance(payload, list):
        raise ValueError("JSON furigana resource must be an object or a list.")

    for entry in payload:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        reading = entry.get("reading")
        if not isinstance(text, str) or not isinstance(reading, str):
            continue
        furigana_parts = _parts_from_json_entry(entry.get("furigana"))
        keyed_map[(text, _normalize_reading_key(reading))] = FuriganaEntry(
            reading=_normalize_reading_key(reading),
            parts=furigana_parts,
        )

    return exact_map, keyed_map


def _parse_text_lexicon(
    text: str,
) -> tuple[dict[str, FuriganaEntry], dict[tuple[str, str], FuriganaEntry]]:
    exact_map: dict[str, FuriganaEntry] = {}
    keyed_map: dict[tuple[str, str], FuriganaEntry] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "|" in line:
            surface, reading, spec = line.split("|", 2)
            normalized_reading = _normalize_reading_key(reading.strip())
            keyed_map[(surface.strip(), normalized_reading)] = FuriganaEntry(
                reading=normalized_reading,
                parts=_parse_compact_furigana_parts(surface.strip(), spec.strip()),
            )
            continue

        parts = [part.strip() for part in re.split(r"\t+", line) if part.strip()]
        if len(parts) >= 2:
            exact_map[parts[0]] = FuriganaEntry(
                reading=_normalize_reading_key(parts[1]),
                parts=[],
            )

    return exact_map, keyed_map


def _parts_from_json_entry(raw_parts: Any) -> list[RubyPart]:
    parts: list[RubyPart] = []
    if not isinstance(raw_parts, list):
        return parts
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            continue
        ruby = str(raw_part.get("ruby", ""))
        rt = raw_part.get("rt")
        parts.append(
            RubyPart(
                ruby=ruby,
                rt=_normalize_reading_key(str(rt)) if isinstance(rt, str) and rt else None,
            )
        )
    return parts


def _parse_compact_furigana_parts(text: str, spec: str) -> list[RubyPart]:
    chars = list(text)
    annotations: list[tuple[int, int, str]] = []
    for chunk in spec.split(";"):
        piece = chunk.strip()
        if not piece:
            continue
        index_spec, reading = piece.split(":", 1)
        if "-" in index_spec:
            start_text, end_text = index_spec.split("-", 1)
            start = int(start_text)
            end = int(end_text)
        else:
            start = int(index_spec)
            end = start
        annotations.append((start, end, _normalize_reading_key(reading)))

    annotations.sort(key=lambda item: item[0])
    parts: list[RubyPart] = []
    cursor = 0
    for start, end, reading in annotations:
        if cursor < start:
            parts.append(RubyPart(ruby="".join(chars[cursor:start]), rt=None))
        parts.append(RubyPart(ruby="".join(chars[start : end + 1]), rt=reading))
        cursor = end + 1

    if cursor < len(chars):
        parts.append(RubyPart(ruby="".join(chars[cursor:]), rt=None))

    return parts


def _build_aligned_parts(
    text: str,
    reading: str,
    processor: JapaneseTextProcessor,
) -> list[RubyPart]:
    if not text:
        return []
    if not has_kanji(text):
        return [RubyPart(ruby=text, rt=None)]

    normalized_reading = _normalize_reading_key(reading)
    if not normalized_reading:
        return [RubyPart(ruby=text, rt=None)]

    runs = _split_surface_runs(text)

    parts: list[RubyPart] = []
    reading_pos = 0

    for index, (segment, is_kanji_run) in enumerate(runs):
        if not is_kanji_run:
            plain_reading = _normalize_surface_reading(segment, processor)
            if plain_reading and normalized_reading[reading_pos:].startswith(plain_reading):
                reading_pos += len(plain_reading)
            parts.append(RubyPart(ruby=segment, rt=None))
            continue

        next_plain = _next_anchor_reading(runs[index + 1 :], processor)
        min_remaining = _minimum_reading_for_runs(runs[index + 1 :], processor)
        max_end = max(reading_pos, len(normalized_reading) - min_remaining)

        if next_plain:
            next_index = normalized_reading.find(next_plain, reading_pos, max_end + 1)
            if next_index == -1:
                next_index = max_end
        else:
            next_index = max_end

        rt = normalized_reading[reading_pos:next_index]
        reading_pos = next_index
        if not rt:
            parts.append(RubyPart(ruby=segment, rt=normalized_reading[reading_pos:] or normalized_reading))
            reading_pos = len(normalized_reading)
            continue

        parts.extend(_segment_kanji_run(segment, rt, processor))

    return _merge_adjacent_parts(parts)


def _normalize_reading_key(text: str) -> str:
    return katakana_to_hiragana(text).replace("\u3000", " ").strip()


def _reading_looks_resolved(surface: str, reading: str) -> bool:
    if not reading:
        return False
    normalized_surface = _normalize_reading_key(surface)
    return not has_kanji(reading) and (
        not has_kanji(surface) or reading != normalized_surface
    )


def _normalize_surface_reading(
    text: str,
    processor: JapaneseTextProcessor,
) -> str:
    return _normalize_reading_key(processor.to_ruby_hiragana(text))


def _split_surface_runs(text: str) -> list[tuple[str, bool]]:
    if not text:
        return []

    runs: list[tuple[str, bool]] = []
    current = text[0]
    current_is_kanji = _is_kanji_like(text[0])
    for char in text[1:]:
        is_kanji = _is_kanji_like(char)
        if is_kanji == current_is_kanji:
            current += char
        else:
            runs.append((current, current_is_kanji))
            current = char
            current_is_kanji = is_kanji
    runs.append((current, current_is_kanji))
    return runs


def _next_anchor_reading(
    runs: list[tuple[str, bool]],
    processor: JapaneseTextProcessor,
) -> str:
    for segment, is_kanji_run in runs:
        if is_kanji_run:
            continue
        anchor = _normalize_surface_reading(segment, processor)
        if anchor:
            return anchor
    return ""


def _minimum_reading_for_runs(
    runs: list[tuple[str, bool]],
    processor: JapaneseTextProcessor,
) -> int:
    minimum = 0
    for segment, is_kanji_run in runs:
        if is_kanji_run:
            minimum += len(segment)
        else:
            minimum += len(_normalize_surface_reading(segment, processor))
    return minimum


def _segment_kanji_run(
    text: str,
    reading: str,
    processor: JapaneseTextProcessor,
) -> list[RubyPart]:
    if not text:
        return []
    if len(text) == 1 or len(reading) <= 1:
        return [RubyPart(ruby=text, rt=reading)]

    @lru_cache(maxsize=None)
    def segment_bonus(ruby: str, rt: str) -> float:
        entry, _ = processor._lookup_entry(ruby, rt)
        if entry is None:
            return 0.0
        return 1.35 if entry.parts else 1.0

    @lru_cache(maxsize=None)
    def solve(text_pos: int, reading_pos: int) -> tuple[float, tuple[RubyPart, ...]] | None:
        if text_pos == len(text) and reading_pos == len(reading):
            return 0.0, ()
        if text_pos >= len(text) or reading_pos >= len(reading):
            return None

        remaining_chars = len(text) - text_pos
        remaining_reading = len(reading) - reading_pos
        if remaining_reading < remaining_chars:
            return None

        best: tuple[float, tuple[RubyPart, ...]] | None = None
        max_group_size = min(remaining_chars, 4)
        for group_size in range(1, max_group_size + 1):
            ruby = text[text_pos : text_pos + group_size]
            min_segment_reading = 1
            max_segment_reading = remaining_reading - (remaining_chars - group_size)
            for segment_reading_len in range(max_segment_reading, min_segment_reading - 1, -1):
                rt = reading[reading_pos : reading_pos + segment_reading_len]
                tail = solve(text_pos + group_size, reading_pos + segment_reading_len)
                if tail is None:
                    continue

                tail_cost, tail_parts = tail
                cost = _segment_cost(
                    ruby=ruby,
                    rt=rt,
                    has_prefix=text_pos > 0,
                    has_tail=bool(tail_parts),
                ) - segment_bonus(ruby, rt) + tail_cost
                candidate = (cost, (RubyPart(ruby=ruby, rt=rt),) + tail_parts)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        return best

    resolved = solve(0, 0)
    if resolved is None:
        return [RubyPart(ruby=text, rt=reading)]
    return list(resolved[1])


def _segment_cost(*, ruby: str, rt: str, has_prefix: bool, has_tail: bool) -> float:
    group_len = len(ruby)
    reading_len = len(rt)
    average = reading_len / max(1, group_len)
    cost = abs(average - 2.0)
    cost += 1.0 * max(0, group_len - 1)

    if rt[0] in LEADING_DISCOURAGED_KANA:
        cost += 3.0
    if has_prefix and rt[0] in {"う", "い"}:
        cost += 0.95
    if has_tail and rt[-1] in SMALL_KANA:
        cost += 0.85
    if reading_len > group_len * 4:
        cost += (reading_len - group_len * 4) * 0.5
    return cost


def _merge_adjacent_parts(parts: list[RubyPart]) -> list[RubyPart]:
    merged: list[RubyPart] = []
    for part in parts:
        if not part.ruby:
            continue
        if merged and merged[-1].rt == part.rt:
            merged[-1] = RubyPart(
                ruby=merged[-1].ruby + part.ruby,
                rt=part.rt,
            )
            continue
        merged.append(part)
    return merged


def _resolve_source(sources: list[str]) -> str:
    if "override" in sources:
        return "override"
    if "resource" in sources:
        return "resource"
    if "builtin" in sources:
        return "builtin"
    return "backend"


def _serialize_parts(parts: list[RubyPart]) -> str:
    return json.dumps(
        [
            {
                "ruby": part.ruby,
                "rt": part.rt,
            }
            for part in parts
        ],
        ensure_ascii=False,
    )


def _deserialize_parts(payload: str) -> list[RubyPart]:
    raw_parts = json.loads(payload)
    return _parts_from_json_entry(raw_parts)


def _build_builtin_furigana_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    assets = _fetch_latest_release_assets()
    temp_dir = Path(tempfile.mkdtemp(prefix="nicokara-furigana-", dir=db_path.parent))
    temp_db_path = temp_dir / "jmdict_furigana.sqlite3"

    try:
        downloaded_files: list[tuple[Path, str]] = []
        for asset_name, source_name in JMDICT_ASSETS.items():
            url = assets.get(asset_name)
            if not url:
                continue
            target_path = temp_dir / asset_name
            urllib.request.urlretrieve(url, target_path)
            downloaded_files.append((target_path, source_name))

        conn = sqlite3.connect(temp_db_path)
        try:
            conn.execute(
                """
                CREATE TABLE entries (
                    text TEXT NOT NULL,
                    reading TEXT NOT NULL,
                    source TEXT NOT NULL,
                    furigana_json TEXT NOT NULL,
                    PRIMARY KEY (text, reading, source)
                )
                """
            )
            conn.execute(
                "CREATE INDEX idx_entries_text_reading ON entries(text, reading)"
            )

            batch: list[tuple[str, str, str, str]] = []
            for file_path, source_name in downloaded_files:
                with file_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line or line.startswith("#") or "|" not in line:
                            continue
                        text, reading, spec = line.split("|", 2)
                        batch.append(
                            (
                                text,
                                _normalize_reading_key(reading),
                                source_name,
                                _serialize_parts(_parse_compact_furigana_parts(text, spec)),
                            )
                        )
                        if len(batch) >= 1000:
                            conn.executemany(
                                "INSERT OR IGNORE INTO entries(text, reading, source, furigana_json) VALUES (?, ?, ?, ?)",
                                batch,
                            )
                            batch.clear()

            if batch:
                conn.executemany(
                    "INSERT OR IGNORE INTO entries(text, reading, source, furigana_json) VALUES (?, ?, ?, ?)",
                    batch,
                )
            conn.commit()
        finally:
            conn.close()

        temp_db_path.replace(db_path)
    finally:
        for path in temp_dir.iterdir():
            try:
                path.unlink()
            except IsADirectoryError:
                pass
            except FileNotFoundError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass


def _fetch_latest_release_assets() -> dict[str, str]:
    request = urllib.request.Request(
        JMDICT_RELEASE_API_URL,
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)

    assets: dict[str, str] = {}
    for asset in payload.get("assets", []):
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if isinstance(name, str) and isinstance(url, str):
            assets[name] = url
    return assets
