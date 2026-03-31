from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from .convert import JapaneseTextProcessor, build_text_processor

try:
    from PIL import ImageFont
except ModuleNotFoundError:  # pragma: no cover - handled by fallback sizing
    ImageFont = None


RUBY_FONT_NAME = "Noto Sans CJK JP"
RUBY_FONT_SIZE = 24
RUBY_MARGIN_V = 88
KARAOKE_FONT_NAME = "Noto Sans CJK JP"
KARAOKE_FONT_SIZE = 52


def format_ass_timestamp(seconds: float) -> str:
    total_centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"


def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _boundary_weight(unit: dict[str, Any]) -> int:
    value = unit.get("total_chars")
    if isinstance(value, int) and value > 0:
        return value
    normalized = str(unit.get("normalized_text", "")).strip()
    if normalized:
        return len(normalized)
    text = str(unit.get("text", "")).strip()
    return max(1, len(text))


def _average_boundary(existing: float | None, candidate: float) -> float:
    if existing is None:
        return candidate
    return (existing + candidate) / 2.0


def _resolve_line_boundaries(line: dict[str, Any]) -> list[tuple[str, float, float]]:
    units = list(line.get("words") or [])
    if not units:
        return []

    line_start = line.get("start")
    line_end = line.get("end")
    if line_start is None:
        for unit in units:
            if unit.get("start") is not None:
                line_start = float(unit["start"])
                break
    if line_end is None:
        for unit in reversed(units):
            if unit.get("end") is not None:
                line_end = float(unit["end"])
                break

    if line_start is None or line_end is None or float(line_end) <= float(line_start):
        return []

    boundaries: list[float | None] = [None] * (len(units) + 1)
    boundaries[0] = float(line_start)
    boundaries[-1] = float(line_end)

    for index, unit in enumerate(units):
        start = unit.get("start")
        end = unit.get("end")
        if start is not None:
            boundaries[index] = _average_boundary(boundaries[index], float(start))
        if end is not None:
            boundaries[index + 1] = _average_boundary(boundaries[index + 1], float(end))

    known_indices = [index for index, value in enumerate(boundaries) if value is not None]
    if not known_indices:
        return []

    weights = [_boundary_weight(unit) for unit in units]
    for left_index, right_index in zip(known_indices, known_indices[1:]):
        left_value = boundaries[left_index]
        right_value = boundaries[right_index]
        if left_value is None or right_value is None or right_index <= left_index + 1:
            continue

        span = max(0.0, right_value - left_value)
        total_weight = sum(weights[left_index:right_index]) or (right_index - left_index)
        accumulated = 0
        for boundary_index in range(left_index + 1, right_index):
            accumulated += weights[boundary_index - 1]
            boundaries[boundary_index] = left_value + span * accumulated / total_weight

    last_known = None
    for index, value in enumerate(boundaries):
        if value is None:
            if last_known is None:
                continue
            boundaries[index] = last_known
        else:
            last_known = value

    next_known = None
    for index in range(len(boundaries) - 1, -1, -1):
        value = boundaries[index]
        if value is None:
            if next_known is None:
                continue
            boundaries[index] = next_known
        else:
            next_known = value

    resolved = [float(value if value is not None else line_start) for value in boundaries]
    if resolved[-1] <= resolved[0]:
        return []

    segments: list[tuple[str, float, float]] = []
    for index, unit in enumerate(units):
        text = str(unit.get("text", ""))
        start = resolved[index]
        end = resolved[index + 1]
        if end < start:
            end = start
        segments.append((text, start, end))
    return segments


def _infer_separator(text: str) -> str:
    if "\u3000" in text:
        return "\u3000"
    if " " in text:
        return " "
    return ""


def _get_ruby_parts(
    unit: dict[str, Any],
    text_processor: JapaneseTextProcessor,
) -> list[dict[str, str | None]]:
    raw_parts = unit.get("ruby_parts")
    if isinstance(raw_parts, list):
        normalized_parts = []
        for raw_part in raw_parts:
            if not isinstance(raw_part, dict):
                continue
            ruby = str(raw_part.get("ruby", ""))
            rt = raw_part.get("rt")
            normalized_parts.append(
                {
                    "ruby": ruby,
                    "rt": str(rt) if isinstance(rt, str) and rt else None,
                }
            )
        if normalized_parts:
            return normalized_parts

    text = str(unit.get("text", ""))
    word_reading = text_processor.resolve_word_reading(text)
    return [
        {
            "ruby": part.ruby,
            "rt": part.rt,
        }
        for part in word_reading.ruby_parts
    ]


def _line_to_karaoke_events(
    line: dict[str, Any],
    *,
    play_res_x: int,
    play_res_y: int,
    text_processor: JapaneseTextProcessor,
) -> tuple[float, float, str, list[str]] | None:
    segments = _resolve_line_boundaries(line)
    if not segments:
        return None

    line_start = segments[0][1]
    line_end = max(end for _, _, end in segments)
    separator = _infer_separator(str(line.get("text", "")))
    units = list(line.get("words") or [])

    karaoke_parts = []
    layout_segments = []
    total_width = 0.0

    for index, ((text, start, end), unit) in enumerate(zip(segments, units)):
        duration_cs = max(1, round((end - start) * 100))
        suffix = separator if separator and index < len(segments) - 1 else ""
        display_text = text + suffix
        karaoke_parts.append(r"{\k" + str(duration_cs) + "}" + escape_ass_text(display_text))

        display_width = _measure_text(display_text, KARAOKE_FONT_NAME, KARAOKE_FONT_SIZE)
        word_width = _measure_text(text, KARAOKE_FONT_NAME, KARAOKE_FONT_SIZE)
        layout_segments.append(
            {
                "text": text,
                "display_text": display_text,
                "display_width": display_width,
                "word_width": word_width,
                "unit": unit,
            }
        )
        total_width += display_width

    ruby_events: list[str] = []
    cursor_x = play_res_x / 2.0 - total_width / 2.0
    ruby_y = play_res_y - RUBY_MARGIN_V

    for segment in layout_segments:
        unit = segment["unit"]
        text = str(segment["text"])
        word_left = cursor_x
        ruby_parts = _get_ruby_parts(unit, text_processor)
        if not ruby_parts:
            cursor_x += float(segment["display_width"])
            continue

        part_cursor = 0.0
        rendered_any = False
        for part in ruby_parts:
            part_text = str(part.get("ruby", ""))
            part_rt = part.get("rt")
            if not part_text:
                continue
            part_width = _measure_text(part_text, KARAOKE_FONT_NAME, KARAOKE_FONT_SIZE)
            part_center_x = word_left + part_cursor + part_width / 2.0
            if part_rt:
                rendered_any = True
                ruby_events.append(
                    "Dialogue: 1,"
                    f"{format_ass_timestamp(line_start)},"
                    f"{format_ass_timestamp(line_end)},"
                    "Ruby,,0,0,0,,"
                    r"{\an2\pos("
                    f"{round(part_center_x, 2)},{round(ruby_y, 2)}"
                    r")}"
                    + escape_ass_text(str(part_rt))
                )
            part_cursor += part_width

        if not rendered_any:
            ruby_text = str(unit.get("ruby_text", "")).strip()
            if ruby_text:
                ruby_events.append(
                    "Dialogue: 1,"
                    f"{format_ass_timestamp(line_start)},"
                    f"{format_ass_timestamp(line_end)},"
                    "Ruby,,0,0,0,,"
                    r"{\an2\pos("
                    f"{round(word_left + float(segment['word_width']) / 2.0, 2)},{round(ruby_y, 2)}"
                    r")}"
                    + escape_ass_text(ruby_text)
                )

        cursor_x += float(segment["display_width"])

    return line_start, line_end, "".join(karaoke_parts), ruby_events


def payload_to_ass_text(
    payload: dict[str, Any],
    *,
    play_res_x: int = 1280,
    play_res_y: int = 720,
    title: str = "Nicokara Karaoke",
    text_processor: JapaneseTextProcessor | None = None,
    reading_backend: str = "auto",
    reading_split_mode: str = "C",
    furigana_resource_path: str | Path | None = None,
    reading_overrides_path: str | Path | None = None,
) -> str:
    processor = text_processor or build_text_processor(
        backend=reading_backend,
        split_mode=reading_split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )
    events = []
    for line in payload.get("lines", []):
        rendered = _line_to_karaoke_events(
            line,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            text_processor=processor,
        )
        if rendered is None:
            continue
        start, end, karaoke_text, ruby_events = rendered
        events.extend(ruby_events)
        events.append(
            "Dialogue: 0,"
            f"{format_ass_timestamp(start)},"
            f"{format_ass_timestamp(end)},"
            "Karaoke,,0,0,0,,"
            f"{karaoke_text}"
        )

    header = f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {play_res_x}
PlayResY: {play_res_y}
Title: {escape_ass_text(title)}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Ruby,{RUBY_FONT_NAME},{RUBY_FONT_SIZE},&H00F8FBFF,&H00F8FBFF,&H00111111,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,56,56,{RUBY_MARGIN_V},1
Style: Karaoke,{KARAOKE_FONT_NAME},{KARAOKE_FONT_SIZE},&H00F8FBFF,&H006A6A6A,&H00111111,&H64000000,-1,0,0,0,100,100,0,0,1,3,0,2,56,56,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = "\n".join(events)
    return header + body + ("\n" if body else "")


@lru_cache(maxsize=8)
def _resolve_font_path(font_name: str) -> str | None:
    try:
        probe = subprocess.run(
            ["fc-match", "-f", "%{file}\n", font_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if probe.returncode != 0:
        return None
    path = probe.stdout.strip().splitlines()
    if not path:
        return None
    return path[0].strip() or None


@lru_cache(maxsize=16)
def _load_font(font_name: str, font_size: int):
    if ImageFont is None:
        return None

    font_path = _resolve_font_path(font_name)
    candidates = [font_path, font_name]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return None


def _measure_text(text: str, font_name: str, font_size: int) -> float:
    if not text:
        return 0.0
    font = _load_font(font_name, font_size)
    if font is None:
        return max(1.0, float(font_size) * len(text) * 0.85)

    try:
        return float(font.getlength(text))
    except AttributeError:
        bbox = font.getbbox(text)
        return float(max(0, bbox[2] - bbox[0]))
