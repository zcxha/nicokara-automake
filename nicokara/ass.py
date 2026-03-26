from __future__ import annotations

from typing import Any

from .convert import build_kakasi_converter


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


def _has_kanji(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _build_ruby_text(unit: dict[str, Any], kakasi) -> str:
    ruby_text = str(unit.get("ruby_text", "")).strip()
    if ruby_text:
        return ruby_text

    text = str(unit.get("text", ""))
    if not _has_kanji(text):
        return ""

    ruby_parts = []
    for item in kakasi.convert(text):
        hira = str(item.get("hira", "")).replace("\u3000", " ").strip()
        if hira:
            ruby_parts.append(hira)
    return "".join(ruby_parts)


def _placeholder_for_text(text: str) -> str:
    visible_chars = [char for char in text if not char.isspace()]
    return "\u3000" * max(1, len(visible_chars))


def _line_to_karaoke_events(line: dict[str, Any]) -> tuple[float, float, str, str | None] | None:
    segments = _resolve_line_boundaries(line)
    if not segments:
        return None

    line_start = segments[0][1]
    line_end = max(end for _, _, end in segments)
    separator = _infer_separator(str(line.get("text", "")))
    kakasi = build_kakasi_converter()
    units = list(line.get("words") or [])

    karaoke_parts = []
    ruby_parts = []
    has_ruby = False

    for index, ((text, start, end), unit) in enumerate(zip(segments, units)):
        duration_cs = max(1, round((end - start) * 100))
        suffix = separator if separator and index < len(segments) - 1 else ""
        karaoke_parts.append(r"{\k" + str(duration_cs) + "}" + escape_ass_text(text + suffix))
        ruby_text = _build_ruby_text(unit, kakasi)
        if ruby_text:
            has_ruby = True
            ruby_parts.append(escape_ass_text(ruby_text + suffix))
        else:
            placeholder = _placeholder_for_text(text) + suffix
            ruby_parts.append(escape_ass_text(placeholder))

    ruby_line = "".join(ruby_parts) if has_ruby else None
    return line_start, line_end, "".join(karaoke_parts), ruby_line


def payload_to_ass_text(
    payload: dict[str, Any],
    *,
    play_res_x: int = 1280,
    play_res_y: int = 720,
    title: str = "Nicokara Karaoke",
) -> str:
    events = []
    for line in payload.get("lines", []):
        rendered = _line_to_karaoke_events(line)
        if rendered is None:
            continue
        start, end, karaoke_text, ruby_text = rendered
        if ruby_text:
            events.append(
                "Dialogue: 0,"
                f"{format_ass_timestamp(start)},"
                f"{format_ass_timestamp(end)},"
                "Ruby,,0,0,0,,"
                f"{ruby_text}"
            )
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
Style: Ruby,Noto Sans CJK JP,24,&H00F8FBFF,&H00F8FBFF,&H00111111,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,56,56,88,1
Style: Karaoke,Noto Sans CJK JP,52,&H00F8FBFF,&H006A6A6A,&H00111111,&H64000000,-1,0,0,0,100,100,0,0,1,3,0,2,56,56,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    body = "\n".join(events)
    return header + body + ("\n" if body else "")
