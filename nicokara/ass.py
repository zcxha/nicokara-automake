from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .convert import JapaneseTextProcessor, build_text_processor
from .fonts import DEFAULT_KARAOKE_FONT_NAME, DEFAULT_RUBY_FONT_NAME, bundled_font_environment

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - handled by fallback sizing
    Image = None


RUBY_FONT_NAME = DEFAULT_RUBY_FONT_NAME
RUBY_FONT_SIZE = 30
RUBY_ALIGNMENT = 2
RUBY_MARGIN_H = 56
RUBY_MARGIN_V = 88
RUBY_OUTLINE = 2
KARAOKE_FONT_NAME = DEFAULT_KARAOKE_FONT_NAME
KARAOKE_FONT_SIZE = 64
KARAOKE_ALIGNMENT = 2
KARAOKE_MARGIN_H = 56
KARAOKE_MARGIN_V = 36
KARAOKE_OUTLINE = 4
MEASUREMENT_MARGIN_X = 48
MEASUREMENT_MARGIN_Y = 48
MEASUREMENT_MIN_WIDTH = 1024
MEASUREMENT_MIN_HEIGHT = 256
UPPER_SLOT_LEFT_MARGIN = 88
LOWER_SLOT_RIGHT_MARGIN = 88
UPPER_SLOT_TOP_RATIO = 0.66
LOWER_SLOT_TOP_RATIO = 0.79
RUBY_VERTICAL_GAP = 18


@dataclass(frozen=True)
class _MeasuredLineLayout:
    """Store cumulative character boundaries for one rendered lyric line."""

    boundaries: tuple[float, ...]


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds as an ASS timestamp string."""
    total_centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"


def escape_ass_text(text: str) -> str:
    """Escape plain text so it is safe to embed in ASS dialogue lines."""
    return (
        text.replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def _boundary_weight(unit: dict[str, Any]) -> int:
    """Estimate how much timing span a word unit should consume."""
    value = unit.get("total_chars")
    if isinstance(value, int) and value > 0:
        return value
    normalized = str(unit.get("normalized_text", "")).strip()
    if normalized:
        return len(normalized)
    text = str(unit.get("text", "")).strip()
    return max(1, len(text))


def _average_boundary(existing: float | None, candidate: float) -> float:
    """Blend two boundary estimates when both are available."""
    if existing is None:
        return candidate
    return (existing + candidate) / 2.0


def _resolve_line_boundaries(line: dict[str, Any]) -> list[tuple[str, float, float]]:
    """Infer per-word timing boundaries for a rendered lyric line."""
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
    """Infer the separator that should be preserved between rendered words."""
    if "\u3000" in text:
        return "\u3000"
    if " " in text:
        return " "
    return ""


def _get_ruby_parts(
    unit: dict[str, Any],
    text_processor: JapaneseTextProcessor,
) -> list[dict[str, str | None]]:
    """Resolve ruby-part annotations for a lyric word unit."""
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


def _resolve_preview_start(lines: list[dict[str, Any]], index: int, line_start: float) -> float:
    """Return when a lyric line should first appear as an upcoming preview."""
    if index <= 0:
        return line_start

    previous_start = lines[index - 1].get("start")
    if previous_start is None:
        return line_start
    return min(line_start, float(previous_start))


def _resolve_line_slot(index: int) -> str:
    """Return which alternating nicokara slot should render the line."""
    return "upper_left" if index % 2 == 0 else "lower_right"


def _resolve_slot_top(slot_name: str, play_res_y: int) -> float:
    """Return the top y-coordinate for one lyric slot."""
    ratio = UPPER_SLOT_TOP_RATIO if slot_name == "upper_left" else LOWER_SLOT_TOP_RATIO
    return round(play_res_y * ratio, 2)


def _resolve_slot_left(slot_name: str, *, total_width: float, play_res_x: int) -> float:
    """Return the left x-coordinate for one lyric slot while keeping the line on screen."""
    right_aligned_left = max(0.0, play_res_x - LOWER_SLOT_RIGHT_MARGIN - total_width)
    if slot_name == "upper_left":
        return min(float(UPPER_SLOT_LEFT_MARGIN), right_aligned_left)
    return right_aligned_left


def _my_line_to_karaoke_events(
    line: dict[str, Any],
    *,
    play_res_x: int,
    play_res_y: int,
    text_processor: JapaneseTextProcessor,
) -> None:
    """Keep a placeholder for experimental karaoke rendering logic."""
    pass


def _line_to_karaoke_events(
    line: dict[str, Any],
    *,
    lines: list[dict[str, Any]],
    line_index: int,
    play_res_x: int,
    play_res_y: int,
    text_processor: JapaneseTextProcessor,
    ass_path: Path | None = None,
) -> tuple[float, float, str, list[str]] | None:
    """Render one aligned lyric line into karaoke and ruby ASS events."""
    segments = _resolve_line_boundaries(line)
    if not segments:
        return None

    line_start = segments[0][1]
    line_end = max(end for _, _, end in segments)
    display_start = _resolve_preview_start(lines, line_index, line_start)
    slot_name = _resolve_line_slot(line_index)
    separator = _infer_separator(str(line.get("text", "")))
    units = list(line.get("words") or [])

    karaoke_parts = []
    layout_segments = []
    plain_line_text = ""
    lead_in_cs = max(0, round((line_start - display_start) * 100))
    if lead_in_cs > 0:
        karaoke_parts.append(r"{\k" + str(lead_in_cs) + "}")

    for index, ((text, start, end), unit) in enumerate(zip(segments, units)):
        duration_cs = max(1, round((end - start) * 100))
        suffix = separator if separator and index < len(segments) - 1 else ""
        display_text = text + suffix
        karaoke_parts.append(r"{\kf" + str(duration_cs) + "}" + escape_ass_text(display_text))
        char_start = len(plain_line_text)
        plain_line_text += display_text
        char_end = char_start + len(text)
        layout_segments.append(
            {
                "text": text,
                "display_text": display_text,
                "char_start": char_start,
                "char_end": char_end,
                "unit": unit,
            }
        )

    ruby_events: list[str] = []
    measured_layout = _measure_line_layout(
        plain_line_text,
        ass_path=ass_path,
    )
    line_boundaries = list(measured_layout.boundaries)
    total_width = line_boundaries[-1] if line_boundaries else 0.0
    cursor_x = _resolve_slot_left(slot_name, total_width=total_width, play_res_x=play_res_x)
    line_top_y = _resolve_slot_top(slot_name, play_res_y)
    ruby_y = line_top_y - RUBY_VERTICAL_GAP
    for segment in layout_segments:
        unit = segment["unit"]
        text = str(segment["text"])
        word_left = cursor_x + line_boundaries[int(segment["char_start"])]
        ruby_parts = _get_ruby_parts(unit, text_processor)
        if not ruby_parts:
            continue
        boundaries = line_boundaries
        part_cursor = 0
        rendered_any = False
        for part in ruby_parts:
            part_text = str(part.get("ruby", ""))
            part_rt = part.get("rt")
            if not part_text:
                continue
            next_cursor = min(len(text), part_cursor + len(part_text))
            part_left = cursor_x + boundaries[int(segment["char_start"]) + part_cursor]
            part_right = cursor_x + boundaries[int(segment["char_start"]) + next_cursor]
            if part_right <= part_left:
                part_right = part_left
            part_center_x = part_left + (part_right - part_left) / 2.0
            if part_rt:
                rendered_any = True
                ruby_events.append(
                    "Dialogue: 1,"
                    f"{format_ass_timestamp(display_start)},"
                    f"{format_ass_timestamp(line_end)},"
                    "Ruby,,0,0,0,,"
                    r"{\an2\pos("
                    f"{round(part_center_x, 2)},{round(ruby_y, 2)}"
                    r")}"
                    + escape_ass_text(str(part_rt))
                )
            part_cursor = next_cursor

        if not rendered_any:
            ruby_text = str(unit.get("ruby_text", "")).strip()
            if ruby_text:
                ruby_events.append(
                    "Dialogue: 1,"
                    f"{format_ass_timestamp(display_start)},"
                    f"{format_ass_timestamp(line_end)},"
                    "Ruby,,0,0,0,,"
                    r"{\an2\pos("
                    f"{round((word_left + cursor_x + boundaries[int(segment['char_end'])]) / 2.0, 2)},{round(ruby_y, 2)}"
                    r")}"
                    + escape_ass_text(ruby_text)
                )

    positioned_karaoke = (
        r"{\an7\pos("
        f"{round(cursor_x, 2)},{round(line_top_y, 2)}"
        r")}"
        + "".join(karaoke_parts)
    )
    return display_start, line_end, positioned_karaoke, ruby_events


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
    ass_path: str | Path | None = None,
) -> str:
    """Convert an aligned nicokara payload into a full ASS subtitle document."""
    processor = text_processor or build_text_processor(
        backend=reading_backend,
        split_mode=reading_split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )
    lines = list(payload.get("lines", []))
    events = []
    for line_index, line in enumerate(lines):
        rendered = _line_to_karaoke_events(
            line,
            lines=lines,
            line_index=line_index,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            text_processor=processor,
            ass_path=Path(ass_path) if ass_path is not None else None,
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

    header = _build_ass_header(play_res_x=play_res_x, play_res_y=play_res_y, title=title)
    body = "\n".join(events)
    return header + body + ("\n" if body else "")


def _build_style_line(
    style_name: str,
    font_name: str,
    font_size: int,
    *,
    primary_colour: str,
    secondary_colour: str,
    outline_colour: str,
    back_colour: str,
    bold: int,
    outline: int,
    shadow: int,
    alignment: int,
    margin_l: int,
    margin_r: int,
    margin_v: int,
) -> str:
    """Build one ASS style definition line."""
    return (
        f"Style: {style_name},{font_name},{font_size},"
        f"{primary_colour},{secondary_colour},{outline_colour},{back_colour},"
        f"{bold},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},"
        f"{margin_l},{margin_r},{margin_v},1"
    )


def _build_ass_header(*, play_res_x: int, play_res_y: int, title: str) -> str:
    """Build the shared ASS document header used for final rendering."""
    ruby_style = _build_style_line(
        "Ruby",
        RUBY_FONT_NAME,
        RUBY_FONT_SIZE,
        primary_colour="&H00F8FBFF",
        secondary_colour="&H00F8FBFF",
        outline_colour="&H00111111",
        back_colour="&H64000000",
        bold=0,
        outline=RUBY_OUTLINE,
        shadow=0,
        alignment=RUBY_ALIGNMENT,
        margin_l=RUBY_MARGIN_H,
        margin_r=RUBY_MARGIN_H,
        margin_v=RUBY_MARGIN_V,
    )
    karaoke_style = _build_style_line(
        "Karaoke",
        KARAOKE_FONT_NAME,
        KARAOKE_FONT_SIZE,
        primary_colour="&H00F8FBFF",
        secondary_colour="&H006A6A6A",
        outline_colour="&H00111111",
        back_colour="&H64000000",
        bold=-1,
        outline=KARAOKE_OUTLINE,
        shadow=0,
        alignment=KARAOKE_ALIGNMENT,
        margin_l=KARAOKE_MARGIN_H,
        margin_r=KARAOKE_MARGIN_H,
        margin_v=KARAOKE_MARGIN_V,
    )
    return f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {play_res_x}
PlayResY: {play_res_y}
Title: {escape_ass_text(title)}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{ruby_style}
{karaoke_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _measurement_line_gap(font_size: int) -> int:
    """Return a vertical gap that keeps stacked measurement rows from overlapping."""
    return max(font_size * 2, 96)


def _estimate_measurement_width(text: str, font_size: int) -> int:
    """Estimate a safe canvas width for rendering one measurement line."""
    return max(MEASUREMENT_MIN_WIDTH, len(text) * font_size * 4 + MEASUREMENT_MARGIN_X * 2)


def _estimate_measurement_height(row_count: int, line_gap: int) -> int:
    """Estimate a safe canvas height for a stack of measurement rows."""
    return max(MEASUREMENT_MIN_HEIGHT, row_count * line_gap + MEASUREMENT_MARGIN_Y * 2)


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path so it can be embedded in an ffmpeg filter graph."""
    escaped = str(path)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    return escaped


def _build_ass_filter(path: Path, *, fonts_dir: Path | None = None) -> str:
    """Build an ffmpeg ass filter string for subtitle rendering."""
    options = [f"filename='{_escape_ffmpeg_filter_path(path)}'"]
    if fonts_dir is not None:
        options.append(f"fontsdir='{_escape_ffmpeg_filter_path(fonts_dir)}'")
    return "ass=" + ":".join(options)


def _build_measurement_document(
    texts: list[str],
    *,
    play_res_x: int,
    play_res_y: int,
    font_name: str,
    font_size: int,
) -> str:
    """Build an ASS document that renders multiple left-aligned text rows for measurement."""
    measure_style = _build_style_line(
        "Measure",
        font_name,
        font_size,
        primary_colour="&H00FFFFFF",
        secondary_colour="&H00FFFFFF",
        outline_colour="&H00111111",
        back_colour="&H64000000",
        bold=-1,
        outline=KARAOKE_OUTLINE,
        shadow=0,
        alignment=7,
        margin_l=0,
        margin_r=0,
        margin_v=0,
    )
    line_gap = _measurement_line_gap(font_size)
    events = []
    for index, text in enumerate(texts):
        y = MEASUREMENT_MARGIN_Y + index * line_gap
        events.append(
            "Dialogue: 0,0:00:00.00,0:00:01.00,Measure,,0,0,0,,"
            r"{\an7\pos("
            f"{MEASUREMENT_MARGIN_X},{y}"
            r")}"
            + escape_ass_text(text)
        )
    return (
        f"""[Script Info]
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {play_res_x}
PlayResY: {play_res_y}
Title: Measurement

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{measure_style}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        + "\n".join(events)
        + ("\n" if events else "")
    )


def _render_ass_image(
    ass_text: str,
    *,
    play_res_x: int,
    play_res_y: int,
    ass_path: Path | None = None,
) -> Image.Image:
    """Render an ASS document through ffmpeg/libass and return the resulting frame image."""
    if Image is None:
        raise RuntimeError("Pillow is required for libass measurement.")

    with tempfile.TemporaryDirectory(prefix="nicokara-ass-measure-") as temp_dir:
        temp_ass_path = Path(temp_dir) / "measurement.ass"
        temp_ass_path.write_text(ass_text, encoding="utf-8")
        with bundled_font_environment(
            [RUBY_FONT_NAME, KARAOKE_FONT_NAME],
            ass_path=ass_path,
        ) as (fonts_dir, font_env):
            command = [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={play_res_x}x{play_res_y}:d=1",
                "-frames:v",
                "1",
                "-vf",
                _build_ass_filter(temp_ass_path, fonts_dir=fonts_dir),
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ]
            frame_bytes = subprocess.check_output(command, env=font_env)

    with Image.open(io.BytesIO(frame_bytes)) as frame:
        return frame.convert("L")


def _extract_visible_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return the bounding box of all non-background pixels in a rendered frame."""
    return image.point(lambda value: 255 if value > 0 else 0).getbbox()


def _extract_row_bbox(
    image: Image.Image,
    *,
    index: int,
    row_count: int,
    line_gap: int,
) -> tuple[int, int, int, int] | None:
    """Return the visible bounding box for one stacked measurement row."""
    anchor_y = MEASUREMENT_MARGIN_Y + index * line_gap
    top = 0 if index == 0 else max(0, anchor_y - line_gap // 2)
    bottom = image.height if index == row_count - 1 else min(image.height, anchor_y + line_gap // 2)
    crop = image.crop((0, top, image.width, bottom))
    bbox = _extract_visible_bbox(crop)
    if bbox is None:
        return None
    return (bbox[0], bbox[1] + top, bbox[2], bbox[3] + top)


def _measure_prefix_boundaries(
    text: str,
    *,
    font_name: str,
    font_size: int,
    ass_path: Path | None = None,
) -> tuple[float, ...]:
    """Measure cumulative rendered widths for every prefix of a line using libass."""
    if not text:
        return (0.0,)

    prefixes = [text[:index] for index in range(len(text) + 1)]
    line_gap = _measurement_line_gap(font_size)
    play_res_x = _estimate_measurement_width(text, font_size)
    play_res_y = _estimate_measurement_height(len(prefixes), line_gap)
    measurement_image = _render_ass_image(
        _build_measurement_document(
            prefixes,
            play_res_x=play_res_x,
            play_res_y=play_res_y,
            font_name=font_name,
            font_size=font_size,
        ),
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        ass_path=ass_path,
    )

    row_boxes = [
        _extract_row_bbox(measurement_image, index=index, row_count=len(prefixes), line_gap=line_gap)
        for index in range(len(prefixes))
    ]
    visible_boxes = [box for box in row_boxes[1:] if box is not None]
    if not visible_boxes:
        return tuple(0.0 for _ in prefixes)

    base_left = min(box[0] for box in visible_boxes)
    boundaries = [0.0]
    for box in row_boxes[1:]:
        if box is None:
            boundaries.append(boundaries[-1])
            continue
        boundaries.append(float(box[2] - base_left))
    return tuple(boundaries)


@lru_cache(maxsize=512)
def _measure_line_layout_cached(
    text: str,
    font_name: str,
    font_size: int,
    ass_path_value: str,
) -> _MeasuredLineLayout:
    """Cache one full libass measurement result for a rendered lyric line."""
    ass_path = Path(ass_path_value) if ass_path_value else None
    boundaries = _measure_prefix_boundaries(
        text,
        font_name=font_name,
        font_size=font_size,
        ass_path=ass_path,
    )
    return _MeasuredLineLayout(boundaries=boundaries)


def _measure_line_layout(
    text: str,
    *,
    ass_path: Path | None = None,
) -> _MeasuredLineLayout:
    """Measure one lyric line with the same libass renderer used during final burn-in."""
    ass_path_value = str(ass_path) if ass_path is not None else ""
    return _measure_line_layout_cached(
        text,
        KARAOKE_FONT_NAME,
        KARAOKE_FONT_SIZE,
        ass_path_value,
    )
