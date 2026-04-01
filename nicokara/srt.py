from __future__ import annotations

from typing import Any


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp string."""
    total_milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def normalize_text(text: str) -> str:
    """Normalize subtitle line endings and trim outer whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def extract_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the first supported subtitle-entry list from a payload."""
    for key in ("segments", "lyrics", "lines"):
        entries = payload.get(key)
        if isinstance(entries, list):
            return entries
    raise ValueError("Input JSON must contain either a 'segments', 'lyrics', or 'lines' list.")


def entries_to_srt(entries: list[dict[str, Any]]) -> str:
    """Convert timestamped entries into SRT text blocks."""
    blocks: list[str] = []
    subtitle_index = 1
    for entry in entries:
        start = entry.get("start")
        end = entry.get("end")
        text = normalize_text(str(entry.get("text", "")))
        if start is None or end is None or not text:
            continue
        blocks.append(
            f"{subtitle_index}\n"
            f"{format_timestamp(float(start))} --> {format_timestamp(float(end))}\n"
            f"{text}\n"
        )
        subtitle_index += 1
    return "\n".join(blocks)


def payload_to_srt_text(payload: dict[str, Any]) -> str:
    """Render a supported JSON payload directly into SRT text."""
    return entries_to_srt(extract_entries(payload))
