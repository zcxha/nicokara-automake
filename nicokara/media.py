from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class ExternalToolError(RuntimeError):
    pass


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    """Run an external command and normalize common subprocess failures."""
    try:
        subprocess.run(args, check=True, env=env)
    except FileNotFoundError as exc:
        raise ExternalToolError(f"Command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        command = " ".join(args)
        raise ExternalToolError(f"Command failed ({exc.returncode}): {command}") from exc


def ensure_directory(path: Path) -> Path:
    """Create a directory tree when needed and return the same path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_audio(video_path: Path, output_audio_path: Path, *, sample_rate: int = 16000) -> Path:
    """Extract mono PCM audio from a source video with ffmpeg."""
    ensure_directory(output_audio_path.parent)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(output_audio_path),
        ]
    )
    return output_audio_path


def probe_video_resolution(video_path: Path) -> tuple[int, int]:
    """Read the first video stream resolution with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe got no streams")
    stream = streams[0]
    return int(stream["width"]), int(stream["height"])


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape an ASS path so it can be embedded in an ffmpeg filter graph."""
    escaped = str(path)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    return f"ass='{escaped}'"


def burn_ass_subtitles(
    video_path: Path,
    ass_path: Path,
    output_video_path: Path,
    *,
    video_codec: str = "libx264",
    audio_codec: str = "copy",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Burn ASS subtitles into a video with ffmpeg and return the output path."""
    ensure_directory(output_video_path.parent)
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            _escape_ffmpeg_filter_path(ass_path),
            "-c:v",
            video_codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            audio_codec,
            str(output_video_path),
        ]
    )
    return output_video_path


def copy_if_needed(source: Path, target: Path) -> Path:
    """Copy a file unless the source already resolves to the target path."""
    ensure_directory(target.parent)
    if source.resolve() == target.resolve():
        return target
    shutil.copy2(source, target)
    return target


def prepend_pythonpath(env: dict[str, str], extra_path: Path) -> dict[str, str]:
    """Return a copy of env with extra_path prepended to PYTHONPATH."""
    updated = dict(env)
    current = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = str(extra_path) if not current else os.pathsep.join([str(extra_path), current])
    return updated
