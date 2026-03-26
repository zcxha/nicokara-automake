from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class ExternalToolError(RuntimeError):
    pass


def run_command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(args, check=True, env=env)
    except FileNotFoundError as exc:
        raise ExternalToolError(f"Command not found: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        command = " ".join(args)
        raise ExternalToolError(f"Command failed ({exc.returncode}): {command}") from exc


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_audio(video_path: Path, output_audio_path: Path, *, sample_rate: int = 16000) -> Path:
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
        return 1280, 720
    stream = streams[0]
    return int(stream.get("width", 1280)), int(stream.get("height", 720))


def _escape_ffmpeg_filter_path(path: Path) -> str:
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
    ensure_directory(target.parent)
    if source.resolve() == target.resolve():
        return target
    shutil.copy2(source, target)
    return target


def prepend_pythonpath(env: dict[str, str], extra_path: Path) -> dict[str, str]:
    updated = dict(env)
    current = updated.get("PYTHONPATH", "")
    updated["PYTHONPATH"] = str(extra_path) if not current else os.pathsep.join([str(extra_path), current])
    return updated
