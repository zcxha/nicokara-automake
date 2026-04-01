from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .alignment import build_word_level_payload
from .ass import payload_to_ass_text
from .convert import build_text_processor
from .media import (
    ExternalToolError,
    burn_ass_subtitles,
    copy_if_needed,
    ensure_directory,
    extract_audio,
    prepend_pythonpath,
    probe_video_resolution,
    run_command,
)
from .srt import payload_to_srt_text


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelineArtifacts:
    extracted_audio: Path | None
    vocals_audio: Path | None
    asr_json: Path
    aligned_json: Path
    line_srt: Path
    karaoke_ass: Path
    output_video: Path | None


def _resolve_demucs_command() -> list[str]:
    """Resolve the best available Demucs invocation command."""
    demucs = shutil.which("demucs")
    if demucs:
        return [demucs]

    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "demucs", "demucs"]

    uv = shutil.which("uv")
    if uv:
        return [uv, "tool", "run", "--from", "demucs", "demucs"]

    raise ExternalToolError(
        "Could not find Demucs. Install it with `uv tool install demucs` or make `demucs` available in PATH."
    )


def _ensure_demucs_runtime_ready(command: list[str]) -> None:
    """Validate that the resolved Demucs runtime has the required dependencies."""
    executable = Path(command[0])
    if not executable.is_file():
        return

    try:
        first_line = executable.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeDecodeError, IndexError):
        return

    if not first_line.startswith("#!"):
        return

    python_executable = first_line[2:].strip()
    if not python_executable:
        return

    probe = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util, json, sys; "
                "payload = {'torchcodec': importlib.util.find_spec('torchcodec') is not None}; "
                "try:\n"
                " import torch\n"
                " payload['torch_version'] = torch.__version__\n"
                " payload['cuda_available'] = torch.cuda.is_available()\n"
                "except Exception:\n"
                " payload['torch_version'] = None\n"
                " payload['cuda_available'] = None\n"
                "print(json.dumps(payload))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return

    try:
        payload = json.loads(probe.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return

    if not payload.get("torchcodec", False):
        raise ExternalToolError(
            "Demucs is installed but its runtime is missing `torchcodec`. "
            "Fix it with `uv tool install --force --with torchcodec demucs`."
        )

    torch_version = str(payload.get("torch_version") or "")
    cuda_available = payload.get("cuda_available")
    if "+cu" in torch_version and cuda_available is False:
        raise ExternalToolError(
            "Demucs is using a CUDA PyTorch build, but CUDA is not usable on this machine. "
            "Reinstall it with `uv tool install --force --torch-backend cpu --with torchcodec demucs`."
        )


def separate_vocals(
    audio_path: Path,
    vocals_output_path: Path,
    *,
    work_dir: Path,
    model: str = "htdemucs_ft",
) -> Path:
    """Run Demucs vocal separation and copy the vocals stem into a stable path."""
    command = _resolve_demucs_command()
    _ensure_demucs_runtime_ready(command)
    demucs_output_dir = ensure_directory(work_dir / "demucs")
    run_command(
        command
        + [
            "--two-stems",
            "vocals",
            "-n",
            model,
            "-o",
            str(demucs_output_dir),
            str(audio_path),
        ]
    )

    generated_vocals = demucs_output_dir / model / audio_path.stem / "vocals.wav"
    if not generated_vocals.exists():
        raise ExternalToolError(f"Demucs finished but `{generated_vocals}` was not produced.")
    return copy_if_needed(generated_vocals, vocals_output_path)


def _resolve_whisper_command() -> tuple[list[str], dict[str, str] | None]:
    """Resolve the best available whisper-timestamped invocation command."""
    executable = shutil.which("whisper_timestamped")
    if executable:
        command = [executable]
        _ensure_whisper_runtime_ready(command)
        return command, None

    local_checkout = REPO_ROOT / "whisper-timestamped"
    if local_checkout.exists():
        env = prepend_pythonpath(os.environ, local_checkout)
        return [sys.executable, "-m", "whisper_timestamped.transcribe"], env

    raise ExternalToolError(
        "Could not find `whisper_timestamped`. Install it with `uv tool install whisper-timestamped`."
    )


def _ensure_whisper_runtime_ready(command: list[str]) -> None:
    """Validate that the resolved whisper runtime can execute successfully."""
    executable = Path(command[0])
    if not executable.is_file():
        return

    try:
        first_line = executable.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, UnicodeDecodeError, IndexError):
        return

    if not first_line.startswith("#!"):
        return

    python_executable = first_line[2:].strip()
    if not python_executable:
        return

    probe = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util, json; "
                "payload = {"
                " 'packaging': importlib.util.find_spec('packaging') is not None,"
                " 'whisper': importlib.util.find_spec('whisper') is not None"
                "}; "
                "try:\n"
                " import torch\n"
                " payload['torch_version'] = torch.__version__\n"
                " payload['cuda_available'] = torch.cuda.is_available()\n"
                "except Exception:\n"
                " payload['torch_version'] = None\n"
                " payload['cuda_available'] = None\n"
                "print(json.dumps(payload))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return

    try:
        payload = json.loads(probe.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return

    if not payload.get("packaging", False):
        raise ExternalToolError(
            "whisper_timestamped is installed but its runtime is missing `packaging`. "
            "Fix it with `uv tool install --force --with packaging --with setuptools whisper-timestamped`."
        )

    if not payload.get("whisper", False):
        raise ExternalToolError(
            "whisper_timestamped is installed but its runtime is incomplete. "
            "Reinstall it with `uv tool install --force --with packaging --with setuptools whisper-timestamped`."
        )

    torch_version = str(payload.get("torch_version") or "")
    cuda_available = payload.get("cuda_available")
    if "+cu" in torch_version and cuda_available is False:
        raise ExternalToolError(
            "whisper_timestamped is using a CUDA PyTorch build, but CUDA is not usable on this machine. "
            "Reinstall it with `uv tool install --force --torch-backend cpu --with packaging --with setuptools whisper-timestamped`."
        )


def run_whisper_timestamped(
    audio_path: Path,
    *,
    output_dir: Path,
    model: str = "small",
    language: str = "ja",
    device: str | None = None,
    vad: bool = False,
) -> Path:
    """Run whisper-timestamped and return the generated word-level JSON path."""
    command, env = _resolve_whisper_command()
    ensure_directory(output_dir)

    def cli_bool(value: bool) -> str:
        """Serialize booleans into the CLI format expected by whisper-timestamped."""
        return "True" if value else "False"

    args = command + [
        str(audio_path),
        "--output_dir",
        str(output_dir),
        "--output_format",
        "json",
        "--model",
        model,
        "--language",
        language,
        "--verbose",
        cli_bool(False),
        "--punctuations_with_words",
        cli_bool(False),
        "--compute_confidence",
        cli_bool(True),
        "--vad",
        cli_bool(vad),
        "--efficient",
    ]
    if device:
        args.extend(["--device", device])

    run_command(args, env=env)
    json_path = output_dir / f"{audio_path.name}.words.json"
    if not json_path.exists():
        raise ExternalToolError(f"Whisper finished but `{json_path}` was not produced.")
    return json_path


def _default_output_dir(video_path: Path) -> Path:
    """Build the default nicokara artifact directory for a source video."""
    return video_path.with_name(f"{video_path.stem}.nicokara")


def build_nicokara_video(
    video_path: str | Path,
    lyrics_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    output_video_path: str | Path | None = None,
    whisper_model: str = "small",
    whisper_language: str = "ja",
    whisper_device: str | None = None,
    demucs_model: str = "htdemucs_ft",
    asr_json_path: str | Path | None = None,
    vocals_audio_path: str | Path | None = None,
    force: bool = False,
    skip_burn: bool = False,
    whisper_vad: bool = False,
    reading_backend: str = "auto",
    reading_split_mode: str = "C",
    furigana_resource_path: str | Path | None = None,
    reading_overrides_path: str | Path | None = None,
) -> PipelineArtifacts:
    """Run the full nicokara pipeline and return the produced artifact paths."""
    video_source = Path(video_path)
    lyrics_source = Path(lyrics_path)
    build_dir = Path(output_dir) if output_dir else _default_output_dir(video_source)
    ensure_directory(build_dir)

    extracted_audio_path = build_dir / f"{video_source.stem}.audio.wav"
    stable_vocals_path = build_dir / f"{video_source.stem}.vocals.wav"
    stable_asr_json_path = build_dir / f"{stable_vocals_path.name}.words.json"
    aligned_json_path = build_dir / f"{video_source.stem}.nicokara.json"
    line_srt_path = build_dir / f"{video_source.stem}.nicokara.srt"
    karaoke_ass_path = build_dir / f"{video_source.stem}.karaoke.ass"
    final_video_path = Path(output_video_path) if output_video_path else build_dir / f"{video_source.stem}.nicokara.mp4"

    if asr_json_path is not None:
        copy_if_needed(Path(asr_json_path), stable_asr_json_path)
    else:
        if vocals_audio_path is not None:
            copy_if_needed(Path(vocals_audio_path), stable_vocals_path)
        else:
            if force or not extracted_audio_path.exists():
                extract_audio(video_source, extracted_audio_path)
            if force or not stable_vocals_path.exists():
                separate_vocals(
                    extracted_audio_path,
                    stable_vocals_path,
                    work_dir=build_dir,
                    model=demucs_model,
                )

    if asr_json_path is None and (force or not stable_asr_json_path.exists()):
        run_whisper_timestamped(
            stable_vocals_path,
            output_dir=build_dir,
            model=whisper_model,
            language=whisper_language,
            device=whisper_device,
            vad=whisper_vad,
        )

    text_processor = build_text_processor(
        backend=reading_backend,
        split_mode=reading_split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )

    payload = build_word_level_payload(
        stable_asr_json_path,
        lyrics_source,
        text_processor=text_processor,
        reading_backend=reading_backend,
        reading_split_mode=reading_split_mode,
        furigana_resource_path=furigana_resource_path,
        reading_overrides_path=reading_overrides_path,
    )
    aligned_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    line_payload = {
        "lines": [
            {
                "line_id": line["line_id"],
                "text": line["text"],
                "start": line["start"],
                "end": line["end"],
            }
            for line in payload["lines"]
        ]
    }
    line_srt_path.write_text(payload_to_srt_text(line_payload), encoding="utf-8")

    width, height = probe_video_resolution(video_source)
    karaoke_ass_path.write_text(
        payload_to_ass_text(
            payload,
            play_res_x=width,
            play_res_y=height,
            title=video_source.stem,
            text_processor=text_processor,
            reading_backend=reading_backend,
            reading_split_mode=reading_split_mode,
            furigana_resource_path=furigana_resource_path,
            reading_overrides_path=reading_overrides_path,
        ),
        encoding="utf-8",
    )

    produced_video: Path | None = None
    if not skip_burn:
        produced_video = burn_ass_subtitles(video_source, karaoke_ass_path, final_video_path)

    return PipelineArtifacts(
        extracted_audio=extracted_audio_path if extracted_audio_path.exists() else None,
        vocals_audio=stable_vocals_path if stable_vocals_path.exists() else None,
        asr_json=stable_asr_json_path,
        aligned_json=aligned_json_path,
        line_srt=line_srt_path,
        karaoke_ass=karaoke_ass_path,
        output_video=produced_video,
    )
