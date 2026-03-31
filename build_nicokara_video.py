#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from nicokara.media import ExternalToolError
from nicokara.pipeline import build_nicokara_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a NicoKara-style MP4 from a source video and official lyrics. "
            "The pipeline extracts audio, separates vocals with Demucs/UVR-style source separation, "
            "runs whisper-timestamped, aligns the recognized text with the official lyrics, "
            "renders karaoke ASS subtitles, and burns them into the source video."
        )
    )
    parser.add_argument("video", help="Input MP4 path")
    parser.add_argument("lyrics", help="Official lyrics TXT path")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory for intermediate files and final outputs (default: <video stem>.nicokara/)",
    )
    parser.add_argument(
        "--output-video",
        help="Optional final burned MP4 path (default: <output-dir>/<video stem>.nicokara.mp4)",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="Whisper model name passed to whisper-timestamped (default: small)",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="Language hint passed to whisper-timestamped (default: ja)",
    )
    parser.add_argument(
        "--whisper-device",
        help="Optional whisper-timestamped device override, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--whisper-vad",
        action="store_true",
        help="Enable VAD for whisper-timestamped",
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs_ft",
        help="Demucs model name for vocal separation (default: htdemucs_ft)",
    )
    parser.add_argument(
        "--asr-json",
        help="Reuse an existing whisper word-level JSON file instead of running ASR again",
    )
    parser.add_argument(
        "--vocals",
        help="Reuse an existing vocals WAV file instead of running source separation again",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild intermediate outputs even if cached files already exist",
    )
    parser.add_argument(
        "--skip-burn",
        action="store_true",
        help="Stop after writing ASS/SRT/JSON and skip final MP4 burning",
    )
    parser.add_argument(
        "--reading-backend",
        default="auto",
        choices=["auto", "fugashi", "sudachi", "pykakasi"],
        help="Japanese reading backend (default: auto, prefers fugashi/unidic over SudachiPy over pykakasi)",
    )
    parser.add_argument(
        "--reading-split-mode",
        default="C",
        choices=["A", "B", "C", "a", "b", "c"],
        help="Sudachi split mode when --reading-backend=sudachi (default: C)",
    )
    parser.add_argument(
        "--furigana-resource",
        help="Optional furigana lexicon path, such as JmdictFurigana json/txt(.gz)",
    )
    parser.add_argument(
        "--reading-overrides",
        help="Optional exact-reading override file path (JSON object or tab-separated TXT)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifacts = build_nicokara_video(
            args.video,
            args.lyrics,
            output_dir=args.output_dir,
            output_video_path=args.output_video,
            whisper_model=args.whisper_model,
            whisper_language=args.language,
            whisper_device=args.whisper_device,
            whisper_vad=args.whisper_vad,
            demucs_model=args.demucs_model,
            asr_json_path=args.asr_json,
            vocals_audio_path=args.vocals,
            force=args.force,
            skip_burn=args.skip_burn,
            reading_backend=args.reading_backend,
            reading_split_mode=args.reading_split_mode.upper(),
            furigana_resource_path=args.furigana_resource,
            reading_overrides_path=args.reading_overrides,
        )
    except (ExternalToolError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if artifacts.extracted_audio is not None:
        print(artifacts.extracted_audio)
    if artifacts.vocals_audio is not None:
        print(artifacts.vocals_audio)
    print(artifacts.asr_json)
    print(artifacts.aligned_json)
    print(artifacts.line_srt)
    print(artifacts.karaoke_ass)
    if artifacts.output_video is not None:
        print(artifacts.output_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
