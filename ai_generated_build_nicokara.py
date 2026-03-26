#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nicokara.alignment import build_word_level_payload
from nicokara.srt import payload_to_srt_text


def default_output_path(json_path: Path) -> Path:
    if json_path.suffix == ".json":
        return json_path.with_name(f"{json_path.stem}.nicokara.json")
    return json_path.with_name(f"{json_path.name}.nicokara.json")


def default_srt_path(output_json: Path) -> Path:
    return output_json.with_suffix(".srt")


def convert(
    json_path: str,
    lyrics_path: str,
    output_path: str | None = None,
    line_srt_path: str | None = None,
) -> tuple[Path, Path | None]:
    source = Path(json_path)
    target = Path(output_path) if output_path else default_output_path(source)

    word_payload = build_word_level_payload(json_path, lyrics_path)
    target.write_text(
        json.dumps(word_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if line_srt_path is None:
        srt_target = default_srt_path(target)
    else:
        srt_target = Path(line_srt_path)

    line_payload = {
        "lines": [
            {
                "line_id": line["line_id"],
                "text": line["text"],
                "start": line["start"],
                "end": line["end"],
            }
            for line in word_payload["lines"]
        ]
    }
    srt_target.write_text(payload_to_srt_text(line_payload), encoding="utf-8")
    return target, srt_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align ASR JSON with official lyrics and build a NicoKara-ready word-level lyric file."
        )
    )
    parser.add_argument("json", help="Input ASR JSON path")
    parser.add_argument("lyrics", help="Input official lyrics TXT path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output NicoKara JSON path (default: input JSON stem + .nicokara.json)",
    )
    parser.add_argument(
        "--line-srt",
        help="Optional line-level SRT path (default: next to the output JSON)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json, output_srt = convert(
        args.json,
        args.lyrics,
        args.output,
        line_srt_path=args.line_srt,
    )
    print(output_json)
    if output_srt is not None:
        print(output_srt)


if __name__ == "__main__":
    main()
