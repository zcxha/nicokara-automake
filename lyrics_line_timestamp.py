#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nicokara.alignment import build_line_level_payload


def default_output_path(json_path: Path) -> Path:
    if json_path.suffix == ".json":
        return json_path.with_name(f"{json_path.stem}.lyric_lines.json")
    return json_path.with_name(f"{json_path.name}.lyric_lines.json")


def convert(json_path: str, lyrics_path: str, output_path: str | None = None) -> Path:
    source = Path(json_path)
    target = Path(output_path) if output_path else default_output_path(source)
    payload = build_line_level_payload(json_path, lyrics_path)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align ASR word characters to lyric lines and estimate line-level timestamps."
    )
    parser.add_argument("json", help="Input word-level JSON path")
    parser.add_argument("lyrics", help="Input lyric txt path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output JSON path (default: input JSON stem + .lyric_lines.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert(args.json, args.lyrics, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
