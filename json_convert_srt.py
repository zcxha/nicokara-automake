#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nicokara.srt import payload_to_srt_text


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".json":
        return input_path.with_suffix(".srt")
    return input_path.with_name(f"{input_path.name}.srt")


def convert(input_path: str, output_path: str | None = None) -> Path:
    source = Path(input_path)
    target = Path(output_path) if output_path else default_output_path(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    target.write_text(payload_to_srt_text(payload), encoding="utf-8")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSON file with segments/lyrics/lines into an SRT subtitle file."
    )
    parser.add_argument("input", help="Input JSON path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output SRT path (default: alongside input with .srt suffix)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert(args.input, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
