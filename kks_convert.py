#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from nicokara.convert import convert_payload_to_hiragana, convert_text_to_hiragana

Mode = Literal["json", "txt"]


def default_output_path(input_path: Path, mode: Mode) -> Path:
    if mode == "json":
        return input_path.with_name(f"{input_path.stem}.hira.json")
    if input_path.suffix == ".txt":
        return input_path.with_name(f"{input_path.stem}.hira.txt")
    return input_path.with_name(f"{input_path.name}.hira")


def infer_mode(input_path: Path) -> Mode:
    if input_path.suffix == ".json":
        return "json"
    if input_path.suffix == ".txt":
        return "txt"
    raise ValueError(
        f"Cannot infer mode from {input_path.name!r}. Use --mode json or --mode txt."
    )


def convert_json_file(input_path: Path, output_path: Path) -> Path:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    converted = convert_payload_to_hiragana(payload)
    output_path.write_text(
        json.dumps(converted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def convert_text_file(
    input_path: Path,
    output_path: Path,
    remove_punctuation: bool = False,
) -> Path:
    text = input_path.read_text(encoding="utf-8")
    output_path.write_text(
        convert_text_to_hiragana(
            text,
            remove_blank_lines=True,
            remove_punctuation=remove_punctuation,
        ),
        encoding="utf-8",
    )
    return output_path


def convert(
    input_path: str,
    output_path: str | None = None,
    mode: Mode | None = None,
    remove_punctuation: bool = False,
) -> Path:
    source = Path(input_path)
    resolved_mode = mode or infer_mode(source)
    target = Path(output_path) if output_path else default_output_path(source, resolved_mode)

    if resolved_mode == "json":
        return convert_json_file(source, target)
    return convert_text_file(source, target, remove_punctuation=remove_punctuation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON or plain text into hiragana with pykakasi."
    )
    parser.add_argument("input", help="Input JSON/TXT path")
    parser.add_argument(
        "--mode",
        choices=("json", "txt"),
        help="Conversion mode. Defaults to inferring from the input suffix.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: alongside input with a .hira.* suffix)",
    )
    parser.add_argument(
        "--remove-punctuation",
        action="store_true",
        help="Remove punctuation from txt output after converting to hiragana.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = convert(
        args.input,
        args.output,
        args.mode,
        remove_punctuation=args.remove_punctuation,
    )
    print(output_path)


if __name__ == "__main__":
    main()
