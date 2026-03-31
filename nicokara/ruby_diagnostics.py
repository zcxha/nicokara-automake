from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .convert import has_kanji


@dataclass
class CandidateAggregate:
    text: str
    ruby_text: str
    ruby_source: str
    reasons: set[str] = field(default_factory=set)
    count: int = 0
    min_coverage: float = 1.0
    max_coverage: float = 0.0
    files: set[str] = field(default_factory=set)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def add(self, candidate: dict[str, Any], *, example_limit: int) -> None:
        self.count += 1
        self.reasons.update(candidate["reasons"])
        self.min_coverage = min(self.min_coverage, candidate["coverage"])
        self.max_coverage = max(self.max_coverage, candidate["coverage"])
        self.files.add(candidate["source_file"])
        if len(self.examples) < example_limit:
            self.examples.append(
                {
                    "source_file": candidate["source_file"],
                    "line_id": candidate["line_id"],
                    "line_text": candidate["line_text"],
                    "coverage": candidate["coverage"],
                    "start": candidate["start"],
                    "end": candidate["end"],
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "ruby_text": self.ruby_text,
            "ruby_source": self.ruby_source,
            "count": self.count,
            "reasons": sorted(self.reasons),
            "min_coverage": round(self.min_coverage, 4),
            "max_coverage": round(self.max_coverage, 4),
            "files": sorted(self.files),
            "examples": self.examples,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export suspicious furigana entries from one or more *.nicokara.json files. "
            "By default, kanji words are flagged when the ruby is backend-generated, "
            "missing, or the alignment coverage is below the threshold."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input *.nicokara.json files or directories containing them",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "tsv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.85,
        help="Flag entries below this alignment coverage (default: 0.85)",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=3,
        help="Maximum example occurrences to keep per unique entry (default: 3)",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Emit every suspicious occurrence instead of grouping by text/ruby/source",
    )
    parser.add_argument(
        "--overrides-template",
        help="Optional path to write a starter overrides JSON mapping text -> ruby_text",
    )
    return parser.parse_args()


def iter_input_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        source = Path(raw_path)
        if source.is_dir():
            files.extend(sorted(source.rglob("*.nicokara.json")))
            continue
        if source.is_file():
            files.append(source)
            continue
        raise FileNotFoundError(f"Input `{source}` does not exist.")
    return files


def collect_candidates(
    payload_path: Path,
    *,
    min_coverage: float,
) -> list[dict[str, Any]]:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    line_map = {
        int(line.get("line_id", -1)): str(line.get("text", ""))
        for line in payload.get("lines", [])
        if isinstance(line, dict)
    }
    reading_backend = str(payload.get("alignment", {}).get("reading_backend", ""))

    candidates: list[dict[str, Any]] = []
    for word in payload.get("words", []):
        if not isinstance(word, dict):
            continue

        text = str(word.get("text", ""))
        if not has_kanji(text):
            continue

        ruby_text = str(word.get("ruby_text", "")).strip()
        ruby_source = str(word.get("ruby_source", "")).strip() or "unknown"
        coverage = float(word.get("coverage", 0.0) or 0.0)
        reasons = build_reasons(
            ruby_text=ruby_text,
            ruby_source=ruby_source,
            coverage=coverage,
            min_coverage=min_coverage,
        )
        if not reasons:
            continue

        line_id = int(word.get("line_id", -1))
        candidates.append(
            {
                "source_file": str(payload_path),
                "reading_backend": reading_backend,
                "line_id": line_id,
                "word_id": int(word.get("word_id", -1)),
                "line_text": line_map.get(line_id, ""),
                "text": text,
                "ruby_text": ruby_text,
                "ruby_source": ruby_source,
                "coverage": coverage,
                "matched_chars": int(word.get("matched_chars", 0) or 0),
                "total_chars": int(word.get("total_chars", 0) or 0),
                "start": word.get("start"),
                "end": word.get("end"),
                "reasons": reasons,
            }
        )

    return candidates


def build_reasons(
    *,
    ruby_text: str,
    ruby_source: str,
    coverage: float,
    min_coverage: float,
) -> list[str]:
    reasons = []
    if not ruby_text:
        reasons.append("missing_ruby")
    if coverage < min_coverage:
        reasons.append("low_coverage")
    if ruby_source == "backend":
        reasons.append("backend_generated")
    return reasons


def dedupe_candidates(
    candidates: list[dict[str, Any]],
    *,
    example_limit: int,
) -> list[dict[str, Any]]:
    aggregates: dict[tuple[str, str, str], CandidateAggregate] = {}
    for candidate in candidates:
        key = (
            candidate["text"],
            candidate["ruby_text"],
            candidate["ruby_source"],
        )
        aggregate = aggregates.get(key)
        if aggregate is None:
            aggregate = CandidateAggregate(
                text=candidate["text"],
                ruby_text=candidate["ruby_text"],
                ruby_source=candidate["ruby_source"],
            )
            aggregates[key] = aggregate
        aggregate.add(candidate, example_limit=example_limit)

    return sorted(
        (aggregate.to_dict() for aggregate in aggregates.values()),
        key=lambda item: (
            -int(item["count"]),
            float(item["min_coverage"]),
            item["text"],
        ),
    )


def write_output(
    rows: list[dict[str, Any]],
    *,
    output_path: str | None,
    output_format: str,
) -> None:
    if output_format == "json":
        text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        if output_path:
            Path(output_path).write_text(text, encoding="utf-8")
        else:
            print(text, end="")
        return

    fieldnames = [
        "text",
        "ruby_text",
        "ruby_source",
        "count",
        "min_coverage",
        "max_coverage",
        "reasons",
        "files",
    ]
    if rows and "source_file" in rows[0]:
        fieldnames = [
            "source_file",
            "reading_backend",
            "line_id",
            "word_id",
            "line_text",
            "text",
            "ruby_text",
            "ruby_source",
            "coverage",
            "matched_chars",
            "total_chars",
            "start",
            "end",
            "reasons",
        ]

    if output_path:
        handle = Path(output_path).open("w", encoding="utf-8", newline="")
        close_handle = True
    else:
        import sys

        handle = sys.stdout
        close_handle = False

    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            for key in ("reasons", "files", "examples"):
                if key in serializable and isinstance(serializable[key], (list, set)):
                    serializable[key] = json.dumps(
                        sorted(serializable[key]) if isinstance(serializable[key], set) else serializable[key],
                        ensure_ascii=False,
                    )
            writer.writerow({name: serializable.get(name, "") for name in fieldnames})
    finally:
        if close_handle:
            handle.close()


def write_overrides_template(rows: list[dict[str, Any]], output_path: str) -> None:
    overrides: dict[str, str] = {}
    for row in rows:
        text = str(row.get("text", "")).strip()
        ruby_text = str(row.get("ruby_text", "")).strip()
        if text and ruby_text:
            overrides.setdefault(text, ruby_text)
    Path(output_path).write_text(
        json.dumps(overrides, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    input_files = iter_input_files(args.inputs)

    candidates = []
    for payload_path in input_files:
        candidates.extend(
            collect_candidates(
                payload_path,
                min_coverage=args.min_coverage,
            )
        )

    if args.no_dedupe:
        rows = sorted(
            candidates,
            key=lambda item: (
                float(item["coverage"]),
                item["text"],
                item["source_file"],
                int(item["line_id"]),
                int(item["word_id"]),
            ),
        )
    else:
        rows = dedupe_candidates(candidates, example_limit=args.example_limit)

    write_output(rows, output_path=args.output, output_format=args.format)

    if args.overrides_template:
        write_overrides_template(rows, args.overrides_template)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
