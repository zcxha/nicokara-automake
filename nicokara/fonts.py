from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_KARAOKE_FONT_NAME = os.environ.get("NICOKARA_KARAOKE_FONT_NAME", "Noto Sans CJK JP")
DEFAULT_RUBY_FONT_NAME = os.environ.get("NICOKARA_RUBY_FONT_NAME", DEFAULT_KARAOKE_FONT_NAME)
FONT_FILE_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}


def _deduplicate_paths(paths: Iterable[Path]) -> list[Path]:
    """Return paths in order while removing duplicates and missing entries."""
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser()
        if not resolved.exists():
            continue
        normalized = resolved.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)
    return unique_paths


def _iter_env_font_dirs() -> list[Path]:
    """Read additional font directories from NICOKARA_FONT_DIR."""
    raw_value = os.environ.get("NICOKARA_FONT_DIR", "").strip()
    if not raw_value:
        return []
    return [
        Path(part)
        for part in raw_value.split(os.pathsep)
        if part.strip()
    ]


def discover_font_dirs(*, ass_path: Path | None = None) -> list[Path]:
    """Discover project-local font directories that should override system fonts."""
    package_root = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        *_iter_env_font_dirs(),
        package_root / "fonts",
        Path.cwd() / "fonts",
    ]
    if ass_path is not None:
        candidate_dirs.append(ass_path.resolve().parent / "fonts")
    return _deduplicate_paths(candidate_dirs)


def _build_fontconfig_xml(font_dirs: Iterable[Path], cache_dir: Path) -> str:
    """Build a minimal fontconfig document for the provided directories."""
    dir_nodes = "\n".join(f"  <dir>{path}</dir>" for path in font_dirs)
    return (
        "<?xml version=\"1.0\"?>\n"
        "<!DOCTYPE fontconfig SYSTEM \"fonts.dtd\">\n"
        "<fontconfig>\n"
        f"{dir_nodes}\n"
        f"  <cachedir>{cache_dir}</cachedir>\n"
        "</fontconfig>\n"
    )


@contextmanager
def fontconfig_environment(font_dirs: Iterable[Path]) -> Iterator[dict[str, str] | None]:
    """Yield an isolated fontconfig environment when custom font directories exist."""
    resolved_dirs = _deduplicate_paths(font_dirs)
    if not resolved_dirs:
        yield None
        return

    with tempfile.TemporaryDirectory(prefix="nicokara-fontconfig-") as temp_dir:
        temp_path = Path(temp_dir)
        config_path = temp_path / "fonts.conf"
        cache_dir = temp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_build_fontconfig_xml(resolved_dirs, cache_dir), encoding="utf-8")

        env = dict(os.environ)
        env["FONTCONFIG_FILE"] = str(config_path)
        env["FONTCONFIG_PATH"] = temp_dir
        yield env


@lru_cache(maxsize=32)
def _resolve_font_path_cached(font_name: str, font_dirs_key: tuple[str, ...]) -> str | None:
    """Resolve a font family name to a concrete file path via fontconfig."""
    font_dirs = [Path(path) for path in font_dirs_key]
    with fontconfig_environment(font_dirs) as env:
        try:
            probe = subprocess.run(
                ["fc-match", "-f", "%{file}\n", font_name],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        except OSError:
            return None

    if probe.returncode != 0:
        return None
    candidates = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    return candidates[0] if candidates else None


def resolve_font_path(font_name: str, *, ass_path: Path | None = None) -> Path | None:
    """Resolve a font family name using project-local fonts before system fonts."""
    font_dirs = discover_font_dirs(ass_path=ass_path)
    resolved = _resolve_font_path_cached(font_name, tuple(str(path) for path in font_dirs))
    return Path(resolved) if resolved else None


def _iter_font_files(font_dir: Path) -> Iterator[Path]:
    """Iterate over font files stored under a directory."""
    for path in sorted(font_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in FONT_FILE_EXTENSIONS:
            yield path.resolve()


def collect_font_files(font_names: Iterable[str], *, ass_path: Path | None = None) -> list[Path]:
    """Collect explicit font files required for ASS measurement and rendering."""
    font_files: list[Path] = []
    for font_name in font_names:
        resolved_path = resolve_font_path(font_name, ass_path=ass_path)
        if resolved_path is not None:
            font_files.append(resolved_path)

    for font_dir in discover_font_dirs(ass_path=ass_path):
        font_files.extend(_iter_font_files(font_dir))

    return _deduplicate_paths(font_files)


@contextmanager
def bundled_font_environment(
    font_names: Iterable[str],
    *,
    ass_path: Path | None = None,
) -> Iterator[tuple[Path | None, dict[str, str] | None]]:
    """Bundle required fonts into one directory for libass and yield its environment."""
    font_files = collect_font_files(font_names, ass_path=ass_path)
    if not font_files:
        yield None, None
        return

    with tempfile.TemporaryDirectory(prefix="nicokara-font-bundle-") as temp_dir:
        bundle_dir = Path(temp_dir) / "fonts"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        for index, font_file in enumerate(font_files):
            target = bundle_dir / f"{index:02d}_{font_file.name}"
            try:
                target.symlink_to(font_file)
            except OSError:
                shutil.copy2(font_file, target)

        with fontconfig_environment([bundle_dir]) as env:
            yield bundle_dir, env
