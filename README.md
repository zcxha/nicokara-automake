# Nicokara Pipeline

This repository builds a NicoKara-style karaoke video from:

- an input `mp4`
- an official lyric `txt`

It is designed for two audiences at the same time:

- people who just want a command they can run
- people who also want to understand what each step is doing

Chinese docs:

- [中文使用说明](./docs/中文使用说明.md)
- [中文架构说明](./docs/中文架构说明.md)

## What This Project Produces

The pipeline can generate:

- aligned lyric timing data
- line subtitles in `SRT`
- karaoke subtitles in `ASS`
- a final burned `MP4`

Typical output files inside `input.nicokara/`:

- `*.audio.wav`: audio extracted from the video
- `*.vocals.wav`: separated vocal stem
- `*.words.json`: word-level ASR output from `whisper-timestamped`
- `*.nicokara.json`: official lyric timing payload after alignment
- `*.nicokara.srt`: line-level subtitles
- `*.karaoke.ass`: karaoke subtitles with furigana and progressive highlight
- `*.nicokara.mp4`: final burned video

The default `ASS` layout now follows a more typical NicoKara presentation:

- larger main lyric text
- larger furigana
- the active lyric area sits a bit above the absolute bottom edge
- consecutive lines alternate between an upper-left slot and a lower-right slot so one line can sing while the next line is already visible

## The Big Picture

The full pipeline is:

1. extract audio from the MP4
2. separate vocals with Demucs
3. transcribe the vocals with `whisper-timestamped`
4. align the recognized text back to the official lyrics
5. render karaoke `ASS` subtitles with furigana and progressive highlight
6. burn the subtitles into the original video with `ffmpeg`

If you are new to the terminology, here is the short version:

- `ffmpeg`: a command-line media toolbox that reads, converts, and writes video/audio
- Demucs: a model that tries to split singing voice from accompaniment
- ASR: automatic speech recognition; here it means turning sung audio into timed words
- alignment: matching the timed ASR words back onto the correct official lyric text
- `SRT`: a simple subtitle format, good for plain lines
- `ASS`: a richer subtitle format that supports karaoke highlight and detailed positioning
- furigana / ruby: small kana shown above kanji to explain pronunciation

## Quick Start

Project dependencies:

```bash
uv sync
```

Activate the project virtual environment first, then install the inference tools into that same environment.
This repository now recommends `uv pip install ... --torch-backend ...` for these packages, because backend selection for CPU / CUDA is explicit there.

```bash
uv pip install --torch-backend auto whisper-timestamped packaging setuptools demucs torchcodec
```

If you want NVIDIA GPU inference, keep the `auto` backend above or replace it with an explicit backend like `cu126` / `cu124`.
After installation, verify that PyTorch can actually see CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If your machine does not have a working CUDA runtime, prefer a CPU environment instead:

```bash
uv pip install --torch-backend cpu whisper-timestamped packaging setuptools demucs torchcodec
```

Then run:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt"
```

You can also use the installed script entrypoint:

```bash
nicokara-build "input.mp4" "lyrics.txt"
```

If you installed a CUDA-enabled PyTorch build, you can ask Whisper to use the GPU explicitly:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --whisper-device cuda
```

## Reusing Existing Intermediate Files

If you already have a vocal stem, skip source separation:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --vocals "song.vocals.wav"
```

If you already have a whisper JSON, skip separation and ASR:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --asr-json "song.vocals.wav.words.json"
```

If you want to iterate on alignment and subtitle styling without burning video again:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --asr-json "song.vocals.wav.words.json" \
  --skip-burn
```

## Requirements

System tools:

- `ffmpeg`
- `ffprobe`

Runtime tools:

- `whisper_timestamped` installed in the active virtual environment or otherwise available in `PATH`
- `demucs` installed in the active virtual environment or otherwise available in `PATH`

On Windows, recent `torchaudio` releases may route Demucs WAV output through `torchcodec`.
The pipeline now runs Demucs through a local compatibility runner so normal `vocals.wav` output can still succeed even when TorchCodec saving is flaky.

Python dependencies:

- `fugashi[unidic-lite]`
- `Pillow`
- `pykakasi`

Optional Python dependencies:

- `SudachiPy`
- `sudachidict_core`

If you want the Sudachi backend:

```bash
uv sync --extra sudachi
```

## Why The Pipeline Uses “Recognized Words + Official Lyrics”

This is the core idea of the project.

Speech recognition is good at estimating time, but it is not always reliable at spelling lyrics perfectly, especially for:

- homophones
- rare names
- stylized lyric writing
- elongated sung vowels

Official lyrics are textually correct, but they do not contain timing.

So the project combines them:

- ASR provides approximate time anchors
- official lyrics provide the final text
- alignment merges the two into usable karaoke timing

This is why the project can often produce better subtitles than either source alone.

## Font Consistency For `ASS` Burn-In

Ruby placement in `*.karaoke.ass` is now measured with `ffmpeg/libass` itself instead of a separate Pillow width estimate.
This makes the generated ruby positions match final burn-in more closely, but it also means ASS generation is a bit slower than before.
To keep the measurement step and the final burn-in step consistent, both stages should see the same font files.

The easiest setup is:

1. put the font file into repo-root `fonts/`
2. make sure the `ASS` font family name matches the font's internal family name
3. run the pipeline normally

The repository already includes a prepared example:

- font file: `fonts/NotoSansCJKjp-Regular.otf`
- internal family name: `Noto Sans CJK JP`
- the default config already uses `Noto Sans CJK JP`

Useful environment variables:

- `NICOKARA_FONT_DIR=/path/to/fonts`
- `NICOKARA_KARAOKE_FONT_NAME="Noto Sans CJK JP"`
- `NICOKARA_RUBY_FONT_NAME="Noto Sans CJK JP"`

If you want to force a different font family:

```bash
export NICOKARA_KARAOKE_FONT_NAME="Your Font Family"
export NICOKARA_RUBY_FONT_NAME="Your Font Family"
python3 build_nicokara_video.py "input.mp4" "lyrics.txt"
```

## Reading Backends

Japanese lyrics without spaces need extra help for segmentation and furigana.

The project supports:

- `auto`
- `fugashi`
- `sudachi`
- `pykakasi`

Example:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --reading-backend auto
```

Useful options:

- `--reading-backend auto|fugashi|sudachi|pykakasi`
- `--reading-split-mode A|B|C` for Sudachi
- `--furigana-resource /path/to/JmdictFurigana.json.gz`
- `--reading-overrides /path/to/overrides.json`

Notes:

- `auto` prefers `fugashi + UniDic`, can fall back to `SudachiPy`, and only uses `pykakasi` as a weak backup
- first run may download and cache official `JmdictFurigana` / `JmnedictFurigana` resources
- set `NICOKARA_DISABLE_AUTO_FURIGANA_DOWNLOAD=1` to force offline behavior

## Ruby Diagnostics

After generating `*.nicokara.json`, you can inspect entries that are likely worth manual review:

```bash
nicokara-ruby-diagnostics song.nicokara.json > suspicious_ruby.json
```

You can also generate a starter overrides file:

```bash
nicokara-ruby-diagnostics \
  song.nicokara.json \
  --overrides-template ruby_overrides.json
```

## Output Format

The NicoKara JSON contains:

- `lines`: per-line timing and coverage
- `words`: karaoke highlight units derived from the official lyrics, including `ruby_text`, `ruby_parts`, and `ruby_source`
- `alignment`: overall match statistics, including the effective `reading_backend`

The texts come from the official lyric file, while the timestamps are inferred from ASR word timings through monotonic character-level alignment.

## Where To Learn More

If you want a beginner-friendly explanation of the concepts, read:

- [中文使用说明](./docs/中文使用说明.md)

If you want the internal design and data flow, read:

- [中文架构说明](./docs/中文架构说明.md)
