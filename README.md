# Nicokara Pipeline

Chinese docs:

- [中文架构说明](./docs/中文架构说明.md)
- [中文使用说明](./docs/中文使用说明.md)

This repository can now build a NicoKara-style finished video from:

- an input `mp4`
- an official lyric `txt`

The full pipeline is:

1. extract audio from the MP4
2. separate vocals with Demucs (UVR-style source separation)
3. transcribe the vocals with `whisper-timestamped`
4. align the recognized text back to the official lyrics
5. render karaoke ASS subtitles with furigana and progressive highlight
6. burn the subtitles into the original video with `ffmpeg`

## Main entrypoint

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt"
```

By default this writes a directory named like `input.nicokara/` containing:

- `*.audio.wav`: audio extracted from the video
- `*.vocals.wav`: separated vocal stem
- `*.words.json`: whisper word-level ASR output
- `*.nicokara.json`: aligned official-lyrics timing payload
- `*.nicokara.srt`: line-level SRT
- `*.karaoke.ass`: karaoke ASS subtitles
- `*.nicokara.mp4`: final burned NicoKara video

## Reusing Existing Intermediate Files

If you already have separated vocals or an ASR JSON, you can skip those expensive steps:

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --vocals "song.vocals.wav"
```

```bash
python3 build_nicokara_video.py \
  "input.mp4" \
  "lyrics.txt" \
  --asr-json "song.vocals.wav.words.json"
```

If you only want the aligned JSON, legacy alignment still works:

```bash
python3 ai_generated_build_nicokara.py \
  "song.vocals.wav.words.json" \
  "lyrics.txt"
```

## Requirements

- `ffmpeg` and `ffprobe` must be available in `PATH`
- `whisper_timestamped` should be available in `PATH`
- `demucs` should be available in `PATH`, or `uvx`/`uv` should be available so the pipeline can launch Demucs automatically
- Python needs `pykakasi`

Typical setup commands:

```bash
uv tool install --with packaging --with setuptools whisper-timestamped
uv tool install --with torchcodec demucs
uv pip install --python .venv/bin/python pykakasi
```

If your machine does not have a working CUDA runtime, prefer:

```bash
uv tool install --force --torch-backend cpu --with packaging --with setuptools whisper-timestamped
uv tool install --force --torch-backend cpu --with torchcodec demucs
```

## Notes On Lyric Segmentation

- If a lyric line already contains spaces, those spaces are treated as strong karaoke boundaries.
- If a lyric line has no spaces, the project now uses `pykakasi` conversion chunks to infer smaller highlight units automatically.
- This means Japanese lyrics without manual spacing no longer collapse into one giant highlighted block.
- If a word contains kanji, the ASS output now also renders a furigana line above the main karaoke line.

## Utility scripts

- `kks_convert.py`: convert JSON or TXT into hiragana
- `lyrics_line_timestamp.py`: generate line-level lyric timing JSON
- `json_convert_srt.py`: convert `segments`, `lyrics`, or `lines` JSON into SRT
- `ai_generated_json_to_srt.py`: thin wrapper for the SRT converter

## Output Format

The NicoKara JSON contains:

- `lines`: per-line timing and coverage
- `words`: karaoke highlight units derived from the official lyrics
- `alignment`: overall match statistics

The texts come from the official lyric file, while the timestamps are inferred from the ASR word timings via monotonic character-level alignment.
