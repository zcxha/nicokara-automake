from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import torch


def _to_pcm16_bytes(wav: torch.Tensor) -> bytes:
    """Convert a Demucs output tensor into interleaved PCM16 WAV payload bytes."""
    clipped = wav.detach().cpu().clamp(-1.0, 1.0)
    pcm = (clipped * 32767.0).round().to(torch.int16)
    interleaved = pcm.transpose(0, 1).contiguous().numpy()
    if interleaved.dtype != np.int16:
        interleaved = interleaved.astype(np.int16, copy=False)
    return interleaved.tobytes()


def _save_wav_audio(
    wav: torch.Tensor,
    path: str | Path,
    samplerate: int,
    bitrate: int = 320,
    clip: str = "rescale",
    bits_per_sample: int = 16,
    as_float: bool = False,
    preset: int = 2,
) -> None:
    """Save Demucs output as PCM16 WAV without relying on TorchCodec."""
    del bitrate
    del preset

    if Path(path).suffix.lower() != ".wav":
        raise ValueError("The nicokara Demucs runner currently supports WAV output only.")
    if bits_per_sample != 16:
        raise ValueError("The nicokara Demucs runner currently supports 16-bit WAV output only.")
    if as_float:
        raise ValueError("The nicokara Demucs runner does not support float32 WAV output.")

    from demucs.audio import prevent_clip

    wav = prevent_clip(wav, mode=clip)
    audio_path = Path(path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(int(wav.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(samplerate))
        handle.writeframes(_to_pcm16_bytes(wav))


def _patch_demucs_save_audio() -> None:
    """Replace Demucs audio saving helpers with the local TorchCodec-free implementation."""
    import demucs.audio
    import demucs.separate

    demucs.audio.save_audio = _save_wav_audio
    demucs.separate.save_audio = _save_wav_audio


def _resolve_device(device: str | None) -> str:
    """Resolve the Demucs execution device, defaulting to CUDA when available."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create a lightweight parser that matches the subset of Demucs CLI flags used by the project."""
    parser = argparse.ArgumentParser(
        prog="python -m nicokara.demucs_runner",
        description="Run Demucs with a WAV writer that avoids TorchCodec runtime issues.",
    )
    parser.add_argument("tracks", nargs="+", type=Path, help="Audio tracks to separate.")
    parser.add_argument("--two-stems", dest="stem", metavar="STEM", help="Only separate one stem.")
    parser.add_argument("-n", "--name", "--model", dest="name", default="htdemucs_ft", help="Demucs model name.")
    parser.add_argument("-o", "--out", type=Path, default=Path("separated"), help="Output directory.")
    parser.add_argument("-d", "--device", default=None, help="Optional device override, e.g. cuda or cpu.")
    parser.add_argument("--shifts", type=int, default=1, help="Number of equivariant shifts.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap between chunks.")
    parser.add_argument("--no-split", action="store_false", dest="split", default=True, help="Disable chunk split.")
    parser.add_argument("--segment", type=int, help="Chunk size override.")
    parser.add_argument("--clip-mode", default="rescale", choices=["rescale", "clamp"], help="Clipping strategy.")
    parser.add_argument("-j", "--jobs", type=int, default=0, help="Worker count.")
    parser.add_argument("--filename", default="{track}/{stem}.{ext}", help="Output naming template.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Demucs separation after patching its audio writer for TorchCodec-free WAV output."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    _patch_demucs_save_audio()
    resolved_device = _resolve_device(args.device)
    print(f"Using Demucs device: {resolved_device}")

    demucs_args: list[str] = []
    if args.stem is not None:
        demucs_args.extend(["--two-stems", args.stem])
    demucs_args.extend(["-n", args.name, "-o", str(args.out)])
    demucs_args.extend(["-d", resolved_device])
    demucs_args.extend(["--shifts", str(args.shifts), "--overlap", str(args.overlap)])
    if not args.split:
        demucs_args.append("--no-split")
    if args.segment is not None:
        demucs_args.extend(["--segment", str(args.segment)])
    if args.clip_mode:
        demucs_args.extend(["--clip-mode", args.clip_mode])
    demucs_args.extend(["-j", str(args.jobs), "--filename", args.filename])
    demucs_args.extend(str(track) for track in args.tracks)

    from demucs.separate import main as demucs_main

    demucs_main(demucs_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
