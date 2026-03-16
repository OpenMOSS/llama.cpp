#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

REF_MAGIC = 0x4652474D  # "MGRF"
REF_VERSION = 1


def resolve_moss_tts_dir() -> Path:
    env_dir = os.getenv("MOSS_TTS_DIR") or os.getenv("MOSS_TTS_ROOT")
    if env_dir:
        path = Path(env_dir).expanduser().resolve()
    else:
        path = Path(__file__).resolve().parents[3] / "MOSS-TTS"

    if not path.is_dir():
        raise FileNotFoundError(
            f"MOSS-TTS repo not found: {path}. Set MOSS_TTS_DIR to the MOSS-TTS checkout root."
        )
    return path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build first-class MOSS-TTS generation input (.bin) from text (+ optional reference audio)."
    )
    ap.add_argument("--tokenizer-dir", required=True, help="Directory containing tokenizer.json")
    ap.add_argument("--output-ref", required=True, help="Output .ref.bin path")
    ap.add_argument("--language", default="zh", help="Language tag passed to prompt builder")
    ap.add_argument("--text", default="", help="Input text (optional when --text-file is used)")
    ap.add_argument("--text-file", default="", help="UTF-8 text file path")
    ap.add_argument("--reference-audio", default="", help="Optional reference wav path (24kHz preferred)")
    ap.add_argument("--encoder-onnx", default="", help="Required when --reference-audio is set")
    ap.add_argument("--decoder-onnx", default="", help="Required when --reference-audio is set")
    ap.add_argument("--cpu-audio-encode", action="store_true", help="Force CPU for ONNX reference encode")
    return ap.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.text_file:
        return Path(args.text_file).read_text(encoding="utf-8")
    if args.text:
        return args.text
    raise ValueError("either --text or --text-file is required")


def _read_reference_codes(args: argparse.Namespace) -> np.ndarray | None:
    if not args.reference_audio:
        return None
    if not args.encoder_onnx or not args.decoder_onnx:
        raise ValueError("--encoder-onnx and --decoder-onnx are required when --reference-audio is set")

    import soundfile as sf
    from moss_audio_tokenizer.onnx import OnnxAudioTokenizer

    wav, sr = sf.read(args.reference_audio, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 24000:
        raise ValueError(f"reference sample rate must be 24000, got {sr}: {args.reference_audio}")

    tokenizer = OnnxAudioTokenizer(
        encoder_path=args.encoder_onnx,
        decoder_path=args.decoder_onnx,
        use_gpu=not args.cpu_audio_encode,
    )
    codes = tokenizer.encode(wav)
    return np.asarray(codes, dtype=np.int64)


def main() -> int:
    args = parse_args()

    sys.path.insert(0, str(resolve_moss_tts_dir()))

    from moss_tts_delay.llama_cpp._constants import AUDIO_PAD_CODE
    from moss_tts_delay.llama_cpp.processor import Tokenizer, build_generation_prompt

    text = _load_text(args)
    reference_codes = _read_reference_codes(args)

    tok = Tokenizer(args.tokenizer_dir)
    input_ids = build_generation_prompt(
        tokenizer=tok,
        text=text,
        reference_codes=reference_codes,
        language=args.language,
    )

    out_ref = Path(args.output_ref)
    out_ref.parent.mkdir(parents=True, exist_ok=True)

    prompt_frames = int(input_ids.shape[0])
    n_vq = int(input_ids.shape[1] - 1)
    with out_ref.open("wb") as f:
        f.write(
            struct.pack(
                "<IIIIIII",
                REF_MAGIC,
                REF_VERSION,
                prompt_frames,
                n_vq,
                int(AUDIO_PAD_CODE),
                prompt_frames,
                0,
            )
        )
        f.write(input_ids.astype(np.int32).tobytes(order="C"))

    ref_frames = 0 if reference_codes is None else int(reference_codes.shape[0])
    print(
        f"wrote {out_ref} prompt_frames={prompt_frames} n_vq={n_vq} reference_frames={ref_frames}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
