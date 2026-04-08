#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
import wave
from pathlib import Path

import numpy as np

from moss_tts_onnx import OnnxAudioTokenizer
from moss_tts_processor import N_VQ, SAMPLE_RATE


CODES_MAGIC = 0x53444F43  # "CODS"
CODES_VERSION = 1


def read_codes(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        hdr = f.read(16)
        if len(hdr) != 16:
            raise RuntimeError("codes header is truncated")
        magic, version, n_frames, n_vq = struct.unpack("<IIII", hdr)
        if magic != CODES_MAGIC or version != CODES_VERSION:
            raise RuntimeError("unexpected codes file format")
        payload = np.frombuffer(f.read(), dtype=np.int32)

    expected = n_frames * n_vq
    if payload.size != expected:
        raise RuntimeError(f"codes payload size mismatch: got {payload.size}, expected {expected}")

    codes = payload.reshape(n_frames, n_vq).astype(np.int64)
    return codes


def write_wav16(path: Path, wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    wav = np.asarray(wav, dtype=np.float32).ravel()
    pcm = np.clip(np.round(wav * 32767.0), -32768, 32767).astype(np.int16)

    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Decode MOSS raw audio codes to wav via Python audio tokenizer")
    ap.add_argument("--codes-bin", required=True)
    ap.add_argument("--wav-out", required=True)
    ap.add_argument("--encoder-onnx", required=True)
    ap.add_argument("--decoder-onnx", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    codes = read_codes(Path(args.codes_bin))
    if codes.ndim != 2 or codes.shape[1] != N_VQ:
        raise RuntimeError(f"expected raw codes with shape (T, {N_VQ}), got {codes.shape}")

    tokenizer = OnnxAudioTokenizer(
        encoder_path=args.encoder_onnx,
        decoder_path=args.decoder_onnx,
        use_gpu=not args.cpu,
    )
    wav = tokenizer.decode(codes)
    write_wav16(Path(args.wav_out), wav, SAMPLE_RATE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
