#!/usr/bin/env python3

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

WORKROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKROOT / "MOSS-TTS"))

from moss_tts_delay.llama_cpp.delay_state import apply_delay_pattern, extract_audio_segments  # noqa: E402


REF_MAGIC = 0x4652444D  # "MDRF"
REF_VERSION = 1


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {sys.argv[0]} <output.ref.bin>\n")
        return 1

    out_path = Path(sys.argv[1])

    n_vq = 32
    audio_pad_code = 1024
    prompt_frames = 3

    ref_prompt = np.full((prompt_frames, n_vq), audio_pad_code, dtype=np.int64)
    ref_prompt[1, 0] = 77
    ref_prompt[2, :2] = [88, 66]

    raw_a = np.stack([np.arange(10, 10 + n_vq), np.arange(110, 110 + n_vq)], axis=0).astype(np.int64)
    raw_b = np.stack([np.arange(210, 210 + n_vq)], axis=0).astype(np.int64)

    delayed_a = apply_delay_pattern(raw_a, audio_pad_code)
    delayed_b = apply_delay_pattern(raw_b, audio_pad_code)

    packed_rows: list[np.ndarray] = []
    for t in range(prompt_frames):
        row = np.full(1 + n_vq, audio_pad_code, dtype=np.int64)
        row[0] = 100 + t
        row[1:] = ref_prompt[t]
        packed_rows.append(row)

    def append_delayed(text_token: int, delayed: np.ndarray) -> None:
        for frame in delayed:
            row = np.full(1 + n_vq, audio_pad_code, dtype=np.int64)
            row[0] = text_token
            row[1:] = frame
            packed_rows.append(row)

    append_delayed(200, delayed_a)

    gap = np.full(1 + n_vq, audio_pad_code, dtype=np.int64)
    gap[0] = 201
    packed_rows.append(gap)

    append_delayed(202, delayed_b)

    packed = np.stack(packed_rows, axis=0)
    generation_audio = packed[prompt_frames:, 1:]
    segments = extract_audio_segments(generation_audio)
    raw_codes = np.concatenate(segments, axis=0) if segments else np.zeros((0, n_vq), dtype=np.int64)

    header = struct.pack(
        "<IIIIIII",
        REF_MAGIC,
        REF_VERSION,
        prompt_frames,
        n_vq,
        audio_pad_code,
        packed.shape[0],
        raw_codes.shape[0],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)
        f.write(packed.astype(np.int32).tobytes())
        f.write(raw_codes.astype(np.int32).tobytes())

    sys.stderr.write(
        "wrote decode parity reference: "
        f"packed_frames={packed.shape[0]} raw_frames={raw_codes.shape[0]} path={out_path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
