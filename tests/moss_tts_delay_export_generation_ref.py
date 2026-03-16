#!/usr/bin/env python3

from __future__ import annotations

import argparse
import struct
import sys
import wave
from pathlib import Path

import numpy as np

WORKROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKROOT / "MOSS-TTS"))

from moss_tts_delay.llama_cpp._constants import AUDIO_PAD_CODE  # noqa: E402
from moss_tts_delay.llama_cpp.pipeline import LlamaCppPipeline, PipelineConfig  # noqa: E402
from moss_tts_delay.llama_cpp.processor import build_generation_prompt, parse_generation_output  # noqa: E402


REF_MAGIC = 0x4652474D  # "MGRF"
REF_VERSION = 1


def write_wav16(path: Path, wav: np.ndarray, sample_rate: int = 24000) -> None:
    wav = np.asarray(wav, dtype=np.float32).ravel()
    pcm = np.clip(np.round(wav * 32767.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description="Export Python generation reference for first-class MOSS parity")
    ap.add_argument("--config", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--output-ref", required=True)
    ap.add_argument("--output-wav", default="")
    ap.add_argument("--reference-audio", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--text-temperature", type=float, default=0.0)
    ap.add_argument("--text-top-p", type=float, default=1.0)
    ap.add_argument("--text-top-k", type=int, default=50)
    ap.add_argument("--audio-temperature", type=float, default=0.0)
    ap.add_argument("--audio-top-p", type=float, default=1.0)
    ap.add_argument("--audio-top-k", type=int, default=25)
    ap.add_argument("--audio-repetition-penalty", type=float, default=1.0)
    args = ap.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    config.max_new_tokens = args.max_new_tokens
    config.text_temperature = args.text_temperature
    config.text_top_p = args.text_top_p
    config.text_top_k = args.text_top_k
    config.audio_temperature = args.audio_temperature
    config.audio_top_p = args.audio_top_p
    config.audio_top_k = args.audio_top_k
    config.audio_repetition_penalty = args.audio_repetition_penalty

    out_ref = Path(args.output_ref)
    out_ref.parent.mkdir(parents=True, exist_ok=True)

    with LlamaCppPipeline(config) as pipeline:
        ref_codes = pipeline._prepare_reference(args.reference_audio)
        input_ids = build_generation_prompt(
            pipeline.tokenizer,
            text=args.text,
            reference_codes=ref_codes,
        )
        prompt_len = input_ids.shape[0]

        backbone = pipeline.backbone
        embedder = pipeline.embedder
        lm_heads = pipeline.lm_heads
        if backbone is None or embedder is None or lm_heads is None:
            raise RuntimeError("pipeline low-memory mode is not supported by this export script")

        backbone.clear_kv()
        pipeline._prefill(input_ids, backbone, embedder)
        generation_ids = pipeline._autoregressive_loop(
            input_ids, config.max_new_tokens, backbone, embedder, lm_heads
        )
        _text, audio_codes = parse_generation_output(pipeline.tokenizer, generation_ids, prompt_len)

        if args.output_wav:
            wav = pipeline.audio_tokenizer.decode(audio_codes)
            write_wav16(Path(args.output_wav), wav, 24000)

    hdr = struct.pack(
        "<IIIIIII",
        REF_MAGIC,
        REF_VERSION,
        prompt_len,
        input_ids.shape[1] - 1,
        AUDIO_PAD_CODE,
        prompt_len,
        audio_codes.shape[0],
    )

    with out_ref.open("wb") as f:
        f.write(hdr)
        f.write(input_ids.astype(np.int32).tobytes())
        f.write(audio_codes.astype(np.int32).tobytes())

    sys.stderr.write(
        "wrote generation reference: "
        f"prompt_frames={prompt_len} raw_frames={audio_codes.shape[0]} ref={out_ref}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
