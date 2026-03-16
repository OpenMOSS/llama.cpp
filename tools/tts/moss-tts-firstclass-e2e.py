#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, env=env, check=False)


def need_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end first-class MOSS-TTS pipeline (llama.cpp backbone + ONNX tokenizer): "
            "text(+ref) -> wav. Intermediate artifacts are stored in a temporary directory "
            "and removed automatically."
        )
    )

    parser.add_argument("--model-gguf", default=os.getenv("MODEL_GGUF", ""))
    parser.add_argument("--moss-tts-dir", default=os.getenv("MOSS_TTS_DIR", os.getenv("MOSS_TTS_ROOT", "")))
    parser.add_argument("--tokenizer-dir", default=os.getenv("TOKENIZER_DIR", ""))
    parser.add_argument("--onnx-encoder", default=os.getenv("ONNX_ENCODER", ""))
    parser.add_argument("--onnx-decoder", default=os.getenv("ONNX_DECODER", ""))
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--reference-audio", default="")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--text-temperature", type=float, default=1.5)
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--llama-bin", default="")
    parser.add_argument("--build", action="store_true", help="Build llama-moss-tts before running")
    parser.add_argument("--n-jobs", type=int, default=(os.cpu_count() or 1))
    parser.add_argument("--audio-decoder-cpu", action="store_true")
    parser.add_argument("--cpu-audio-encode", action="store_true")

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", default="")
    text_group.add_argument("--text-file", default="")

    args = parser.parse_args()

    if not args.model_gguf:
        parser.error("--model-gguf is required (or set MODEL_GGUF)")
    if not args.tokenizer_dir:
        parser.error("--tokenizer-dir is required (or set TOKENIZER_DIR)")
    if not args.onnx_encoder:
        parser.error("--onnx-encoder is required (or set ONNX_ENCODER)")
    if not args.onnx_decoder:
        parser.error("--onnx-decoder is required (or set ONNX_DECODER)")

    return args


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    build_ref_script = repo_root / "tools/tts/moss-tts-build-generation-ref.py"
    decode_script = repo_root / "tools/tts/moss-tts-audio-decode.py"
    llama_bin = Path(args.llama_bin) if args.llama_bin else repo_root / "build/bin/llama-moss-tts"

    model_gguf = Path(args.model_gguf).expanduser().resolve()
    tokenizer_dir = Path(args.tokenizer_dir).expanduser().resolve()
    onnx_encoder = Path(args.onnx_encoder).expanduser().resolve()
    onnx_decoder = Path(args.onnx_decoder).expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser().resolve()
    output_wav = Path(args.output_wav).expanduser().resolve()
    moss_tts_dir = Path(args.moss_tts_dir).expanduser().resolve() if args.moss_tts_dir else None

    need_file(python_bin, "python binary")
    need_file(model_gguf, "first-class model gguf")
    need_file(tokenizer_dir / "tokenizer.json", "tokenizer.json")
    need_file(onnx_encoder, "ONNX encoder")
    need_file(onnx_decoder, "ONNX decoder")
    need_file(build_ref_script, "generation-ref builder")
    need_file(decode_script, "audio decode helper")
    if moss_tts_dir is not None and not moss_tts_dir.is_dir():
        raise FileNotFoundError(f"missing MOSS-TTS repo: {moss_tts_dir}")
    if args.text_file:
        need_file(Path(args.text_file).expanduser().resolve(), "text file")
    if args.reference_audio:
        need_file(Path(args.reference_audio).expanduser().resolve(), "reference audio")

    if args.build:
        rc = run_cmd(["cmake", "-S", str(repo_root), "-B", str(repo_root / "build")]).returncode
        if rc != 0:
            raise RuntimeError(f"cmake configure failed with rc={rc}")
        rc = run_cmd(
            [
                "cmake",
                "--build",
                str(repo_root / "build"),
                "--target",
                "llama-moss-tts",
                "-j",
                str(args.n_jobs),
            ]
        ).returncode
        if rc != 0:
            raise RuntimeError(f"cmake build failed with rc={rc}")

    need_file(llama_bin, "llama-moss-tts binary")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    shared_env = os.environ.copy()
    if moss_tts_dir is not None:
        shared_env["MOSS_TTS_DIR"] = str(moss_tts_dir)
        old_pythonpath = shared_env.get("PYTHONPATH")
        shared_env["PYTHONPATH"] = (
            f"{moss_tts_dir}{os.pathsep}{old_pythonpath}" if old_pythonpath else str(moss_tts_dir)
        )

    with tempfile.TemporaryDirectory(prefix="moss-tts-firstclass-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        generation_ref = tmpdir_path / "generation.ref.bin"
        raw_codes = tmpdir_path / "raw.codes.bin"

        build_ref_cmd = [
            str(python_bin),
            str(build_ref_script),
            "--tokenizer-dir",
            str(tokenizer_dir),
            "--output-ref",
            str(generation_ref),
            "--language",
            args.language,
        ]
        if args.text_file:
            build_ref_cmd.extend(["--text-file", str(Path(args.text_file).expanduser().resolve())])
        else:
            build_ref_cmd.extend(["--text", args.text])

        if args.reference_audio:
            build_ref_cmd.extend(
                [
                    "--reference-audio",
                    str(Path(args.reference_audio).expanduser().resolve()),
                    "--encoder-onnx",
                    str(onnx_encoder),
                    "--decoder-onnx",
                    str(onnx_decoder),
                ]
            )
            if args.cpu_audio_encode:
                build_ref_cmd.append("--cpu-audio-encode")

        rc = run_cmd(build_ref_cmd, env=shared_env).returncode
        if rc != 0:
            raise RuntimeError(f"generation-ref build failed with rc={rc}")

        run_args = [
            str(llama_bin),
            "-m",
            str(model_gguf),
            "--generation-input",
            str(generation_ref),
            "--n-gpu-layers",
            str(args.n_gpu_layers),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--text-temperature",
            str(args.text_temperature),
            "--audio-temperature",
            str(args.audio_temperature),
            "--dump-raw-codes",
            str(raw_codes),
            "--audio-decoder-script",
            str(decode_script),
            "--audio-encoder-onnx",
            str(onnx_encoder),
            "--audio-decoder-onnx",
            str(onnx_decoder),
            "--wav-out",
            str(output_wav),
            "--python-bin",
            str(python_bin),
        ]
        if args.audio_decoder_cpu:
            run_args.append("--audio-decoder-cpu")
        llama_rc = run_cmd(run_args, env=shared_env).returncode

        if not output_wav.is_file():
            raise RuntimeError(f"llama-moss-tts did not produce wav: {output_wav} (rc={llama_rc})")
        if llama_rc != 0:
            print(
                f"warning: llama-moss-tts exited with rc={llama_rc}, but wav was produced.",
                file=sys.stderr,
            )

    with wave.open(str(output_wav), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        ch = f.getnchannels()

    print("done")
    print(f"wav     : {output_wav}")
    print(f"wav_info: sr={sr} ch={ch} frames={n} sec={n/max(sr,1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
