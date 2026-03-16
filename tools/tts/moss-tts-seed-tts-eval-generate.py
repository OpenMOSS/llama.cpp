#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> int:
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None, check=False).returncode


def need_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")


def need_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {name}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MOSS-TTS first-class generation over a seed-tts-eval meta list."
    )
    parser.add_argument("--meta", required=True, help="seed-tts-eval meta.lst path")
    parser.add_argument("--output-dir", required=True, help="Directory to write <utt>.wav outputs")

    parser.add_argument("--skip-generate", action="store_true")

    parser.add_argument("--model-gguf", default=os.getenv("MODEL_GGUF", ""))
    parser.add_argument("--moss-tts-dir", default=os.getenv("MOSS_TTS_DIR", os.getenv("MOSS_TTS_ROOT", "")))
    parser.add_argument("--tokenizer-dir", default=os.getenv("TOKENIZER_DIR", ""))
    parser.add_argument("--onnx-encoder", default=os.getenv("ONNX_ENCODER", ""))
    parser.add_argument("--onnx-decoder", default=os.getenv("ONNX_DECODER", ""))
    parser.add_argument("--language", default="zh")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--text-temperature", type=float, default=1.5)
    parser.add_argument("--audio-temperature", type=float, default=1.7)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--llama-bin", default="")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--audio-decoder-cpu", action="store_true")
    parser.add_argument("--cpu-audio-encode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Only synthesize the first N items when > 0")
    parser.add_argument("--skip-missing-reference", action="store_true")
    parser.add_argument("--e2e-script", default="")

    args = parser.parse_args()

    if not args.skip_generate:
        for key in ("model_gguf", "tokenizer_dir", "onnx_encoder", "onnx_decoder"):
            if not getattr(args, key):
                parser.error(f"--{key.replace('_', '-')} is required unless --skip-generate is set")

    return args


def parse_meta_line(line: str) -> tuple[str, str, str | None]:
    fields = line.rstrip("\n").split("|")
    if len(fields) == 5:
        utt, _prompt_text, prompt_wav, infer_text, _infer_wav = fields
    elif len(fields) == 4:
        utt, _prompt_text, prompt_wav, infer_text = fields
    elif len(fields) == 3:
        utt, infer_text, prompt_wav = fields
    elif len(fields) == 2:
        utt, infer_text = fields
        prompt_wav = None
    else:
        raise ValueError(f"unsupported meta format: {line.rstrip()}")

    utt = utt[:-4] if utt.endswith(".wav") else utt
    return utt, infer_text, prompt_wav


def resolve_prompt_wav(meta_path: Path, prompt_wav: str | None) -> Path | None:
    if not prompt_wav:
        return None
    path = Path(prompt_wav).expanduser()
    if not path.is_absolute():
        path = (meta_path.parent / path).resolve()
    else:
        path = path.resolve()
    return path


def build_generation_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.moss_tts_dir:
        moss_tts_dir = Path(args.moss_tts_dir).expanduser().resolve()
        need_dir(moss_tts_dir, "MOSS-TTS repo")
        env["MOSS_TTS_DIR"] = str(moss_tts_dir)
        old_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{moss_tts_dir}{os.pathsep}{old_pythonpath}" if old_pythonpath else str(moss_tts_dir)
    return env


def generate_wavs(args: argparse.Namespace, meta_path: Path, output_dir: Path, e2e_script: Path) -> None:
    env = build_generation_env(args)
    built = False
    count = 0

    for raw_line in meta_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue

        utt, infer_text, prompt_wav = parse_meta_line(raw_line)
        reference_audio = resolve_prompt_wav(meta_path, prompt_wav)
        if reference_audio is not None and not reference_audio.is_file():
            if args.skip_missing_reference:
                print(f"skip missing reference: {reference_audio}", file=sys.stderr)
                continue
            raise FileNotFoundError(f"missing reference audio: {reference_audio}")

        output_wav = output_dir / f"{utt}.wav"
        if output_wav.exists() and not args.overwrite:
            print(f"skip existing: {output_wav}", file=sys.stderr)
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
            continue

        cmd = [
            str(args.python_bin),
            str(e2e_script),
            "--model-gguf",
            args.model_gguf,
            "--tokenizer-dir",
            args.tokenizer_dir,
            "--onnx-encoder",
            args.onnx_encoder,
            "--onnx-decoder",
            args.onnx_decoder,
            "--output-wav",
            str(output_wav),
            "--language",
            args.language,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--text-temperature",
            str(args.text_temperature),
            "--audio-temperature",
            str(args.audio_temperature),
            "--n-gpu-layers",
            str(args.n_gpu_layers),
            "--python-bin",
            args.python_bin,
            "--text",
            infer_text,
        ]
        if args.moss_tts_dir:
            cmd.extend(["--moss-tts-dir", args.moss_tts_dir])
        if args.llama_bin:
            cmd.extend(["--llama-bin", args.llama_bin])
        if args.build and not built:
            cmd.append("--build")
            built = True
        if args.audio_decoder_cpu:
            cmd.append("--audio-decoder-cpu")
        if args.cpu_audio_encode:
            cmd.append("--cpu-audio-encode")
        if reference_audio is not None:
            cmd.extend(["--reference-audio", str(reference_audio)])

        rc = run_cmd(cmd, env=env)
        if rc != 0:
            raise RuntimeError(f"failed to synthesize {utt} with rc={rc}")

        count += 1
        if args.limit > 0 and count >= args.limit:
            break

    print(f"generation done: {count} items in {output_dir}")


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    e2e_script = Path(args.e2e_script).expanduser().resolve() if args.e2e_script else repo_root / "tools/tts/moss-tts-firstclass-e2e.py"
    meta_path = Path(args.meta).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    need_file(meta_path, "seed-tts-eval meta")
    need_file(e2e_script, "moss-tts firstclass e2e script")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        generate_wavs(args, meta_path, output_dir, e2e_script)

    print(f"done: meta={meta_path} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
