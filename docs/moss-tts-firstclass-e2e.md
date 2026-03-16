# MOSS-TTS First-Class End-to-End Inference Pipeline

[English](moss-tts-firstclass-e2e.md) | [简体中文](moss-tts-firstclass-e2e_zh.md)

This document describes the **first-class** MOSS-TTS end-to-end inference pipeline in the current `llama.cpp` repository.

This pipeline uses:

- **llama.cpp** and `llama-moss-tts` to run the first-class MOSS-TTS-Delay GGUF model
- **ONNX Runtime** for reference-audio encoding and final waveform decoding
- **Python helper scripts** for prompt construction and end-to-end orchestration
- A local **MOSS-TTS** checkout that provides the prompt builder and ONNX tokenizer Python modules

Unlike the older `moss_tts_delay/llama_cpp` backend in the `MOSS-TTS` repository, this path moves multi-channel inputs, the transformer backbone, multi-head outputs, and delay-pattern decoding into `llama.cpp`. Python is only responsible for preparing inputs and invoking the ONNX audio tokenizer.

## Prerequisites

1. **llama.cpp** built from source with the `llama-moss-tts` target
2. **Python >= 3.10**
3. A local **MOSS-TTS** checkout, provided in any of the following ways:
   - available at `../MOSS-TTS` relative to the repository root
   - passed through `--moss-tts-dir`
   - passed through `MOSS_TTS_DIR` or `MOSS_TTS_ROOT`
4. Python packages required by the helper scripts:
   - `numpy`
   - `soundfile`
   - `onnxruntime`

## Build

```bash
cd /path/to/llama.cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-moss-tts -j
```

The resulting binary is:

- `build/bin/llama-moss-tts`

If you want to build at runtime, you can also pass `--build` to the e2e script.

## Weight Preparation

### Step 1: Prepare the first-class GGUF model

You need a first-class MOSS-TTS-Delay GGUF model that already contains:

- text embedding tables
- 32 audio embedding tables
- Qwen3 backbone weights
- a text output head
- 32 audio output heads

For example:

- `out/stage1a_moss_delay_firstclass_f16.gguf`

### Step 2: Prepare the tokenizer directory

You need a tokenizer directory containing at least:

- `tokenizer.json`

For example:

- `weights/extracted/qwen3_backbone/`

### Step 3: Prepare the ONNX audio tokenizer

You need both ONNX files:

- `encoder.onnx`
- `decoder.onnx`

For example:

- `weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx`
- `weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx`

### Step 4: Make the MOSS-TTS repository visible

The helper scripts import:

- `moss_tts_delay.llama_cpp.processor`
- `moss_audio_tokenizer.onnx`

You can provide the repository path like this:

```bash
export MOSS_TTS_DIR=/path/to/MOSS-TTS
```

or:

```bash
python tools/tts/moss-tts-firstclass-e2e.py --moss-tts-dir /path/to/MOSS-TTS ...
```

## Usage

### CLI

```bash
# Voice cloning: text + reference audio -> wav
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --moss-tts-dir /path/to/MOSS-TTS \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --output-wav /path/to/output.wav

# Direct generation without reference audio
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --moss-tts-dir /path/to/MOSS-TTS \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text "Hello, world!" \
    --output-wav /path/to/output.wav

# Build llama-moss-tts before running
python tools/tts/moss-tts-firstclass-e2e.py \
    --build \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --moss-tts-dir /path/to/MOSS-TTS \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text "Hello!" \
    --output-wav /path/to/output.wav
```

## Key Options

| Option | Values | Description |
|------|------|------|
| `--model-gguf` | path | First-class MOSS-TTS GGUF model |
| `--moss-tts-dir` | path | Local `MOSS-TTS` repository root |
| `--tokenizer-dir` | path | Directory containing `tokenizer.json` |
| `--onnx-encoder` | path | Audio tokenizer encoder ONNX |
| `--onnx-decoder` | path | Audio tokenizer decoder ONNX |
| `--text` / `--text-file` | string / path | Input text, choose exactly one |
| `--reference-audio` | path | Optional 24 kHz reference audio |
| `--language` | `zh` / `en` / tag | Language tag passed to the prompt builder |
| `--max-new-tokens` | int | Maximum generation steps |
| `--text-temperature` | float | Text-channel sampling temperature, default `1.5` |
| `--audio-temperature` | float | Audio-channel sampling temperature, default `1.7` |
| `--n-gpu-layers` | `-1` / `0` / `N` | GPU offload layers, default `-1` |
| `--audio-decoder-cpu` | flag | Force ONNX waveform decoding on CPU |
| `--cpu-audio-encode` | flag | Force ONNX reference-audio encoding on CPU |
| `--build` | flag | Build `llama-moss-tts` before running |

## Architecture

```text
Input text (+ optional reference wav)
  |
  v
moss-tts-build-generation-ref.py
  |
  |- tokenizes text with the Qwen3 tokenizer
  |- optionally encodes the reference wav into audio codes with ONNX
  |- calls the prompt builder from the local MOSS-TTS repo
  v
generation.ref.bin
  |
  v
llama-moss-tts
  |
  |- loads the first-class GGUF model
  |- performs multi-channel embedding lookup in-graph
  |- runs the Qwen3 backbone inside llama.cpp
  |- samples multi-head logits
  |- performs delay-pattern decoding in C++
  v
raw.codes.bin
  |
  v
moss-tts-audio-decode.py
  |
  |- decodes raw audio codes into waveform with ONNX
  v
wav
```

## Temporary Artifacts

The e2e script creates a temporary directory and removes it automatically after the run.

The following intermediate files are not kept:

- `generation.ref.bin`
- `raw.codes.bin`

The only visible artifact after the run is the output wav you requested.

## Output

At the end of a successful run, the script prints:

- `wav` — output path
- `wav_info` — sample rate, channel count, frame count, and duration

## File Structure

```text
llama.cpp/
├── docs/
│   ├── moss-tts-firstclass-e2e.md
│   └── moss-tts-firstclass-e2e_zh.md
├── tools/tts/
│   ├── moss-tts-firstclass-e2e.py       # End-to-end wrapper
│   ├── moss-tts-build-generation-ref.py # Prompt / input builder
│   ├── moss-tts-audio-decode.py         # ONNX audio decode helper
│   └── moss-tts.cpp                     # llama-moss-tts implementation
└── build/bin/
    └── llama-moss-tts
```
