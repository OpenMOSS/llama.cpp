# MOSS-TTS First-Class End-to-End Inference Pipeline

[English](moss-tts-firstclass-e2e.md) | [简体中文](moss-tts-firstclass-e2e_zh.md)

This document describes the **first-class** MOSS-TTS end-to-end inference pipeline in the current `llama.cpp` repository.

There are currently two ways to run it:

- **Recommended native path**: all three models run inside `llama.cpp`
  - `moss-tts-delay` backbone via `llama_decode()`
  - `moss-tts-audio-encoder` via `llama_encode()`
  - `moss-tts-audio-decoder` via `llama_encode()`
- **Hybrid wrapper path**: backbone in `llama.cpp`, audio tokenizer in ONNX, orchestrated by Python

Unlike the older `moss_tts_delay/llama_cpp` backend in the `MOSS-TTS` repository, this path moves multi-channel inputs, the transformer backbone, multi-head outputs, and delay-pattern decoding into `llama.cpp`.

## Prerequisites

1. **llama.cpp** built from source with the `llama-moss-tts` target
2. **Python >= 3.10** if you want to use the hybrid wrapper or the converter scripts
3. Python packages required by the hybrid helper scripts:
   - `numpy`
   - `soundfile`
   - `tokenizers`
   - `onnxruntime`

## Build

### CPU-only build

```bash
cd /path/to/llama.cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-moss-tts -j
```

Binary:

- `build/bin/llama-moss-tts`

### CUDA build

```bash
cd /path/to/llama.cpp

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build-cuda --target llama-moss-tts -j
```

Binary:

- `build-cuda/bin/llama-moss-tts`

If you want to build the hybrid wrapper at runtime, you can also pass `--build` to the e2e script.

## Weight Preparation

### Step 1: Prepare the backbone GGUF

You need a first-class MOSS-TTS-Delay GGUF model that already contains:

- text embedding tables
- 32 audio embedding tables
- Qwen3 backbone weights
- a text output head
- 32 audio output heads

For example:

- `out/moss_delay_firstclass_f16.gguf`

You can generate it directly from the full Hugging Face MOSS-TTS model directory:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS --local-dir /path/to/MOSS-TTS-hf

python convert_hf_to_gguf.py \
    /path/to/MOSS-TTS-hf \
    --outfile /path/to/moss_delay_firstclass_f16.gguf \
    --outtype f16
```

Important:

- The `--model-gguf` file used by this e2e pipeline is a **special first-class MOSS-TTS-Delay GGUF** generated from the full `OpenMOSS-Team/MOSS-TTS` Hugging Face model directory with the command above.
- It is **not** the same thing as a generic GGUF downloaded from `OpenMOSS/MOSS-TTS-GGUF`.
- Do not point this pipeline at a file from `OpenMOSS/MOSS-TTS-GGUF` unless that file was explicitly produced as a first-class MOSS-TTS-Delay GGUF for this `llama.cpp` implementation.

### Step 2: Prepare the native audio encoder / decoder GGUFs

You need two additional GGUF files:

- `moss-tts-audio-encoder`
- `moss-tts-audio-decoder`

They can be generated from the Hugging Face `MOSS-Audio-Tokenizer` directory with:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir /path/to/MOSS-Audio-Tokenizer-hf

python convert_moss_audio_tokenizer_split_to_gguf.py \
    /path/to/MOSS-Audio-Tokenizer-hf \
    --outdir /path/to/out \
    --outtype f16
```

Typical outputs:

- `/path/to/out/moss_tts_audio_encoder_f16.gguf`
- `/path/to/out/moss_tts_audio_decoder_f16.gguf`

### Step 3: Prepare the tokenizer directory for the hybrid wrapper

You need a tokenizer directory containing at least:

- `tokenizer.json`

For example:

- `weights/extracted/qwen3_backbone/`

### Step 4: Prepare the ONNX audio tokenizer for the hybrid wrapper

You need both ONNX files:

- `encoder.onnx`
- `decoder.onnx`

For example:

- `weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx`
- `weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx`

## Usage

### Current Native Runtime: Three GGUFs

This is the current recommended path.

#### CPU

```bash
# Text-only TTS on CPU
build/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text "Hello, world!" \
    --wav-out /path/to/output.wav

# Voice cloning on CPU
build/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-encoder-model /path/to/moss_tts_audio_encoder_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --wav-out /path/to/output.wav
```

#### GPU

```bash
# Text-only TTS on GPU
build-cuda/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text "Hello, world!" \
    --wav-out /path/to/output.wav \
    -ngl -1

# Voice cloning on GPU
build-cuda/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-encoder-model /path/to/moss_tts_audio_encoder_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --wav-out /path/to/output.wav \
    -ngl -1
```

Notes:

- `--reference-audio` must be a 24 kHz mono wav.
- `-ngl -1` means "offload all eligible layers to GPU".
- If you built `build-cuda/bin/llama-moss-tts` but want to force CPU execution, use `-ngl 0`.

### Hybrid Wrapper: Backbone in GGUF, Audio Tokenizer in ONNX

This path remains useful for parity checks and intermediate artifact inspection.

#### CLI

```bash
# Voice cloning: text + reference audio -> wav
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --output-wav /path/to/output.wav

# Direct generation without reference audio
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text "Hello, world!" \
    --output-wav /path/to/output.wav

# Build llama-moss-tts before running
python tools/tts/moss-tts-firstclass-e2e.py \
    --build \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
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
| `--moss-tts-dir` | path | Deprecated compatibility flag; no longer required |
| `--tokenizer-dir` | path | Directory containing `tokenizer.json` |
| `--onnx-encoder` | path | Audio tokenizer encoder ONNX |
| `--onnx-decoder` | path | Audio tokenizer decoder ONNX |
| `--text` / `--text-file` | string / path | Input text, choose exactly one |
| `--reference-audio` | path | Optional reference audio; if provided, it must be 24 kHz |
| `--language` | `zh` / `en` / tag | Language tag passed to the prompt builder |
| `--max-new-tokens` | int | Maximum generation steps |
| `--text-temperature` | float | Text-channel sampling temperature, default `1.5` |
| `--audio-temperature` | float | Audio-channel sampling temperature, default `1.7` |
| `--n-gpu-layers` | `-1` / `0` / `N` | GPU offload layers, default `-1` |
| `--audio-decoder-cpu` | flag | Force ONNX waveform decoding on CPU |
| `--cpu-audio-encode` | flag | Force ONNX reference-audio encoding on CPU |
| `--build` | flag | Build `llama-moss-tts` before running |

### Native Runtime Options

| Option | Values | Description |
|------|------|------|
| `-m` | path | Backbone `moss-tts-delay` GGUF |
| `--audio-encoder-model` | path | Native `moss-tts-audio-encoder` GGUF |
| `--audio-decoder-model` | path | Native `moss-tts-audio-decoder` GGUF |
| `--text` / `--text-file` | string / path | Input text, choose exactly one |
| `--reference-audio` | path | Optional 24 kHz reference wav |
| `--language` | `zh` / `en` / tag | Language tag passed to the prompt builder |
| `--max-new-tokens` | int | Maximum generation steps |
| `--gpu-layers` / `-ngl` | `-1` / `0` / `N` | GPU offload layers |
| `--wav-out` | path | Output wav path |

## Architecture

### Native Three-GGUF Path

```text
Input text (+ optional reference wav)
  |
  v
llama-moss-tts
  |
  |- text prompt packing
  |- optional reference wav -> moss-tts-audio-encoder -> reference audio codes
  |- moss-tts-delay backbone via llama_decode()
  |- multi-head sampling + C++ delay-pattern decoding
  |- raw audio codes -> moss-tts-audio-decoder -> waveform
  v
wav
```

### Hybrid Wrapper Path

```text
Input text (+ optional reference wav)
  |
  v
moss-tts-build-generation-ref.py
  |
  |- tokenizes text with the Qwen3 tokenizer
  |- optionally encodes the reference wav into audio codes with ONNX
  |- builds the packed prompt with the local lightweight MOSS-TTS processor
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
├── convert_moss_audio_tokenizer_split_to_gguf.py
├── tools/tts/
│   ├── moss-tts-firstclass-e2e.py       # End-to-end wrapper
│   ├── moss-tts-build-generation-ref.py # Prompt / input builder
│   ├── moss-tts-audio-decode.py         # ONNX audio decode helper
│   └── run-moss-tts-delay.cpp           # llama-moss-tts implementation
├── build/bin/
│   └── llama-moss-tts
└── build-cuda/bin/
    └── llama-moss-tts
```
