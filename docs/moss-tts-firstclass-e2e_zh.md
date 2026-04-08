# MOSS-TTS First-Class 端到端推理流水线

[English](moss-tts-firstclass-e2e.md) | [简体中文](moss-tts-firstclass-e2e_zh.md)

本文档说明当前 `llama.cpp` 仓库中的 **first-class** MOSS-TTS 端到端推理链路。

目前有两种运行方式：

- **推荐的原生路径**：三个模型都在 `llama.cpp` 里运行
  - `moss-tts-delay` backbone 通过 `llama_decode()`
  - `moss-tts-audio-encoder` 通过 `llama_encode()`
  - `moss-tts-audio-decoder` 通过 `llama_encode()`
- **Hybrid wrapper 路径**：backbone 在 `llama.cpp`，音频 tokenizer 仍走 ONNX，由 Python 统一编排

与 `MOSS-TTS` 仓库中较早的 `moss_tts_delay/llama_cpp` 后端不同，这条链路把多通道输入、transformer backbone、多头输出以及 delay-pattern decode 都放进了 `llama.cpp`。

## 前置条件

1. **llama.cpp** 已从源码编译，并包含 `llama-moss-tts` 目标
2. **Python >= 3.10**，如果你要使用 hybrid wrapper 或转换脚本
3. hybrid helper scripts 需要的 Python 包：
   - `numpy`
   - `soundfile`
   - `tokenizers`
   - `onnxruntime`

## 编译

### 仅 CPU 构建

```bash
cd /path/to/llama.cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-moss-tts -j
```

产物：

- `build/bin/llama-moss-tts`

### CUDA 构建

```bash
cd /path/to/llama.cpp

cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build-cuda --target llama-moss-tts -j
```

产物：

- `build-cuda/bin/llama-moss-tts`

如果你希望在 hybrid wrapper 运行时自动构建，也可以传 `--build`。

## 权重准备

### 第一步：准备 backbone GGUF

需要一个已经包含以下内容的 first-class MOSS-TTS-Delay GGUF：

- 文本 embedding 表
- 32 个音频 embedding 表
- Qwen3 backbone 权重
- 文本输出头
- 32 个音频输出头

例如：

- `out/moss_delay_firstclass_f16.gguf`

你可以直接从完整的 Hugging Face MOSS-TTS 模型目录生成它：

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS --local-dir /path/to/MOSS-TTS-hf

python convert_hf_to_gguf.py \
    /path/to/MOSS-TTS-hf \
    --outfile /path/to/moss_delay_firstclass_f16.gguf \
    --outtype f16
```

重要说明：

- 这里 `--model-gguf` 使用的是一个**特殊的 first-class MOSS-TTS-Delay GGUF**，它需要像上面这样，从完整的 `OpenMOSS-Team/MOSS-TTS` Hugging Face 模型目录直接转换得到。
- 它**不是** `OpenMOSS/MOSS-TTS-GGUF` 仓库里的通用 GGUF 文件。
- 除非某个文件被明确说明为适配这套 `llama.cpp` first-class 实现的 MOSS-TTS-Delay GGUF，否则不要把 `OpenMOSS/MOSS-TTS-GGUF` 里的文件直接拿来给这条 e2e 流水线使用。

### 第二步：准备原生 audio encoder / decoder GGUF

还需要两个额外的 GGUF 文件：

- `moss-tts-audio-encoder`
- `moss-tts-audio-decoder`

它们可以从 Hugging Face 的 `MOSS-Audio-Tokenizer` 目录转换得到：

```bash
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer --local-dir /path/to/MOSS-Audio-Tokenizer-hf

python convert_moss_audio_tokenizer_split_to_gguf.py \
    /path/to/MOSS-Audio-Tokenizer-hf \
    --outdir /path/to/out \
    --outtype f16
```

典型输出：

- `/path/to/out/moss_tts_audio_encoder_f16.gguf`
- `/path/to/out/moss_tts_audio_decoder_f16.gguf`

### 第三步：为 hybrid wrapper 准备 tokenizer 目录

需要一个至少包含以下文件的 tokenizer 目录：

- `tokenizer.json`

例如：

- `weights/extracted/qwen3_backbone/`

### 第四步：为 hybrid wrapper 准备 ONNX 音频编解码器

需要同时提供两个 ONNX 文件：

- `encoder.onnx`
- `decoder.onnx`

例如：

- `weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx`
- `weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx`

## 使用方式

### 当前原生运行方式：三个 GGUF

这是当前推荐路径。

#### CPU

```bash
# 纯文本 TTS，CPU 运行
build/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text "你好，世界！" \
    --wav-out /path/to/output.wav

# 音色克隆，CPU 运行
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
# 纯文本 TTS，GPU 运行
build-cuda/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text "你好，世界！" \
    --wav-out /path/to/output.wav \
    -ngl -1

# 音色克隆，GPU 运行
build-cuda/bin/llama-moss-tts \
    -m /path/to/moss_delay_firstclass_f16.gguf \
    --audio-encoder-model /path/to/moss_tts_audio_encoder_f16.gguf \
    --audio-decoder-model /path/to/moss_tts_audio_decoder_f16.gguf \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --wav-out /path/to/output.wav \
    -ngl -1
```

说明：

- `--reference-audio` 必须是 24 kHz 单声道 wav。
- `-ngl -1` 表示尽可能把可 offload 的层全部放到 GPU。
- 如果你使用的是 `build-cuda/bin/llama-moss-tts` 但想强制走 CPU，可以传 `-ngl 0`。

### Hybrid wrapper：backbone 走 GGUF，音频 tokenizer 走 ONNX

这条路径仍然适合做 parity 检查和中间产物调试。

#### 命令行

```bash
# 音色克隆：text + reference audio -> wav
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --output-wav /path/to/output.wav

# 不带参考音频的直接生成
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text "你好，世界！" \
    --output-wav /path/to/output.wav

# 运行前自动构建 llama-moss-tts
python tools/tts/moss-tts-firstclass-e2e.py \
    --build \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text "你好！" \
    --output-wav /path/to/output.wav
```


## 关键参数

| 参数 | 取值 | 说明 |
|------|------|------|
| `--model-gguf` | path | first-class MOSS-TTS GGUF 模型 |
| `--moss-tts-dir` | path | 已废弃的兼容参数；不再需要 |
| `--tokenizer-dir` | path | 含 `tokenizer.json` 的目录 |
| `--onnx-encoder` | path | 音频 tokenizer encoder ONNX |
| `--onnx-decoder` | path | 音频 tokenizer decoder ONNX |
| `--text` / `--text-file` | string / path | 输入文本，二选一 |
| `--reference-audio` | path | 可选参考音频；如果提供，必须是 24 kHz |
| `--language` | `zh` / `en` / tag | 传给 prompt builder 的语言标签 |
| `--max-new-tokens` | int | 最大生成步数 |
| `--text-temperature` | float | 文本通道采样温度，默认 `1.5` |
| `--audio-temperature` | float | 音频通道采样温度，默认 `1.7` |
| `--n-gpu-layers` | `-1` / `0` / `N` | GPU offload 层数，默认 `-1` |
| `--audio-decoder-cpu` | flag | 强制 ONNX 波形解码走 CPU |
| `--cpu-audio-encode` | flag | 强制 ONNX 参考音频编码走 CPU |
| `--build` | flag | 运行前构建 `llama-moss-tts` |

### 原生运行参数

| 参数 | 取值 | 说明 |
|------|------|------|
| `-m` | path | `moss-tts-delay` backbone GGUF |
| `--audio-encoder-model` | path | 原生 `moss-tts-audio-encoder` GGUF |
| `--audio-decoder-model` | path | 原生 `moss-tts-audio-decoder` GGUF |
| `--text` / `--text-file` | string / path | 输入文本，二选一 |
| `--reference-audio` | path | 可选 24 kHz reference wav |
| `--language` | `zh` / `en` / tag | 传给 prompt builder 的语言标签 |
| `--max-new-tokens` | int | 最大生成步数 |
| `--gpu-layers` / `-ngl` | `-1` / `0` / `N` | GPU offload 层数 |
| `--wav-out` | path | 输出 wav 路径 |

## 架构

### 原生三 GGUF 路径

```text
输入文本（+ 可选 reference wav）
  |
  v
llama-moss-tts
  |
  |- 文本 prompt 打包
  |- 可选：reference wav -> moss-tts-audio-encoder -> reference audio codes
  |- moss-tts-delay backbone，经由 llama_decode()
  |- 多头采样 + C++ delay-pattern decode
  |- raw audio codes -> moss-tts-audio-decoder -> waveform
  v
wav
```

### Hybrid wrapper 路径

```text
输入文本（+ 可选 reference wav）
  |
  v
moss-tts-build-generation-ref.py
  |
  |- 用 Qwen3 tokenizer 处理文本
  |- 可选：用 ONNX 把 reference wav 编成 audio codes
  |- 用仓库内置的轻量 MOSS-TTS processor 构建 packed prompt
  v
generation.ref.bin
  |
  v
llama-moss-tts
  |
  |- 加载 first-class GGUF 模型
  |- 在图内完成多通道 embedding lookup
  |- 在 llama.cpp 中执行 Qwen3 backbone
  |- 对多头 logits 做采样
  |- 在 C++ 中完成 delay-pattern decode
  v
raw.codes.bin
  |
  v
moss-tts-audio-decode.py
  |
  |- 用 ONNX 把 raw audio codes 解码成波形
  v
wav
```

## 临时产物

e2e 脚本会创建临时目录，并在流程结束后自动删除。

以下中间文件不会保留：

- `generation.ref.bin`
- `raw.codes.bin`

最终对外可见的产物只有你指定的输出 wav。

## 输出

成功结束时，脚本会打印：

- `wav` — 输出路径
- `wav_info` — 采样率、声道数、帧数和时长

## 文件结构

```text
llama.cpp/
├── docs/
│   ├── moss-tts-firstclass-e2e.md
│   └── moss-tts-firstclass-e2e_zh.md
├── convert_moss_audio_tokenizer_split_to_gguf.py
├── tools/tts/
│   ├── moss-tts-firstclass-e2e.py       # 端到端 wrapper
│   ├── moss-tts-build-generation-ref.py # prompt / input 构建器
│   ├── moss-tts-audio-decode.py         # ONNX 音频解码 helper
│   └── run-moss-tts-delay.cpp           # llama-moss-tts 实现
├── build/bin/
│   └── llama-moss-tts
└── build-cuda/bin/
    └── llama-moss-tts
```
