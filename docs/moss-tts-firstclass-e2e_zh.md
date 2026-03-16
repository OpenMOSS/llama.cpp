# MOSS-TTS First-Class 端到端推理流水线

[English](moss-tts-firstclass-e2e.md) | [简体中文](moss-tts-firstclass-e2e_zh.md)

本文档说明当前 `llama.cpp` 仓库中的 **first-class** MOSS-TTS 端到端推理链路。

这条链路使用：

- **llama.cpp** 和 `llama-moss-tts` 运行 first-class MOSS-TTS-Delay GGUF 模型
- **ONNX Runtime** 完成参考音频编码和最终波形解码
- **Python helper scripts** 负责 prompt 构建和整条链路编排
- 本地 **MOSS-TTS** 仓库 checkout 提供 prompt builder 和 ONNX tokenizer Python 模块

与 `MOSS-TTS` 仓库中较早的 `moss_tts_delay/llama_cpp` 后端不同，这条链路把多通道输入、transformer backbone、多头输出以及 delay-pattern decode 都放进了 `llama.cpp`。Python 只负责准备输入和调用 ONNX 音频编解码器。

## 前置条件

1. **llama.cpp** 已从源码编译，并包含 `llama-moss-tts` 目标
2. **Python >= 3.10**
3. 本地存在一个 **MOSS-TTS** checkout，可以通过以下任一方式提供：
   - 位于当前仓库根目录旁边的 `../MOSS-TTS`
   - 通过 `--moss-tts-dir` 指定
   - 通过 `MOSS_TTS_DIR` 或 `MOSS_TTS_ROOT` 指定
4. helper scripts 需要的 Python 包：
   - `numpy`
   - `soundfile`
   - `onnxruntime`

## 编译

```bash
cd /path/to/llama.cpp

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --target llama-moss-tts -j
```

编译产物为：

- `build/bin/llama-moss-tts`

如果你希望在运行时自动构建，也可以在 e2e 脚本里传 `--build`。

## 权重准备

### 第一步：准备 first-class GGUF 模型

需要一个已经包含以下内容的 first-class MOSS-TTS-Delay GGUF：

- 文本 embedding 表
- 32 个音频 embedding 表
- Qwen3 backbone 权重
- 文本输出头
- 32 个音频输出头

例如：

- `out/stage1a_moss_delay_firstclass_f16.gguf`

### 第二步：准备 tokenizer 目录

需要一个至少包含以下文件的 tokenizer 目录：

- `tokenizer.json`

例如：

- `weights/extracted/qwen3_backbone/`

### 第三步：准备 ONNX 音频编解码器

需要同时提供两个 ONNX 文件：

- `encoder.onnx`
- `decoder.onnx`

例如：

- `weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx`
- `weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx`

### 第四步：让脚本能找到 MOSS-TTS 仓库

helper scripts 会导入：

- `moss_tts_delay.llama_cpp.processor`
- `moss_audio_tokenizer.onnx`

可以通过以下方式提供 repo 路径：

```bash
export MOSS_TTS_DIR=/path/to/MOSS-TTS
```

或者：

```bash
python tools/tts/moss-tts-firstclass-e2e.py --moss-tts-dir /path/to/MOSS-TTS ...
```

## 使用方式

### 命令行

```bash
# 音色克隆：text + reference audio -> wav
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --moss-tts-dir /path/to/MOSS-TTS \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text-file /path/to/text.txt \
    --reference-audio /path/to/reference_24k.wav \
    --output-wav /path/to/output.wav

# 不带参考音频的直接生成
python tools/tts/moss-tts-firstclass-e2e.py \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --moss-tts-dir /path/to/MOSS-TTS \
    --tokenizer-dir /path/to/tokenizer_dir \
    --onnx-encoder /path/to/encoder.onnx \
    --onnx-decoder /path/to/decoder.onnx \
    --text "你好，世界！" \
    --output-wav /path/to/output.wav

# 运行前自动构建 llama-moss-tts
python tools/tts/moss-tts-firstclass-e2e.py \
    --build \
    --model-gguf /path/to/moss_delay_firstclass.gguf \
    --moss-tts-dir /path/to/MOSS-TTS \
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
| `--moss-tts-dir` | path | 本地 `MOSS-TTS` 仓库根目录 |
| `--tokenizer-dir` | path | 含 `tokenizer.json` 的目录 |
| `--onnx-encoder` | path | 音频 tokenizer encoder ONNX |
| `--onnx-decoder` | path | 音频 tokenizer decoder ONNX |
| `--text` / `--text-file` | string / path | 输入文本，二选一 |
| `--reference-audio` | path | 可选的 24 kHz 参考音频 |
| `--language` | `zh` / `en` / tag | 传给 prompt builder 的语言标签 |
| `--max-new-tokens` | int | 最大生成步数 |
| `--text-temperature` | float | 文本通道采样温度，默认 `1.5` |
| `--audio-temperature` | float | 音频通道采样温度，默认 `1.7` |
| `--n-gpu-layers` | `-1` / `0` / `N` | GPU offload 层数，默认 `-1` |
| `--audio-decoder-cpu` | flag | 强制 ONNX 波形解码走 CPU |
| `--cpu-audio-encode` | flag | 强制 ONNX 参考音频编码走 CPU |
| `--build` | flag | 运行前构建 `llama-moss-tts` |

## 架构

```text
输入文本（+ 可选 reference wav）
  |
  v
moss-tts-build-generation-ref.py
  |
  |- 用 Qwen3 tokenizer 处理文本
  |- 可选：用 ONNX 把 reference wav 编成 audio codes
  |- 调用本地 MOSS-TTS repo 的 prompt builder
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
├── tools/tts/
│   ├── moss-tts-firstclass-e2e.py       # 端到端 wrapper
│   ├── moss-tts-build-generation-ref.py # prompt / input 构建器
│   ├── moss-tts-audio-decode.py         # ONNX 音频解码 helper
│   └── moss-tts.cpp                     # llama-moss-tts 实现
└── build/bin/
    └── llama-moss-tts
```
