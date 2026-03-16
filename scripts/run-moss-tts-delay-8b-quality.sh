#!/usr/bin/env bash
set -euo pipefail

# End-to-end setup for MOSS-TTS-Delay 8B quality smoke test on a fresh machine.
#
# What it does:
# 1. Clones this llama.cpp fork and the official OpenMOSS/MOSS-TTS repo.
# 2. Creates/uses a conda env and installs the minimal Python stack.
# 3. Downloads the official GGUF backbone + embeddings/lm_heads/tokenizer.
# 4. Downloads the official ONNX audio tokenizer.
# 5. Builds llama-moss-tts and runs the C++ vs Python de-delay/raw-code parity test.
# 6. Runs the official Python llama_cpp backend to synthesize wavs for listening.
#
# Defaults target a CUDA machine. For CPU-only ONNX Runtime:
#   ORT_PKG=onnxruntime USE_GPU_AUDIO=false bash run-moss-tts-delay-8b-quality.sh

WORKDIR="${WORKDIR:-$HOME/moss-tts-delay-8b-eval}"
CONDA_ENV="${CONDA_ENV:-moss-tts-delay-8b}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/expectqwq/llama.cpp.git}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
MOSS_TTS_REPO="${MOSS_TTS_REPO:-https://github.com/OpenMOSS/MOSS-TTS.git}"
MOSS_TTS_REF="${MOSS_TTS_REF:-main}"

ORT_PKG="${ORT_PKG:-onnxruntime-gpu}"
USE_GPU_AUDIO="${USE_GPU_AUDIO:-true}"
N_JOBS="${N_JOBS:-$(nproc)}"

TEXT_ZH="${TEXT_ZH:-今天天气很好，我们来测试一下 MOSS-TTS Delay 8B 的音质和稳定性。}"
TEXT_EN="${TEXT_EN:-Hello, this is a quality smoke test for the MOSS-TTS Delay 8B pipeline running with llama.cpp and the ONNX audio tokenizer.}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-}"

HF_MODEL_REPO="${HF_MODEL_REPO:-OpenMOSS-Team/MOSS-TTS-GGUF}"
HF_AUDIO_REPO="${HF_AUDIO_REPO:-OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX}"

LLAMA_CPP_DIR="$WORKDIR/llama.cpp"
MOSS_TTS_DIR="$WORKDIR/MOSS-TTS"
WEIGHTS_DIR="$WORKDIR/weights"
GGUF_DIR="$WEIGHTS_DIR/MOSS-TTS-GGUF"
AUDIO_ORT_DIR="$WEIGHTS_DIR/MOSS-Audio-Tokenizer-ONNX"
OUT_DIR="$WORKDIR/out"
CONFIG_PATH="$WORKDIR/moss_delay_8b_eval.yaml"

mkdir -p "$WORKDIR" "$WEIGHTS_DIR" "$OUT_DIR"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: required command not found: $1" >&2
        exit 1
    }
}

git_clone_or_update() {
    local repo_url="$1"
    local repo_dir="$2"
    local repo_ref="$3"

    if [[ ! -d "$repo_dir/.git" ]]; then
        git clone "$repo_url" "$repo_dir"
    fi

    git -C "$repo_dir" fetch --all --tags
    git -C "$repo_dir" checkout "$repo_ref"
    git -C "$repo_dir" pull --ff-only || true
}

need_cmd git
need_cmd cmake
need_cmd conda

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "huggingface_hub[cli]>=0.30"

git_clone_or_update "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR" "$LLAMA_CPP_REF"
git_clone_or_update "$MOSS_TTS_REPO" "$MOSS_TTS_DIR" "$MOSS_TTS_REF"
git -C "$MOSS_TTS_DIR" submodule update --init --recursive

if [[ "$ORT_PKG" == "onnxruntime-gpu" ]]; then
    python -m pip install -e "${MOSS_TTS_DIR}[llama-cpp-onnx]"
else
    python -m pip install -e "${MOSS_TTS_DIR}[llama-cpp]"
    python -m pip install --upgrade "${ORT_PKG}>=1.19"
fi

huggingface-cli download "$HF_MODEL_REPO" --local-dir "$GGUF_DIR"
huggingface-cli download "$HF_AUDIO_REPO" --local-dir "$AUDIO_ORT_DIR"

if [[ -z "$REFERENCE_AUDIO" ]]; then
    REFERENCE_AUDIO="$MOSS_TTS_DIR/assets/audio/reference_zh.wav"
fi

if [[ ! -f "$GGUF_DIR/MOSS_TTS_Q4_K_M.gguf" ]]; then
    echo "error: expected backbone file missing: $GGUF_DIR/MOSS_TTS_Q4_K_M.gguf" >&2
    exit 1
fi

if [[ ! -f "$AUDIO_ORT_DIR/encoder.onnx" || ! -f "$AUDIO_ORT_DIR/decoder.onnx" ]]; then
    echo "error: expected ONNX audio tokenizer files missing in $AUDIO_ORT_DIR" >&2
    exit 1
fi

cat > "$CONFIG_PATH" <<EOF
backbone_gguf: $GGUF_DIR/MOSS_TTS_Q4_K_M.gguf
embedding_dir: $GGUF_DIR/embeddings
lm_head_dir: $GGUF_DIR/lm_heads
tokenizer_dir: $GGUF_DIR/tokenizer

audio_backend: onnx
audio_encoder_onnx: $AUDIO_ORT_DIR/encoder.onnx
audio_decoder_onnx: $AUDIO_ORT_DIR/decoder.onnx

heads_backend: auto

n_ctx: 4096
n_batch: 512
n_threads: 8
n_gpu_layers: -1
max_new_tokens: 2000
use_gpu_audio: $USE_GPU_AUDIO

text_temperature: 1.5
text_top_p: 1.0
text_top_k: 50

audio_temperature: 1.7
audio_top_p: 0.8
audio_top_k: 25
audio_repetition_penalty: 1.0
EOF

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
cmake --build "$LLAMA_CPP_DIR/build" --target llama-moss-tts -j"$N_JOBS"

echo
echo "== Running C++ vs Python decode parity =="
"$LLAMA_CPP_DIR/scripts/run-moss-tts-delay-decode-parity.sh"

echo
echo "== Generating zero-shot quality sample (ZH) =="
python -m moss_tts_delay.llama_cpp \
    --config "$CONFIG_PATH" \
    --text "$TEXT_ZH" \
    --output "$OUT_DIR/zero_shot_zh.wav" \
    --profile

echo
echo "== Generating zero-shot quality sample (EN) =="
python -m moss_tts_delay.llama_cpp \
    --config "$CONFIG_PATH" \
    --text "$TEXT_EN" \
    --output "$OUT_DIR/zero_shot_en.wav" \
    --profile

if [[ -f "$REFERENCE_AUDIO" ]]; then
    echo
    echo "== Generating voice-clone quality sample =="
    python -m moss_tts_delay.llama_cpp \
        --config "$CONFIG_PATH" \
        --reference "$REFERENCE_AUDIO" \
        --text "$TEXT_ZH" \
        --output "$OUT_DIR/clone_zh.wav" \
        --profile
fi

python - <<PY
from pathlib import Path
import wave

out_dir = Path("$OUT_DIR")
print("\\n== Output summary ==")
for wav_path in sorted(out_dir.glob("*.wav")):
    with wave.open(str(wav_path), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        ch = f.getnchannels()
    dur = n / max(sr, 1)
    print(f"{wav_path}: sr={sr}Hz channels={ch} duration={dur:.2f}s frames={n}")
print("\\nListen to the generated wavs above for subjective quality.")
PY

echo
echo "Done."
echo "Artifacts:"
echo "  config: $CONFIG_PATH"
echo "  outputs: $OUT_DIR"
echo "  gguf weights: $GGUF_DIR"
echo "  onnx tokenizer: $AUDIO_ORT_DIR"
