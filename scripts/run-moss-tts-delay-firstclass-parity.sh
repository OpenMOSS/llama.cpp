#!/usr/bin/env bash
set -euo pipefail

# Full hybrid parity runner for a machine with enough RAM / disk:
# - downloads OpenMOSS-Team/MOSS-TTS (HF)
# - extracts hybrid llama_cpp weights for the official Python pipeline
# - converts the same HF checkpoint to a first-class MOSS-TTS-Delay GGUF
# - runs deterministic generation on both paths
# - compares raw audio codes exactly
# - decodes both sides to wav through the same ONNX audio tokenizer

WORKDIR="/home/taoji/data/zlwang/workwork"
CONDA_ENV="${CONDA_ENV:-moss-tts-firstclass}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
N_JOBS="${N_JOBS:-$(nproc)}"

TEXT="${TEXT:-今天天气很好，我们来验证 first-class MOSS-TTS Delay 在 llama.cpp 中的端到端一致性。}"
REFERENCE_AUDIO="${REFERENCE_AUDIO:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
USE_GPU_AUDIO="${USE_GPU_AUDIO:-true}"

HF_MODEL_ID="${HF_MODEL_ID:-OpenMOSS-Team/MOSS-TTS}"
HF_AUDIO_REPO="${HF_AUDIO_REPO:-OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX}"
ORT_PKG="${ORT_PKG:-onnxruntime-gpu}"

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$WORKDIR/llama.cpp}"
MOSS_TTS_DIR="${MOSS_TTS_DIR:-$WORKDIR/MOSS-TTS}"
HF_MODEL_DIR="${HF_MODEL_DIR:-$WORKDIR/weights/MOSS-TTS-hf}"
EXTRACT_DIR="${EXTRACT_DIR:-$WORKDIR/weights/extracted}"
ONNX_DIR="${ONNX_DIR:-$WORKDIR/weights/MOSS-Audio-Tokenizer-ONNX}"
PY_CONFIG="${PY_CONFIG:-$WORKDIR/moss_delay_python_ref.yaml}"

BACKBONE_GGUF="${BACKBONE_GGUF:-$WORKDIR/weights/backbone_f16.gguf}"
FIRSTCLASS_GGUF="${FIRSTCLASS_GGUF:-$WORKDIR/weights/moss_delay_firstclass_f16.gguf}"

GEN_REF_BIN="${GEN_REF_BIN:-$WORKDIR/out/python_generation.ref.bin}"
PY_WAV="${PY_WAV:-$WORKDIR/out/python_reference.wav}"
CPP_CODES_BIN="${CPP_CODES_BIN:-$WORKDIR/out/cpp_raw_codes.bin}"
CPP_WAV="${CPP_WAV:-$WORKDIR/out/cpp_firstclass.wav}"

mkdir -p "$WORKDIR" "$WORKDIR/weights" "$WORKDIR/out"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "error: missing command: $1" >&2
        exit 1
    }
}

clone_or_update() {
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

export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade "huggingface_hub[cli]>=0.30"

clone_or_update "https://github.com/expectqwq/llama.cpp.git" "$LLAMA_CPP_DIR" master
clone_or_update "https://github.com/OpenMOSS/MOSS-TTS.git" "$MOSS_TTS_DIR" main
git -C "$MOSS_TTS_DIR" submodule update --init --recursive

if [[ "$ORT_PKG" == "onnxruntime-gpu" ]]; then
    python -m pip install -e "${MOSS_TTS_DIR}[llama-cpp-onnx]"
else
    python -m pip install -e "${MOSS_TTS_DIR}[llama-cpp]"
    python -m pip install --upgrade "${ORT_PKG}>=1.19"
fi

huggingface-cli download "$HF_MODEL_ID" --local-dir "$HF_MODEL_DIR"
huggingface-cli download "$HF_AUDIO_REPO" --local-dir "$ONNX_DIR"

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
cmake --build "$LLAMA_CPP_DIR/build" --target llama-moss-tts llama-quantize -j"$N_JOBS"

bash "$MOSS_TTS_DIR/moss_tts_delay/llama_cpp/build_bridge.sh" "$LLAMA_CPP_DIR"

python "$MOSS_TTS_DIR/moss_tts_delay/llama_cpp/conversion/extract_weights.py" \
    --model "$HF_MODEL_DIR" \
    --output "$EXTRACT_DIR"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$EXTRACT_DIR/qwen3_backbone" \
    --outfile "$BACKBONE_GGUF" \
    --outtype f16

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$HF_MODEL_DIR" \
    --outfile "$FIRSTCLASS_GGUF" \
    --outtype f16

cat > "$PY_CONFIG" <<EOF
backbone_gguf: $BACKBONE_GGUF
embedding_dir: $EXTRACT_DIR/embeddings
lm_head_dir: $EXTRACT_DIR/lm_heads
tokenizer_dir: $EXTRACT_DIR/qwen3_backbone

audio_backend: onnx
audio_encoder_onnx: $ONNX_DIR/encoder.onnx
audio_decoder_onnx: $ONNX_DIR/decoder.onnx

heads_backend: numpy

n_ctx: 4096
n_batch: 512
n_threads: 8
n_gpu_layers: -1
max_new_tokens: $MAX_NEW_TOKENS
use_gpu_audio: $USE_GPU_AUDIO

text_temperature: 0.0
text_top_p: 1.0
text_top_k: 50

audio_temperature: 0.0
audio_top_p: 1.0
audio_top_k: 25
audio_repetition_penalty: 1.0
EOF

if [[ -z "$REFERENCE_AUDIO" ]]; then
    REF_ARGS=()
else
    REF_ARGS=(--reference-audio "$REFERENCE_AUDIO")
fi

python "$LLAMA_CPP_DIR/tests/moss_tts_delay_export_generation_ref.py" \
    --config "$PY_CONFIG" \
    --text "$TEXT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-ref "$GEN_REF_BIN" \
    --output-wav "$PY_WAV" \
    "${REF_ARGS[@]}"

"$LLAMA_CPP_DIR/build/bin/llama-moss-tts" \
    -m "$FIRSTCLASS_GGUF" \
    --generation-ref "$GEN_REF_BIN" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --text-temperature 0.0 \
    --audio-temperature 0.0 \
    --dump-raw-codes "$CPP_CODES_BIN" \
    --audio-decoder-script "$LLAMA_CPP_DIR/tools/tts/moss-tts-audio-decode.py" \
    --audio-encoder-onnx "$ONNX_DIR/encoder.onnx" \
    --audio-decoder-onnx "$ONNX_DIR/decoder.onnx" \
    --wav-out "$CPP_WAV" \
    --python-bin python \
    $( [[ "$USE_GPU_AUDIO" == "true" ]] || echo --audio-decoder-cpu )

python - <<PY
from pathlib import Path
import hashlib
import wave
import numpy as np

py_wav = Path("$PY_WAV")
cpp_wav = Path("$CPP_WAV")

def read_pcm(path: Path):
    with wave.open(str(path), "rb") as f:
        sr = f.getframerate()
        n = f.getnframes()
        data = np.frombuffer(f.readframes(n), dtype=np.int16)
    return sr, data

py_sr, py_pcm = read_pcm(py_wav)
cpp_sr, cpp_pcm = read_pcm(cpp_wav)
same_len = py_pcm.shape == cpp_pcm.shape
max_abs = None
if same_len:
    max_abs = int(np.max(np.abs(py_pcm.astype(np.int32) - cpp_pcm.astype(np.int32)))) if py_pcm.size else 0

print("python wav:", py_wav)
print("cpp wav   :", cpp_wav)
print("sample_rate_equal:", py_sr == cpp_sr, "py_sr=", py_sr, "cpp_sr=", cpp_sr)
print("pcm_length_equal:", same_len, "py_samples=", py_pcm.size, "cpp_samples=", cpp_pcm.size)
print("pcm_max_abs_diff:", max_abs)
print("python_md5:", hashlib.md5(py_wav.read_bytes()).hexdigest())
print("cpp_md5   :", hashlib.md5(cpp_wav.read_bytes()).hexdigest())
PY

echo
echo "Parity run finished."
echo "Python reference wav: $PY_WAV"
echo "First-class C++ wav: $CPP_WAV"
echo "Generation ref file : $GEN_REF_BIN"
