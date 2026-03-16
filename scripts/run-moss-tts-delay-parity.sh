#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKROOT="$(cd "${LLAMA_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/home/expec/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llama-cpp}"

HF_DIR="${HF_DIR:-${WORKROOT}/tmp/moss_tts_delay_test_hf_kv4}"
GGUF_PATH="${GGUF_PATH:-${WORKROOT}/tmp/moss_tts_delay_test_kv4_f32.gguf}"
REF_PATH="${REF_PATH:-${WORKROOT}/tmp/moss_tts_delay_test_kv4.ref.bin}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_DIR}/build}"
TEST_BIN="${TEST_BIN:-${BUILD_DIR}/bin/test-moss-tts-delay-forward}"

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "error: conda init script not found: ${CONDA_SH}" >&2
    exit 1
fi

if [[ ! -d "${HF_DIR}" ]]; then
    echo "error: tiny HF fixture not found: ${HF_DIR}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

echo "[1/4] building parity test target"
cmake --build "${BUILD_DIR}" --target test-moss-tts-delay-forward -j2

echo "[2/4] converting tiny HF fixture to F32 GGUF"
python "${LLAMA_DIR}/convert_hf_to_gguf.py" \
    "${HF_DIR}" \
    --outfile "${GGUF_PATH}" \
    --outtype f32

echo "[3/4] exporting PyTorch reference"
python "${LLAMA_DIR}/tests/moss_tts_delay_export_ref.py" \
    "${HF_DIR}" \
    "${REF_PATH}"

echo "[4/4] running forward parity"
"${TEST_BIN}" "${GGUF_PATH}" "${REF_PATH}"

echo "PASS: moss-tts-delay forward parity verified"
