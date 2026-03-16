#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/expec/workwork/llama.cpp"
REF_PATH="${REF_PATH:-/home/expec/workwork/tmp/moss_tts_delay_decode.ref.bin}"

source /home/expec/miniconda3/etc/profile.d/conda.sh
conda activate llama-cpp

python "$ROOT/tests/moss_tts_delay_export_decode_ref.py" "$REF_PATH"

cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target llama-moss-tts -j4

"$ROOT/build/bin/llama-moss-tts" \
  --python-bin python \
  --decode-parity-ref "$REF_PATH"

echo "PASS: moss-tts delay decode parity verified"
