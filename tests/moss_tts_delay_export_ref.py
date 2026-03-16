#!/usr/bin/env python3

import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch
import transformers

# Keep compatibility with environments that do not provide
# transformers.initialization (older transformers releases).
try:
    from transformers import initialization as _hf_init  # noqa: F401
except Exception:
    import torch.nn.init as nn_init

    def _guard_torch_init_functions():
        def _decorator(fn):
            return fn
        return _decorator

    shim = types.SimpleNamespace(
        normal_=nn_init.normal_,
        zeros_=nn_init.zeros_,
        guard_torch_init_functions=_guard_torch_init_functions,
    )
    transformers.initialization = shim
    sys.modules["transformers.initialization"] = shim

WORKROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKROOT / "MOSS-TTS"))

from moss_tts_delay.configuration_moss_tts import MossTTSDelayConfig  # noqa: E402
from moss_tts_delay.modeling_moss_tts import MossTTSDelayModel  # noqa: E402

REF_MAGIC = 0x4D545452  # "RTTM"
REF_VERSION = 1


def build_text_ids(length: int, vocab_size: int) -> np.ndarray:
    if vocab_size < 8:
        raise ValueError(f"vocab_size must be >= 8, got {vocab_size}")

    # Keep away from the first few special ids and generate a deterministic but
    # non-trivial pattern that works for both tiny toy models and full exports.
    ids = np.zeros(length, dtype=np.int32)
    span = vocab_size - 4
    for i in range(length):
        ids[i] = 4 + ((i * 7 + 3) % span)
    return ids


def build_audio_ids(n_tokens: int, n_vq: int, audio_vocab_size: int) -> np.ndarray:
    audio = np.zeros((n_tokens, n_vq), dtype=np.int32)
    for t in range(n_tokens):
        for q in range(n_vq):
            audio[t, q] = (t * 37 + q * 53) % audio_vocab_size
    return audio


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write(f"usage: {sys.argv[0]} <hf-model-dir> <reference.bin>\n")
        return 1

    model_dir = sys.argv[1]
    out_path = sys.argv[2]

    config = MossTTSDelayConfig.from_pretrained(model_dir)
    model = MossTTSDelayModel.from_pretrained(model_dir, local_files_only=True).eval()

    n_tokens = 4
    text_ids = build_text_ids(n_tokens, config.language_config.vocab_size)
    audio_ids = build_audio_ids(n_tokens, config.n_vq, config.audio_vocab_size)
    input_ids = np.concatenate([text_ids[:, None], audio_ids], axis=1)[None, :, :]

    with torch.no_grad():
        outputs = model(
            input_ids=torch.from_numpy(input_ids).long(),
            use_cache=False,
        )

    ref_embd = outputs.hidden_states[-1][0, -1].float().cpu().numpy().astype(np.float32, copy=False)
    ref_logits = np.concatenate(
        [head[0, -1].float().cpu().numpy() for head in outputs.logits],
        axis=0,
    ).astype(np.float32, copy=False)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<6I", REF_MAGIC, REF_VERSION, n_tokens, config.n_vq, ref_embd.shape[0], ref_logits.shape[0]))
        f.write(text_ids.astype(np.int32, copy=False).tobytes())
        f.write(audio_ids.reshape(-1).astype(np.int32, copy=False).tobytes())
        f.write(ref_embd.tobytes())
        f.write(ref_logits.tobytes())

    sys.stderr.write(
        "exported moss-tts-delay reference: "
        f"n_tokens={n_tokens} n_vq={config.n_vq} "
        f"n_embd={ref_embd.shape[0]} n_logits={ref_logits.shape[0]} -> {out_path}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
