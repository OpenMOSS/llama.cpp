from __future__ import annotations

from pathlib import Path

import numpy as np

N_QUANTIZERS = 32
DOWNSAMPLE_RATE = 1920


def _load_ort_session(model_path: str | Path, use_gpu: bool):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for MOSS audio tokenizer ONNX inference") from exc

    providers = ["CPUExecutionProvider"]
    if use_gpu:
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session_options = ort.SessionOptions()
    return ort.InferenceSession(str(model_path), sess_options=session_options, providers=providers)


class OnnxAudioTokenizer:
    """Minimal ONNX wrapper for the MOSS audio tokenizer."""

    def __init__(self, encoder_path: str | Path, decoder_path: str | Path, use_gpu: bool = True):
        self.encoder_session = _load_ort_session(encoder_path, use_gpu)
        self.decoder_session = _load_ort_session(decoder_path, use_gpu)
        self.encoder_inputs = [item.name for item in self.encoder_session.get_inputs()]
        self.encoder_outputs = [item.name for item in self.encoder_session.get_outputs()]
        self.decoder_inputs = [item.name for item in self.decoder_session.get_inputs()]
        self.decoder_outputs = [item.name for item in self.decoder_session.get_outputs()]

    def encode(self, waveform: np.ndarray, n_quantizers: int = N_QUANTIZERS) -> np.ndarray:
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, np.newaxis, :]
        elif waveform.ndim == 2:
            waveform = waveform[np.newaxis, :]

        t = waveform.shape[-1]
        padded = ((t + DOWNSAMPLE_RATE - 1) // DOWNSAMPLE_RATE) * DOWNSAMPLE_RATE
        if padded != t:
            waveform = np.concatenate(
                [waveform, np.zeros((waveform.shape[0], waveform.shape[1], padded - t), dtype=np.float32)],
                axis=-1,
            )

        result = self.encoder_session.run(
            self.encoder_outputs,
            {
                self.encoder_inputs[0]: waveform.astype(np.float32),
                self.encoder_inputs[1]: np.array(n_quantizers, dtype=np.int64),
            },
        )
        return result[0][:, 0, :int(result[1][0])].T.astype(np.int64)

    def decode(self, audio_codes: np.ndarray, n_quantizers: int = N_QUANTIZERS) -> np.ndarray:
        if audio_codes.ndim == 2:
            if audio_codes.shape[1] == N_QUANTIZERS and audio_codes.shape[0] != N_QUANTIZERS:
                audio_codes = audio_codes.T
            audio_codes = audio_codes[:, np.newaxis, :]

        result = self.decoder_session.run(
            self.decoder_outputs,
            {
                self.decoder_inputs[0]: audio_codes.astype(np.int64),
                self.decoder_inputs[1]: np.array(n_quantizers, dtype=np.int64),
            },
        )
        return result[0][0, 0, :int(result[1][0])].astype(np.float32)
