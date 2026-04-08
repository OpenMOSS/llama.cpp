#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import types
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# Older local Python envs can ship a NumPy build without numpy.typing.
try:
    import numpy.typing  # type: ignore  # noqa: F401
except Exception:
    numpy_typing = types.ModuleType("numpy.typing")
    numpy_typing.DTypeLike = object
    sys.modules["numpy.typing"] = numpy_typing

sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))
import gguf  # noqa: E402


logger = logging.getLogger("convert_moss_audio_tokenizer_to_gguf")

ARCH = "moss-audio-tokenizer"

DEFAULT_SAMPLING_RATE = 24_000
DEFAULT_DOWNSAMPLE_RATE = 1_920
DEFAULT_CONTEXT_DURATION = 10.0

SUPPORTED_MODULE_TYPES = {"PatchedPretransform", "Transformer"}
SUPPORTED_GATING = {"none"}
SUPPORTED_POSITIONAL_EMBEDDINGS = {"rope"}
SUPPORTED_QUANTIZER_TYPES = {"rlfq"}

_SAFETENSORS_DTYPES: dict[str, np.dtype[Any]] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "I16": np.dtype(np.int16),
    "U16": np.dtype(np.uint16),
    "I32": np.dtype(np.int32),
    "U32": np.dtype(np.uint32),
    "I64": np.dtype(np.int64),
    "U64": np.dtype(np.uint64),
    "F16": np.dtype(np.float16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
}


@dataclass(frozen=True)
class TensorLocation:
    name: str
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    data_start: int


class SafeTensorsIndex:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.locations: OrderedDict[str, TensorLocation] = OrderedDict()
        self._headers: dict[Path, dict[str, Any]] = {}

        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = index["weight_map"]
            for tensor_name, shard_name in weight_map.items():
                shard_path = model_dir / shard_name
                header, data_start = self._load_header(shard_path)
                meta = header[tensor_name]
                self.locations[tensor_name] = TensorLocation(
                    name=tensor_name,
                    shard=shard_path,
                    dtype=meta["dtype"],
                    shape=tuple(int(v) for v in meta["shape"]),
                    data_offsets=(int(meta["data_offsets"][0]), int(meta["data_offsets"][1])),
                    data_start=data_start,
                )
            return

        shard_paths = sorted(model_dir.glob("*.safetensors"))
        if not shard_paths:
            raise FileNotFoundError(f"No safetensors files found under {model_dir}")

        for shard_path in shard_paths:
            header, data_start = self._load_header(shard_path)
            for tensor_name, meta in header.items():
                if tensor_name == "__metadata__":
                    continue
                self.locations[tensor_name] = TensorLocation(
                    name=tensor_name,
                    shard=shard_path,
                    dtype=meta["dtype"],
                    shape=tuple(int(v) for v in meta["shape"]),
                    data_offsets=(int(meta["data_offsets"][0]), int(meta["data_offsets"][1])),
                    data_start=data_start,
                )

    def _load_header(self, shard_path: Path) -> tuple[dict[str, Any], int]:
        cached = self._headers.get(shard_path)
        if cached is not None:
            return cached, cached["__data_start__"]

        with shard_path.open("rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))

        data_start = 8 + header_len
        header["__data_start__"] = data_start
        self._headers[shard_path] = header
        return header, data_start

    def __contains__(self, name: str) -> bool:
        return name in self.locations

    def __iter__(self) -> Iterator[str]:
        return iter(self.locations.keys())

    def load(self, name: str) -> np.ndarray[Any, Any]:
        loc = self.locations[name]
        shape = tuple(loc.shape)
        offset = loc.data_start + loc.data_offsets[0]

        if loc.dtype == "BF16":
            raw = np.memmap(loc.shard, mode="r", dtype=np.uint16, offset=offset, shape=shape)
            return bf16_to_float32(raw)

        dtype = _SAFETENSORS_DTYPES.get(loc.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype {loc.dtype!r} for tensor {name!r}")

        tensor = np.memmap(loc.shard, mode="r", dtype=dtype, offset=offset, shape=shape)
        return np.asarray(tensor)


def bf16_to_float32(raw: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    u32 = raw.astype(np.uint32) << 16
    return u32.view(np.float32)


def to_serializable_config_value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported config value type: {type(value)!r}")


def add_config_value(writer: gguf.GGUFWriter, key: str, value: Any) -> None:
    value = to_serializable_config_value(value)
    if isinstance(value, bool):
        writer.add_bool(key, value)
    elif isinstance(value, int):
        if value >= 0:
            writer.add_uint32(key, value)
        else:
            writer.add_int32(key, value)
    elif isinstance(value, float):
        writer.add_float32(key, value)
    elif isinstance(value, str):
        writer.add_string(key, value)
    else:
        raise TypeError(f"Unsupported config value type for {key!r}: {type(value)!r}")


def load_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json under {model_dir}")
    return json.loads(config_path.read_text())


def validate_config(config: dict[str, Any]) -> None:
    quantizer_type = config.get("quantizer_type") or config.get("quantizer_kwargs", {}).get("quantizer_type")
    if quantizer_type not in SUPPORTED_QUANTIZER_TYPES:
        raise ValueError(
            f"Unsupported quantizer_type {quantizer_type!r}. "
            f"This converter currently supports: {sorted(SUPPORTED_QUANTIZER_TYPES)}"
        )

    for section in ("encoder_kwargs", "decoder_kwargs"):
        for idx, module_cfg in enumerate(config.get(section, [])):
            module_type = module_cfg.get("module_type")
            if module_type not in SUPPORTED_MODULE_TYPES:
                raise ValueError(f"Unsupported {section}[{idx}].module_type={module_type!r}")
            if module_type != "Transformer":
                continue
            gating = module_cfg.get("gating", "none")
            if gating not in SUPPORTED_GATING:
                raise ValueError(f"Unsupported {section}[{idx}].gating={gating!r}")
            positional_embedding = module_cfg.get("positional_embedding", "rope")
            if positional_embedding not in SUPPORTED_POSITIONAL_EMBEDDINGS:
                raise ValueError(
                    f"Unsupported {section}[{idx}].positional_embedding={positional_embedding!r}"
                )
            if "weights_per_step" in module_cfg and module_cfg["weights_per_step"]:
                raise ValueError(f"Unsupported {section}[{idx}].weights_per_step={module_cfg['weights_per_step']!r}")
            if "weights_per_step_schedule" in module_cfg and module_cfg["weights_per_step_schedule"]:
                raise ValueError(
                    f"Unsupported {section}[{idx}].weights_per_step_schedule="
                    f"{module_cfg['weights_per_step_schedule']!r}"
                )


def add_metadata(
    writer: gguf.GGUFWriter,
    config: dict[str, Any],
    model_name: str,
    *,
    include_general_fields: bool = True,
) -> None:
    if include_general_fields:
        writer.add_type("audio_tokenizer")
        writer.add_name(model_name)

    sampling_rate = int(config.get("sampling_rate", DEFAULT_SAMPLING_RATE))
    downsample_rate = int(config.get("downsample_rate", DEFAULT_DOWNSAMPLE_RATE))
    context_duration = float(config.get("causal_transformer_context_duration", DEFAULT_CONTEXT_DURATION))

    writer.add_uint32(f"{ARCH}.sampling_rate", sampling_rate)
    writer.add_uint32(f"{ARCH}.downsample_rate", downsample_rate)
    writer.add_float32(f"{ARCH}.causal_transformer_context_duration", context_duration)

    if "code_dim" in config:
        writer.add_uint32(f"{ARCH}.code_dim", int(config["code_dim"]))

    quantizer_type = config.get("quantizer_type") or config.get("quantizer_kwargs", {}).get("quantizer_type", "rlfq")
    writer.add_string(f"{ARCH}.quantizer_type", quantizer_type)

    quantizer_cfg = dict(config.get("quantizer_kwargs", {}))
    writer.add_uint32(f"{ARCH}.quantizer.input_dim", int(quantizer_cfg["input_dim"]))
    writer.add_uint32(f"{ARCH}.quantizer.rvq_dim", int(quantizer_cfg.get("rvq_dim", quantizer_cfg["input_dim"])))
    writer.add_uint32(f"{ARCH}.quantizer.output_dim", int(quantizer_cfg.get("output_dim", quantizer_cfg["input_dim"])))
    writer.add_uint32(f"{ARCH}.quantizer.num_quantizers", int(quantizer_cfg["num_quantizers"]))
    writer.add_uint32(f"{ARCH}.quantizer.codebook_size", int(quantizer_cfg["codebook_size"]))
    writer.add_uint32(f"{ARCH}.quantizer.codebook_dim", int(quantizer_cfg["codebook_dim"]))

    write_module_list_metadata(
        writer=writer,
        arch=ARCH,
        section_name="encoder",
        module_cfgs=list(config.get("encoder_kwargs", [])),
        initial_frame_rate=float(sampling_rate),
        context_duration=context_duration,
        is_encoder=True,
    )

    encoder_final_frame_rate = compute_final_encoder_frame_rate(
        module_cfgs=list(config.get("encoder_kwargs", [])),
        sampling_rate=float(sampling_rate),
    )
    write_module_list_metadata(
        writer=writer,
        arch=ARCH,
        section_name="decoder",
        module_cfgs=list(config.get("decoder_kwargs", [])),
        initial_frame_rate=encoder_final_frame_rate,
        context_duration=context_duration,
        is_encoder=False,
    )


def compute_final_encoder_frame_rate(module_cfgs: list[dict[str, Any]], sampling_rate: float) -> float:
    frame_rate = sampling_rate
    for module_cfg in module_cfgs:
        if module_cfg.get("module_type") == "PatchedPretransform":
            frame_rate /= int(module_cfg["patch_size"])
    return frame_rate


def write_module_list_metadata(
    writer: gguf.GGUFWriter,
    arch: str,
    section_name: str,
    module_cfgs: list[dict[str, Any]],
    initial_frame_rate: float,
    context_duration: float,
    is_encoder: bool,
) -> None:
    writer.add_uint32(f"{arch}.{section_name}.block_count", len(module_cfgs))

    frame_rate = initial_frame_rate
    for idx, module_cfg in enumerate(module_cfgs):
        prefix = f"{arch}.{section_name}.{idx}"
        module_type = module_cfg["module_type"]
        writer.add_string(f"{prefix}.module_type", module_type)

        if module_type == "PatchedPretransform":
            patch_size = int(module_cfg["patch_size"])
            writer.add_uint32(f"{prefix}.patch_size", patch_size)
            if is_encoder:
                frame_rate /= patch_size
            else:
                frame_rate *= patch_size
            continue

        context = int(frame_rate * context_duration)
        add_config_value(writer, f"{prefix}.context", context)
        for key in (
            "input_dimension",
            "output_dimension",
            "d_model",
            "num_heads",
            "num_layers",
            "dim_feedforward",
            "causal",
            "norm",
            "positional_embedding",
            "max_period",
            "layer_scale",
            "conv_layout",
            "gating",
        ):
            if key in module_cfg:
                add_config_value(writer, f"{prefix}.{key}", module_cfg[key])


def map_tensor_name(name: str) -> str | None:
    if ".parametrizations.weight.original0" in name:
        return name.replace(".parametrizations.weight.original0", ".weight")
    if ".parametrizations.weight.original1" in name:
        return None
    return name


def is_float_tensor(tensor: np.ndarray[Any, Any]) -> bool:
    return np.issubdtype(tensor.dtype, np.floating)


def choose_output_dtype(tensor: np.ndarray[Any, Any], outtype: str) -> np.dtype[Any] | None:
    if not is_float_tensor(tensor):
        return None
    if outtype == "f32":
        return np.dtype(np.float32)
    if outtype == "f16":
        if tensor.ndim <= 1:
            return np.dtype(np.float32)
        return np.dtype(np.float16)
    raise ValueError(f"Unsupported outtype {outtype!r}")


def convert_tensor_dtype(tensor: np.ndarray[Any, Any], outtype: str) -> np.ndarray[Any, Any]:
    dst_dtype = choose_output_dtype(tensor, outtype)
    if dst_dtype is None:
        return np.ascontiguousarray(tensor)
    if tensor.dtype == dst_dtype:
        return np.ascontiguousarray(tensor)
    return np.ascontiguousarray(tensor.astype(dst_dtype, copy=False))


def merge_weight_norm(g: np.ndarray[Any, Any], v: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    axes = tuple(range(1, v.ndim))
    norm = np.linalg.norm(v.astype(np.float32), axis=axes, keepdims=True)
    norm = np.maximum(norm, np.finfo(np.float32).eps)
    return g.astype(np.float32) * v.astype(np.float32) / norm


def iter_converted_tensors(index: SafeTensorsIndex, outtype: str) -> Iterator[tuple[str, np.ndarray[Any, Any]]]:
    emitted: set[str] = set()

    for name in index:
        mapped_name = map_tensor_name(name)
        if mapped_name is None or mapped_name in emitted:
            continue

        if ".parametrizations.weight.original0" in name:
            prefix = name.replace(".parametrizations.weight.original0", "")
            g_name = f"{prefix}.parametrizations.weight.original0"
            v_name = f"{prefix}.parametrizations.weight.original1"
            weight = merge_weight_norm(index.load(g_name), index.load(v_name))
            yield mapped_name, convert_tensor_dtype(weight, outtype)
            emitted.add(mapped_name)
            continue

        tensor = index.load(name)
        yield mapped_name, convert_tensor_dtype(tensor, outtype)
        emitted.add(mapped_name)


def count_output_tensors(index: SafeTensorsIndex) -> int:
    seen: set[str] = set()
    for name in index:
        mapped_name = map_tensor_name(name)
        if mapped_name is not None:
            seen.add(mapped_name)
    return len(seen)


def default_outfile(model_dir: Path, outtype: str) -> Path:
    return model_dir / f"{model_dir.name}-{outtype}.gguf"


def build_writer(outfile: Path, outtype: str, model_name: str, config: dict[str, Any]) -> gguf.GGUFWriter:
    ftype_map = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
    }
    writer = gguf.GGUFWriter(path=outfile, arch=ARCH)
    writer.add_file_type(ftype_map[outtype])
    add_metadata(writer, config, model_name)
    return writer


def convert(model_dir: Path, outfile: Path, outtype: str, model_name: str, dry_run: bool) -> None:
    config = load_config(model_dir)
    validate_config(config)

    index = SafeTensorsIndex(model_dir)
    total_tensors = count_output_tensors(index)
    logger.info("Found %d input tensors, %d output tensors", len(index.locations), total_tensors)

    if dry_run:
        logger.info("Dry-run only, not writing %s", outfile)
        return

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = build_writer(outfile=outfile, outtype=outtype, model_name=model_name, config=config)
    try:
        for i, (name, tensor) in enumerate(iter_converted_tensors(index, outtype), start=1):
            logger.debug("[%4d / %4d] %s %s", i, total_tensors, name, list(tensor.shape))
            writer.add_tensor(name, tensor)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        logger.info("Wrote %s", outfile)
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face MOSS Audio Tokenizer checkpoint to GGUF without modifying convert_hf_to_gguf.py."
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to a local MOSS Audio Tokenizer HF checkpoint directory containing config.json and safetensors shards.",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=None,
        help="Output GGUF path. Defaults to <model_dir>/<model_dir.name>-<outtype>.gguf",
    )
    parser.add_argument(
        "--outtype",
        choices=("f16", "f32"),
        default="f16",
        help="GGUF floating-point storage type. f16 keeps 1D float tensors in f32, matching MOSTLY_F16 semantics.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional GGUF general.name override. Defaults to the checkpoint directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the checkpoint and tensor mapping without writing the GGUF file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable per-tensor logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    model_dir = args.model_dir.resolve()
    outfile = args.outfile.resolve() if args.outfile is not None else default_outfile(model_dir, args.outtype)
    model_name = args.model_name or model_dir.name

    convert(
        model_dir=model_dir,
        outfile=outfile,
        outtype=args.outtype,
        model_name=model_name,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
