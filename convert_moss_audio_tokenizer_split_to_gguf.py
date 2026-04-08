#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from convert_moss_audio_tokenizer_to_gguf import SafeTensorsIndex
from convert_moss_audio_tokenizer_to_gguf import convert_tensor_dtype
from convert_moss_audio_tokenizer_to_gguf import load_config
from convert_moss_audio_tokenizer_to_gguf import map_tensor_name
from convert_moss_audio_tokenizer_to_gguf import merge_weight_norm
from convert_moss_audio_tokenizer_to_gguf import validate_config
from convert_moss_audio_tokenizer_to_gguf import write_module_list_metadata

import sys
import types

# Older local Python envs can ship a NumPy build without numpy.typing.
try:
    import numpy.typing  # type: ignore  # noqa: F401
except Exception:
    numpy_typing = types.ModuleType("numpy.typing")
    numpy_typing.DTypeLike = object
    sys.modules["numpy.typing"] = numpy_typing

sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))
import gguf  # noqa: E402


logger = logging.getLogger("convert_moss_audio_tokenizer_split_to_gguf")

ARCH_ENCODER = "moss-tts-audio-encoder"
ARCH_DECODER = "moss-tts-audio-decoder"

DEFAULT_SAMPLING_RATE = 24_000
DEFAULT_DOWNSAMPLE_RATE = 1_920
DEFAULT_CONTEXT_DURATION = 10.0


def default_encoder_outfile(model_dir: Path, outtype: str) -> Path:
    return model_dir / f"{model_dir.name}-encoder-{outtype}.gguf"


def default_decoder_outfile(model_dir: Path, outtype: str) -> Path:
    return model_dir / f"{model_dir.name}-decoder-{outtype}.gguf"


def build_transformer_block_index_map(module_cfgs: list[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    tensor_block = 0
    for module_idx, module_cfg in enumerate(module_cfgs):
        if module_cfg.get("module_type") != "Transformer":
            continue
        result[module_idx] = tensor_block
        tensor_block += 1
    return result


def map_transformer_tensor_name(tensor_block: int, tail: str) -> str | None:
    if tail == "input_proj.weight":
        return f"blk.{tensor_block}.input_proj.weight"
    if tail == "output_proj.weight":
        return f"blk.{tensor_block}.output_proj.weight"

    parts = tail.split(".")
    if len(parts) < 5 or parts[0] != "transformer" or parts[1] != "layers":
        return None

    layer_idx = int(parts[2])
    layer_prefix = f"blk.{tensor_block}.layer.{layer_idx}"
    layer_tail = ".".join(parts[3:])

    if layer_tail == "layer_scale_1.scale":
        return f"{layer_prefix}.attn_scale.scale"
    if layer_tail == "layer_scale_2.scale":
        return f"{layer_prefix}.ffn_scale.scale"
    if layer_tail == "linear1.weight":
        return f"{layer_prefix}.ffn_up.weight"
    if layer_tail == "linear2.weight":
        return f"{layer_prefix}.ffn_down.weight"
    if layer_tail == "norm1.weight":
        return f"{layer_prefix}.attn_norm.weight"
    if layer_tail == "norm1.bias":
        return f"{layer_prefix}.attn_norm.bias"
    if layer_tail == "norm2.weight":
        return f"{layer_prefix}.ffn_norm.weight"
    if layer_tail == "norm2.bias":
        return f"{layer_prefix}.ffn_norm.bias"
    if layer_tail == "self_attn.in_projs.0.weight":
        return f"{layer_prefix}.attn_qkv.weight"
    if layer_tail == "self_attn.out_projs.0.weight":
        return f"{layer_prefix}.attn_output.weight"

    return None


def map_split_tensor_name(
    name: str,
    encoder_block_map: dict[int, int],
    decoder_block_map: dict[int, int],
) -> str | None:
    mapped = map_tensor_name(name)
    if mapped is None:
        return None

    if mapped.startswith("encoder."):
        rest = mapped[len("encoder."):]
        module_idx_str, tail = rest.split(".", 1)
        return map_transformer_tensor_name(encoder_block_map[int(module_idx_str)], tail)

    if mapped.startswith("decoder."):
        rest = mapped[len("decoder."):]
        module_idx_str, tail = rest.split(".", 1)
        return map_transformer_tensor_name(decoder_block_map[int(module_idx_str)], tail)

    return mapped


def _count_path(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "quantizer" and parts[1] == "quantizers":
        return ".".join(parts[:4])
    if len(parts) >= 4 and parts[0] == "blk" and parts[2] == "layer":
        return ".".join(parts[:4])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return name


def is_encoder_tensor(name: str) -> bool:
    if name.startswith("encoder."):
        return True
    if name.startswith("quantizer.input_proj."):
        return True
    if name.startswith("quantizer.quantizers.") and (
        ".in_proj." in name or ".out_proj." in name or ".codebook." in name
    ):
        return True
    return False


def is_decoder_tensor(name: str) -> bool:
    if name.startswith("decoder."):
        return True
    if name.startswith("quantizer.output_proj."):
        return True
    if name.startswith("quantizer.quantizers.") and (
        ".out_proj." in name or ".codebook." in name
    ):
        return True
    return False


def count_filtered_output_tensors(
    index: SafeTensorsIndex,
    include_fn: Callable[[str], bool],
    rename_fn: Callable[[str], str | None],
) -> int:
    seen: set[str] = set()
    for name in index:
        mapped_name = map_tensor_name(name)
        renamed_name = rename_fn(name)
        if mapped_name is None or renamed_name is None or not include_fn(mapped_name):
            continue
        seen.add(renamed_name)
    return len(seen)


def iter_filtered_tensors(
    index: SafeTensorsIndex,
    outtype: str,
    include_fn: Callable[[str], bool],
    rename_fn: Callable[[str], str | None],
) -> Iterator[tuple[str, np.ndarray[Any, Any]]]:
    emitted: set[str] = set()

    for name in index:
        mapped_name = map_tensor_name(name)
        renamed_name = rename_fn(name)
        if (
            mapped_name is None
            or renamed_name is None
            or renamed_name in emitted
            or not include_fn(mapped_name)
        ):
            continue

        if ".parametrizations.weight.original0" in name:
            prefix = name.replace(".parametrizations.weight.original0", "")
            g_name = f"{prefix}.parametrizations.weight.original0"
            v_name = f"{prefix}.parametrizations.weight.original1"
            weight = merge_weight_norm(index.load(g_name), index.load(v_name))
            yield renamed_name, convert_tensor_dtype(weight, outtype)
            emitted.add(renamed_name)
            continue

        tensor = index.load(name)
        yield renamed_name, convert_tensor_dtype(tensor, outtype)
        emitted.add(renamed_name)


def add_common_metadata(
    writer: gguf.GGUFWriter,
    arch: str,
    config: dict[str, Any],
    model_name: str,
) -> None:
    writer.add_type("model")
    writer.add_name(model_name)

    sampling_rate = int(config.get("sampling_rate", DEFAULT_SAMPLING_RATE))
    downsample_rate = int(config.get("downsample_rate", DEFAULT_DOWNSAMPLE_RATE))
    context_duration = float(config.get("causal_transformer_context_duration", DEFAULT_CONTEXT_DURATION))
    quantizer_cfg = dict(config.get("quantizer_kwargs", {}))
    quantizer_type = config.get("quantizer_type") or quantizer_cfg.get("quantizer_type", "rlfq")

    writer.add_uint32(f"{arch}.sampling_rate", sampling_rate)
    writer.add_uint32(f"{arch}.downsample_rate", downsample_rate)
    writer.add_float32(f"{arch}.causal_transformer_context_duration", context_duration)
    writer.add_uint32(f"{arch}.code_dim", int(config.get("code_dim", quantizer_cfg.get("output_dim", 0))))
    writer.add_string(f"{arch}.quantizer_type", quantizer_type)
    writer.add_uint32(f"{arch}.quantizer.input_dim", int(quantizer_cfg["input_dim"]))
    writer.add_uint32(f"{arch}.quantizer.rvq_dim", int(quantizer_cfg.get("rvq_dim", quantizer_cfg["input_dim"])))
    writer.add_uint32(f"{arch}.quantizer.output_dim", int(quantizer_cfg.get("output_dim", quantizer_cfg["input_dim"])))
    writer.add_uint32(f"{arch}.quantizer.num_quantizers", int(quantizer_cfg["num_quantizers"]))
    writer.add_uint32(f"{arch}.quantizer.codebook_size", int(quantizer_cfg["codebook_size"]))
    writer.add_uint32(f"{arch}.quantizer.codebook_dim", int(quantizer_cfg["codebook_dim"]))


def add_encoder_metadata(writer: gguf.GGUFWriter, config: dict[str, Any], model_name: str) -> None:
    add_common_metadata(writer, ARCH_ENCODER, config, model_name)
    write_module_list_metadata(
        writer=writer,
        arch=ARCH_ENCODER,
        section_name="encoder",
        module_cfgs=list(config.get("encoder_kwargs", [])),
        initial_frame_rate=float(config.get("sampling_rate", DEFAULT_SAMPLING_RATE)),
        context_duration=float(config.get("causal_transformer_context_duration", DEFAULT_CONTEXT_DURATION)),
        is_encoder=True,
    )


def add_decoder_metadata(writer: gguf.GGUFWriter, config: dict[str, Any], model_name: str) -> None:
    add_common_metadata(writer, ARCH_DECODER, config, model_name)

    encoder_frame_rate = float(config.get("sampling_rate", DEFAULT_SAMPLING_RATE))
    for module_cfg in list(config.get("encoder_kwargs", [])):
        if module_cfg.get("module_type") == "PatchedPretransform":
            encoder_frame_rate /= int(module_cfg["patch_size"])

    write_module_list_metadata(
        writer=writer,
        arch=ARCH_DECODER,
        section_name="decoder",
        module_cfgs=list(config.get("decoder_kwargs", [])),
        initial_frame_rate=encoder_frame_rate,
        context_duration=float(config.get("causal_transformer_context_duration", DEFAULT_CONTEXT_DURATION)),
        is_encoder=False,
    )


def build_writer(outfile: Path, arch: str, outtype: str, config: dict[str, Any], model_name: str) -> gguf.GGUFWriter:
    ftype_map = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
    }
    writer = gguf.GGUFWriter(path=outfile, arch=arch)
    writer.add_file_type(ftype_map[outtype])
    if arch == ARCH_ENCODER:
        add_encoder_metadata(writer, config, model_name)
    elif arch == ARCH_DECODER:
        add_decoder_metadata(writer, config, model_name)
    else:
        raise ValueError(f"unexpected split arch {arch!r}")
    return writer


def convert_one(
    model_dir: Path,
    outfile: Path,
    outtype: str,
    model_name: str,
    include_fn: Callable[[str], bool],
    arch: str,
    dry_run: bool,
) -> None:
    config = load_config(model_dir)
    validate_config(config)
    index = SafeTensorsIndex(model_dir)
    encoder_block_map = build_transformer_block_index_map(list(config.get("encoder_kwargs", [])))
    decoder_block_map = build_transformer_block_index_map(list(config.get("decoder_kwargs", [])))
    rename_fn = lambda name: map_split_tensor_name(name, encoder_block_map, decoder_block_map)
    total_tensors = count_filtered_output_tensors(index, include_fn, rename_fn)
    logger.info(
        "%s: selected %d output tensors for %s",
        arch,
        total_tensors,
        outfile,
    )

    if dry_run:
        paths: dict[str, int] = {}
        for name in index:
            mapped_name = map_tensor_name(name)
            renamed_name = rename_fn(name)
            if mapped_name is None or renamed_name is None or not include_fn(mapped_name):
                continue
            key = _count_path(renamed_name)
            paths[key] = paths.get(key, 0) + 1

        for key in sorted(paths):
            logger.debug("%s keeps %3d tensors under %s", arch, paths[key], key)
        logger.info("%s: dry-run only, not writing %s", arch, outfile)
        return

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer = build_writer(outfile, arch, outtype, config, model_name)
    try:
        for i, (name, tensor) in enumerate(iter_filtered_tensors(index, outtype, include_fn, rename_fn), start=1):
            logger.debug("[%4d / %4d] %s %s", i, total_tensors, name, list(tensor.shape))
            writer.add_tensor(name, tensor)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        logger.info("%s: wrote %s", arch, outfile)
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a Hugging Face MOSS Audio Tokenizer checkpoint into "
            "moss-tts-audio-encoder and moss-tts-audio-decoder GGUF files."
        )
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Path to a local MOSS Audio Tokenizer HF checkpoint directory.",
    )
    parser.add_argument(
        "--encoder-outfile",
        type=Path,
        default=None,
        help="Output path for the moss-tts-audio-encoder GGUF.",
    )
    parser.add_argument(
        "--decoder-outfile",
        type=Path,
        default=None,
        help="Output path for the moss-tts-audio-decoder GGUF.",
    )
    parser.add_argument(
        "--outtype",
        choices=("f16", "f32"),
        default="f16",
        help="GGUF floating-point storage type.",
    )
    parser.add_argument(
        "--model-name-prefix",
        type=str,
        default=None,
        help="Optional prefix for general.name. Defaults to the checkpoint directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and tensor split without writing GGUF files.",
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
    name_prefix = args.model_name_prefix or model_dir.name
    encoder_outfile = (
        args.encoder_outfile.resolve()
        if args.encoder_outfile is not None
        else default_encoder_outfile(model_dir, args.outtype)
    )
    decoder_outfile = (
        args.decoder_outfile.resolve()
        if args.decoder_outfile is not None
        else default_decoder_outfile(model_dir, args.outtype)
    )

    convert_one(
        model_dir=model_dir,
        outfile=encoder_outfile,
        outtype=args.outtype,
        model_name=f"{name_prefix} Encoder",
        include_fn=is_encoder_tensor,
        arch=ARCH_ENCODER,
        dry_run=args.dry_run,
    )
    convert_one(
        model_dir=model_dir,
        outfile=decoder_outfile,
        outtype=args.outtype,
        model_name=f"{name_prefix} Decoder",
        include_fn=is_decoder_tensor,
        arch=ARCH_DECODER,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
