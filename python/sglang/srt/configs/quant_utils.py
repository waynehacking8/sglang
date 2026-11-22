"""Read a checkpoint's quantization settings from its HF config files."""

import json
import os

# ModelConfig picks the Marlin method for these itself only when --quantization is
# unset or names a Marlin method, so a model override must not write them into
# server_args.quantization.
LOADER_RESOLVED_QUANT_METHODS = frozenset({"awq", "gptq"})


def has_hf_quant_config(model_path: str) -> bool:
    """Check if the model path contains hf_quant_config.json file.

    Args:
        model_path: Path to the model, can be local path or remote URL.

    Returns:
        True if hf_quant_config.json exists, False otherwise.
    """
    # Local paths are decided on the filesystem; the hub helpers below
    # reject them as invalid repo ids.
    if os.path.isdir(model_path):
        return os.path.isfile(os.path.join(model_path, "hf_quant_config.json"))

    from huggingface_hub import try_to_load_from_cache

    # Check if the model_path is a HuggingFace model ID and exists locally
    result = try_to_load_from_cache(model_path, "hf_quant_config.json")
    if isinstance(result, str):
        return True

    # Check if the model_path is a remote URL and exists on the HuggingFace Hub
    try:
        from huggingface_hub import HfApi

        hf_api = HfApi()
        return hf_api.file_exists(model_path, "hf_quant_config.json")
    except Exception:
        return False


def get_quantization_config(hf_config) -> str | None:
    """Extract quantization method from HuggingFace config."""
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is not None:
        return quantization_config.get("quant_method")
    return None


def _modelopt_section(config) -> dict:
    """The ModelOpt settings of a nested or flat quantization config."""
    from sglang.srt.configs.model_config import (
        _quant_config_to_dict,
        unwrap_modelopt_quantization_config,
    )

    config = _quant_config_to_dict(config)
    if not isinstance(config, dict):
        return {}
    return unwrap_modelopt_quantization_config(config)


def _load_hf_quant_config(model_path: str, revision: str | None) -> dict | None:
    """hf_quant_config.json from a local model directory or the HF cache."""
    if os.path.isdir(model_path):
        path = os.path.join(model_path, "hf_quant_config.json")
    else:
        from huggingface_hub import try_to_load_from_cache

        try:
            path = try_to_load_from_cache(
                model_path, "hf_quant_config.json", revision=revision
            )
        except Exception:  # not a repo id, e.g. a missing local path
            return None
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def get_modelopt_quant_section(
    hf_config, model_path: str | None = None, revision: str | None = None
) -> dict:
    """ModelOpt settings from config.json, or from hf_quant_config.json when
    config.json has none, or has a MIXED_PRECISION config without the per-layer
    map or the KV-cache settings: the cases where the weight loader reads the
    file (weight_utils.get_quant_config). The file is read when it is local or
    in the HF cache (ModelConfig downloads it when config.json declares no
    quantization); otherwise config.json's settings are used."""
    section = _modelopt_section(getattr(hf_config, "quantization_config", None))
    needs_file = not section or (
        str(section.get("quant_algo", "")).upper() == "MIXED_PRECISION"
        and (
            "quantized_layers" not in section
            or (
                "kv_cache_quant_algo" not in section
                and "kv_cache_scheme" not in section
            )
        )
    )
    if needs_file and model_path is not None:
        section = (
            _modelopt_section(_load_hf_quant_config(model_path, revision)) or section
        )
    return section


def get_mixed_precision_moe_quant_algos(
    hf_config, model_path: str | None = None, revision: str | None = None
) -> set:
    """quant_algo values ModelOpt MIXED_PRECISION assigns to `*.experts` layers."""
    section = get_modelopt_quant_section(hf_config, model_path, revision)
    return {
        str(info.get("quant_algo", "")).upper()
        for name, info in (section.get("quantized_layers") or {}).items()
        if ".experts" in name and isinstance(info, dict)
    }


def has_fp8_weights_in_checkpoint(model_path: str) -> bool:
    """Check if a model checkpoint actually contains FP8 (float8_e4m3fn) expert
    weight tensors by reading safetensors metadata headers.

    This is needed because some models (e.g. DeepSeek V3/R1) use native FP8 MoE
    experts without declaring it in quantization_config, while other models
    sharing the same architecture (e.g. Moonlight) are purely BF16.

    Accepts a local directory or a HuggingFace repo ID. For remote repos, only
    safetensors headers (a few KB) are fetched via byte-range reads; full
    shards are never downloaded.
    """
    import json
    import struct

    try:
        if os.path.isdir(model_path):

            def _open(name):
                return open(os.path.join(model_path, name), "rb")

            def _exists(name):
                return os.path.exists(os.path.join(model_path, name))

        else:
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()

            def _open(name):
                return fs.open(f"{model_path}/{name}", "rb")

            def _exists(name):
                return fs.exists(f"{model_path}/{name}")

        if _exists("model.safetensors.index.json"):
            with _open("model.safetensors.index.json") as f:
                weight_map = json.loads(f.read()).get("weight_map", {})
            expert_files = sorted(
                {v for k, v in weight_map.items() if "experts" in k and "weight" in k}
            )
            shard_file = (
                expert_files[0]
                if expert_files
                else next(iter(sorted(set(weight_map.values()))), None)
            )
            if shard_file is None:
                return False
        elif _exists("model.safetensors"):
            shard_file = "model.safetensors"
        else:
            return False

        with _open(shard_file) as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))

        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if "experts" in key and "weight" in key:
                return meta.get("dtype") == "F8_E4M3"
        return False
    except Exception:
        return False
