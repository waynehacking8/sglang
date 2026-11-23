import itertools
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

import huggingface_hub

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="extra-test-cpu-intel")


class TestQuantLogString(CustomTestCase):
    def test_qwen_fp8_config(self):
        # Example from Qwen/Qwen3-4B-Thinking-2507-FP8
        quant_config = {
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["lm_head"],
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }

        # Create a raw instance
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=fp8, fmt=e4m3"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Qwen FP8] Result: {result}")
        self.assertEqual(result, expected)

    def test_llama_gptq_int4_config(self):
        # Example from hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
        quant_config = {"bits": 4, "quant_method": "gptq", "group_size": 128}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=gptq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test Llama GPTQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_awq_config(self):
        quant_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
        }
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=awq, bits=4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test AWQ] Result: {result}")
        self.assertEqual(result, expected)

    def test_modelopt_nvfp4(self):
        quant_config = {"quant_method": "modelopt_fp4", "quant_algo": "NVFP4"}
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=quant_config)

        expected = "quant=modelopt_fp4, quant_algo=NVFP4"
        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test ModelOpt] Result: {result}")
        self.assertEqual(result, expected)

    def test_no_quant_config(self):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config._parse_quant_hf_config = MagicMock(return_value=None)

        result = model_config.get_quantization_config_log_str()
        print(f"\n[Test No Quant] Result: {result}")
        self.assertIsNone(result)


class TestW4A16Nvfp4MoeLayers(CustomTestCase):
    """ModelOpt exports declare the quantization in config.json or only in
    hf_quant_config.json; the check reads both, as the loader does."""

    def _model_config(self, model_path, quantization, quantization_config=None):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.model_path = model_path
        # What _verify_quantization resolves for these checkpoints.
        model_config.quantization = quantization
        model_config.hf_config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"])
        if quantization_config is not None:
            model_config.hf_config.quantization_config = quantization_config
        return model_config

    def test_reads_config_json_and_hf_quant_config_json(self):
        def mixed(algo):
            return {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.mlp.experts.0.gate_proj": {"quant_algo": algo},
                    "model.layers.0.self_attn.q_proj": {"quant_algo": "W4A16_NVFP4"},
                },
            }

        for name, quant, quantization, expected in (
            ("W4A16_NVFP4", {"quant_algo": "W4A16_NVFP4"}, "modelopt_fp4", True),
            ("NVFP4", {"quant_algo": "NVFP4"}, "modelopt_fp4", False),
            ("mixed W4A16 experts", mixed("W4A16_NVFP4"), "modelopt_mixed", True),
            ("mixed NVFP4 experts", mixed("NVFP4"), "modelopt_mixed", False),
        ):
            with tempfile.TemporaryDirectory() as model_path:
                with self.subTest(source="config.json", checkpoint=name):
                    model_config = self._model_config(
                        model_path, quantization, {"quant_method": "modelopt", **quant}
                    )
                    self.assertIs(model_config.has_w4a16_nvfp4_moe_layers, expected)
                with open(os.path.join(model_path, "hf_quant_config.json"), "w") as f:
                    json.dump({"quantization": quant}, f)
                with self.subTest(source="hf_quant_config.json", checkpoint=name):
                    model_config = self._model_config(model_path, quantization)
                    self.assertIs(model_config.has_w4a16_nvfp4_moe_layers, expected)
        with tempfile.TemporaryDirectory() as model_path:
            self.assertFalse(
                self._model_config(model_path, None).has_w4a16_nvfp4_moe_layers
            )
        # A modelopt_mixed config.json without the per-layer map or the KV-cache
        # metadata defers to hf_quant_config.json, its algorithm and its layer
        # map, as the loader does; a complete one is read as is.
        inline = {"quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION"}
        attention_only = {
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {"quant_algo": "NVFP4"}
            }
        }
        for (name, inline_extra, expected), (
            sidecar_name,
            sidecar,
        ) in itertools.product(
            (
                ("no layer map", {}, True),
                ("attention-only map, no KV metadata", attention_only, True),
                (
                    "complete attention-only map",
                    {**attention_only, "kv_cache_quant_algo": None},
                    False,
                ),
            ),
            (
                ("W4A16 experts", mixed("W4A16_NVFP4")),
                ("W4A16_NVFP4 checkpoint", {"quant_algo": "W4A16_NVFP4"}),
            ),
        ):
            with tempfile.TemporaryDirectory() as model_path:
                with open(os.path.join(model_path, "hf_quant_config.json"), "w") as f:
                    json.dump({"quantization": sidecar}, f)
                with self.subTest(inline=name, sidecar=sidecar_name):
                    model_config = self._model_config(
                        model_path, "modelopt_mixed", {**inline, **inline_extra}
                    )
                    self.assertIs(model_config.has_w4a16_nvfp4_moe_layers, expected)

    def test_probes_the_requested_revision(self):
        """hf_quant_config.json is looked up on the revision the weights come
        from, not the default branch."""
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.model_path = "org/remote-model"
        model_config.revision = "w4a16-branch"
        with (
            mock.patch.object(huggingface_hub.constants, "HF_HUB_OFFLINE", False),
            mock.patch.object(
                huggingface_hub.HfApi, "file_exists", return_value=False
            ) as file_exists,
        ):
            self.assertIsNone(model_config._parse_hf_quant_config_json())
        self.assertEqual(file_exists.call_args.kwargs.get("revision"), "w4a16-branch")


if __name__ == "__main__":
    unittest.main()
