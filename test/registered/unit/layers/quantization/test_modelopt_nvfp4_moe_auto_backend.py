"""CPU unit tests for how ModelOptNvFp4FusedMoEMethod resolves ``auto``.

With ``--moe-runner-backend auto`` the method used to compute its TRT-LLM
flags from the live global (which still said ``auto``) while create_moe_runner
resolved ``auto`` to FlashInfer TRT-LLM on Blackwell. create_weights and
process_weights_after_loading then prepared a non-TRT-LLM layout (no
g1_scale_c) and apply() dispatched to the TRT-LLM kernel, which read
``layer.g1_scale_c`` and raised AttributeError on the first forward. These
tests pin that the flags and the cached runner backend come from one
resolution, for every entry point of the method.

The platform and device checks are stubbed so the tests stay on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.test.test_utils import CustomTestCase


@contextmanager
def _device(capability: tuple, live: MoeRunnerBackend, tp_ep: tuple = (1, 1)):
    """A CUDA device of ``capability`` with ``live`` as the process-wide backend
    and ``tp_ep`` as the tensor and expert parallel sizes."""
    is_blackwell = capability >= (10, 0)
    platform = SimpleNamespace(
        is_blackwell=is_blackwell,
        is_sm110=capability[0] == 11,
        is_sm120=capability[0] == 12,
    )
    parallel = SimpleNamespace(tp_size=tp_ep[0], moe_ep_size=tp_ep[1])
    with (
        mock.patch.object(modelopt_quant, "get_moe_runner_backend", return_value=live),
        mock.patch.object(modelopt_quant, "get_platform", return_value=platform),
        mock.patch.object(modelopt_quant, "get_parallel", return_value=parallel),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=True),
        mock.patch.object(
            modelopt_quant, "get_device_capability", return_value=capability
        ),
    ):
        yield


def _method(is_w4a16: bool = False) -> ModelOptNvFp4FusedMoEMethod:
    return ModelOptNvFp4FusedMoEMethod(
        SimpleNamespace(use_per_token_activation=False, is_w4a16=is_w4a16)
    )


def _create_runner(method: ModelOptNvFp4FusedMoEMethod) -> None:
    with mock.patch.object(modelopt_quant, "MoeRunner", return_value="runner"):
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                activation="silu",
                is_gated=True,
            ),
        )


class TestNvFp4MoeAutoBackend(CustomTestCase):
    def test_auto_on_blackwell_prepares_trtllm_weights_for_the_trtllm_runner(self):
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = _method()
                    _create_runner(method)
                # The weight prep flags and the cached runner must agree.
                self.assertTrue(method.enable_flashinfer_trtllm_moe)
                self.assertTrue(method.use_flashinfer_trtllm_weight_layout)
                self.assertTrue(method._moe_runner_backend.is_flashinfer_trtllm())

    def test_auto_on_hopper_falls_back_to_marlin(self):
        with _device((9, 0), MoeRunnerBackend.AUTO):
            method = _method()
            _create_runner(method)
        self.assertFalse(method.enable_flashinfer_trtllm_moe)
        self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
        self.assertTrue(method._moe_runner_backend.is_marlin())

    def test_explicit_backend_is_kept(self):
        for live, is_trtllm in (
            (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, True),
            (MoeRunnerBackend.FLASHINFER_TRTLLM, True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, False),
        ):
            with self.subTest(live=live.value):
                with _device((10, 0), live):
                    method = _method()
                    _create_runner(method)
                self.assertIs(method._moe_runner_backend, live)
                self.assertEqual(method.enable_flashinfer_trtllm_moe, is_trtllm)

    def test_auto_on_sm110_and_sm120_avoids_trtllm_gen(self):
        """trtllm-gen MoE kernels are SM100-only: W4A4 layers run on FlashInfer
        CUTLASS where an explicit flashinfer_cutlass is accepted (EP 1 or TP),
        the rest on Marlin, and the weight prep follows the same resolution."""
        cutlass, marlin = MoeRunnerBackend.FLASHINFER_CUTLASS, MoeRunnerBackend.MARLIN
        for capability, is_w4a16, tp_ep, expected in (
            ((12, 0), False, (1, 1), cutlass),
            ((12, 1), False, (4, 4), cutlass),
            ((12, 0), False, (4, 1), cutlass),
            ((12, 0), False, (4, 2), marlin),
            ((12, 1), True, (1, 1), marlin),
            ((11, 0), False, (1, 1), cutlass),
            ((11, 0), True, (1, 1), marlin),
        ):
            with self.subTest(capability=capability, w4a16=is_w4a16, tp_ep=tp_ep):
                with _device(capability, MoeRunnerBackend.AUTO, tp_ep):
                    method = _method(is_w4a16)
                    _create_runner(method)
                self.assertIs(method._moe_runner_backend, expected)
                self.assertFalse(method.enable_flashinfer_trtllm_moe)
                self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
                uses_cutlass = expected is cutlass
                self.assertIs(method.enable_flashinfer_cutlass_moe, uses_cutlass)
                with mock.patch.object(
                    modelopt_quant,
                    "get_moe_runner_backend",
                    return_value=MoeRunnerBackend.AUTO,
                ):
                    self.assertIs(method.load_up_proj_weight_first, uses_cutlass)

    def test_mixed_precision_marks_its_w4a16_layers(self):
        config = modelopt_quant.ModelOptMixedPrecisionConfig.from_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "model.layers.0.mlp.experts": {
                            "quant_algo": "W4A16_NVFP4",
                            "group_size": 16,
                        },
                        "model.layers.1.mlp.experts": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                    },
                }
            }
        )
        self.assertTrue(config.nvfp4a16_config.is_w4a16)
        self.assertFalse(config.nvfp4_config.is_w4a16)


if __name__ == "__main__":
    unittest.main()
