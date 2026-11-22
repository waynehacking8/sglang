"""CPU unit tests for how ModelOptNvFp4FusedMoEMethod resolves ``auto``.

The flags that pick the weight layout (create_weights,
process_weights_after_loading) and the runner create_moe_runner builds must
come from one resolution: a layer prepared for one runner and dispatched to
another fails on its first forward. The device and parallel state are stubbed
so the tests stay on CPU.
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
    platform = SimpleNamespace(is_blackwell=is_blackwell)
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


def _create_runner(
    method: ModelOptNvFp4FusedMoEMethod, apply_router_weight_on_input: bool = False
) -> MoeRunnerBackend:
    """Run create_moe_runner and return the backend it built MoeRunner with."""
    with mock.patch.object(modelopt_quant, "MoeRunner") as runner:
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                activation="silu",
                is_gated=True,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ),
        )
    runner.assert_called_once()
    return runner.call_args.args[0]


class TestNvFp4MoeAutoBackend(CustomTestCase):
    def test_auto_on_blackwell_prepares_trtllm_weights_for_the_trtllm_runner(self):
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = _method()
                    runner = _create_runner(method)
                # The weight prep flags and the runner must agree.
                self.assertTrue(method.enable_flashinfer_trtllm_moe)
                self.assertTrue(method.use_flashinfer_trtllm_weight_layout)
                self.assertTrue(runner.is_flashinfer_trtllm())

    def test_auto_on_hopper_falls_back_to_marlin(self):
        with _device((9, 0), MoeRunnerBackend.AUTO):
            method = _method()
            runner = _create_runner(method)
        self.assertFalse(method.enable_flashinfer_trtllm_moe)
        self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
        self.assertTrue(runner.is_marlin())

    def test_explicit_backend_is_kept(self):
        for live, is_trtllm in (
            (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, True),
            (MoeRunnerBackend.FLASHINFER_TRTLLM, True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, False),
        ):
            with self.subTest(live=live.value):
                with _device((10, 0), live):
                    method = _method()
                    runner = _create_runner(method)
                self.assertIs(runner, live)
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
                    runner = _create_runner(method)
                self.assertIs(runner, expected)
                self.assertIs(method.get_moe_runner_backend(None), expected)
                self.assertFalse(method.enable_flashinfer_trtllm_moe)
                self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
                uses_cutlass = expected is cutlass
                self.assertIs(method.enable_flashinfer_cutlass_moe, uses_cutlass)
                self.assertIs(method.load_up_proj_weight_first, uses_cutlass)

    def test_auto_keeps_w4a16_layers_on_marlin(self):
        """trtllm-gen has only the W4A4 NVFP4 MoE path, so W4A16 layers resolve
        to Marlin on SM100 as well, with the weight prep that goes with it."""
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = _method(is_w4a16=True)
                    runner = _create_runner(method)
                self.assertIs(runner, MoeRunnerBackend.MARLIN)
                self.assertIs(
                    method.get_moe_runner_backend(None), MoeRunnerBackend.MARLIN
                )
                self.assertFalse(method.enable_flashinfer_trtllm_moe)
                self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
                self.assertFalse(method.enable_flashinfer_cutlass_moe)

    def test_runner_follows_the_init_resolution(self):
        """create_moe_runner builds the runner __init__ resolved, which the
        weight layout follows, whatever the process-wide backend reads when
        it runs."""
        with _device((12, 0), MoeRunnerBackend.AUTO):
            method = _method()
        with _device((12, 0), MoeRunnerBackend.FLASHINFER_TRTLLM):
            runner = _create_runner(method)
        self.assertIs(runner, MoeRunnerBackend.FLASHINFER_CUTLASS)

    def test_marlin_refuses_router_weights_on_the_input(self):
        """Marlin applies router weights to the expert output, so a layer that
        applies them to its input (Llama 4) stops at startup wherever its
        backend resolves to Marlin, and still builds on CUTLASS."""
        for capability, tp_ep, is_w4a16 in (
            ((9, 0), (1, 1), False),
            ((12, 0), (4, 2), False),
            ((12, 0), (1, 1), True),
        ):
            with self.subTest(capability=capability, tp_ep=tp_ep, w4a16=is_w4a16):
                with _device(capability, MoeRunnerBackend.AUTO, tp_ep):
                    method = _method(is_w4a16)
                    with self.assertRaisesRegex(
                        ValueError, "apply_router_weight_on_input"
                    ):
                        _create_runner(method, apply_router_weight_on_input=True)
        with _device((12, 0), MoeRunnerBackend.AUTO):
            method = _method()
            runner = _create_runner(method, apply_router_weight_on_input=True)
        self.assertIs(runner, MoeRunnerBackend.FLASHINFER_CUTLASS)

    def test_online_quantization_resolves_like_a_w4a4_checkpoint(self):
        """Experts quantized at load time (a BF16 checkpoint with
        --quantization modelopt_fp4, or FP8 MTP experts converted to NVFP4)
        build the method from the online NVFP4 config, which resolves auto as a
        serialized W4A4 checkpoint does."""
        from sglang.srt.layers.quantization.nvfp4_online import (
            ModelOptNvFp4OnlineFusedMoEMethod,
        )

        config = modelopt_quant.ModelOptFp4Config.for_online_weight_quantization()
        for capability, expected in (
            ((10, 0), MoeRunnerBackend.FLASHINFER_TRTLLM),
            ((12, 0), MoeRunnerBackend.FLASHINFER_CUTLASS),
        ):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = ModelOptNvFp4OnlineFusedMoEMethod(
                        config, "model.layers.0.mlp.experts"
                    )
                self.assertIs(method.get_moe_runner_backend(None), expected)

    def test_mixed_precision_w4a16_experts_resolve_to_marlin(self):
        """MIXED_PRECISION hands each MoE layer the config of its own
        quant_algo, so on SM120 the W4A16 experts take Marlin and the NVFP4
        experts FlashInfer CUTLASS."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

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
        layer = object.__new__(FusedMoE)
        with _device((12, 0), MoeRunnerBackend.AUTO):
            w4a16 = config.get_quant_method(layer, "model.layers.0.mlp.experts")
            nvfp4 = config.get_quant_method(layer, "model.layers.1.mlp.experts")
        self.assertIs(w4a16.get_moe_runner_backend(None), MoeRunnerBackend.MARLIN)
        self.assertIs(
            nvfp4.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_CUTLASS
        )


if __name__ == "__main__":
    unittest.main()
