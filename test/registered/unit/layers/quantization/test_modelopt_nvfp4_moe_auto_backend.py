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

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.test.test_utils import CustomTestCase


@contextmanager
def _device(
    capability: tuple,
    live: MoeRunnerBackend,
    tp_ep: tuple = (1, 1),
    device_0_capability: tuple | None = None,
):
    """The worker's CUDA device 1 of ``capability`` (device 0 may differ), with
    ``live`` as the process-wide backend and ``tp_ep`` as the tensor and expert
    parallel sizes."""
    # The platform flags describe device 0.
    is_blackwell = (device_0_capability or capability) >= (10, 0)
    platform = SimpleNamespace(is_blackwell=is_blackwell)
    parallel = SimpleNamespace(tp_size=tp_ep[0], moe_ep_size=tp_ep[1])
    capabilities = {0: device_0_capability or capability, 1: capability}
    with (
        mock.patch.object(modelopt_quant, "get_moe_runner_backend", return_value=live),
        mock.patch.object(modelopt_quant, "get_platform", return_value=platform),
        mock.patch.object(modelopt_quant, "get_parallel", return_value=parallel),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=True),
        mock.patch.object(modelopt_quant, "get_cuda_version", return_value=(12, 8)),
        mock.patch.object(modelopt_quant.torch.cuda, "current_device", return_value=1),
        mock.patch.object(
            modelopt_quant,
            "get_device_capability",
            side_effect=lambda device_id=0: capabilities[device_id],
        ),
    ):
        yield


def _method(
    is_w4a16: bool = False, per_token: bool = False
) -> ModelOptNvFp4FusedMoEMethod:
    return ModelOptNvFp4FusedMoEMethod(
        SimpleNamespace(use_per_token_activation=per_token, is_w4a16=is_w4a16)
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

    def test_w4a16_layers_with_an_a2a_backend_refuse_explicit_fp4_runners(self):
        """With an A2A backend, an explicit backend that quantizes activations
        to FP4 would run W4A16 layers as W4A4, so they stop at startup on it,
        as under auto; the runners that keep their 16-bit activations are kept.
        Without an A2A backend, an explicit backend is kept as on main."""
        env = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
        a2a = SimpleNamespace(is_none=lambda: False, is_megamoe=lambda: False)
        for capability in ((10, 0), (12, 0)):
            for live, cutedsl_w4a16, kept in (
                (MoeRunnerBackend.FLASHINFER_CUTLASS, True, False),
                (MoeRunnerBackend.FLASHINFER_TRTLLM, True, False),
                (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, False, False),
                (MoeRunnerBackend.FLASHINFER_CUTEDSL, False, False),
                (MoeRunnerBackend.FLASHINFER_MEGAMOE, False, False),
                (MoeRunnerBackend.FLASHINFER_CUTEDSL, True, True),
                (MoeRunnerBackend.FLASHINFER_MEGAMOE, True, True),
                (MoeRunnerBackend.MARLIN, False, True),
                (MoeRunnerBackend.LORA_MARLIN, False, True),
            ):
                with self.subTest(
                    capability=capability, live=live.value, env=cutedsl_w4a16
                ):
                    with _device(capability, live), env.override(cutedsl_w4a16):
                        self.assertIs(
                            _method(is_w4a16=True).get_moe_runner_backend(None),
                            live,
                        )
                        with mock.patch.object(
                            modelopt_quant, "get_moe_a2a_backend", return_value=a2a
                        ):
                            # W4A4 layers keep any explicit backend.
                            self.assertIs(_method().get_moe_runner_backend(None), live)
                            if kept:
                                self.assertIs(
                                    _method(is_w4a16=True).get_moe_runner_backend(None),
                                    live,
                                )
                            else:
                                with self.assertRaisesRegex(
                                    ValueError, "quantizes the activations"
                                ):
                                    _method(is_w4a16=True)

    def test_auto_outside_sm100_avoids_trtllm_gen(self):
        """trtllm-gen MoE kernels are SM100-only: elsewhere W4A4 layers run on
        FlashInfer CUTLASS and W4A16 layers on Marlin, and the weight prep
        follows the same resolution."""
        cutlass, marlin = MoeRunnerBackend.FLASHINFER_CUTLASS, MoeRunnerBackend.MARLIN
        for capability, is_w4a16, tp_ep, expected in (
            ((12, 0), False, (1, 1), cutlass),
            ((12, 1), False, (4, 4), cutlass),
            ((12, 0), False, (4, 1), cutlass),
            ((12, 1), True, (1, 1), marlin),
            ((12, 0), True, (4, 2), marlin),
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

    def test_auto_stops_at_startup_where_no_runner_fits(self):
        """At an expert parallel size flashinfer_cutlass rejects, auto has no
        runner for W4A4 layers (Marlin runs them only when asked), so the
        method stops with the sizes instead of picking one silently."""
        for capability in ((12, 0), (11, 0)):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO, tp_ep=(4, 2)):
                    with self.assertRaisesRegex(
                        ValueError,
                        "expert parallel size of 2 and a tensor parallel size of 4",
                    ):
                        _method()

    def test_auto_reads_the_workers_gpu(self):
        """With --base-gpu-id, the worker's GPU is not device 0, for the
        support check as for the resolution."""
        with _device((12, 0), MoeRunnerBackend.AUTO, device_0_capability=(9, 0)):
            method = _method()
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_CUTLASS
        )
        with _device((12, 0), MoeRunnerBackend.AUTO, device_0_capability=(10, 0)):
            method = _method()
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_CUTLASS
        )
        with _device((10, 0), MoeRunnerBackend.AUTO, device_0_capability=(12, 0)):
            method = _method()
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_TRTLLM
        )

    def test_per_token_activation_refuses_flashinfer_cutlass(self):
        """Weight prep replaces the static activation scales with ones for
        per-token scaling, which flashinfer_cutlass never computes, so its
        layers stop before the weights are allocated."""
        layer = SimpleNamespace(moe_runner_config=SimpleNamespace(is_gated=True))
        for capability, live in (
            ((12, 0), MoeRunnerBackend.AUTO),
            ((10, 0), MoeRunnerBackend.FLASHINFER_CUTLASS),
        ):
            with self.subTest(capability=capability, live=live.value):
                with _device(capability, live):
                    method = _method(per_token=True)
                with self.assertRaisesRegex(ValueError, "Per-token NVFP4"):
                    method.create_weights(
                        layer=layer,
                        num_experts=8,
                        hidden_size=64,
                        intermediate_size_per_partition=128,
                        params_dtype=None,
                    )
        for capability, is_w4a16, expected in (
            ((10, 0), False, MoeRunnerBackend.FLASHINFER_TRTLLM),
            ((12, 0), True, MoeRunnerBackend.MARLIN),
        ):
            with self.subTest(capability=capability, w4a16=is_w4a16):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = _method(is_w4a16=is_w4a16, per_token=True)
                self.assertIs(method.get_moe_runner_backend(None), expected)
                self.assertFalse(method.enable_flashinfer_cutlass_moe)

    def test_w4a16_with_an_a2a_backend_stops_at_startup(self):
        """Only Marlin keeps W4A16 layers' 16-bit activations, and it has no
        A2A path, so auto refuses the A2A backend instead of quantizing the
        activations to FP4."""
        for capability in ((10, 0), (12, 0)):
            with self.subTest(capability=capability):
                with (
                    _device(capability, MoeRunnerBackend.AUTO),
                    mock.patch.object(
                        modelopt_quant,
                        "get_moe_a2a_backend",
                        return_value=SimpleNamespace(
                            is_none=lambda: False, is_megamoe=lambda: False
                        ),
                    ),
                    self.assertRaisesRegex(ValueError, "--moe-a2a-backend none"),
                ):
                    _method(is_w4a16=True)

    def test_auto_leaves_the_swizzled_scales_to_weight_prep(self):
        """Weight prep binds the swizzled block scales for every runner that
        reads them, so under auto create_weights allocates none (36 MiB per
        Qwen3-30B-A3B layer); an explicit flashinfer_cutlass allocates them."""
        for live, allocated in (
            (MoeRunnerBackend.AUTO, False),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, True),
        ):
            with self.subTest(live=live.value):
                with _device((12, 0), live):
                    method = ModelOptNvFp4FusedMoEMethod(
                        modelopt_quant.ModelOptFp4Config(
                            is_checkpoint_nvfp4_serialized=True, group_size=16
                        )
                    )
                    layer = torch.nn.Module()
                    layer.moe_runner_config = SimpleNamespace(is_gated=True)
                    layer.num_experts = layer.num_local_experts = 2
                    method.create_weights(
                        layer=layer,
                        num_experts=2,
                        hidden_size=64,
                        intermediate_size_per_partition=64,
                        params_dtype=torch.bfloat16,
                    )
                self.assertIs(method.enable_flashinfer_cutlass_moe, True)
                self.assertIs(layer.w13_blockscale_swizzled is not None, allocated)
                self.assertIs(layer.w2_blockscale_swizzled is not None, allocated)

    def test_marlin_shares_one_global_scale_for_w1_and_w3(self):
        """Marlin takes one global weight scale per expert for w1 and w3. Where
        the checkpoint's differ, the larger one is kept and the other half's
        block scales absorb the ratio, so each half dequantizes to its
        checkpoint value up to one E4M3 rounding of its block scales."""
        torch.manual_seed(0)
        experts, half, blocks = 4, 32, 8
        block_scales = (torch.rand(experts, 2 * half, blocks) * 400 + 1).to(
            torch.float8_e4m3fn
        )
        gate = torch.rand(experts) + 0.5
        for up in (gate.clone(), gate * 3, gate / 3):
            layer = torch.nn.Module()
            layer.w13_weight_scale = torch.nn.Parameter(
                block_scales.clone(), requires_grad=False
            )
            layer.w13_weight_scale_2 = torch.nn.Parameter(
                torch.stack([gate, up], dim=1), requires_grad=False
            )
            shared = modelopt_quant._share_w13_global_scale(layer)
            new = layer.w13_weight_scale.float() * shared[:, None, None]
            old = block_scales.float() * torch.cat(
                [
                    gate[:, None, None].expand(-1, half, 1),
                    up[:, None, None].expand(-1, half, 1),
                ],
                dim=1,
            )
            with self.subTest(ratio=(up / gate)[0].item()):
                self.assertTrue(torch.equal(shared, torch.maximum(gate, up)))
                # E4M3 keeps 3 mantissa bits: one rounding is within 2^-4.
                torch.testing.assert_close(new, old, rtol=2**-4, atol=0)
                if torch.equal(gate, up):
                    self.assertTrue(
                        torch.equal(layer.w13_weight_scale.data, block_scales)
                    )
        # A checkpoint may store one shared column for w1 and w3.
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            gate[:, None].clone(), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            block_scales.clone(), requires_grad=False
        )
        self.assertTrue(
            torch.equal(modelopt_quant._share_w13_global_scale(layer), gate)
        )
        self.assertTrue(torch.equal(layer.w13_weight_scale.data, block_scales))

    def test_cutlass_refuses_gated_shards_it_cannot_pad(self):
        """flashinfer_cutlass pads only non-gated experts, so a per-rank gated
        intermediate size that is not a multiple of 64 (Qwen3-30B-A3B at TP8:
        768 / 8 = 96) stops before the weights are allocated."""
        with _device((12, 0), MoeRunnerBackend.AUTO, tp_ep=(8, 1)):
            method = _method()
        layer = SimpleNamespace(moe_runner_config=SimpleNamespace(is_gated=True))
        with self.assertRaisesRegex(ValueError, "multiple of 64, got 96"):
            method.create_weights(
                layer=layer,
                num_experts=128,
                hidden_size=2048,
                intermediate_size_per_partition=96,
                params_dtype=None,
            )

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
            ((12, 0), (4, 2), True),
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
