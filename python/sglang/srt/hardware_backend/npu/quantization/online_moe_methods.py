"""Online (config-driven) quantized FusedMoE methods for Ascend NPU.

These are the ``--quantization <scheme>`` entry points: the checkpoint holds
BF16/FP16 expert weights and the per-gmm kernels quantize them at load time.
Offline (msmodelslim) checkpoints go through the ModelSlim schemes instead and
reuse the same kernels.

Kept out of ``moe_methods.py`` because ``unquant.py`` imports that module at
module scope, so subclassing ``UnquantizedFusedMoEMethod`` there would be a
circular import.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUMXFP8MoEMethod
from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


class NPUMXFP8OnlineMoEMethod(UnquantizedFusedMoEMethod):
    """Online MXFP8 FusedMoE entry point (``--quantization mxfp8`` on A5).

    Weight creation, weight post-processing and the forward pass are identical
    to the unquantized Ascend path — the only difference is which per-gmm kernel
    the layer gets, so everything but ``create_moe_runner`` is inherited.
    ``NPUMXFP8MoEMethod`` then quantizes the BF16 expert weights to MXFP8 in
    ``process_weights_after_loading``.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.quant_config = quant_config
        # True when the checkpoint holds FP8 weights + float32 scales and needs
        # dequantization to BF16 before MXFP8 requantization.
        self._fp8_checkpoint = bool(
            quant_config is not None
            and getattr(quant_config, "is_checkpoint_fp8_serialized", False)
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        if self._fp8_checkpoint:
            # FP8 checkpoint: create float8_e4m3fn weight + float32 scale params
            # so the loader can populate them; dequant→BF16→MXFP8 happens in
            # process_weights_after_loading.
            from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

            quant_config = self.quant_config
            block_quant = quant_config.weight_block_size is not None
            Fp8MoEMethod.create_fp8_moe_weight_(
                layer=layer,
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size_per_partition,
                block_quant=block_quant,
                quant_config=quant_config,
                use_mxfp8=False,  # standard FP8: float32 scales, not uint8 e8m0
                is_checkpoint_fp8_serialized=True,
                is_fp4_expert=False,
                params_dtype=params_dtype,
                with_bias=with_bias,
                **extra_weight_attrs,
            )
            return
        # BF16 checkpoint: create BF16 params; MXFP8 quant happens in
        # process_weights_after_loading (inherited from UnquantizedFusedMoEMethod
        # → NPUMXFP8MoEMethod).
        super().create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            with_bias=with_bias,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._fp8_checkpoint:
            # Reinterpret [128,128] block-FP8 float32 scales as A5 MXFP8 e8m0
            # scales (same technique as NPUMXFP8LinearMethod). Extract the 8-bit
            # exponent from the float32 bits, expand from per-128 to per-32 (K)
            # and replicate across 128 (N), producing uint8 [E, N, K//32]. Then
            # rename weight_scale_inv → weight_scale so the inherited offline
            # path in NPUMXFP8MoEMethod picks it up (reshape pairs + transpose).
            block_n, block_k = self.quant_config.weight_block_size
            for prefix in ("w13", "w2"):
                scale_inv = getattr(layer, f"{prefix}_weight_scale_inv")
                # [E, N//bn, K//bk] float32 → [E, N//bn, K//bk] uint8 exponent
                scale_u8 = (
                    scale_inv.data.view(torch.int32) >> 23 & 0xFF
                ).to(torch.uint8)
                # Expand K: 128→4×32, then N: replicate across 128 rows
                scale_u8 = scale_u8.repeat_interleave(
                    block_k // 32, dim=2
                ).repeat_interleave(block_n, dim=1)
                # [E, N, K//32] uint8 — same layout as offline ModelSlim scales
                delattr(layer, f"{prefix}_weight_scale_inv")
                layer.register_parameter(
                    f"{prefix}_weight_scale",
                    torch.nn.Parameter(scale_u8, requires_grad=False),
                )
        # Inherited path: NPUMXFP8MoEMethod.process_weights_after_loading
        # sees float8_e4m3fn weight + uint8 weight_scale → offline layout path.
        super().process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_ascend()):
            # Not merely a wrong-runner check. Because this method subclasses
            # UnquantizedFusedMoEMethod it matches FusedMoE's shard-swap list, so
            # a flashinfer backend would make the weight loader exchange the
            # w1/w3 shards ("flashinfer assumes w31 format"). Every expert would
            # then load with gate and up swapped, and gmm1's fused swiglu would
            # compute silu(up) * gate — no error, just degenerate output.
            raise ValueError(
                "MXFP8 MoE on Ascend requires --moe-runner-backend 'auto' or "
                f"'ascend', got {backend.value!r}."
            )

        # The kernels must be attached before the runner is built:
        # AscendRunnerCore.__init__ reads layer.w2_kernel to pick its activation.
        layer.w13_kernel = NPUMXFP8MoEMethod("w13")
        layer.w2_kernel = NPUMXFP8MoEMethod("w2")
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)
        # Inherited apply() consults this; aiter is CUDA/ROCm-only.
        self._aiter_runner = None
