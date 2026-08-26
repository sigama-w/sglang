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

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP8MoEMethod,
    NPUW4A8MXFP4MoEMethod,
)
from sglang.srt.layers.moe.moe_runner import MoeRunner
from sglang.srt.layers.moe.utils import MoeRunnerBackend, get_moe_runner_backend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

import logging

logger = logging.getLogger(__name__)


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


class NPUMXFP4OnlineMoEMethod(UnquantizedFusedMoEMethod):
    """Online MXFP4 (W4A8) FusedMoE entry for pre-packed MXFP4 checkpoints.

    Covers checkpoints that declare ``quant_method="fp8"`` with
    ``store_dtype="mxfp4"`` (e.g. Xiaomi MiMo-V2.5-Pro-FP4-DFlash): routed
    experts are stored as packed MXFP4 (2 fp4 per byte) with per-32 e8m0 block
    scales, while the non-expert linears stay FP8 (routed to
    ``NPUMXFP8LinearMethod``).

    Weight creation reuses ``Fp8MoEMethod.create_fp8_moe_weight_`` so the
    existing fp8 FP4 weight loader reads the HF mxfp4 checkpoint unchanged
    (same layout as the DSV4 mxfp4 path). ``process_weights_after_loading``
    then bitcasts the int8 packed weight to uint8, reinterprets the float32
    e8m0 scales as uint8, and delegates to ``NPUW4A8MXFP4MoEMethod`` for the
    Ascend layout (``npu_format_cast`` + transpose + scale reshape). The
    forward pass is the kernel's own ``apply`` (grouped matmul, fp4 weight +
    e8m0 scale, dynamic MXFP8 activation), invoked by the Ascend runner via
    ``layer.w13_kernel``/``layer.w2_kernel``.
    """

    def __init__(self, quant_config: Optional["QuantizationConfig"] = None):
        super().__init__()
        self.quant_config = quant_config

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
            use_mxfp8=False,
            is_checkpoint_fp8_serialized=True,
            is_fp4_expert=True,
            params_dtype=params_dtype,
            with_bias=with_bias,
            **extra_weight_attrs,
        )

        # MiMo's expert loader (mimo_v2.load_weights via
        # DeepEPMoE.make_expert_params_mapping) maps the HF checkpoint tensor
        # ...experts.{id}.down_proj.weight_scale to a param named
        # {prefix}_weight_scale (no "_inv"), whereas the fp8 FP4 helper
        # registers {prefix}_weight_scale_inv (DSV4 convention). Re-register
        # under the loader-expected name, reusing the same Parameter object so
        # its weight_loader / shard attrs survive.
        for prefix in ("w13", "w2"):
            inv_param = getattr(layer, f"{prefix}_weight_scale_inv")
            delattr(layer, f"{prefix}_weight_scale_inv")
            layer.register_parameter(f"{prefix}_weight_scale", inv_param)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        backend = get_moe_runner_backend()
        if not (backend.is_auto() or backend.is_ascend()):
            raise ValueError(
                "MXFP4 MoE on Ascend requires --moe-runner-backend 'auto' or "
                f"'ascend', got {backend.value!r}."
            )
        # Attach kernels before building the runner: AscendRunnerCore.__init__
        # reads layer.w2_kernel to pick its activation path.
        layer.w13_kernel = NPUW4A8MXFP4MoEMethod()
        layer.w2_kernel = NPUW4A8MXFP4MoEMethod()
        moe_runner_config.layer = layer
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.ASCEND, moe_runner_config)
        self._aiter_runner = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for prefix in ("w13", "w2"):
            # Packed MXFP4 weight is int8 (2 fp4 per byte); the NPU W4A8 kernel
            # expects uint8. Identical byte layout -> zero-copy bitcast.
            weight = getattr(layer, f"{prefix}_weight")
            # Diagnostic: log weight byte distribution before bitcast
            w_int8 = weight.data.detach().view(torch.uint8).flatten()
            w_unique = torch.unique(w_int8)
            logger.warning_once(
                "NPUMXFP4OnlineMoEMethod %s_weight: dtype=%s shape=%s "
                "num_unique_bytes=%d byte_min=%d byte_max=%d "
                "sample_bytes=%s",
                prefix,
                weight.data.dtype,
                tuple(weight.data.shape),
                int(w_unique.numel()),
                int(w_int8.min()),
                int(w_int8.max()),
                tuple(w_int8[:8].to("cpu").tolist()),
            )
            weight.data = weight.data.contiguous().view(torch.uint8)

            # create_fp8_moe_weight_ allocates per-32 scales as float32 on
            # non-aiter (NPU). Each float32 holds an e8m0 scale value; extract
            # its 8-bit exponent to get the uint8 e8m0 byte the kernel consumes
            # (same technique as NPUMXFP8's [128,128] scale path). If the
            # loader ever returns float8_e8m0fnu directly, a plain view works.
            scale = getattr(layer, f"{prefix}_weight_scale")
            # Diagnostic: log scale dtype/shape/range to identify the checkpoint
            # convention. Two valid encodings exist:
            #   (a) float32 holding the e8m0 scale VALUE 2^(e-127) (power of 2,
            #       e.g. 1.0/0.5/2.0). Bit-extraction of the IEEE-754 exponent
            #       field recovers e.
            #   (b) float32 holding the raw e8m0 BYTE as an integer 0..255
            #       (uint8 storage cast to float32 by the loader). Direct
            #       `.to(uint8)` recovers e; bit-extraction would corrupt it.
            # If values are integers in [0, 255] we are in case (b); otherwise
            # the bit-extraction path handles case (a) and float8_e8m0fnu.
            if scale.data.dtype == torch.float32:
                # Detect case (b): all values are non-negative integers in
                # [0, 255]. cast uint8->float32 yields exact integers, while
                # 2^(e-127) values are never integers except for e=127 (1.0).
                s_flat = scale.data.detach().float().flatten()
                is_int_like = bool(
                    bool(torch.all(s_flat >= 0))
                    and bool(torch.all(s_flat <= 255))
                    and bool(torch.allclose(s_flat, s_flat.round()))
                )
                logger.warning_once(
                    "NPUMXFP4OnlineMoEMethod %s_scale: dtype=%s shape=%s "
                    "min=%.6f max=%.6f is_int_like=%s sample=%s",
                    prefix,
                    scale.data.dtype,
                    tuple(scale.data.shape),
                    float(s_flat.min()),
                    float(s_flat.max()),
                    is_int_like,
                    tuple(s_flat[:5].to("cpu").tolist()),
                )
                if is_int_like:
                    # Case (b): raw e8m0 byte stored as float32 integer.
                    scale_u8 = scale.data.to(torch.uint8)
                else:
                    # Case (a): float32 holding 2^(e-127); extract exponent.
                    scale_u8 = (
                        scale.data.view(torch.int32) >> 23 & 0xFF
                    ).to(torch.uint8)
            elif scale.data.dtype == torch.float8_e8m0fnu:
                scale_u8 = scale.data.view(torch.uint8).clone()
            else:
                scale_u8 = scale.data.to(torch.uint8)
            scale.data = scale_u8

            # NPU offline layout: npu_format_cast(weight) + transpose + scale
            # reshape + dispatcher dtype. The kernel reads {prefix}_weight and
            # {prefix}_weight_scale off the layer.
            getattr(layer, f"{prefix}_kernel").process_weights_after_loading(
                layer, prefix
            )
