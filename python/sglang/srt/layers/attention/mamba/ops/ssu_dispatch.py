from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MambaSSUBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name used for logging."""

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based selective-state-update backend."""

    def __init__(self) -> None:
        from sglang.srt.layers.attention.mamba.ops.mamba_ssm import (
            selective_state_update,
        )

        self._kernel = selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            intermediate_state_indices=intermediate_state_indices,
        )


def _selective_state_update_native(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    out: torch.Tensor | None = None,
    disable_state_update: bool = False,
) -> None:
    """
    Native PyTorch implementation of selective_state_update.
    Matches Triton kernel behavior exactly, including TIE_HDIM optimization.
    All intermediate computation done in float32.

    Args:
        state: (batch, nheads, dim, dstate) or (batch, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim) or (batch, T, nheads, dim)
        dt: same shape as x
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate) or (batch, T, ngroups, dstate)
        C: same shape as B
        D: (dim,) or (nheads, dim)
        z: same shape as x
        dt_bias: (dim,) or (nheads, dim)
        out: same shape as x, in-place updated
    """
    # Normalize shapes to 4D: state (batch, nheads, dim, dstate)
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.dim() == 3:
            z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if out.dim() == 3:
        out = out.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch, T, _, _ = x.shape
    ngroups = B.shape[2]
    nheads_ngroups_ratio = nheads // ngroups

    # Detect TIE_HDIM: A and dt are scalar per head (stride along dim/dstate = 0)
    tie_hdim = (
        A.stride(-1) == 0
        and A.stride(-2) == 0
        and dt.stride(-1) == 0
        and (dt_bias is None or dt_bias.stride(-1) == 0)
    )

    for t_idx in range(T):
        # Get current time step data
        x_t = x[:, t_idx, :, :].float()  # (batch, nheads, dim)
        dt_t = dt[:, t_idx, :, :].float()  # (batch, nheads, dim)
        B_t = B[:, t_idx, :, :].float()  # (batch, ngroups, dstate)
        C_t = C[:, t_idx, :, :].float()  # (batch, ngroups, dstate)

        for b_idx in range(batch):
            # Skip padded entries
            if state_batch_indices is not None:
                state_b_idx = state_batch_indices[b_idx].item()
                if state_b_idx == pad_slot_id:
                    continue
            else:
                state_b_idx = b_idx

            for h_idx in range(nheads):
                group_idx = h_idx // nheads_ngroups_ratio

                # Load state: (dim, dstate)
                s = state[state_b_idx, h_idx].float()  # (dim, dstate)

                # Get dt value and apply bias + softplus
                if tie_hdim:
                    # Scalar dt per head
                    dt_val = dt_t[b_idx, h_idx, 0]  # scalar tensor
                    if dt_bias is not None:
                        dt_val = dt_val + dt_bias[h_idx, 0].float()
                    if dt_softplus:
                        # Match Triton Kernel: only apply softplus when dt <= 20 to avoid overflow
                        # Use torch.where to handle 0-d tensor
                        dt_val = torch.where(
                            dt_val <= 20.0,
                            F.softplus(dt_val),
                            dt_val
                        )
                    # Scalar A per head
                    A_val = A[h_idx, 0, 0].float()  # scalar
                    dA = torch.exp(A_val * dt_val)  # scalar
                    # dB = B * dt (vector)
                    dB = B_t[b_idx, group_idx] * dt_val  # (dstate,)
                    # state update: state = state * dA + outer(x, dB)
                    x_h = x_t[b_idx, h_idx]  # (dim,)
                    s = s * dA + x_h.unsqueeze(-1) * dB.unsqueeze(0)  # (dim, dstate)
                else:
                    # Per-dim dt
                    dt_h = dt_t[b_idx, h_idx]  # (dim,)
                    if dt_bias is not None:
                        dt_h = dt_h + dt_bias[h_idx].float()  # (dim,)
                    if dt_softplus:
                        dt_h = torch.where(dt_h <= 20.0, F.softplus(dt_h), dt_h)
                    # Per-element A: (dim, dstate)
                    A_h = A[h_idx].float()  # (dim, dstate)
                    dA = torch.exp(A_h * dt_h.unsqueeze(-1))  # (dim, dstate)
                    # dB: (dim, dstate)
                    dB = B_t[b_idx, group_idx].unsqueeze(0) * dt_h.unsqueeze(-1)  # (dim, dstate)
                    # state update
                    x_h = x_t[b_idx, h_idx]  # (dim,)
                    s = s * dA + dB * x_h.unsqueeze(-1)  # (dim, dstate)

                # Compute output: out = sum(state * C, dim=-1)
                C_h = C_t[b_idx, group_idx]  # (dstate,)
                out_h = torch.sum(s * C_h.unsqueeze(0), dim=-1)  # (dim,)

                # Add D term
                if D is not None:
                    out_h = out_h + D[h_idx].float() * x_t[b_idx, h_idx]

                # Apply z gating (SiLU)
                if z is not None:
                    z_h = z[b_idx, t_idx, h_idx].float()  # (dim,)
                    out_h = out_h * z_h * torch.sigmoid(z_h)

                # Write output
                out[b_idx, t_idx, h_idx] = out_h.to(out.dtype)

                # Update state (unless disabled)
                if not disable_state_update:
                    state[state_b_idx, h_idx] = s.to(state.dtype)


class NativeSSUBackend(MambaSSUBackend):
    """Native PyTorch selective-state-update backend for Ascend NPU.
    
    Avoids Triton compilation issues on Ascend NPU by using standard
    PyTorch operations. All computation done in float32 to match
    Triton kernel precision.
    """

    @property
    def name(self) -> str:
        return "native"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        _selective_state_update_native(
            state, x, dt, A, B, C,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id, out=out,
            disable_state_update=disable_state_update,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based selective-state-update backend."""

    def __init__(self) -> None:
        from flashinfer.mamba import selective_state_update

        self._kernel = selective_state_update

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        pad_slot_id: int = -1,
        out: torch.Tensor | None = None,
        disable_state_update: bool = False,
        intermediate_states_buffer: torch.Tensor | None = None,
        cache_steps: int | None = None,
        retrieve_parent_token: torch.Tensor | None = None,
        intermediate_state_indices: torch.Tensor | None = None,
    ) -> None:
        if retrieve_parent_token is not None:
            raise ValueError(
                "FlashInfer backend does not support retrieve_parent_token. "
                "Use --mamba-backend triton for EAGLE tree attention."
            )
        # FlashInfer expects cache_steps as an int (0 when unused).
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=pad_slot_id,
            out=out,
            disable_state_update=disable_state_update,
            intermediate_states_buffer=intermediate_states_buffer,
            cache_steps=0 if cache_steps is None else cache_steps,
            intermediate_state_indices=intermediate_state_indices,
        )


_BACKEND_REGISTRY: dict[str, type[MambaSSUBackend]] = {
    "triton": TritonSSUBackend,
    "flashinfer": FlashInferSSUBackend,
    "native": NativeSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


def initialize_mamba_selective_state_update_backend(server_args: ServerArgs) -> None:
    """Instantiate the selective-state-update backend from server config.

    This should be called once during scheduler initialization.

    Args:
        server_args: Server arguments containing ``mamba_backend`` setting.

    Raises:
        ValueError: If the requested backend is unavailable or cannot be imported.
    """
    global _mamba_ssu_backend

    # On NPU, default to native backend to avoid Triton issues
    from sglang.srt.utils import is_npu
    if is_npu():
        requested = server_args.mamba_backend or "native"
    else:
        requested = server_args.mamba_backend or "triton"

    backend_cls = _BACKEND_REGISTRY.get(requested)
    if backend_cls is None:
        raise ValueError(
            f"Unknown mamba backend '{requested}'. "
            f"Available backends: {list(_BACKEND_REGISTRY.keys())}"
        )

    try:
        _mamba_ssu_backend = backend_cls()
    except ImportError:
        raise ValueError(
            f"Mamba backend '{requested}' requested but its dependencies are not "
            f"available. Install the required package or use a different "
            f"--mamba-backend value."
        )

    logger.debug(
        "Mamba selective_state_update backend initialized: %s",
        _mamba_ssu_backend.name,
    )


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
    out: torch.Tensor | None = None,
    disable_state_update: bool = False,
    intermediate_states_buffer: torch.Tensor | None = None,
    cache_steps: int | None = None,
    retrieve_parent_token: torch.Tensor | None = None,
    intermediate_state_indices: torch.Tensor | None = None,
) -> None:
    """Dispatch selective-state-update to the configured backend.

    This function provides a unified interface regardless of the underlying
    backend. Backend-specific argument adaptation is handled inside each
    :class:`MambaSSUBackend` subclass.

    Args:
        state: SSM state tensor (batch, nheads, dim, dstate)
        x: Input tensor
        dt: Delta time tensor
        A: A matrix
        B: B matrix
        C: C matrix
        D: Optional D vector
        z: Optional z tensor for gating
        dt_bias: Optional dt bias
        dt_softplus: Whether to apply softplus to dt
        state_batch_indices: Optional batch indices for state
        out: Preallocated output tensor (in-place updated)
        disable_state_update: If True, don't write back to state (for speculative verify)
        intermediate_states_buffer: Buffer to cache intermediate states
        cache_steps: Total number of steps in the buffer
        retrieve_parent_token: (batch, T) tensor of parent token indices for EAGLE tree attention
        intermediate_state_indices: (batch,) tensor of indices for intermediate_states_buffer operations.
            If provided, uses these indices instead of state_batch_indices for the buffer.
    """
    assert _mamba_ssu_backend is not None, (
        "Mamba selective_state_update backend not initialized. "
        "Call initialize_mamba_selective_state_update_backend() first."
    )

    _mamba_ssu_backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        pad_slot_id=pad_slot_id,
        out=out,
        disable_state_update=disable_state_update,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
        intermediate_state_indices=intermediate_state_indices,
    )
