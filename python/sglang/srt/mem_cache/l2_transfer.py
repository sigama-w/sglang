from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import cache
from typing import Any, Callable, NamedTuple, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.base import shared_host_layout_domains
from sglang.srt.utils import get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


@cache
def _timing_events_supported() -> bool:
    try:
        device_module.Event(enable_timing=True)
        return True
    except (TypeError, NotImplementedError):
        logger.warning(
            "%s.Event does not support timing; L2 transfer timing is disabled",
            device_module.__name__,
        )
        return False


def make_timing_event_pair():
    timing_enabled = _timing_events_supported()
    if not timing_enabled:
        # Without event timing the start event would never be read (only
        # elapsed_time consumers touch it), so keep the completion signal
        # alone instead of creating and recording a dead event.
        return None, device_module.Event(), False
    kwargs = {"enable_timing": True}
    return device_module.Event(**kwargs), device_module.Event(**kwargs), True


class L2Transfer(NamedTuple):
    host_pool: Any
    device_pool: Any
    host_indices: torch.Tensor
    device_indices: torch.Tensor
    layer_mapper: Optional[Callable[[int], Optional[int]]] = None
    is_draft: bool = False


class TransferCompletion(NamedTuple):
    start_event: Any
    finish_event: Any
    timing_enabled: bool


class L2TransferEngine:
    """Runs resolved device↔host transfers without owning cache state."""

    def __init__(self, io_backend: str):
        self.io_backend = io_backend
        self.device_to_host_stream = device_module.Stream()
        self.host_to_device_stream = device_module.Stream()

    @staticmethod
    def _resolve_device_indices(transfer: L2Transfer) -> torch.Tensor:
        """Resolve controller IDs into this pool's device-buffer indices.

        Call on the transfer stream after the producer event. The host-transfer
        move gate prevents relocation until the transfer completes.
        """
        # Mamba state pools do not inherit KVCache's default attributes.
        translate = getattr(transfer.device_pool, "host_transfer_translate", None)
        if translate is None:
            return transfer.device_indices
        original_device = transfer.device_indices.device
        # The direct backend supplies CPU indices even for a CUDA pool.
        # Translate alongside the v2p table, then restore the backend's device.
        indices = transfer.device_indices.to(
            getattr(transfer.device_pool, "device", original_device)
        ).contiguous()
        dcp_size = getattr(transfer.host_pool, "dcp_size", 1)
        if dcp_size > 1:
            # The MLA host pool selects this rank and collapses logical IDs.
            # Translate in local virtual space, then preserve that widened
            # interface (including token order) for the host pool.
            resolved = translate(indices // dcp_size) * dcp_size + indices % dcp_size
        else:
            resolved = translate(indices)
        return resolved.to(original_device)

    def _prepare_transfers(self, transfers: list[L2Transfer]) -> list[L2Transfer]:
        prepared = []
        for transfer in transfers:
            host_indices, device_indices = transfer.host_pool.prepare_transfer_indices(
                transfer.host_indices,
                self._resolve_device_indices(transfer),
                self.io_backend,
            )
            prepared.append(
                transfer._replace(
                    host_indices=host_indices,
                    device_indices=device_indices,
                )
            )
        return prepared

    @contextmanager
    def _submission(self, transfers, stream, transfer_key, start_event=None):
        ack_start, ack_finish, timing_enabled = make_timing_event_pair()
        completion = TransferCompletion(ack_start, ack_finish, timing_enabled)
        domains = shared_host_layout_domains(t.host_pool for t in transfers)
        for domain in domains:
            domain.acquire_layout()
        # Index preparation can read producer-owned device indices; without a
        # caller-supplied producer event, order the transfer against the
        # caller's current stream through one native stream wait instead of
        # creating and recording a throwaway event.
        caller_stream = None if start_event is not None else device_module.current_stream()
        finish_recorded = False
        try:
            with device_module.stream(stream):
                if start_event is not None:
                    start_event.wait(stream)
                else:
                    stream.wait_stream(caller_stream)
                prepared = self._prepare_transfers(transfers)
                if ack_start is not None:
                    ack_start.record()
                yield prepared, completion
                ack_finish.record()
                finish_recorded = True
                self._record_stream(transfers + prepared, stream)
        except Exception:
            stream.synchronize()
            raise
        finally:
            for domain in reversed(domains):
                domain.release_layout(
                    ack_finish if finish_recorded else None,
                    (id(self), transfer_key),
                )

    def submit_device_to_host(self, transfers: list[L2Transfer]) -> TransferCompletion:
        with self._submission(
            transfers, self.device_to_host_stream, "device_to_host"
        ) as (transfers, completion):
            for transfer in transfers:
                transfer.host_pool.backup_from_device_all_layer_physical(
                    transfer.device_pool,
                    transfer.host_indices,
                    transfer.device_indices,
                    self.io_backend,
                )
        return completion

    def submit_host_to_device(
        self,
        transfers: list[L2Transfer],
        *,
        transfer_layer_id_max: int,
        start_event=None,
        on_layer_done=None,
    ) -> TransferCompletion:
        with self._submission(
            transfers, self.host_to_device_stream, "host_to_device", start_event
        ) as (transfers, completion):
            primary = transfers[0] if transfers else None
            delay_cycles = envs.SGLANG_TEST_HICACHE_H2D_DELAY_CYCLES.get()
            for layer_id in range(transfer_layer_id_max):
                if delay_cycles:
                    torch.cuda._sleep(delay_cycles)
                for transfer in transfers:
                    local_layer_id = (
                        transfer.layer_mapper(layer_id)
                        if transfer.layer_mapper is not None
                        else layer_id
                    )
                    if local_layer_id is None or (
                        transfer is not primary
                        and transfer.layer_mapper is None
                        and layer_id >= transfer.host_pool.layer_num
                    ):
                        continue
                    transfer.host_pool.load_to_device_per_layer_physical(
                        transfer.device_pool,
                        transfer.host_indices,
                        transfer.device_indices,
                        local_layer_id,
                        self.io_backend,
                        is_draft=transfer.is_draft,
                    )
                if on_layer_done is not None:
                    on_layer_done(layer_id)
        return completion

    @staticmethod
    def _record_stream(transfers: list[L2Transfer], stream, resolved=()) -> None:
        tensors = []
        for transfer in transfers:
            tensors.extend((transfer.host_indices, transfer.device_indices))
        # Keep temporary translated indices alive until the transfer completes.
        tensors.extend(resolved)
        for indices in tensors:
            if indices is not None and (
                indices.is_cuda or indices.device.type == "xpu"
            ):
                indices.record_stream(stream)
