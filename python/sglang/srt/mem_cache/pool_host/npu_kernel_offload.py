"""sgl-kernel-npu-backed host DRAM and AIV sparse-copy IO path (NPU only).

Kernel-side replacement for the Memfabric acc_offload path: the host pool is
allocated through the sgl-kernel-npu hugepage host allocator (1GB hugepage
mmap + halHostRegister, see csrc/hicache_offload/hugepage_host_allocator) and
the L2<->L1 IO uses the ported AIV kv_exchange kernel
(torch.ops.npu.kv_exchange).

SGLANG_HICACHE_HOST_MEM_BACKEND=kernel_npu is the single switch for this
whole path; SGLANG_HICACHE_HOST_MEM_BACKEND=memfabric keeps the original
Memfabric behavior. Both backends share the same meta layout and the same
AscendC IO control flow (see ascendc_io_enabled).
"""

from __future__ import annotations

import logging
import os
from math import prod
from typing import List, Sequence

import torch

from sglang.srt.mem_cache.pool_host.npu_memfabric import (
    memfabric_host_memory_enabled,
    track_pinned_staging,
)

logger = logging.getLogger(__name__)

_KERNEL_NPU_GB = 1024**3
_kernel_npu_state = {
    "registered_dev_ptrs": {},  # host_ptr (int) -> dev_ptr (int)
    "allocated_bytes": 0,
    "device_id": None,
    "logged": False,
}


def kernel_npu_host_memory_enabled() -> bool:
    """Single switch for the sgl-kernel-npu host pool + AIV IO path."""
    return os.environ.get("SGLANG_HICACHE_HOST_MEM_BACKEND", "").lower() == (
        "kernel_npu"
    )


def ascendc_io_enabled() -> bool:
    """Use the AIV sparse-copy kernel for HiCache L2<->L1 IO.

    True for both NPU host-memory backends (kernel_npu and memfabric): the
    AIV kernel de-references host pool pointers, which requires registered
    host memory from either allocator.
    """
    return (
        memfabric_host_memory_enabled() or kernel_npu_host_memory_enabled()
    )


def ensure_kernel_npu_capacity(total_bytes: int, device_id: int) -> None:
    """Bookkeeping-only capacity check (allocations are per-buffer mmaps).

    Mirrors ensure_memfabric_capacity so mla.py can size both backends with
    the same combined-bytes computation; the hugepage allocator has no global
    reserve, so this only logs once and raises on absurd requests.
    """
    if not _kernel_npu_state["logged"]:
        _kernel_npu_state["logged"] = True
        _kernel_npu_state["device_id"] = device_id
        logger.info(
            "[HiCache] sgl-kernel-npu hugepage host memory backend enabled: "
            "device=%d (first pool needs %.2fGB; backed by 1GB hugepages, "
            "falling back to 2MB/4KB pages — configure /proc/sys/vm/nr_hugepages)",
            device_id,
            total_bytes / _KERNEL_NPU_GB,
        )


def alloc_with_kernel_npu(
    dims: tuple,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """Allocate a host tensor backed by registered hugepage memory.

    The returned CPU tensor's data_ptr maps to a device-visible address (DVA)
    which the kv_exchange kernel receives through the meta host_base field
    (see get_dev_ptr).
    """
    from sgl_kernel_npu.hicache_offload import create_hugepage_tensor

    numel = int(prod(int(d) for d in dims))
    if numel <= 0:
        raise ValueError(f"hugepage tensor needs a positive numel, got dims={dims}")
    tensor, host_ptr, dev_ptr = create_hugepage_tensor(
        tuple(int(d) for d in dims), dtype
    )
    _kernel_npu_state["registered_dev_ptrs"][host_ptr] = dev_ptr
    _kernel_npu_state["allocated_bytes"] += numel * dtype.itemsize
    return tensor


def get_dev_ptr(host_ptr: int) -> int:
    """Return the device-visible address registered for a host pointer."""
    dev_ptr = _kernel_npu_state["registered_dev_ptrs"].get(int(host_ptr))
    if dev_ptr is None:
        raise ValueError(
            f"kv_exchange: host_base 0x{int(host_ptr):x} is not a "
            "hugepage-registered allocation (kernel_npu backend)"
        )
    return dev_ptr


def launch_kv_exchange(vals: Sequence[int], device) -> None:
    """Upload the packed meta vector and launch the AIV kv_exchange kernel.

    ``vals`` is the same layout Memfabric's kv_exchange_copy consumes (see
    build_kv_exchange_meta / mla.py's AscendC helpers). Host bases are
    resolved to DVAs through the hugepage allocator registry before launch.
    """
    from sgl_kernel_npu.hicache_offload import kv_exchange as kernel_kv_exchange

    _KV_EXCHANGE_META_HEADER = 8
    _KV_EXCHANGE_META_STRIDE = 9
    _KV_EXCHANGE_HOST_BASE_OFFSET = 1

    meta_vals: List[int] = list(vals)
    num_components = int(meta_vals[0])
    if not 0 < num_components <= 4:
        raise ValueError(
            f"kv_exchange: invalid num_components {num_components} in meta"
        )
    for c in range(num_components):
        idx = (
            _KV_EXCHANGE_META_HEADER
            + _KV_EXCHANGE_META_STRIDE * c
            + _KV_EXCHANGE_HOST_BASE_OFFSET
        )
        meta_vals[idx] = get_dev_ptr(meta_vals[idx])

    pinned_meta = torch.tensor(meta_vals, dtype=torch.int64, pin_memory=True)
    meta = torch.empty(pinned_meta.shape, dtype=torch.int64, device=device)
    meta.copy_(pinned_meta, non_blocking=True)
    track_pinned_staging(pinned_meta)
    kernel_kv_exchange(meta)
