# MemFabric H2D Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 A3/A5 上 MemFabric acc_offload H2D 使用的末尾两个 AIV 能与 DeepseekV2/GLM-5.2 attention 计算重叠，同时保持 MLP/MoE 全核执行。

**Architecture:** 保留现有 `L2TransferEngine`、独立 H2D Stream 和逐层 event。用一个 MemFabric attention 上下文在有效 H2D consumer 期间限制 compute current stream：A3 使用 `cube_num=23, vector_num=46`，A5 使用 `cube_num=31, vector_num=62`；退出 attention 后恢复，并在 MLP 前等待下一层 H2D。非 MemFabric、无有效 consumer 和 graph capture 路径不变。

**Tech Stack:** Python、PyTorch/torch_npu、pytest。

---

### Task 1: MemFabric attention overlap context

**Files:**
- Modify: `python/sglang/srt/mem_cache/pool_host/npu_memfabric.py`
- Test: `python/sglang/test/ascend/test_npu_memfabric.py`

- [ ] **Step 1: Write failing tests**

测试用 fake NPU stream 和 fake `LayerDoneCounter` 验证：有效路径调用顺序为 `set_stream_limit -> attention -> reset_stream_limit -> wait_until(next_layer)`；最后一层不等待下一层；禁用路径不调用 NPU API；attention 异常时仍恢复 stream 且不等待下一层。

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest python/sglang/test/ascend/test_npu_memfabric.py -q`
Expected: collection fails because `memfabric_h2d_attention_overlap` does not exist.

- [ ] **Step 3: Implement the minimal context manager**

```python
@contextmanager
def memfabric_h2d_attention_overlap(kv_pool):
    counter = getattr(kv_pool, "layer_transfer_counter", None)
    if counter is None or counter.consumer_index < 0 or not ascendc_io_enabled():
        yield None
        return

    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    cube_num, vector_num = (31, 62) if is_npu_arch35() else (23, 46)
    stream = torch.npu.current_stream()
    torch.npu.set_stream_limit(stream, cube_num=cube_num, vector_num=vector_num)
    try:
        yield counter
    finally:
        torch.npu.reset_stream_limit(stream)


def wait_for_next_h2d_layer(kv_pool, layer_id: int, counter) -> None:
    if counter is None:
        return
    next_layer_id = layer_id - kv_pool.start_layer + 1
    if next_layer_id < counter.num_layers:
        counter.wait_until(next_layer_id)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest python/sglang/test/ascend/test_npu_memfabric.py -q`
Expected: all tests pass.

### Task 2: Apply the context around decoder attention

**Files:**
- Modify: `python/sglang/srt/models/deepseek_v2.py`

- [ ] **Step 1: Add the capture guard and attention scope**

在非 capture 的 NPU forward 中创建 `memfabric_h2d_attention_overlap(get_token_to_kv_pool())`，其他平台使用 `nullcontext()`。仅将现有 `self_attn` 放入该上下文，并在原有 `maybe_prefetch_next_full_attention_kv` 之后调用 `wait_for_next_h2d_layer`，保持 `attn_boundary.prepare/finish`、返回张量、MoE boundary 和后续 MLP 逻辑不变。

- [ ] **Step 2: Verify syntax and formatting**

Run: `python -m py_compile python/sglang/srt/models/deepseek_v2.py python/sglang/srt/mem_cache/pool_host/npu_memfabric.py python/sglang/test/ascend/test_npu_memfabric.py`
Run: `git diff --check`
Expected: both commands succeed.

### Task 3: Regression and device validation

**Files:**
- Review: all files above.

- [ ] **Step 1: Run focused tests and inspect the final diff**

Run: `python -m pytest python/sglang/test/ascend/test_npu_memfabric.py -q`
Expected: all tests pass; diff contains no MemFabric kernel changes.

- [ ] **Step 2: Validate on A3/A5 servers**

使用相同 shared-prefix 压测分别采集修改前后 profiling，确认 `acc_kv_exchange` 与 attention 计算 kernel 的时间区间相交；检查输出一致性和端到端吞吐。full prefill graph replay 仍使用当前整批等待路径，不在本次改动中扩展。
