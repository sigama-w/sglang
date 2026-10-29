from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.pool_host import npu_memfabric


class _FakeStream:
    def __init__(self, capturing=False):
        self.capturing = capturing

    def is_capturing(self):
        return self.capturing


class _FakeNPU:
    def __init__(self, events, *, capturing=False):
        self.events = events
        self.stream = _FakeStream(capturing)

    def current_stream(self):
        self.events.append("current_stream")
        return self.stream

    def set_stream_limit(self, stream, *, cube_num, vector_num):
        assert stream is self.stream
        self.events.append(("set_stream_limit", cube_num, vector_num))

    def reset_stream_limit(self, stream):
        assert stream is self.stream
        self.events.append("reset_stream_limit")


def _make_pool(events, *, consumer_index=0, num_layers=3):
    counter = SimpleNamespace(
        consumer_index=consumer_index,
        num_layers=num_layers,
        wait_until=lambda layer_id: events.append(("wait_until", layer_id)),
    )
    return SimpleNamespace(start_layer=8, layer_transfer_counter=counter)


def _enable_a5_memfabric(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_HOST_MEM_BACKEND", "memfabric")
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.npu.utils.is_npu_arch35", lambda: True
    )


def test_attention_overlap_reserves_aiv_and_waits_for_next_layer(monkeypatch):
    events = []
    fake_npu = _FakeNPU(events)
    pool = _make_pool(events)
    _enable_a5_memfabric(monkeypatch)
    monkeypatch.setattr(npu_memfabric.torch, "npu", fake_npu, raising=False)

    with npu_memfabric.memfabric_h2d_attention_overlap(pool) as counter:
        events.append("attention")
    events.append("prefetch_next_full_attention")
    npu_memfabric.wait_for_next_h2d_layer(pool, layer_id=8, counter=counter)

    assert events == [
        "current_stream",
        ("set_stream_limit", 31, 62),
        "attention",
        "reset_stream_limit",
        "prefetch_next_full_attention",
        ("wait_until", 1),
    ]


def test_attention_overlap_does_not_wait_past_last_layer(monkeypatch):
    events = []
    pool = _make_pool(events)
    _enable_a5_memfabric(monkeypatch)
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )

    with npu_memfabric.memfabric_h2d_attention_overlap(pool) as counter:
        events.append("attention")
    npu_memfabric.wait_for_next_h2d_layer(pool, layer_id=10, counter=counter)

    assert not any(
        isinstance(event, tuple) and event[0] == "wait_until" for event in events
    )


def test_attention_overlap_restores_stream_on_attention_error(monkeypatch):
    events = []
    pool = _make_pool(events)
    _enable_a5_memfabric(monkeypatch)
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )

    with pytest.raises(RuntimeError, match="attention failed"):
        with npu_memfabric.memfabric_h2d_attention_overlap(pool):
            raise RuntimeError("attention failed")

    assert events[-1] == "reset_stream_limit"
    assert not any(
        isinstance(event, tuple) and event[0] == "wait_until" for event in events
    )


@pytest.mark.parametrize("consumer_index", [-1, 0])
def test_attention_overlap_is_noop_when_disabled(monkeypatch, consumer_index):
    events = []
    pool = _make_pool(events, consumer_index=consumer_index)
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )
    if consumer_index == -1:
        _enable_a5_memfabric(monkeypatch)
    else:
        monkeypatch.delenv("SGLANG_HICACHE_HOST_MEM_BACKEND", raising=False)

    with npu_memfabric.memfabric_h2d_attention_overlap(pool):
        events.append("attention")

    assert events == ["attention"]


def test_attention_overlap_reserves_aiv_on_a3(monkeypatch):
    events = []
    pool = _make_pool(events)
    monkeypatch.setenv("SGLANG_HICACHE_HOST_MEM_BACKEND", "memfabric")
    monkeypatch.setattr(
        "sglang.srt.hardware_backend.npu.utils.is_npu_arch35", lambda: False
    )
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )

    with npu_memfabric.memfabric_h2d_attention_overlap(pool):
        events.append("attention")

    assert events == [
        "current_stream",
        ("set_stream_limit", 23, 46),
        "attention",
        "reset_stream_limit",
    ]


def test_attention_overlap_skips_runtime_stream_capture(monkeypatch):
    events = []
    pool = _make_pool(events)
    _enable_a5_memfabric(monkeypatch)
    monkeypatch.setattr(
        npu_memfabric.torch,
        "npu",
        _FakeNPU(events, capturing=True),
        raising=False,
    )

    with npu_memfabric.memfabric_h2d_attention_overlap(pool) as counter:
        assert counter is None
        events.append("attention")

    assert events == ["current_stream", "attention"]
