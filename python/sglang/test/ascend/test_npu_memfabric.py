from contextlib import nullcontext
from types import SimpleNamespace

from sglang.srt.mem_cache import l2_transfer
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


def _enable_memfabric(monkeypatch, *, is_a5):
    monkeypatch.setenv("SGLANG_HICACHE_HOST_MEM_BACKEND", "memfabric")
    from sglang.srt.hardware_backend.npu import utils as npu_utils

    monkeypatch.setattr(npu_utils, "is_npu_arch35", lambda: is_a5)


def test_a5_stream_limit_reserves_last_two_aivs(monkeypatch):
    events = []
    _enable_memfabric(monkeypatch, is_a5=True)
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )

    npu_memfabric.set_memfabric_h2d_stream_limit()
    npu_memfabric.reset_memfabric_h2d_stream_limit(
        SimpleNamespace(consumer_index=0)
    )

    assert events == [
        "current_stream",
        ("set_stream_limit", 31, 62),
        "current_stream",
        "reset_stream_limit",
    ]


def test_a3_stream_limit_reserves_last_two_aivs(monkeypatch):
    events = []
    _enable_memfabric(monkeypatch, is_a5=False)
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )

    npu_memfabric.set_memfabric_h2d_stream_limit()
    npu_memfabric.reset_memfabric_h2d_stream_limit(
        SimpleNamespace(consumer_index=0)
    )

    assert events == [
        "current_stream",
        ("set_stream_limit", 23, 46),
        "current_stream",
        "reset_stream_limit",
    ]


def test_stream_limit_skips_runtime_capture(monkeypatch):
    events = []
    _enable_memfabric(monkeypatch, is_a5=False)
    monkeypatch.setattr(
        npu_memfabric.torch,
        "npu",
        _FakeNPU(events, capturing=True),
        raising=False,
    )

    npu_memfabric.set_memfabric_h2d_stream_limit()
    npu_memfabric.reset_memfabric_h2d_stream_limit(
        SimpleNamespace(consumer_index=0)
    )

    assert events == ["current_stream", "current_stream"]


def test_stream_limit_is_noop_without_memfabric(monkeypatch):
    events = []
    monkeypatch.delenv("SGLANG_HICACHE_HOST_MEM_BACKEND", raising=False)
    monkeypatch.setattr(
        npu_memfabric.torch, "npu", _FakeNPU(events), raising=False
    )

    npu_memfabric.set_memfabric_h2d_stream_limit()
    npu_memfabric.reset_memfabric_h2d_stream_limit(
        SimpleNamespace(consumer_index=0)
    )

    assert events == []


def test_on_demand_submission_finishes_only_after_last_layer(monkeypatch):
    events = []
    monkeypatch.setattr(
        l2_transfer.device_module, "stream", lambda stream: nullcontext()
    )

    class FakeEvent:
        def record(self):
            events.append("finish")

    class FakeDomain:
        def release_layout(self, event, key):
            events.append(("release", event))

    finish_event = FakeEvent()
    engine = SimpleNamespace(
        _submit_h2d_layer=lambda transfers, layer_id: events.append(
            ("layer", layer_id)
        ),
        _record_stream=lambda transfers, stream: events.append("record_stream"),
    )
    submission = l2_transfer.OnDemandH2DSubmission(
        engine=engine,
        original_transfers=[object()],
        transfers=[object()],
        stream=object(),
        completion=SimpleNamespace(finish_event=finish_event),
        domains=[FakeDomain()],
        transfer_layer_id_max=2,
        on_layer_done=lambda layer_id: events.append(("done", layer_id)),
    )

    submission.submit_next()
    assert not submission.finished
    assert events == [("layer", 0), ("done", 0)]

    submission.submit_next()
    submission.submit_next()
    assert submission.finished
    assert events == [
        ("layer", 0),
        ("done", 0),
        ("layer", 1),
        ("done", 1),
        "finish",
        "record_stream",
        ("release", finish_event),
    ]
