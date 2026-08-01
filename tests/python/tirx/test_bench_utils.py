# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Tests for tvm.tirx.bench utilities."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")  # tvm.tirx.bench imports triton.profiler

from tvm.testing import env
from tvm.tirx.bench import DistributedBenchContext, bench

bench_module = importlib.import_module("tvm.tirx.bench")


class _FakeStream:
    cuda_stream = 123

    def synchronize(self):
        pass


def _distributed_context(max_reduce=lambda value: value):
    return DistributedBenchContext(
        rank=0,
        world_size=4,
        barrier=lambda: None,
        max_reduce=max_reduce,
        stream=_FakeStream(),
    )


def test_bench_cooldown_precedes_every_impl(monkeypatch):
    """cooldown_s sleeps immediately before each impl's warmup+measurement.

    2 impls x 2 rounds = 4 timed calls, so 4 sleeps (the first impl in the
    first round is included). Pins the #29 per-impl cooldown semantics.
    """
    calls = []
    sleeps = []

    def fake_timer(fn, warmup=25, rep=100):
        del warmup, rep
        fn()
        return 0.001

    monkeypatch.setattr(bench_module, "_do_bench_event", fake_timer)
    monkeypatch.setattr(bench_module.time, "sleep", sleeps.append)

    results = bench(
        {"a": lambda: calls.append("a"), "b": lambda: calls.append("b")},
        warmup=0,
        repeat=1,
        timer="event",
        cooldown_s=1.0,
        rounds=2,
    )

    assert calls == ["a", "b", "a", "b"]
    assert sleeps == [1.0, 1.0, 1.0, 1.0]
    assert results["benchmark_protocol"]["cooldown_s"] == 1.0
    assert results["benchmark_protocol"]["round_aggregate"] == "mean"


def test_bench_retains_round_samples_and_uses_arithmetic_mean(monkeypatch):
    values = iter([0.001, 0.002, 0.100])

    def fake_timer(_fn, warmup=25, rep=100):
        del warmup, rep
        return next(values)

    monkeypatch.setattr(bench_module, "_do_bench_event", fake_timer)

    results = bench({"tir": lambda: None}, timer="event", cooldown_s=0, rounds=3)

    assert results["round_samples"] == {"tir": [1.0, 2.0, 100.0]}
    assert results["impls"] == {"tir": 103.0 / 3.0}


def test_bench_l2_flush_buffer_matches_triton_256_mib(monkeypatch):
    captured = {}

    def fake_empty(size, *, dtype, device):
        captured.update(size=size, dtype=dtype, device=device)
        return object()

    monkeypatch.setattr(bench_module.torch, "empty", fake_empty)

    bench_module._empty_cache_for_benchmark()

    assert captured == {
        "size": 256 * 1024 * 1024 // 4,
        "dtype": torch.int,
        "device": "cuda",
    }


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_event_pure_launch():
    """New Triton-standard bench(): no-arg launch closures, event timer."""
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10, timer="event")
    assert "mm" in results["impls"]
    assert results["impls"]["mm"] > 0
    assert results["timer"] == "event"


def test_bench_default_timer_is_proton(monkeypatch):
    """Omitting timer resolves to Proton and invokes only the Proton timer."""
    calls = []

    def fake_proton(fn, warmup=25, rep=100):
        calls.append((warmup, rep))
        fn()
        return 0.001

    monkeypatch.setattr(bench_module, "_do_bench_proton", fake_proton)

    results = bench({"noop": lambda: None}, cooldown_s=0)

    assert results["timer"] == "proton"
    assert results["impls"] == {"noop": 1.0}
    assert calls == [(25, 100)]


def test_distributed_kineto_span_uses_cross_stream_samples_and_rank_max(monkeypatch):
    reductions = []
    local_samples = [float(index + 1) for index in range(30)]

    def fake_profile(name, func, prepare, distributed):
        assert name == "impl"
        assert callable(func)
        assert callable(prepare["impl"])
        assert distributed.world_size == 4
        return local_samples, [2] * len(local_samples)

    def max_reduce(value):
        reductions.append(value)
        return value + 100.0

    monkeypatch.setattr(bench_module, "_profile_distributed_kineto_span", fake_profile)

    result = bench(
        {"impl": lambda: None},
        distributed=_distributed_context(max_reduce),
        prepare={"impl": lambda: None},
        cooldown_s=0,
    )

    assert result["timer"] == "kineto"
    assert result["impls"] == {"impl": 115.5}
    assert reductions == local_samples
    protocol = result["benchmark_protocol"]
    assert protocol["timing_scope"] == "complete correlated GPU activity span"
    assert protocol["span_definition"] == "latest activity end minus earliest activity start"
    assert protocol["rank_aggregate"] == "sample_wise_max"
    assert protocol["rank_local_scope_stream_counts"] == {"impl": [{"min": 2, "max": 2}]}


def test_kineto_span_collector_uses_earliest_start_and_latest_end():
    def event(name, device_type, stream, start, end):
        return SimpleNamespace(
            name=name,
            device_type=device_type,
            device_resource_id=stream,
            time_range=SimpleNamespace(start=start, end=end),
        )

    profiler = SimpleNamespace(
        events=lambda: [
            event("sample.0", torch.autograd.DeviceType.CPU, 0, 0.0, 10.0),
            event("sample.0", torch.autograd.DeviceType.CUDA, 11, 2.0, 5.0),
            event("sample.0", torch.autograd.DeviceType.CUDA, 17, 3.0, 8.0),
            event("sample.1", torch.autograd.DeviceType.CUDA, 11, 20.0, 21.5),
        ]
    )

    samples, stream_counts = bench_module._collect_kineto_span_samples(
        profiler, ["sample.0", "sample.1"]
    )

    assert samples == [6.0, 1.5]
    assert stream_counts == [2, 1]


def test_distributed_kineto_span_keeps_fixed_order_without_ab_ba():
    source = inspect.getsource(bench_module._bench_distributed_kineto_span)
    assert "reversed" not in source
    assert "round_orders" not in source


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"timer": "event"}, "only timer='kineto'"),
        ({"timer": "proton"}, "only timer='kineto'"),
        ({"warmup": 1}, "rejects overrides: warmup"),
        ({"repeat": 1}, "rejects overrides: repeat"),
        ({"cudagraph_rep": 1}, "rejects overrides: cudagraph_rep"),
    ],
)
def test_distributed_timers_reject_invalid_timer_and_budgets(kwargs, match):
    with pytest.raises(ValueError, match=match):
        bench(
            {"noop": lambda: None},
            distributed=_distributed_context(),
            cooldown_s=0,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        ({"world_size": 0}, ValueError, "world_size must be a positive integer"),
        ({"world_size": True}, ValueError, "world_size must be a positive integer"),
        ({"rank": -1}, ValueError, "rank must be in"),
        ({"rank": 4}, ValueError, "rank must be in"),
        ({"barrier": None}, TypeError, "barrier must be callable"),
        ({"max_reduce": None}, TypeError, "max_reduce must be callable"),
        ({"stream": object()}, TypeError, "stream must be an actual CUDA stream"),
    ],
)
def test_distributed_timer_rejects_invalid_context(overrides, error, match):
    values = {
        "rank": 0,
        "world_size": 4,
        "barrier": lambda: None,
        "max_reduce": lambda value: value,
        "stream": _FakeStream(),
    }
    values.update(overrides)

    with pytest.raises(error, match=match):
        bench(
            {"noop": lambda: None},
            distributed=DistributedBenchContext(**values),
            cooldown_s=0,
        )


def test_prepare_requires_distributed_context():
    with pytest.raises(ValueError, match="only with a distributed context"):
        bench({"noop": lambda: None}, prepare={"noop": lambda: None}, cooldown_s=0)


def test_bench_cudagraph_proton_wiring(monkeypatch):
    calls = []

    def fake_cudagraph_proton(fn, rep=20):
        calls.append(rep)
        fn()
        return 0.002

    monkeypatch.setattr(bench_module, "_do_bench_cudagraph_proton", fake_cudagraph_proton)

    results = bench({"noop": lambda: None}, timer="cudagraph_proton", cudagraph_rep=7, cooldown_s=0)

    assert results["impls"] == {"noop": 2.0}
    assert results["timer"] == "cudagraph_proton"
    assert calls == [7]


def test_bench_never_silently_falls_back_from_proton(monkeypatch):
    def unavailable(_fn, warmup=25, rep=100):
        del warmup, rep
        raise RuntimeError("Proton profiler session could not be created")

    monkeypatch.setattr(bench_module, "_do_bench_proton", unavailable)

    with pytest.raises(RuntimeError, match="Proton profiler session"):
        bench({"noop": lambda: None}, timer="proton", cooldown_s=0)


@pytest.mark.parametrize(
    ("timer", "alternative"),
    [("proton", "event"), ("cudagraph_proton", "event")],
)
def test_missing_proton_session_is_an_explicit_error(monkeypatch, timer, alternative):
    monkeypatch.setattr(bench_module.proton, "start", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match=rf"{timer}.*timer='{alternative}'"):
        bench_module._start_proton_session("profile", timer=timer, explicit_alternative=alternative)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_references_pure_launch():
    """New bench(): reference builders return no-arg callables and get timed."""
    M, N = 128, 128
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"tir": lambda: torch.mm(A, B)}

    def _addmm():
        C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
        return lambda: torch.addmm(C, A, B)

    results = bench(funcs, warmup=5, repeat=10, timer="event", references={"addmm": _addmm})
    assert set(results["impls"].keys()) == {"tir", "addmm"}
    assert all(v > 0 for v in results["impls"].values())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_rejects_unknown_timer():
    """Unknown timer names fail instead of changing measurement method."""
    A = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        bench({"mm": lambda: torch.mm(A, A)}, timer="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
