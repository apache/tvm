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

import argparse
import gc
import inspect
import json
import math
import os
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import triton.profiler as proton
import tvm_ffi

import tvm
from tvm.script import tirx as T
from tvm.support import nvcc

_DISTRIBUTED_KINETO_WARMUP_ITERATIONS = 5
_DISTRIBUTED_KINETO_REPEAT_ITERATIONS = 30
_DISTRIBUTED_FLUSH_L2_BYTES = int(8e9)
_DISTRIBUTED_GPU_SLEEP_CYCLES = int(2e7)


@dataclass(frozen=True)
class DistributedBenchContext:
    """Rank-local synchronization needed by the distributed Kineto timer.

    Parameters
    ----------
    rank : int
        Rank of the current process.
    world_size : int
        Number of ranks participating in the benchmark.
    barrier : callable
        Host callback that blocks until every rank reaches the same point.
    max_reduce : callable
        Host callback that returns the maximum of one floating-point value over
        all ranks.  The distributed timer calls it once per measured launch.
    stream : torch.cuda.Stream
        The actual CUDA stream whose launch scope is being timed.  Launch
        closures that use auxiliary streams must make this stream wait for them
        before returning.
    """

    rank: int
    world_size: int
    barrier: Callable[[], None]
    max_reduce: Callable[[float], float]
    stream: object


def is_running_under_pytest():
    """Check if the code is being executed within a pytest session."""
    return "PYTEST_CURRENT_TEST" in os.environ


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-ptx", type=str, help="Dump PTX code to specified file")
    parser.add_argument("--dump-source", action="store_true", help="Dump source code")
    args = parser.parse_args()

    if args.dump_ptx:

        @tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)
        def tvm_callback_cuda_compile(code, target):
            ptx = nvcc.compile_cuda(code, target_format="ptx")
            with open(args.dump_ptx, "w", encoding="utf-8") as f:
                f.write(ptx.decode())
            return ptx

    return args


# ---------------------------------------------------------------------------
# Triton-standard benchmark path.
#
# Faithful in-repo port of triton.testing.do_bench / do_bench_proton /
# do_bench_cudagraph_proton
# (see https://github.com/triton-lang/triton, python/triton/testing.py). This is
# a torch CUDA timer port: Triton runtime-driver calls (get_device_interface /
# get_empty_cache_for_benchmark / clear_cache) are replaced with torch.cuda and
# a torch L2-flush buffer. The Proton variants still use triton.profiler for
# attribution. The timed function is a *pure no-arg launch closure* (inputs
# captured once, allocated outside the timed region) -- exactly how Triton
# times a function.
# ---------------------------------------------------------------------------


def _quantile(a, q):
    # pure-Python np.quantile / torch.quantile (port of triton.testing._quantile)
    n = len(a)
    a = sorted(a)

    def get_quantile(qi):
        if not (0 <= qi <= 1):
            raise ValueError("Quantiles must be in the range [0, 1]")
        point = qi * (n - 1)
        lower = math.floor(point)
        upper = math.ceil(point)
        t = point - lower
        return (1 - t) * a[lower] + t * a[upper]

    return [get_quantile(qi) for qi in q]


def _summarize_statistics(times, quantiles, return_mode):
    # port of triton.testing._summarize_statistics
    if quantiles is not None:
        ret = _quantile(times, quantiles)
        if len(ret) == 1:
            ret = ret[0]
        return ret
    if return_mode == "all":
        return times
    elif return_mode == "min":
        return min(times)
    elif return_mode == "max":
        return max(times)
    elif return_mode == "mean":
        return statistics.mean(times)
    elif return_mode == "median":
        return statistics.median(times)


@contextmanager
def _cuda_graph_without_gc(*args, **kwargs):
    # port of triton.testing.cuda_graph_without_gc. A loaded kernel may be
    # finalized by Python's cyclic GC; its destructor unloads the CUDA module,
    # which is illegal during CUDA stream capture and invalidates the graph.
    # Keep GC disabled only for the capture window and restore afterwards.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        with torch.cuda.graph(*args, **kwargs) as graph:
            yield graph
    finally:
        if gc_was_enabled:
            gc.enable()


def _empty_cache_for_benchmark():
    # torch equivalent of triton's driver.get_empty_cache_for_benchmark(): a
    # 256 MiB buffer whose .zero_() evicts the L2 cache between measured iters.
    return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")


def _do_bench_event(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    return_mode="mean",
):
    """Faithful port of triton.testing.do_bench (CUDA-event timing, per-iter L2 flush).

    ``warmup`` and ``rep`` are millisecond time budgets, not iteration counts.
    Returns the runtime in milliseconds (mean by default).
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()
    torch.cuda.synchronize()

    cache = _empty_cache_for_benchmark()

    # Estimate the runtime of the function.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Compute number of warmup and repeat iterations from the ms budgets.
    # Keep this identical to triton.testing.do_bench. Unlike Triton's Proton
    # and CUDA-graph helpers, do_bench deliberately has no zero-time fallback.
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    # Warm-up.
    for _ in range(n_warmup):
        fn()
    # Benchmark.
    for i in range(n_repeat):
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # Clear the L2 cache before each run.
        cache.zero_()
        start_event[i].record()
        fn()
        end_event[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]
    return _summarize_statistics(times, quantiles, return_mode)


def _collect_proton_scope_times(database, prefix):
    """Port of triton.testing._collect_proton_scope_times.

    Walk the Proton hatchet JSON tree and, for each scope whose frame name starts
    with ``prefix``, sum the GPU ``time (ns)`` of all its leaf kernels. Returns the
    per-scope times (ms), sorted by scope name.
    """
    scope_times = []

    def kernel_time_ms(node):
        children = node.get("children", [])
        if len(children) == 0:
            return node.get("metrics", {}).get("time (ns)", 0) / 1e6
        return sum(kernel_time_ms(child) for child in children)

    def visit(node):
        name = node.get("frame", {}).get("name", "")
        if name.startswith(prefix):
            time_ms = kernel_time_ms(node)
            if time_ms > 0:
                scope_times.append((name, time_ms))
            return
        for child in node.get("children", []):
            visit(child)

    for node in database:
        # The hatchet top-level list may carry a device_info dict (no "frame"/
        # "children"); the name-prefix walk simply ignores it.
        if isinstance(node, dict):
            visit(node)
    return [t for _, t in sorted(scope_times)]


def _start_proton_session(profile_path, *, timer, explicit_alternative):
    """Start Proton or fail without changing the requested measurement method."""
    session = proton.start(profile_path, context="shadow", data="tree")
    if session is None:
        raise RuntimeError(
            f"{timer}: Proton profiler session could not be created. "
            f"Use timer={explicit_alternative!r} explicitly if that timing is intended."
        )
    return session


def _do_bench_proton(
    fn,
    warmup=25,
    rep=100,
    grad_to_none=None,
    quantiles=None,
    return_mode="mean",
):
    """Port of triton.testing.do_bench_proton, aligned with ``_do_bench_event``.

    IDENTICAL to ``_do_bench_event`` in everything -- warmup/rep millisecond budgets,
    the 5-call estimate, per-iter L2 flush, the untimed warmup loop -- EXCEPT the
    timing mechanism: each timed call runs inside a Proton scope and the per-kernel
    GPU time (read from the hatchet tree) is used instead of the CUDA-event wall.
    Cold cache. NVIDIA + Proton only. No CUDA graph (so it works for references that
    can't be graph-captured, e.g. CuTeDSL flash-attention).

    A missing Proton session is an error. Silently switching to event timing would
    leave the result labelled ``proton`` while changing the measured quantity.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    fn()
    torch.cuda.synchronize()

    cache = _empty_cache_for_benchmark()

    # Estimate the runtime of the function (identical to _do_bench_event).
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    if estimate_ms == 0:
        n_warmup = 1000
        n_repeat = 1000
    else:
        n_warmup = max(1, int(warmup / estimate_ms))
        n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up (untimed), same as _do_bench_event.
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    with tempfile.TemporaryDirectory(prefix=f"tirx-proton-{uuid.uuid4().hex}-") as tmpdir:
        profile_path = os.path.join(tmpdir, "profile")
        session = _start_proton_session(profile_path, timer="proton", explicit_alternative="event")
        scope_prefix = f"proton.{uuid.uuid4().hex}."
        # finalize() MUST run even if fn() raises mid-loop; otherwise the global
        # Proton profiler stays active and the next session in this process starts
        # dirty (corrupted attribution). Mirrors triton.testing._proton_bench_session
        # (finalize in a finally), adapted to read the .hatchet finalize writes.
        try:
            for i in range(n_repeat):
                if grad_to_none is not None:
                    for x in grad_to_none:
                        x.grad = None
                # Flush L2 OUTSIDE the scope so it is excluded from the measured time
                # -- identical cold-cache behavior to _do_bench_event.
                cache.zero_()
                with proton.scope(f"{scope_prefix}{i:08d}"):
                    fn()
            torch.cuda.synchronize()
        finally:
            proton.finalize(session)
        with open(profile_path + ".hatchet") as f:
            database = json.load(f)
        times = _collect_proton_scope_times(database, scope_prefix)

    if not times:
        raise RuntimeError(
            "proton: Proton attributed no kernel time to the captured scopes. "
            "Use timer='event' instead."
        )
    return _summarize_statistics(times, quantiles, return_mode)


def _do_bench_cudagraph_proton(fn, rep=20, grad_to_none=None, quantiles=None, return_mode="mean"):
    """Faithful port of triton.testing.do_bench_cudagraph_proton.

    CUDA-graph replay (kills per-launch CPU overhead) + Proton per-kernel GPU time
    + per-iter L2 flush. Best accuracy for short / multi-kernel workloads. NVIDIA
    only (Proton cannot reliably attribute graph-replay launches to scopes on HIP).
    ``rep`` (ms) sets the graph unroll count (``n_repeat = rep / estimate_ms``); the
    measurement is 10 graph replays. Triton's default is ``rep=20``. Returns ms.

    Adapted to the installed proton (3.6.0): there is no ``proton.data.get`` /
    ``deactivate(flushing=True)`` here, so we read the ``.hatchet`` JSON that
    ``finalize`` writes (the same tree the in-memory getter would return).

    A missing Proton session is an error; this timer never changes measurement
    method while retaining the ``cudagraph_proton`` label.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    if torch.version.hip is not None:
        raise RuntimeError(
            "cudagraph_proton requires the NVIDIA backend because Proton does not "
            "reliably attribute CUDA graph replay launches to scopes on HIP."
        )

    with torch.cuda.stream(torch.cuda.Stream()):
        # warmup
        fn()
        if grad_to_none is not None:
            for x in grad_to_none:
                x.detach_()
                x.requires_grad_(True)
                x.grad = None
        # estimate single-call time
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        torch.cuda.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        with tempfile.TemporaryDirectory(prefix=f"tirx-cgproton-{uuid.uuid4().hex}-") as tmpdir:
            profile_path = os.path.join(tmpdir, "profile")
            # shadow/CUPTI captures kernel GPU activity itself; the triton launch
            # hook only adds flops/bytes metadata, so it is omitted here.
            session = _start_proton_session(
                profile_path,
                timer="cudagraph_proton",
                explicit_alternative="event",
            )

            cache = _empty_cache_for_benchmark()
            scope_prefix = f"proton.{uuid.uuid4().hex}."
            g = torch.cuda.CUDAGraph()
            n_retries = 10
            # finalize() MUST run even if capture/replay raises, or the global Proton
            # profiler stays active and poisons the next session (see _do_bench_proton).
            try:
                with _cuda_graph_without_gc(g):
                    for i in range(n_repeat):
                        if grad_to_none is not None:
                            for x in grad_to_none:
                                x.grad = None
                        # Flush L2 OUTSIDE the scope so it is excluded from the timed span.
                        cache.zero_()
                        with proton.scope(f"{scope_prefix}{i:08d}"):
                            fn()
                torch.cuda.synchronize()
                for _ in range(n_retries):
                    g.replay()
                torch.cuda.synchronize()
            finally:
                # finalize flushes the replay data and writes <profile>.hatchet.
                proton.finalize(session)
            with open(profile_path + ".hatchet") as f:
                database = json.load(f)
            times = [t / n_retries for t in _collect_proton_scope_times(database, scope_prefix)]

    if not times:
        raise RuntimeError(
            "cudagraph_proton: Proton attributed no kernel time to the captured scopes "
            "(CUDA-graph replay scope attribution may be unsupported in this "
            "environment). Use timer='event' or 'proton' instead."
        )
    return _summarize_statistics(times, quantiles, return_mode)


def _sleep_before_impl(cooldown_s):
    if cooldown_s > 0:
        time.sleep(cooldown_s)


def _validate_distributed_context(distributed):
    if not isinstance(distributed, DistributedBenchContext):
        raise TypeError("distributed must be a DistributedBenchContext")
    if (
        not isinstance(distributed.world_size, int)
        or isinstance(distributed.world_size, bool)
        or distributed.world_size < 1
    ):
        raise ValueError("distributed.world_size must be a positive integer")
    if (
        not isinstance(distributed.rank, int)
        or isinstance(distributed.rank, bool)
        or not 0 <= distributed.rank < distributed.world_size
    ):
        raise ValueError("distributed.rank must be in [0, world_size)")
    if not callable(distributed.barrier):
        raise TypeError("distributed.barrier must be callable")
    if not callable(distributed.max_reduce):
        raise TypeError("distributed.max_reduce must be callable")
    if not hasattr(distributed.stream, "synchronize"):
        raise TypeError("distributed.stream must be an actual CUDA stream")


def _distributed_cold_start(distributed):
    """Apply DeepGEMM's cold-cache protocol before one distributed launch."""
    torch.empty(
        _DISTRIBUTED_FLUSH_L2_BYTES // 4,
        dtype=torch.int,
        device="cuda",
    ).zero_()
    torch.cuda._sleep(_DISTRIBUTED_GPU_SLEEP_CYCLES)
    distributed.barrier()


def _collect_kineto_span_samples(profiler, scope_names):
    """Collect one cross-stream GPU activity span for every record-function scope."""
    scope_events = {scope_name: [] for scope_name in scope_names}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and event.name in scope_events:
            scope_events[event.name].append(event)

    samples = []
    stream_counts = []
    for scope_name in scope_names:
        events = scope_events[scope_name]
        if not events:
            raise RuntimeError(
                f"kineto: profiler attributed no CUDA activity to scope {scope_name!r}"
            )
        start_us = min(event.time_range.start for event in events)
        end_us = max(event.time_range.end for event in events)
        elapsed_us = float(end_us - start_us)
        if not math.isfinite(elapsed_us) or elapsed_us <= 0:
            raise RuntimeError(f"kineto: scope {scope_name!r} has invalid GPU span {elapsed_us} us")
        samples.append(elapsed_us)
        stream_counts.append(len({event.device_resource_id for event in events}))
    return samples, stream_counts


def _profile_distributed_kineto_span(name, func, prepare, distributed):
    """Profile complete correlated GPU activity spans for one implementation."""
    prepare_fn = prepare.get(name)
    with torch.cuda.stream(distributed.stream):
        for _ in range(_DISTRIBUTED_KINETO_WARMUP_ITERATIONS):
            _distributed_cold_start(distributed)
            if prepare_fn is not None:
                prepare_fn()
            func()
            distributed.stream.synchronize()

        scope_names = [
            f"tirx.bench.{name}.{index:08d}"
            for index in range(_DISTRIBUTED_KINETO_REPEAT_ITERATIONS)
        ]
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        with profiler:
            for scope_name in scope_names:
                _distributed_cold_start(distributed)
                if prepare_fn is not None:
                    prepare_fn()
                with torch.profiler.record_function(scope_name):
                    func()
                distributed.stream.synchronize()

    return _collect_kineto_span_samples(profiler, scope_names)


def _distributed_stream_id(stream):
    value = getattr(stream, "cuda_stream", None)
    return int(value) if value is not None else repr(stream)


def _bench_distributed_kineto_span(funcs, prepare, distributed, rounds, cooldown_s):
    """Time complete correlated GPU activity spans across all participating streams."""
    items = list(funcs.items())
    round_samples = {name: [] for name in funcs}
    stream_counts = {name: [] for name in funcs}
    for _ in range(rounds):
        for name, func in items:
            _sleep_before_impl(cooldown_s)
            local_samples, local_stream_counts = _profile_distributed_kineto_span(
                name, func, prepare, distributed
            )
            samples = []
            for local_us in local_samples:
                elapsed_us = float(distributed.max_reduce(local_us))
                if not math.isfinite(elapsed_us) or elapsed_us <= 0:
                    raise RuntimeError(
                        "kineto: distributed max-reduce returned invalid activity span "
                        f"for {name!r}: {elapsed_us}"
                    )
                samples.append(elapsed_us)
            round_samples[name].append(statistics.median(samples))
            stream_counts[name].append(
                {"min": min(local_stream_counts), "max": max(local_stream_counts)}
            )
            distributed.barrier()

    return {
        "impls": {name: statistics.mean(samples) for name, samples in round_samples.items()},
        "round_samples": round_samples,
        "errors": {},
        "timer": "kineto",
        "benchmark_protocol": {
            "source": "torch.profiler Kineto correlated CUDA activities",
            "timing_scope": "complete correlated GPU activity span",
            "span_definition": "latest activity end minus earliest activity start",
            "timing_stream": _distributed_stream_id(distributed.stream),
            "all_correlated_streams": True,
            "warmup_iterations": _DISTRIBUTED_KINETO_WARMUP_ITERATIONS,
            "repeat_iterations": _DISTRIBUTED_KINETO_REPEAT_ITERATIONS,
            "flush_l2": True,
            "flush_l2_bytes": _DISTRIBUTED_FLUSH_L2_BYTES,
            "gpu_sleep_cycles": _DISTRIBUTED_GPU_SLEEP_CYCLES,
            "rank_barrier_before_each_launch": True,
            "cold_start_outside_scope": True,
            "prepare_outside_scope": True,
            "rank_aggregate": "sample_wise_max",
            "sample_aggregate": "median",
            "round_aggregate": "mean",
            "world_size": distributed.world_size,
            "rounds": rounds,
            "cooldown_s": cooldown_s,
            "order": [name for name, _ in items],
            "rank_local_scope_stream_counts": stream_counts,
        },
    }


def bench(
    funcs,
    *,
    warmup=None,
    repeat=None,
    cudagraph_rep=None,
    timer=None,
    references=None,
    cooldown_s=1.0,
    rounds=1,
    distributed=None,
    prepare=None,
):
    """Benchmark pure-launch implementations using Triton-standard timing.

    Each callable in ``funcs`` is a *no-arg launch closure* (inputs allocated
    once and captured in the closure), timed exactly the way
    ``triton.testing.do_bench`` / ``do_bench_proton`` /
    ``do_bench_cudagraph_proton`` time a function.
    The timing core is a faithful in-repo port (torch-only, no Triton runtime
    driver dependency); see the corresponding ``_do_bench_*`` functions.

    Parameters
    ----------
    funcs : dict[str, callable]
        Map of implementation name to a no-arg callable that launches our
        kernel.  This should hold only *our* kernel(s); external baselines go in
        ``references``.
    references : dict[str, callable], optional
        Map of reference-impl name to a no-arg *builder* that does the heavy
        import/setup and returns the no-arg run callable.  A builder that raises
        is recorded as a ``BASELINE_ERROR`` instead of failing the workload.
    warmup : int, optional
        Warmup time budget (ms) for the ``event`` and ``proton`` timers. ``None``
        (default) defers to the selected timer's own default; pass a value only
        to override. Ignored by ``cudagraph_proton``, which has no warmup argument.
    repeat : int, optional
        Rep time budget (ms) for the ``event`` and ``proton`` timers. ``None``
        (default) defers to the selected timer's own default.
    cudagraph_rep : int, optional
        ``rep`` (ms) for ``cudagraph_proton`` (graph unroll length). ``None``
        (default) defers to that timer's own default. Each timer default lives only
        in its own signature (Triton: do_bench 25/100, graph 20); nothing is hardcoded
        here.
    timer : {None, "event", "proton", "cudagraph_proton", "kineto"}
        ``None`` (default) resolves to ``proton`` for a local benchmark and to
        ``kineto`` when ``distributed`` is present.  The default is defined in
        exactly one place so kernels pass ``None`` to inherit.
        ``event`` -> ported ``do_bench`` (CUDA-event wall of each call).
        ``proton`` -> ported ``do_bench_proton``: same setup as ``event``, differs
        ONLY in timing -- Proton per-kernel GPU time instead of the event wall (so
        launch/host overhead of the ref is excluded). No graph, so it works for
        references that can't be CUDA-graph-captured (e.g. CuTeDSL flash-attention).
        NVIDIA + Proton only.
        ``cudagraph_proton`` -> ported ``do_bench_cudagraph_proton`` (graph replay +
        Proton kernel time, with L2 flushes captured outside Proton scopes); like
        ``proton`` it reports kernel time, but also removes launch overhead via
        graph replay. Graph capture or attribution is not universal.
        Prefer ``proton`` when the reference has heavy host dispatch (flashinfer,
        CuTeDSL) -- ``event`` would over/under-credit us by measuring the wall, not the
        kernel. Timer failures are propagated; a result is never silently relabelled
        after falling back to a different measurement method.
        A distributed benchmark supports only ``kineto``.  It measures the complete
        correlated GPU activity span across all streams, performs sample-wise
        slowest-rank aggregation, and applies DeepGEMM's 8 GB L2 flush, GPU sleep,
        and rank barrier before every launch.  Distributed timing uses fixed
        iteration counts and rejects local timer budget overrides.
    distributed : DistributedBenchContext, optional
        Rank-local barrier, max-reduce callback, and actual CUDA timing stream.
        Launch closures that use auxiliary streams must make the timing stream
        wait for them before returning.
    prepare : dict[str, callable], optional
        Per-implementation reset/preparation callbacks.  Kineto scopes begin after
        preparation, so preparation activities are excluded from the measured span.
    rounds : int
        Independent measurement rounds; per-impl times are averaged across
        rounds. (Triton times a single fn with no rounds; this is our sampling
        layer on top.)
    cooldown_s : float
        Seconds to sleep immediately before each implementation in every round.

    Returns
    -------
    dict
        ``{"impls": {name: us}, "round_samples": {name: [us, ...]}, ...}``.
        Times are stored in microseconds (same unit as the pinned tir-bench baselines).
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be >= 1")
    if cooldown_s < 0:
        raise ValueError("cooldown_s must be non-negative")

    if distributed is None:
        if prepare is not None:
            raise ValueError("prepare is supported only with a distributed context")
        if repeat is not None and repeat <= 0:
            raise ValueError("repeat must be positive")
        if warmup is not None and warmup < 0:
            raise ValueError("warmup must be non-negative")
        if timer is None:
            timer = "proton"
        if timer not in {"event", "proton", "cudagraph_proton"}:
            raise ValueError(
                f"unsupported timer {timer!r}; expected event, proton, or cudagraph_proton"
            )
    else:
        _validate_distributed_context(distributed)
        if timer is None:
            timer = "kineto"
        elif timer != "kineto":
            raise ValueError("a distributed context supports only timer='kineto'")
        overrides = [
            name
            for name, value in (
                ("warmup", warmup),
                ("repeat", repeat),
                ("cudagraph_rep", cudagraph_rep),
            )
            if value is not None
        ]
        if overrides:
            raise ValueError(
                f"timer={timer!r} uses fixed iteration counts and rejects overrides: "
                + ", ".join(overrides)
            )
    if not isinstance(funcs, Mapping) or not funcs:
        raise TypeError("funcs must be a non-empty mapping of name to no-arg callable")
    for name, func in funcs.items():
        if not isinstance(name, str):
            raise TypeError("func names must be strings")
        if not callable(func):
            raise TypeError(f"funcs[{name!r}] must be callable")

    # ``funcs`` holds our own kernel(s); external baselines are passed as
    # ``references`` (name -> no-arg builder). A builder that fails is recorded
    # as a BASELINE_ERROR rather than failing the workload.
    build_errors: dict[str, str] = {}
    for ref_name, builder in (references or {}).items():
        if not isinstance(ref_name, str) or not callable(builder):
            raise TypeError("references must map a name to a no-arg builder callable")
        try:
            ref_fn = builder()
        except Exception as e:
            build_errors[ref_name] = str(e)
            print(f"BASELINE_ERROR: {ref_name}: {e}", file=sys.stderr)
            continue
        if ref_fn is None:
            continue
        if not callable(ref_fn):
            raise TypeError(f"references[{ref_name!r}] builder must return a callable")
        funcs = {**funcs, ref_name: ref_fn}

    if distributed is not None:
        if prepare is None:
            prepare = {}
        if not isinstance(prepare, Mapping):
            raise TypeError("prepare must map implementation names to no-arg callables")
        for name, prepare_fn in prepare.items():
            if name not in funcs:
                raise ValueError(f"prepare contains unknown implementation {name!r}")
            if not callable(prepare_fn):
                raise TypeError(f"prepare[{name!r}] must be callable")
        result = _bench_distributed_kineto_span(
            funcs, prepare, distributed, rounds=rounds, cooldown_s=cooldown_s
        )
        result["errors"] = build_errors
        return result

    # Resolve the timer function once. Only forward warmup/repeat/cudagraph_rep when a
    # caller explicitly overrode them; otherwise the _do_bench_* signature default
    # applies, so each default lives in exactly ONE place (its timer signature). The
    # effective value is read back via inspect, so the recorded protocol tracks that
    # default automatically even if it later changes -- no value is duplicated here.
    def _sig_default(fn, param):
        return inspect.signature(fn).parameters[param].default

    if timer in ("event", "proton"):
        # event and proton share the exact same warmup/rep setup; they differ only in
        # how the timed calls are measured (CUDA-event wall vs Proton per-kernel time).
        _timer_fn = _do_bench_event if timer == "event" else _do_bench_proton
        _timer_kwargs = {}
        if warmup is not None:
            _timer_kwargs["warmup"] = warmup
        if repeat is not None:
            _timer_kwargs["rep"] = repeat
        _eff = {
            "warmup": _timer_kwargs.get("warmup", _sig_default(_timer_fn, "warmup")),
            "repeat": _timer_kwargs.get("rep", _sig_default(_timer_fn, "rep")),
        }
    else:  # cudagraph_proton has no warmup; rep is the graph-unroll budget
        _timer_fn = _do_bench_cudagraph_proton
        _timer_kwargs = {}
        if cudagraph_rep is not None:
            _timer_kwargs["rep"] = cudagraph_rep
        _eff = {"cudagraph_rep": _timer_kwargs.get("rep", _sig_default(_timer_fn, "rep"))}

    protocol = {
        **_eff,
        "cooldown_s": cooldown_s,
        "rounds": rounds,
        "round_aggregate": "mean",
        "order": list(funcs.keys()),
    }

    round_samples: dict[str, list[float]] = {}
    for _ in range(rounds):
        for name, func in funcs.items():
            _sleep_before_impl(cooldown_s)
            ms = _timer_fn(func, **_timer_kwargs)
            # ms -> microseconds (matches pinned baseline units).
            round_samples.setdefault(name, []).append(ms * 1000.0)

    aggregated = {impl: statistics.mean(samples) for impl, samples in round_samples.items()}

    return {
        "impls": aggregated,
        "round_samples": round_samples,
        "errors": build_errors,
        "timer": timer,
        "benchmark_protocol": protocol,
    }


# utils for tg4perfetto profiler, adapted from https://github.com/flashinfer-ai/flashinfer


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2
    kFinalize = 3


def decode_tag(tag, num_groups):
    block_group_tag = tag >> 12
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    return (block_group_tag // num_groups, block_group_tag % num_groups, event_idx, event_type)


def export_to_perfetto_trace(
    profiler_buffer: np.ndarray, file_name: str, event_type_names: list[str]
) -> None:
    if is_running_under_pytest():
        return

    import torch

    # pip install git+https://github.com/ihavnoid/tg4perfetto.git
    from tg4perfetto import TraceGenerator

    profiler_buffer_host = torch.tensor(profiler_buffer)
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)
    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    finish_idx = set()
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type = decode_tag(tag, num_groups)

        if event_type == EventType.kFinalize.value:
            finish_idx.add((block_idx, group_idx))
            if len(finish_idx) == num_blocks * num_groups:
                break
        else:
            if (block_idx, group_idx) in finish_idx:
                continue

        event = event_type_names[event_idx]
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()


@T.meta_class
class CudaProfiler:
    """A lightweight wrapper around T.timer_* CUDA intrinsics.

    Stores repeated arguments used by timer_init/start/end/finalize so users can
    call concise methods in kernels. Intended to mirror Pipeline/TileScheduler helpers.

    When ``profiler_enabled`` is False (or a false-y Expr), calls to
    ``init/start/end/finalize`` become no-ops. This allows constructing a
    profiler unconditionally and eliminating external ``if PROFILER_ON:`` guards.
    """

    def __init__(
        self,
        profiler_buffer: T.Buffer,
        write_stride: int,
        num_groups: int,
        default_leader: None | tvm.tirx.Expr | bool = None,
        profiler_enabled: bool | tvm.tirx.Expr = True,
    ):
        self.buffer = profiler_buffer
        self.write_stride = write_stride
        self.num_groups = num_groups
        self.default_leader = default_leader
        # Accept either a Python bool or a Expr; normalize simple bools to T.bool
        # so we can use it uniformly inside macros for conditional emission.
        if isinstance(profiler_enabled, bool | np.bool_):
            self.profiler_enabled = T.bool(bool(profiler_enabled))
        else:
            # Assume Expr-like input; use as-is
            self.profiler_enabled = profiler_enabled  # type: ignore[assignment]

        self.profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
        self.profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)

    def _leader(self, leader: None | tvm.tirx.Expr | bool):
        if leader is not None:
            if isinstance(leader, bool | np.bool_):
                return T.bool(bool(leader))
            return leader
        if self.default_leader is not None:
            return self.default_leader
        return T.bool(True)

    @T.inline
    def init(self, group_id: tvm.tirx.Expr):
        if self.profiler_enabled:
            T.cuda.timer_init(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.num_groups,
                group_id,
            )

    @T.inline
    def start(self, event_type: Enum, leader: None | tvm.tirx.Expr | bool = None):
        if self.profiler_enabled:
            T.cuda.timer_start(
                event_type,
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )

    @T.inline
    def end(self, event_type: Enum, leader: None | tvm.tirx.Expr | bool = None):
        if self.profiler_enabled:
            T.cuda.timer_end(
                event_type,
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )

    @T.inline
    def finalize(self, leader: None | tvm.tirx.Expr | bool = None):
        if self.profiler_enabled:
            T.cuda.timer_finalize(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )
