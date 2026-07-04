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
import re
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum

import numpy as np
import torch
import triton.profiler as proton
import tvm_ffi

import tvm
from tvm.script import tirx as T
from tvm.support import nvcc


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


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# proton-viewer -m avg_time/us prints average kernel time in microseconds (see
# triton/profiler/viewer.py avg_time_factor_dict). Store microseconds as-is.
PROTON_AVG_TIME_METRIC = "avg_time/us"


def _parse_proton_tree(text, *, kernel: str = ""):
    """Parse proton-viewer tree output into {impl: time_us}.

    Accepts ALL depth-1 nodes (no KNOWN_IMPLS filter). For each depth-1 impl,
    takes the slowest depth-2 child kernel time.

    Tree numbers are microseconds when ProtonContext uses avg_time/us.

    Returns (impl_times, baseline_errors) where:
      impl_times: {str: float} — impl name to avg time in microseconds
      baseline_errors: {str: str} — impl name to error message
    """
    _ = kernel  # kept for callers; unit does not depend on workload
    impl = None
    results = {}
    baseline_errors = {}
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", raw).rstrip()
        if not line:
            continue
        if line.startswith("BASELINE_ERROR:"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                baseline_errors[parts[1].strip()] = parts[2].strip()
            continue
        # Depth-1 impl header: starts with tree drawing chars
        if line and line[0] in "\u251c\u2514":  # ├ └
            parts = line.split("\u2500", 1)[-1].split()  # split on ─
            if len(parts) >= 2:
                impl = parts[1]
            else:
                impl = None
            continue
        # Depth-2 kernel: contains tree drawing chars at deeper indent
        if impl and ("\u251c\u2500" in line or "\u2514\u2500" in line):  # ├─ └─
            parts = line.split("\u2500", 1)[-1].split()
            if len(parts) >= 2:
                name = parts[1]
                if (
                    "vectorized_elementwise_kernel" in name
                    or "elementwise_kernel_with_index" in name
                ):
                    continue
                try:
                    t = float(parts[0])
                    results[impl] = max(results.get(impl, 0), t)
                except ValueError:
                    pass
    return results, baseline_errors


class ProtonContext:
    """Context manager for Proton profiling sessions.

    Always captures proton-viewer output and parses impl times so that
    get_impl_times() / get_baseline_errors() work after exiting the context.

    The proton tree is printed to **stdout** by default (visible on screen
    when running kernels interactively).  When the environment variable
    ``TIRX_BENCH_JSON=1`` is set (done automatically by ``--json`` mode),
    the tree goes to **stderr** instead so it does not corrupt the JSON on
    stdout.
    """

    def __init__(
        self,
        name="kernel",
        hook="triton",
        debug=False,
        nsight=False,
        metric=PROTON_AVG_TIME_METRIC,
        kernel="",
    ):
        self.name = name
        self.hook = hook
        self.debug = debug
        self.nsight = nsight
        self.metric = metric
        self.kernel = kernel
        self._impl_times = {}
        self._baseline_errors = {}

    def __enter__(self):
        if not is_running_under_pytest() and not self.debug and not self.nsight:
            proton.start(self.name, hook=self.hook)
            proton.deactivate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not is_running_under_pytest() and not self.debug and not self.nsight:
            proton.finalize()

            hatchet = f"{self.name}.hatchet"
            result = subprocess.run(
                ["proton-viewer", "-m", self.metric, hatchet],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self._impl_times, self._baseline_errors = _parse_proton_tree(
                    result.stdout, kernel=self.kernel
                )
                out = sys.stderr if os.environ.get("TIRX_BENCH_JSON") else sys.stdout
                print(f"# proton {PROTON_AVG_TIME_METRIC} (microseconds)\n", file=out, end="")
                print(result.stdout, file=out, end="")
            else:
                print(
                    f"proton-viewer failed (rc={result.returncode}): {result.stderr}",
                    file=sys.stderr,
                )

            if os.path.exists(hatchet):
                os.remove(hatchet)

    def get_impl_times(self):
        """Return {impl_name: avg_time_us} parsed from proton-viewer output."""
        return dict(self._impl_times)

    def get_baseline_errors(self):
        """Return {impl_name: error_message} from BASELINE_ERROR lines."""
        return dict(self._baseline_errors)


def _get_l2_cache_bytes():
    """Query L2 cache size from the current CUDA device, fallback to 128MB."""
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        if hasattr(props, "l2_cache_size") and props.l2_cache_size > 0:
            return props.l2_cache_size
    except Exception:
        pass
    return 128 * 1024 * 1024  # 128MB default (B200)


def _tensor_bytes(args, _seen=None):
    """Sum the byte size of all torch/tvm tensors in a nested value."""
    if _seen is None:
        _seen = set()
    total = 0
    if isinstance(args, list | tuple):
        for a in args:
            total += _tensor_bytes(a, _seen)
    elif isinstance(args, Mapping):
        for a in args.values():
            total += _tensor_bytes(a, _seen)
    elif isinstance(args, torch.Tensor):
        key = ("torch", args.device.type, args.device.index, int(args.data_ptr()))
        if key not in _seen:
            _seen.add(key)
            total += args.nelement() * args.element_size()
    elif hasattr(args, "numpy"):  # tvm.runtime.NDArray
        try:
            key = ("tvm", int(args.handle.value))
        except Exception:
            key = ("tvm", id(args))
        if key not in _seen:
            _seen.add(key)
            try:
                total += int(np.prod(args.shape)) * np.dtype(str(args.dtype)).itemsize
            except Exception:
                total += args.numpy().nbytes
    return total


def tensor_bytes(*values):
    """Return unique torch/tvm tensor bytes for kernel-owned byte accounting.

    The benchmark driver does not use this implicitly.  Kernel benchmark
    factories may call it when their invocation footprint is exactly the set of
    tensors in ``values``.
    """
    if len(values) == 1:
        return _tensor_bytes(values[0])
    return _tensor_bytes(values)


def _compute_group_count(input_bytes, l2_bytes=None):
    """Return TK-style input-group count from one invocation's byte footprint."""
    if input_bytes <= 0:
        return 1
    if l2_bytes is None:
        l2_bytes = _get_l2_cache_bytes()
    threshold = l2_bytes * 3
    if input_bytes >= threshold:
        return 1
    return int(threshold // input_bytes) + 1


def _make_bench_input(input_factory):
    value = input_factory()
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("input_factory must return (case, input_bytes)")

    case, input_bytes = value
    try:
        input_bytes = int(input_bytes)
    except (TypeError, ValueError) as err:
        raise TypeError("input_factory input_bytes must be an integer") from err
    if input_bytes < 0:
        raise ValueError("input_factory input_bytes must be non-negative")
    return case, input_bytes


def prepare_input_groups(input_factory, l2_bytes=None):
    """Materialize TK-style input groups from a single-group factory.

    ``input_factory`` must return ``(case, input_bytes)``.  ``case`` is passed
    back to every benchmark function unchanged.  ``input_bytes`` defines one
    invocation's L2-eviction footprint and is intentionally owned by the kernel
    benchmark harness instead of inferred here.
    """
    if not callable(input_factory):
        raise TypeError("input_factory must be callable")
    if l2_bytes is None:
        l2_bytes = _get_l2_cache_bytes()

    sample, input_bytes = _make_bench_input(input_factory)
    num_groups = _compute_group_count(input_bytes, l2_bytes)
    groups = [sample]
    for _ in range(num_groups - 1):
        case, _ = _make_bench_input(input_factory)
        groups.append(case)

    return groups, {
        "num_groups": num_groups,
        "input_bytes": input_bytes,
        "l2_bytes": l2_bytes,
        "l2_eviction_factor": 3,
        "flush_l2": False,
    }


def _bench_event_groups(funcs, groups, warmup, repeat, cooldown_s):
    num_groups = len(groups)
    results = {}

    for idx, (name, func) in enumerate(funcs.items()):
        if idx > 0:
            time.sleep(cooldown_s)

        for i in range(warmup):
            func(groups[i % num_groups])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start_event.record()
        for i in range(repeat):
            func(groups[i % num_groups])
        end_event.record()

        torch.cuda.synchronize()
        results[name] = start_event.elapsed_time(end_event) / repeat * 1000.0

        time.sleep(cooldown_s)

    return results


def _bench_proton_groups(
    funcs, groups, warmup, repeat, cooldown_s, proton_name, debug, nsight, *, kernel=""
):
    num_groups = len(groups)
    with ProtonContext(proton_name, debug=debug, nsight=nsight, kernel=kernel) as ctx:
        for idx, (name, func) in enumerate(funcs.items()):
            if idx > 0:
                time.sleep(cooldown_s)

            for i in range(warmup):
                func(groups[i % num_groups])
            torch.cuda.synchronize()

            if not is_running_under_pytest() and not debug and not nsight:
                proton.activate()
                with proton.scope(name, metrics={}):
                    for i in range(repeat):
                        func(groups[i % num_groups])
                proton.deactivate()
            else:
                for i in range(repeat):
                    func(groups[i % num_groups])
            torch.cuda.synchronize()

            time.sleep(cooldown_s)

    return ctx.get_impl_times(), ctx.get_baseline_errors()


def _flush_l2_legacy(flush_l2_size):
    if flush_l2_size > 0:
        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()


def _bench_legacy_callable(func, warmup, repeat, proton_name, debug, nsight, flush_l2_size):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def timed_loop():
        start_event.record()
        for _ in range(repeat):
            _flush_l2_legacy(flush_l2_size)
            func()
        end_event.record()

    for _ in range(warmup):
        _flush_l2_legacy(flush_l2_size)
        func()
    torch.cuda.synchronize()
    if not is_running_under_pytest() and not debug and not nsight:
        proton.activate()
        with proton.scope(proton_name, metrics={}):
            timed_loop()
        proton.deactivate()
    else:
        timed_loop()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / repeat * 1000.0


def bench_tk(
    funcs,
    input_factory=None,
    warmup=500,
    repeat=100,
    cooldown_s=1.0,
    timer="proton",
    proton_name="kernel",
    l2_bytes=None,
    debug=False,
    nsight=False,
    flush_l2_size=int(8e8 // 4),
    references=None,
    rounds=1,
    round_cooldown_s=1.0,
    validate_case=None,
):
    """Benchmark implementations with a factory-owned input footprint.

    This is the ThunderKittens-style TIRx benchmark API.  It follows the
    multi-input protocol for L2 eviction (adapted from ThunderKittens,
    https://github.com/HazyResearch/ThunderKittens) and supports either
    Proton/CUPTI or CUDA-event timing.  The benchmark driver never infers which
    tensors belong to a workload; ``input_factory`` owns that definition by
    returning ``(case, input_bytes)``.

    For the Triton-standard, pure-launch benchmark path (``do_bench`` /
    ``do_bench_cudagraph`` semantics, no group protocol), use ``bench`` instead.

    Parameters
    ----------
    funcs : dict[str, callable]
        Map of implementation name to callable.  Each callable receives one
        ``case`` returned by ``input_factory``.  This should hold only *our*
        kernel(s); external baselines go in ``references``.
    references : dict[str, callable], optional
        Map of reference-impl name to a no-arg *builder* that does the heavy
        import/setup and returns the run callable.  A builder that raises is
        recorded as a ``BASELINE_ERROR`` instead of failing the workload.
    input_factory : callable
        Factory returning ``(case, input_bytes)`` for one benchmark group.
    warmup : int
        Number of untimed warmup iterations per implementation.
    repeat : int
        Number of timed iterations per round.
    cooldown_s : float
        Seconds to sleep between impls for thermal cooldown.
    rounds : int
        Independent benchmark rounds (compile + inputs once; each round runs
        warmup + repeat for every selected impl).
    round_cooldown_s : float
        Seconds to sleep between rounds (ignored when ``rounds == 1``).
    validate_case : callable, optional
        Called once on the first prepared ``case`` (after ``prepare_input_groups``,
        before warmup/repeat rounds). Under tir-bench, ``run_kernel_bench`` holds
        the per-GPU lock for the whole ``run_bench()`` call.
    timer : {"event", "proton"}
        Timing backend.

    Returns
    -------
    dict
        ``{"impls": {name: us}, "round_samples": {name: [us, ...]}, ...}``.
        Times are stored in microseconds (same unit as pinned tir-bench baselines).
    """
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if round_cooldown_s < 0:
        raise ValueError("round_cooldown_s must be non-negative")
    if timer not in {"event", "proton"}:
        raise ValueError(f"unsupported timer {timer!r}; expected event or proton")

    if callable(funcs) and input_factory is None:
        return _bench_legacy_callable(
            funcs,
            warmup=warmup,
            repeat=repeat,
            proton_name=proton_name,
            debug=debug,
            nsight=nsight,
            flush_l2_size=flush_l2_size,
        )

    if input_factory is None:
        raise TypeError("input_factory is required when funcs is a mapping")
    if not isinstance(funcs, Mapping) or not funcs:
        raise TypeError("funcs must be a non-empty mapping of name to callable")
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

    inputs, protocol = prepare_input_groups(input_factory, l2_bytes=l2_bytes)
    num_groups = len(inputs)
    if num_groups == 0:
        return {
            "impls": {},
            "round_samples": {},
            "errors": build_errors,
            "timer": timer,
            "benchmark_protocol": {
                **protocol,
                "warmup": warmup,
                "repeat": repeat,
                "cooldown_s": cooldown_s,
                "rounds": rounds,
                "round_cooldown_s": round_cooldown_s,
                "order": list(funcs.keys()),
            },
        }

    if validate_case is not None:
        validate_case(inputs[0])

    errors = dict(build_errors)
    round_samples: dict[str, list[float]] = {}
    for round_idx in range(rounds):
        if round_idx > 0:
            time.sleep(round_cooldown_s)
        if timer == "event":
            impls = _bench_event_groups(funcs, inputs, warmup, repeat, cooldown_s)
            proton_errors = {}
        else:
            impls, proton_errors = _bench_proton_groups(
                funcs,
                inputs,
                warmup,
                repeat,
                cooldown_s,
                proton_name,
                debug,
                nsight,
                kernel=proton_name,
            )
        errors.update(proton_errors)
        for impl, sec in impls.items():
            round_samples.setdefault(impl, []).append(sec)

    if not round_samples:
        aggregated = {}
    else:
        aggregated = {impl: statistics.mean(samples) for impl, samples in round_samples.items()}

    return {
        "impls": aggregated,
        "round_samples": round_samples,
        "errors": errors,
        "timer": timer,
        "benchmark_protocol": {
            **protocol,
            "warmup": warmup,
            "repeat": repeat,
            "cooldown_s": cooldown_s,
            "rounds": rounds,
            "round_cooldown_s": round_cooldown_s,
            "order": list(funcs.keys()),
        },
    }


# ---------------------------------------------------------------------------
# Triton-standard benchmark path.
#
# Faithful in-repo port of triton.testing.do_bench / do_bench_cudagraph
# (see https://github.com/triton-lang/triton, python/triton/testing.py). This is
# torch-only and does NOT import or call into triton at runtime: Triton driver
# calls (get_device_interface / get_empty_cache_for_benchmark / clear_cache) are
# replaced with direct torch.cuda + a torch L2-flush buffer. The timed function
# is a *pure no-arg launch closure* (inputs captured once, allocated outside the
# timed region) -- exactly how Triton times a function.
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
    # 256 MB buffer whose .zero_() evicts the L2 cache between measured iters.
    return torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")


def _do_bench_event(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
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
    if estimate_ms == 0:
        n_warmup = 1000
        n_repeat = 1000
    else:
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


def _do_bench_proton(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean"):
    """Port of triton.testing.do_bench_proton, aligned with ``_do_bench_event``.

    IDENTICAL to ``_do_bench_event`` in everything -- warmup/rep millisecond budgets,
    the 5-call estimate, per-iter L2 flush, the untimed warmup loop -- EXCEPT the
    timing mechanism: each timed call runs inside a Proton scope and the per-kernel
    GPU time (read from the hatchet tree) is used instead of the CUDA-event wall.
    Cold cache. NVIDIA + Proton only. No CUDA graph (so it works for references that
    can't be graph-captured, e.g. CuTeDSL flash-attention).

    Falls back to ``_do_bench_event`` under pytest or when a Proton session cannot be
    created.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    if is_running_under_pytest():
        return _do_bench_event(
            fn,
            warmup=warmup,
            rep=rep,
            grad_to_none=grad_to_none,
            quantiles=quantiles,
            return_mode=return_mode,
        )

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
        session = proton.start(profile_path, context="shadow", data="tree")
        if session is None:
            print(
                "proton: Proton session unavailable; falling back to event timing", file=sys.stderr
            )
            return _do_bench_event(
                fn,
                warmup=warmup,
                rep=rep,
                grad_to_none=grad_to_none,
                quantiles=quantiles,
                return_mode=return_mode,
            )
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

    Falls back to ``_do_bench_event`` (cold-cache CUDA-event timing) under pytest or
    when a Proton session cannot be created -- staying on a cold-cache timer keeps it
    consistent with the rest of the baseline.
    """
    assert return_mode in ["min", "max", "mean", "median", "all"]

    if is_running_under_pytest():
        return _do_bench_event(
            fn, grad_to_none=grad_to_none, quantiles=quantiles, return_mode=return_mode
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
            session = proton.start(profile_path, context="shadow", data="tree")
            if session is None:
                print(
                    "cudagraph_proton: Proton session unavailable; falling back to event timing",
                    file=sys.stderr,
                )
                return _do_bench_event(
                    fn, grad_to_none=grad_to_none, quantiles=quantiles, return_mode=return_mode
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


def bench(
    funcs,
    *,
    warmup=None,
    repeat=None,
    cudagraph_rep=None,
    timer=None,
    references=None,
    rounds=1,
    round_cooldown_s=1.0,
):
    """Benchmark pure-launch implementations using Triton-standard timing.

    Each callable in ``funcs`` is a *no-arg launch closure* (inputs allocated
    once and captured in the closure), timed exactly the way
    ``triton.testing.do_bench`` / ``do_bench_cudagraph_proton`` time a function. The
    timing core is a faithful in-repo port (torch-only, no triton dependency);
    see ``_do_bench_event`` / ``_do_bench_cudagraph_proton``.

    For the ThunderKittens-style multi-input group protocol (``input_factory``,
    per-workload L2 eviction, Proton per-kernel attribution), use ``bench_tk``.

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
        Warmup time budget (ms) for the ``event`` timer. ``None`` (default) defers
        to ``_do_bench_event``'s own default; pass a value only to override. Ignored
        by the graph timers (which have no warmup).
    repeat : int, optional
        Rep time budget (ms) for the ``event`` timer. ``None`` (default) defers to
        ``_do_bench_event``'s own default; pass a value only to override.
    cudagraph_rep : int, optional
        ``rep`` (ms) for the graph timers (graph unroll length). ``None`` (default)
        defers to ``_do_bench_cudagraph_proton``'s own default. Each timer default
        lives only in its own signature (Triton: do_bench 25/100, graph 20); nothing
        is hardcoded here.
    timer : {None, "event", "proton", "cudagraph_proton"}
        ``None`` (default) resolves to ``proton`` -- see the resolution in the body;
        the default is defined in exactly one place so kernels pass ``None`` to inherit.
        All three are **cold-cache** (per-iter L2 flush) so results are comparable.
        ``event`` -> ported ``do_bench`` (CUDA-event wall of each call).
        ``proton`` -> ported ``do_bench_proton``: same setup as ``event``, differs
        ONLY in timing -- Proton per-kernel GPU time instead of the event wall (so
        launch/host overhead of the ref is excluded). No graph, so it works for
        references that can't be CUDA-graph-captured (e.g. CuTeDSL flash-attention).
        NVIDIA + Proton only.
        ``cudagraph_proton`` -> ``do_bench_cudagraph_proton`` (graph replay + Proton +
        L2 flush); like ``proton`` but also removes launch overhead via graph replay --
        best for tiny back-to-back kernels, but the graph capture fails/misattributes
        for some references (CuTeDSL) so it is not universal.
        Prefer ``proton`` when the reference has heavy host dispatch (flashinfer,
        CuTeDSL) -- ``event`` would over/under-credit us by measuring the wall, not the
        kernel. (Plain no-flush ``do_bench_cudagraph`` is intentionally NOT offered.)
    rounds : int
        Independent measurement rounds; per-impl times are averaged across
        rounds. (Triton times a single fn with no rounds; this is our sampling
        layer on top.)
    round_cooldown_s : float
        Seconds to sleep between rounds (ignored when ``rounds == 1``).

    Returns
    -------
    dict
        ``{"impls": {name: us}, "round_samples": {name: [us, ...]}, ...}``.
        Times are stored in microseconds (same unit as ``bench_tk`` and the
        pinned tir-bench baselines).
    """
    # warmup/repeat/cudagraph_rep default to None = "use the timer function's own
    # (Triton-aligned) default". They are only forwarded to the timer below when a
    # caller explicitly overrides, so the defaults live in exactly one place: the
    # _do_bench_* signatures.
    if repeat is not None and repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup is not None and warmup < 0:
        raise ValueError("warmup must be non-negative")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if round_cooldown_s < 0:
        raise ValueError("round_cooldown_s must be non-negative")
    # ``timer=None`` means "use the default timer". The default lives in exactly one
    # place -- here -- so kernels forward ``timer=None`` to inherit it and a future
    # change is a one-line edit. proton = pure per-kernel GPU time; it is honest for
    # references with heavy host dispatch (flashinfer, CuTeDSL) where the ``event``
    # wall would over/under-credit us, and matches event within a few % on
    # compute-bound kernels. It auto-falls back to ``event`` when Proton/CUPTI is
    # unavailable (non-NVIDIA, or under pytest).
    if timer is None:
        timer = "proton"
    if timer not in {"event", "proton", "cudagraph_proton"}:
        raise ValueError(
            f"unsupported timer {timer!r}; expected event, proton, or cudagraph_proton"
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
    else:  # cudagraph_proton -- no warmup; rep is the graph unroll budget
        _timer_fn = _do_bench_cudagraph_proton
        _timer_kwargs = {}
        if cudagraph_rep is not None:
            _timer_kwargs["rep"] = cudagraph_rep
        _eff = {"cudagraph_rep": _timer_kwargs.get("rep", _sig_default(_timer_fn, "rep"))}

    protocol = {
        **_eff,
        "rounds": rounds,
        "round_cooldown_s": round_cooldown_s,
        "order": list(funcs.keys()),
    }

    round_samples: dict[str, list[float]] = {}
    for round_idx in range(rounds):
        if round_idx > 0:
            time.sleep(round_cooldown_s)
        for name, func in funcs.items():
            ms = _timer_fn(func, **_timer_kwargs)
            # ms -> microseconds (matches bench_tk unit and pinned baselines).
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

    When ``profiler_enabled`` is False (or a false-y PrimExpr), calls to
    ``init/start/end/finalize`` become no-ops. This allows constructing a
    profiler unconditionally and eliminating external ``if PROFILER_ON:`` guards.
    """

    def __init__(
        self,
        profiler_buffer: T.Buffer,
        write_stride: int,
        num_groups: int,
        default_leader: None | tvm.tirx.PrimExpr | bool = None,
        profiler_enabled: bool | tvm.tirx.PrimExpr = True,
    ):
        self.buffer = profiler_buffer
        self.write_stride = write_stride
        self.num_groups = num_groups
        self.default_leader = default_leader
        # Accept either a Python bool or a PrimExpr; normalize simple bools to T.bool
        # so we can use it uniformly inside macros for conditional emission.
        if isinstance(profiler_enabled, bool | np.bool_):
            self.profiler_enabled = T.bool(bool(profiler_enabled))
        else:
            # Assume PrimExpr-like input; use as-is
            self.profiler_enabled = profiler_enabled  # type: ignore[assignment]

        self.profiler_tag = T.alloc_buffer([1], "uint64", scope="local", align=8)
        self.profiler_write_offset = T.alloc_buffer([1], "uint32", scope="local", align=8)

    def _leader(self, leader: None | tvm.tirx.PrimExpr | bool):
        if leader is not None:
            if isinstance(leader, bool | np.bool_):
                return T.bool(bool(leader))
            return leader
        if self.default_leader is not None:
            return self.default_leader
        return T.bool(True)

    @T.inline
    def init(self, group_id: tvm.tirx.PrimExpr):
        if self.profiler_enabled:
            T.cuda.timer_init(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.num_groups,
                group_id,
            )

    @T.inline
    def start(self, event_type: Enum, leader: None | tvm.tirx.PrimExpr | bool = None):
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
    def end(self, event_type: Enum, leader: None | tvm.tirx.PrimExpr | bool = None):
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
    def finalize(self, leader: None | tvm.tirx.PrimExpr | bool = None):
        if self.profiler_enabled:
            T.cuda.timer_finalize(
                self.buffer.data,
                self.profiler_tag.data,
                self.profiler_write_offset.data,
                self.write_stride,
                self._leader(leader),
            )
