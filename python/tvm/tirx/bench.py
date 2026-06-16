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
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
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


def _parse_proton_tree(text, value_scale=1.0):
    """Parse proton-viewer tree output into {impl: time_ms}.

    Accepts ALL depth-1 nodes (no KNOWN_IMPLS filter). For each depth-1 impl,
    takes the slowest depth-2 child kernel time.

    ``value_scale`` converts the displayed metric to milliseconds.  For
    example, use ``1e-3`` when parsing ``avg_time/us`` output.

    Returns (impl_times, baseline_errors) where:
      impl_times: {str: float} — impl name to avg time in ms
      baseline_errors: {str: str} — impl name to error message
    """
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
                    t = float(parts[0]) * value_scale
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
        metric="avg_time/us",
        metric_scale=1e-3,
    ):
        self.name = name
        self.hook = hook
        self.debug = debug
        self.nsight = nsight
        self.metric = metric
        self.metric_scale = metric_scale
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
                    result.stdout, value_scale=self.metric_scale
                )
                out = sys.stderr if os.environ.get("TIRX_BENCH_JSON") else sys.stdout
                print(result.stdout, file=out, end="")
            else:
                print(
                    f"proton-viewer failed (rc={result.returncode}): {result.stderr}",
                    file=sys.stderr,
                )

            if os.path.exists(hatchet):
                os.remove(hatchet)

    def get_impl_times(self):
        """Return {impl_name: avg_time_ms} parsed from proton-viewer output."""
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
        results[name] = start_event.elapsed_time(end_event) / repeat

        time.sleep(cooldown_s)

    return results


def _bench_proton_groups(funcs, groups, warmup, repeat, cooldown_s, proton_name, debug, nsight):
    num_groups = len(groups)
    with ProtonContext(proton_name, debug=debug, nsight=nsight) as ctx:
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
    for _ in range(warmup):
        _flush_l2_legacy(flush_l2_size)
        func()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    def timed_loop():
        start_event.record()
        for _ in range(repeat):
            _flush_l2_legacy(flush_l2_size)
            func()
        end_event.record()

    if not is_running_under_pytest() and not debug and not nsight:
        proton.activate()
        with proton.scope(proton_name, metrics={}):
            timed_loop()
        proton.deactivate()
    else:
        timed_loop()

    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / repeat


def bench(
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
):
    """Benchmark implementations with a factory-owned input footprint.

    This is the single TIRx benchmark API.  It follows the ThunderKittens-style
    multi-input protocol for L2 eviction and supports either Proton/CUPTI or
    CUDA-event timing.  The benchmark driver never infers which tensors belong
    to a workload; ``input_factory`` owns that definition by returning
    ``(case, input_bytes)``.

    Parameters
    ----------
    funcs : dict[str, callable]
        Map of implementation name to callable.  Each callable receives one
        ``case`` returned by ``input_factory``.
    input_factory : callable
        Factory returning ``(case, input_bytes)`` for one benchmark group.
    warmup : int
        Number of untimed warmup iterations per implementation.
    repeat : int
        Number of timed iterations.
    cooldown_s : float
        Seconds to sleep between impls for thermal cooldown.
    timer : {"event", "proton"}
        Timing backend.

    Returns
    -------
    dict
        ``{"impls": {name: ms}, "errors": {}, "timer": ..., ...}``.
    """
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
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

    inputs, protocol = prepare_input_groups(input_factory, l2_bytes=l2_bytes)
    num_groups = len(inputs)
    if num_groups == 0:
        return {
            "impls": {},
            "errors": {},
            "timer": timer,
            "benchmark_protocol": {
                **protocol,
                "warmup": warmup,
                "repeat": repeat,
                "cooldown_s": cooldown_s,
                "order": list(funcs.keys()),
            },
        }

    errors = {}
    if timer == "event":
        impls = _bench_event_groups(funcs, inputs, warmup, repeat, cooldown_s)
    else:
        impls, errors = _bench_proton_groups(
            funcs, inputs, warmup, repeat, cooldown_s, proton_name, debug, nsight
        )

    return {
        "impls": impls,
        "errors": errors,
        "timer": timer,
        "benchmark_protocol": {
            **protocol,
            "warmup": warmup,
            "repeat": repeat,
            "cooldown_s": cooldown_s,
            "order": list(funcs.keys()),
        },
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
