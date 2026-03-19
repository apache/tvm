#!/usr/bin/env python3
"""Measure a small programmer-facing performance reference for the RK3588 NPU.

This tool focuses on the current clean 2-D subset:
- matmul
- matmul + bias
- matmul + bias + relu
- add
- mul
- residual MLP MVP block

It collects four classes of information:
1. latency atlas for small and large shapes
2. penalty for fused task vs split tasks vs split submits
3. effective throughput for the measured op families
4. task-template facts already encoded in the backend
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
sys.modules.setdefault("torch", None)

import tvm
from tvm import relax

import tvm.relax.backend.contrib.rknpu as rknpu
from tvm.relax.backend.contrib.rknpu.npu_core import hardware


DTYPE = "float16"
DEFAULT_DEVFREQ_DIR = "/sys/class/devfreq/fdab0000.npu"
REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_RUNNER_SRC = REPO_ROOT / "tools" / "rknpu_vm_cpp_runner.cc"
CPP_RUNNER_BIN = REPO_ROOT / "build" / "rknpu_vm_cpp_runner"
DRIVER_PROBE = REPO_ROOT / "tools" / "rknpu_driver_probe.py"
STAGE_NAME_BY_ID = {
    1: "matmul",
    2: "add",
    3: "relu",
    4: "relu_4d",
    5: "conv2d",
    6: "matmul_bias_relu",
    7: "add_relu",
    8: "conv2d_relu",
    9: "mul",
    10: "exp",
    11: "matmul_bias",
    12: "reciprocal",
    13: "gelu",
}


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    size_tag: str
    ordered_input_names: Sequence[str]
    build_mod: Callable[[], tvm.IRModule]
    make_inputs: Callable[[np.random.Generator], Dict[str, np.ndarray]]
    ref: Callable[[Dict[str, np.ndarray]], np.ndarray]
    metrics: Callable[[], Dict[str, float]]
    env_overrides: Dict[str, str]
    comparison_group: str = ""
    comparison_mode: str = "base"
    max_err: float = 1e-3


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _parse_total_transitions(text: str | None) -> int | None:
    if not text:
        return None
    for line in text.splitlines():
        if "Total transition" in line:
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _read_devfreq_snapshot(devfreq_dir: Path | None) -> Dict[str, object]:
    if devfreq_dir is None:
        return {}
    snap = {
        "path": str(devfreq_dir),
        "cur_freq": _read_text(devfreq_dir / "cur_freq"),
        "target_freq": _read_text(devfreq_dir / "target_freq"),
        "min_freq": _read_text(devfreq_dir / "min_freq"),
        "max_freq": _read_text(devfreq_dir / "max_freq"),
        "governor": _read_text(devfreq_dir / "governor"),
        "load": _read_text(devfreq_dir / "load"),
        "available_frequencies": _read_text(devfreq_dir / "available_frequencies"),
        "available_governors": _read_text(devfreq_dir / "available_governors"),
    }
    snap["transitions_total"] = _parse_total_transitions(_read_text(devfreq_dir / "trans_stat"))
    return snap


def _read_driver_probe_snapshot(
    *,
    enabled: bool,
    device: str,
    debugfs_root: str,
    procfs_root: str,
    devfreq_dir: str,
    include_unsupported: bool,
) -> Dict[str, object]:
    if not enabled:
        return {}
    cmd = [sys.executable, str(DRIVER_PROBE), "--json"]
    if device:
        cmd.extend(["--device", device])
    if debugfs_root:
        cmd.extend(["--debugfs-root", debugfs_root])
    if procfs_root:
        cmd.extend(["--procfs-root", procfs_root])
    if devfreq_dir:
        cmd.extend(["--devfreq-dir", devfreq_dir])
    if include_unsupported:
        cmd.append("--include-unsupported")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "error": "invalid_json",
        }
    payload["ok"] = True
    return payload


def _run_vm(vm: relax.VirtualMachine, ordered_inputs: Sequence[np.ndarray]) -> np.ndarray:
    args = [tvm.runtime.tensor(x) for x in ordered_inputs]
    return vm["main"](*args).numpy()


@contextlib.contextmanager
def _temp_env(overrides: Dict[str, str]):
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _base_env() -> Dict[str, str]:
    return {
        "TVM_RKNPU_PC_CHAIN_INCLUDE_SINGLETONS": "1",
        "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "0",
        "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "0",
        "TVM_RKNPU_BRIDGE_REAL_SUBMIT": "1",
        "TVM_RKNPU_BRIDGE_USE_RELOCS": "1",
        "TVM_RKNPU_BRIDGE_RUN_CPU_AFTER_SUBMIT": "0",
        "TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK": "1",
        "TVM_RKNPU_BRIDGE_VALIDATE_RELOCS": "0",
        "TVM_RKNPU_BRIDGE_CHECK_OUTPUTS": "0",
    }


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def _ensure_cpp_runner_built() -> Path:
    runner_inputs = [
        CPP_RUNNER_SRC,
        REPO_ROOT / "build" / "libtvm.so",
        REPO_ROOT / "build" / "libtvm_runtime.so",
        REPO_ROOT / "build" / "lib" / "libtvm_ffi.so",
    ]
    if CPP_RUNNER_BIN.exists():
        bin_mtime = CPP_RUNNER_BIN.stat().st_mtime
        if all(path.exists() and path.stat().st_mtime <= bin_mtime for path in runner_inputs):
            return CPP_RUNNER_BIN
    cmd = [
        "g++",
        "-std=c++17",
        "-O2",
        f"-I{REPO_ROOT / 'include'}",
        f"-I{REPO_ROOT / '3rdparty' / 'tvm-ffi' / 'include'}",
        f"-I{REPO_ROOT / '3rdparty' / 'tvm-ffi' / '3rdparty' / 'dlpack' / 'include'}",
        str(CPP_RUNNER_SRC),
        f"-L{REPO_ROOT / 'build'}",
        f"-L{REPO_ROOT / 'build' / 'lib'}",
        "-ltvm",
        "-ltvm_ffi",
        f"-Wl,-rpath,{REPO_ROOT / 'build'}",
        f"-Wl,-rpath,{REPO_ROOT / 'build' / 'lib'}",
        "-o",
        str(CPP_RUNNER_BIN),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return CPP_RUNNER_BIN


def _export_case_cpp_bundle(
    bundle_dir: Path,
    case: Case,
    lowered: tvm.IRModule,
    executable,
    inputs: Dict[str, np.ndarray],
    expected: np.ndarray,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    executable.export_library(str(bundle_dir / "exec.so"))
    (bundle_dir / "chain_blob.bin").write_bytes(rknpu.get_bridge_chain_blob(lowered))
    manifest_lines = [
        "entry=main",
        f"dtype={DTYPE}",
        f"num_inputs={len(case.ordered_input_names)}",
    ]
    for i, name in enumerate(case.ordered_input_names):
        arr = inputs[name]
        (bundle_dir / f"{name}.bin").write_bytes(arr.tobytes(order="C"))
        manifest_lines.extend(
            [
                f"input{i}_name={name}",
                f"input{i}_file={name}.bin",
                f"input{i}_shape={_shape_text(arr.shape)}",
            ]
        )
    (bundle_dir / "expected_y.bin").write_bytes(expected.tobytes(order="C"))
    manifest_lines.extend(
        [
            "num_outputs=1",
            "output0_name=out",
            "output0_expected_file=expected_y.bin",
            f"output0_shape={_shape_text(expected.shape)}",
        ]
    )
    (bundle_dir / "bundle.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def _schedule_summary(report: Dict[str, object]) -> Dict[str, object]:
    submit_stage_ids = [
        [int(stage_id) for stage_id in submit]
        for submit in report.get("submit_stage_ids", [])
    ]
    return {
        "num_submits": int(report.get("num_submits", 0)),
        "total_tasks": int(report.get("total_tasks", 0)),
        "submit_task_counts": [int(x) for x in report.get("submit_task_counts", [])],
        "submit_stage_ids": submit_stage_ids,
        "submit_stage_names": [
            [STAGE_NAME_BY_ID.get(stage_id, f"stage_{stage_id}") for stage_id in submit]
            for submit in submit_stage_ids
        ],
        "blocked_boundary_count": int(
            report.get("chain_compatibility", {}).get("blocked_boundary_count", 0)
        ),
    }


def _runtime_sanity(report: Dict[str, object], stats: Dict[str, object], iters: int) -> List[str]:
    errors: List[str] = []
    expected_submit_calls = int(report.get("num_submits", 0)) * int(iters)
    expected_task_calls = int(report.get("total_tasks", 0)) * int(iters)
    if int(stats.get("real_submit_fail", 0)) != 0:
        errors.append(f"real_submit_fail={stats.get('real_submit_fail')}")
    if int(stats.get("touch_fallback", 0)) != 0:
        errors.append(f"touch_fallback={stats.get('touch_fallback')}")
    if int(stats.get("reloc_submit_fallbacks", 0)) != 0:
        errors.append(f"reloc_submit_fallbacks={stats.get('reloc_submit_fallbacks')}")
    if int(stats.get("reloc_semantic_mismatch", 0)) != 0:
        errors.append(f"reloc_semantic_mismatch={stats.get('reloc_semantic_mismatch')}")
    if int(stats.get("reloc_range_mismatch", 0)) != 0:
        errors.append(f"reloc_range_mismatch={stats.get('reloc_range_mismatch')}")
    if int(stats.get("real_submit_ok", -1)) != expected_submit_calls:
        errors.append(
            f"real_submit_ok={stats.get('real_submit_ok')} expected={expected_submit_calls}"
        )
    if int(stats.get("submitted_tasks", -1)) != expected_task_calls:
        errors.append(
            f"submitted_tasks={stats.get('submitted_tasks')} expected={expected_task_calls}"
        )
    return errors


def _hw_elapsed_ns_total(stats: Dict[str, object]) -> int:
    total = 0
    for bucket in stats.get("submit_timing_buckets", []):
        total += int(bucket.get("hw_elapsed_ns", 0))
    return total


def _submit_timing_metric_total(stats: Dict[str, object], key: str) -> int:
    total = 0
    for bucket in stats.get("submit_timing_buckets", []):
        total += int(bucket.get(key, 0))
    return total


def _aggregate_runtime_stats(sample_stats: Sequence[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {}
    bucket_by_key: Dict[tuple, Dict[str, object]] = {}
    for stats in sample_stats:
        for key, value in stats.items():
            if key == "submit_timing_buckets":
                for bucket in value:
                    bkey = (
                        int(bucket.get("submit_slot", 0)),
                        int(bucket.get("stage_count", 0)),
                        int(bucket.get("task_count", 0)),
                    )
                    acc = bucket_by_key.setdefault(
                        bkey,
                        {
                            "submit_slot": bkey[0],
                            "stage_count": bkey[1],
                            "task_count": bkey[2],
                        },
                    )
                    for bfield, bvalue in bucket.items():
                        if bfield in ("submit_slot", "stage_count", "task_count"):
                            continue
                        acc[bfield] = int(acc.get(bfield, 0)) + int(bvalue)
                continue
            if not isinstance(value, (int, bool)):
                continue
            aggregate[key] = int(aggregate.get(key, 0)) + int(value)
    aggregate["submit_timing_buckets"] = [
        bucket_by_key[key] for key in sorted(bucket_by_key.keys())
    ]
    return aggregate


def _steady_state_reached(signal_ns: Sequence[int], window: int, rel_tol: float) -> bool:
    if window <= 0 or len(signal_ns) < window:
        return False
    recent = [max(int(x), 1) for x in signal_ns[-window:]]
    hi = max(recent)
    lo = min(recent)
    return ((hi - lo) / float(hi)) <= rel_tol


def _measure_case(
    case: Case,
    warmup: int,
    iters: int,
    seed: int,
    devfreq_dir: Path | None = None,
    capture_driver_probe: bool = False,
    driver_probe_device: str = "",
    driver_probe_debugfs_root: str = "",
    driver_probe_procfs_root: str = "",
    driver_probe_devfreq_dir: str = "",
    driver_probe_include_unsupported: bool = False,
    global_env_overrides: Dict[str, str] | None = None,
    warmup_mode: str = "steady",
    steady_state_window: int = 4,
    steady_state_rtol: float = 0.10,
    steady_state_max_extra_warmup: int = 24,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    inputs = case.make_inputs(rng)
    ordered_inputs = [inputs[name] for name in case.ordered_input_names]
    env = dict(_base_env())
    env.update(case.env_overrides)
    if global_env_overrides:
        env.update(global_env_overrides)

    with _temp_env(env):
        mod = case.build_mod()
        lowered = rknpu.plan_rknpu_tir_memory(rknpu.lower_to_rknpu_tir_with_pc_chain(mod))
        report = rknpu.get_rknpu_schedule_report(lowered)
        vm = rknpu.build_vm_with_runtime_bridge(lowered, tvm.cpu(), target="llvm")
        driver_probe_before = _read_driver_probe_snapshot(
            enabled=capture_driver_probe,
            device=driver_probe_device,
            debugfs_root=driver_probe_debugfs_root,
            procfs_root=driver_probe_procfs_root,
            devfreq_dir=driver_probe_devfreq_dir,
            include_unsupported=driver_probe_include_unsupported,
        )
        warmup_wall_ns: List[int] = []
        warmup_hw_ns: List[int] = []
        warmup_iters = 0
        steady_state_reached = warmup_mode != "steady"
        max_warmup_iters = max(warmup, 0)
        if warmup_mode == "steady":
            max_warmup_iters += max(steady_state_max_extra_warmup, 0)
        while warmup_iters < max_warmup_iters:
            rknpu.reset_runtime_bridge_stats()
            t0 = time.perf_counter_ns()
            _ = _run_vm(vm, ordered_inputs)
            t1 = time.perf_counter_ns()
            stats = rknpu.get_runtime_bridge_stats()
            warmup_wall_ns.append(t1 - t0)
            warmup_hw_ns.append(_hw_elapsed_ns_total(stats))
            warmup_iters += 1
            if warmup_mode != "steady" or warmup_iters < max(warmup, 0):
                continue
            signal_ns = [
                hw if hw > 0 else wall for hw, wall in zip(warmup_hw_ns, warmup_wall_ns)
            ]
            if _steady_state_reached(signal_ns, steady_state_window, steady_state_rtol):
                steady_state_reached = True
                break
        rknpu.reset_runtime_bridge_stats()

        wall_samples_ns: List[int] = []
        runtime_total_samples_ns: List[int] = []
        runtime_submit_samples_ns: List[int] = []
        runtime_hw_samples_ns: List[int] = []
        devfreq_before_samples: List[Dict[str, object]] = []
        devfreq_after_samples: List[Dict[str, object]] = []
        timed_runtime_stats: List[Dict[str, object]] = []
        out = None
        for _ in range(max(iters, 1)):
            if devfreq_dir is not None:
                devfreq_before_samples.append(_read_devfreq_snapshot(devfreq_dir))
            rknpu.reset_runtime_bridge_stats()
            t0 = time.perf_counter_ns()
            out = _run_vm(vm, ordered_inputs)
            t1 = time.perf_counter_ns()
            iter_stats = rknpu.get_runtime_bridge_stats()
            if devfreq_dir is not None:
                devfreq_after_samples.append(_read_devfreq_snapshot(devfreq_dir))
            wall_samples_ns.append(t1 - t0)
            runtime_total_samples_ns.append(_submit_timing_metric_total(iter_stats, "total_ns"))
            runtime_submit_samples_ns.append(_submit_timing_metric_total(iter_stats, "submit_ns"))
            runtime_hw_samples_ns.append(_submit_timing_metric_total(iter_stats, "hw_elapsed_ns"))
            timed_runtime_stats.append(iter_stats)

        ref = case.ref(inputs)
        max_err = float(np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32))))
        stats = _aggregate_runtime_stats(timed_runtime_stats)
        runtime_errors = _runtime_sanity(report, stats, max(iters, 1))
        if max_err > case.max_err:
            runtime_errors.append(f"max_err={max_err:.6g} limit={case.max_err:.6g}")
        if runtime_errors:
            raise RuntimeError(f"{case.name}: " + "; ".join(runtime_errors))

        wall_sample_array = np.array(wall_samples_ns, dtype=np.int64)
        runtime_total_array = np.array(runtime_total_samples_ns, dtype=np.int64)
        runtime_submit_array = np.array(runtime_submit_samples_ns, dtype=np.int64)
        runtime_hw_array = np.array(runtime_hw_samples_ns, dtype=np.int64)
        tail_count = max(1, len(wall_samples_ns) // 2)
        wall_ns_median = int(np.median(wall_sample_array))
        wall_ns_tail_median = int(np.median(wall_sample_array[-tail_count:]))
        runtime_total_ns_median = int(np.median(runtime_total_array))
        runtime_total_ns_tail_median = int(np.median(runtime_total_array[-tail_count:]))
        runtime_submit_ns_median = int(np.median(runtime_submit_array))
        runtime_submit_ns_tail_median = int(np.median(runtime_submit_array[-tail_count:]))
        runtime_hw_ns_median = int(np.median(runtime_hw_array))
        runtime_hw_ns_tail_median = int(np.median(runtime_hw_array[-tail_count:]))
        primary_latency_kind = "runtime_total_tail"
        latency_ns = runtime_total_ns_tail_median
        schedule = _schedule_summary(report)
        metrics = dict(case.metrics())
        if "macs" in metrics:
            macs = float(metrics["macs"])
            latency_s = latency_ns / 1e9
            metrics["effective_tmac_s"] = macs / latency_s / 1e12
            metrics["effective_tflop_s"] = (2.0 * macs) / latency_s / 1e12
        if "elements" in metrics:
            elems = float(metrics["elements"])
            latency_s = latency_ns / 1e9
            metrics["effective_gelem_s"] = elems / latency_s / 1e9

        record = {
            "host": "python",
            "name": case.name,
            "family": case.family,
            "size_tag": case.size_tag,
            "comparison_group": case.comparison_group,
            "comparison_mode": case.comparison_mode,
            "latency_kind": primary_latency_kind,
            "latency_ns_median": latency_ns,
            "latency_us_median": latency_ns / 1e3,
            "latency_ms_median": latency_ns / 1e6,
            "cold_start": {
                "wall_ns": int(wall_samples_ns[0]),
                "runtime_total_ns": int(runtime_total_samples_ns[0]),
                "runtime_submit_ns": int(runtime_submit_samples_ns[0]),
                "runtime_hw_ns": int(runtime_hw_samples_ns[0]),
            },
            "steady_tail": {
                "tail_count": int(tail_count),
                "wall_ns_median": wall_ns_tail_median,
                "runtime_total_ns_median": runtime_total_ns_tail_median,
                "runtime_submit_ns_median": runtime_submit_ns_tail_median,
                "runtime_hw_ns_median": runtime_hw_ns_tail_median,
            },
            "wall_ns_median": wall_ns_median,
            "wall_us_median": wall_ns_median / 1e3,
            "wall_ms_median": wall_ns_median / 1e6,
            "wall_ns_tail_median": wall_ns_tail_median,
            "wall_us_tail_median": wall_ns_tail_median / 1e3,
            "wall_ms_tail_median": wall_ns_tail_median / 1e6,
            "runtime_total_ns_median": runtime_total_ns_median,
            "runtime_total_us_median": runtime_total_ns_median / 1e3,
            "runtime_total_ms_median": runtime_total_ns_median / 1e6,
            "runtime_total_ns_tail_median": runtime_total_ns_tail_median,
            "runtime_total_us_tail_median": runtime_total_ns_tail_median / 1e3,
            "runtime_total_ms_tail_median": runtime_total_ns_tail_median / 1e6,
            "runtime_submit_ns_median": runtime_submit_ns_median,
            "runtime_submit_ns_tail_median": runtime_submit_ns_tail_median,
            "runtime_hw_ns_median": runtime_hw_ns_median,
            "runtime_hw_ns_tail_median": runtime_hw_ns_tail_median,
            "timed_samples": [
                {
                    "wall_ms": wall / 1e6,
                    "runtime_total_ms": runtime_total / 1e6,
                    "runtime_submit_ms": runtime_submit / 1e6,
                    "runtime_hw_ms": runtime_hw / 1e6,
                }
                for wall, runtime_total, runtime_submit, runtime_hw in zip(
                    wall_samples_ns,
                    runtime_total_samples_ns,
                    runtime_submit_samples_ns,
                    runtime_hw_samples_ns,
                )
            ],
            "max_err": max_err,
            "metrics": metrics,
            "schedule": schedule,
            "runtime_bridge_stats": stats,
            "warmup": {
                "mode": warmup_mode,
                "minimum_iterations": int(max(warmup, 0)),
                "iterations_run": int(warmup_iters),
                "steady_state_window": int(steady_state_window),
                "steady_state_rtol": float(steady_state_rtol),
                "steady_state_max_extra_warmup": int(max(steady_state_max_extra_warmup, 0)),
                "steady_state_reached": bool(steady_state_reached),
                "wall_ms_samples": [x / 1e6 for x in warmup_wall_ns],
                "hw_ms_samples": [x / 1e6 for x in warmup_hw_ns],
            },
        }
        if timed_runtime_stats and isinstance(timed_runtime_stats[-1].get("host_dma_debug"), list):
            record["runtime_bridge_debug_last"] = {
                "host_dma_debug": timed_runtime_stats[-1]["host_dma_debug"]
            }
        if devfreq_dir is not None:
            record["devfreq"] = {
                "before_samples": devfreq_before_samples,
                "after_samples": devfreq_after_samples,
                "unique_cur_freq_before": sorted(
                    {str(x.get("cur_freq")) for x in devfreq_before_samples}
                ),
                "unique_cur_freq_after": sorted(
                    {str(x.get("cur_freq")) for x in devfreq_after_samples}
                ),
                "unique_target_freq_before": sorted(
                    {str(x.get("target_freq")) for x in devfreq_before_samples}
                ),
                "unique_load_before": sorted(
                    {str(x.get("load")) for x in devfreq_before_samples}
                ),
                "unique_governor_before": sorted(
                    {str(x.get("governor")) for x in devfreq_before_samples}
                ),
                "unique_transition_totals_before": sorted(
                    {
                        -1 if x.get("transitions_total") is None else int(x["transitions_total"])
                        for x in devfreq_before_samples
                    }
                ),
                "unique_transition_totals_after": sorted(
                    {
                        -1 if x.get("transitions_total") is None else int(x["transitions_total"])
                        for x in devfreq_after_samples
                    }
                ),
            }
        if capture_driver_probe:
            record["driver_probe"] = {
                "before": driver_probe_before,
                "after": _read_driver_probe_snapshot(
                    enabled=True,
                    device=driver_probe_device,
                    debugfs_root=driver_probe_debugfs_root,
                    procfs_root=driver_probe_procfs_root,
                    devfreq_dir=driver_probe_devfreq_dir,
                    include_unsupported=driver_probe_include_unsupported,
                ),
            }
        return record


def _measure_case_cpp(
    case: Case,
    warmup: int,
    iters: int,
    seed: int,
    devfreq_dir: Path | None = None,
    capture_driver_probe: bool = False,
    driver_probe_device: str = "",
    driver_probe_debugfs_root: str = "",
    driver_probe_procfs_root: str = "",
    driver_probe_devfreq_dir: str = "",
    driver_probe_include_unsupported: bool = False,
    global_env_overrides: Dict[str, str] | None = None,
    warmup_mode: str = "steady",
    steady_state_window: int = 4,
    steady_state_rtol: float = 0.10,
    steady_state_max_extra_warmup: int = 24,
) -> Dict[str, object]:
    if warmup_mode != "fixed":
        raise RuntimeError("cpp host currently supports warmup_mode=fixed only")

    rng = np.random.default_rng(seed)
    inputs = case.make_inputs(rng)
    env = dict(_base_env())
    env.update(case.env_overrides)
    if global_env_overrides:
        env.update(global_env_overrides)

    with _temp_env(env):
        mod = case.build_mod()
        lowered = rknpu.plan_rknpu_tir_memory(rknpu.lower_to_rknpu_tir_with_pc_chain(mod))
        report = rknpu.get_rknpu_schedule_report(lowered)
        executable = rknpu.build_with_runtime_bridge(lowered, target="llvm")
        expected = case.ref(inputs)
        runner = _ensure_cpp_runner_built()
        driver_probe_before = _read_driver_probe_snapshot(
            enabled=capture_driver_probe,
            device=driver_probe_device,
            debugfs_root=driver_probe_debugfs_root,
            procfs_root=driver_probe_procfs_root,
            devfreq_dir=driver_probe_devfreq_dir,
            include_unsupported=driver_probe_include_unsupported,
        )
        with tempfile.TemporaryDirectory(prefix=f"rknpu_cpp_bundle_{case.name}_") as bundle_tmp:
            bundle_dir = Path(bundle_tmp)
            _export_case_cpp_bundle(bundle_dir, case, lowered, executable, inputs, expected)
            runner_json = bundle_dir / "runner.json"
            cmd = [
                str(runner),
                str(bundle_dir),
                "--warmup",
                str(max(warmup, 0)),
                "--iters",
                str(max(iters, 1)),
                "--json-out",
                str(runner_json),
            ]
            runner_env = os.environ.copy()
            ld_paths = [str(REPO_ROOT / "build"), str(REPO_ROOT / "build" / "lib")]
            if runner_env.get("LD_LIBRARY_PATH"):
                ld_paths.append(runner_env["LD_LIBRARY_PATH"])
            runner_env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
            runner_env.setdefault("TVM_LIBRARY_PATH", str(REPO_ROOT / "build"))
            devfreq_before_samples: List[Dict[str, object]] = []
            devfreq_after_samples: List[Dict[str, object]] = []
            if devfreq_dir is not None:
                devfreq_before_samples.append(_read_devfreq_snapshot(devfreq_dir))
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=runner_env,
                text=True,
                capture_output=True,
                check=False,
            )
            if devfreq_dir is not None:
                devfreq_after_samples.append(_read_devfreq_snapshot(devfreq_dir))
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.returncode != 0:
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr, end="")
                raise RuntimeError(f"cpp runner failed: {case.name} rc={proc.returncode}")
            payload = json.loads(runner_json.read_text(encoding="utf-8"))

        warmup_payload = payload.get("warmup", {})
        samples = payload.get("timed_samples", [])
        wall_samples_ns = [int(sample["wall_ns"]) for sample in samples]
        runtime_total_samples_ns = [int(sample["runtime_total_ns"]) for sample in samples]
        runtime_submit_samples_ns = [int(sample["runtime_submit_ns"]) for sample in samples]
        runtime_hw_samples_ns = [int(sample["runtime_hw_ns"]) for sample in samples]
        timed_runtime_stats = [sample["runtime_bridge_stats"] for sample in samples]
        max_err = float(payload.get("max_err", float("inf")))
        stats = _aggregate_runtime_stats(timed_runtime_stats)
        runtime_errors = _runtime_sanity(report, stats, max(iters, 1))
        if max_err > case.max_err:
            runtime_errors.append(f"max_err={max_err:.6g} limit={case.max_err:.6g}")
        if runtime_errors:
            raise RuntimeError(f"{case.name}: " + "; ".join(runtime_errors))

        wall_sample_array = np.array(wall_samples_ns, dtype=np.int64)
        runtime_total_array = np.array(runtime_total_samples_ns, dtype=np.int64)
        runtime_submit_array = np.array(runtime_submit_samples_ns, dtype=np.int64)
        runtime_hw_array = np.array(runtime_hw_samples_ns, dtype=np.int64)
        tail_count = max(1, len(wall_samples_ns) // 2)
        wall_ns_median = int(np.median(wall_sample_array))
        wall_ns_tail_median = int(np.median(wall_sample_array[-tail_count:]))
        runtime_total_ns_median = int(np.median(runtime_total_array))
        runtime_total_ns_tail_median = int(np.median(runtime_total_array[-tail_count:]))
        runtime_submit_ns_median = int(np.median(runtime_submit_array))
        runtime_submit_ns_tail_median = int(np.median(runtime_submit_array[-tail_count:]))
        runtime_hw_ns_median = int(np.median(runtime_hw_array))
        runtime_hw_ns_tail_median = int(np.median(runtime_hw_array[-tail_count:]))
        primary_latency_kind = "runtime_total_tail"
        latency_ns = runtime_total_ns_tail_median
        schedule = _schedule_summary(report)
        metrics = dict(case.metrics())
        if "macs" in metrics:
            macs = float(metrics["macs"])
            latency_s = latency_ns / 1e9
            metrics["effective_tmac_s"] = macs / latency_s / 1e12
            metrics["effective_tflop_s"] = (2.0 * macs) / latency_s / 1e12
        if "elements" in metrics:
            elems = float(metrics["elements"])
            latency_s = latency_ns / 1e9
            metrics["effective_gelem_s"] = elems / latency_s / 1e9

        record = {
            "host": "cpp",
            "name": case.name,
            "family": case.family,
            "size_tag": case.size_tag,
            "comparison_group": case.comparison_group,
            "comparison_mode": case.comparison_mode,
            "latency_kind": primary_latency_kind,
            "latency_ns_median": latency_ns,
            "latency_us_median": latency_ns / 1e3,
            "latency_ms_median": latency_ns / 1e6,
            "cold_start": {
                "wall_ns": int(wall_samples_ns[0]),
                "runtime_total_ns": int(runtime_total_samples_ns[0]),
                "runtime_submit_ns": int(runtime_submit_samples_ns[0]),
                "runtime_hw_ns": int(runtime_hw_samples_ns[0]),
            },
            "steady_tail": {
                "tail_count": int(tail_count),
                "wall_ns_median": wall_ns_tail_median,
                "runtime_total_ns_median": runtime_total_ns_tail_median,
                "runtime_submit_ns_median": runtime_submit_ns_tail_median,
                "runtime_hw_ns_median": runtime_hw_ns_tail_median,
            },
            "wall_ns_median": wall_ns_median,
            "wall_us_median": wall_ns_median / 1e3,
            "wall_ms_median": wall_ns_median / 1e6,
            "wall_ns_tail_median": wall_ns_tail_median,
            "wall_us_tail_median": wall_ns_tail_median / 1e3,
            "wall_ms_tail_median": wall_ns_tail_median / 1e6,
            "runtime_total_ns_median": runtime_total_ns_median,
            "runtime_total_us_median": runtime_total_ns_median / 1e3,
            "runtime_total_ms_median": runtime_total_ns_median / 1e6,
            "runtime_total_ns_tail_median": runtime_total_ns_tail_median,
            "runtime_total_us_tail_median": runtime_total_ns_tail_median / 1e3,
            "runtime_total_ms_tail_median": runtime_total_ns_tail_median / 1e6,
            "runtime_submit_ns_median": runtime_submit_ns_median,
            "runtime_submit_ns_tail_median": runtime_submit_ns_tail_median,
            "runtime_hw_ns_median": runtime_hw_ns_median,
            "runtime_hw_ns_tail_median": runtime_hw_ns_tail_median,
            "timed_samples": [
                {
                    "wall_ms": wall / 1e6,
                    "runtime_total_ms": runtime_total / 1e6,
                    "runtime_submit_ms": runtime_submit / 1e6,
                    "runtime_hw_ms": runtime_hw / 1e6,
                }
                for wall, runtime_total, runtime_submit, runtime_hw in zip(
                    wall_samples_ns,
                    runtime_total_samples_ns,
                    runtime_submit_samples_ns,
                    runtime_hw_samples_ns,
                )
            ],
            "max_err": max_err,
            "metrics": metrics,
            "schedule": schedule,
            "runtime_bridge_stats": stats,
            "warmup": {
                "mode": warmup_mode,
                "minimum_iterations": int(max(warmup, 0)),
                "iterations_run": int(warmup_payload.get("iterations_run", max(warmup, 0))),
                "steady_state_window": int(steady_state_window),
                "steady_state_rtol": float(steady_state_rtol),
                "steady_state_max_extra_warmup": int(max(steady_state_max_extra_warmup, 0)),
                "steady_state_reached": True,
                "wall_ms_samples": [
                    int(x) / 1e6 for x in warmup_payload.get("wall_ns_samples", [])
                ],
                "hw_ms_samples": [
                    int(x) / 1e6 for x in warmup_payload.get("runtime_hw_ns_samples", [])
                ],
            },
            "cpp_runner": {
                "chain_blob_source": str(payload.get("chain_blob_source", "")),
            },
        }
        if timed_runtime_stats and isinstance(timed_runtime_stats[-1].get("host_dma_debug"), list):
            record["runtime_bridge_debug_last"] = {
                "host_dma_debug": timed_runtime_stats[-1]["host_dma_debug"]
            }
        if devfreq_dir is not None:
            record["devfreq"] = {
                "before_samples": devfreq_before_samples,
                "after_samples": devfreq_after_samples,
                "unique_cur_freq_before": sorted(
                    {str(x.get("cur_freq")) for x in devfreq_before_samples}
                ),
                "unique_cur_freq_after": sorted(
                    {str(x.get("cur_freq")) for x in devfreq_after_samples}
                ),
                "unique_target_freq_before": sorted(
                    {str(x.get("target_freq")) for x in devfreq_before_samples}
                ),
                "unique_load_before": sorted(
                    {str(x.get("load")) for x in devfreq_before_samples}
                ),
                "unique_governor_before": sorted(
                    {str(x.get("governor")) for x in devfreq_before_samples}
                ),
                "unique_transition_totals_before": sorted(
                    {
                        -1 if x.get("transitions_total") is None else int(x["transitions_total"])
                        for x in devfreq_before_samples
                    }
                ),
                "unique_transition_totals_after": sorted(
                    {
                        -1 if x.get("transitions_total") is None else int(x["transitions_total"])
                        for x in devfreq_after_samples
                    }
                ),
            }
        if capture_driver_probe:
            record["driver_probe"] = {
                "before": driver_probe_before,
                "after": _read_driver_probe_snapshot(
                    enabled=True,
                    device=driver_probe_device,
                    debugfs_root=driver_probe_debugfs_root,
                    procfs_root=driver_probe_procfs_root,
                    devfreq_dir=driver_probe_devfreq_dir,
                    include_unsupported=driver_probe_include_unsupported,
                ),
            }
        return record


def _build_matmul_mod(m: int, k: int, n: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, k), DTYPE))
    w = relax.Var("w", relax.TensorStructInfo((k, n), DTYPE))
    with bb.function("main", [x, w]):
        with bb.dataflow():
            out = bb.emit(relax.op.matmul(x, w))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_matmul_bias_mod(m: int, k: int, n: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, k), DTYPE))
    w = relax.Var("w", relax.TensorStructInfo((k, n), DTYPE))
    b = relax.Var("b", relax.TensorStructInfo((n,), DTYPE))
    with bb.function("main", [x, w, b]):
        with bb.dataflow():
            mm = bb.emit(relax.op.matmul(x, w))
            out = bb.emit(relax.op.add(mm, b))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_matmul_bias_relu_mod(m: int, k: int, n: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, k), DTYPE))
    w = relax.Var("w", relax.TensorStructInfo((k, n), DTYPE))
    b = relax.Var("b", relax.TensorStructInfo((n,), DTYPE))
    with bb.function("main", [x, w, b]):
        with bb.dataflow():
            mm = bb.emit(relax.op.matmul(x, w))
            add = bb.emit(relax.op.add(mm, b))
            out = bb.emit(relax.op.nn.relu(add))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_add_mod(m: int, n: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorStructInfo((m, n), DTYPE))
    b = relax.Var("b", relax.TensorStructInfo((m, n), DTYPE))
    with bb.function("main", [a, b]):
        with bb.dataflow():
            out = bb.emit(relax.op.add(a, b))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_add_chain_identity_mod(m: int, n: int, depth: int) -> tvm.IRModule:
    if depth < 1:
        raise ValueError("depth must be >= 1")
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, n), DTYPE))
    zeros = [
        relax.Var(f"z{i + 1}", relax.TensorStructInfo((m, n), DTYPE)) for i in range(depth)
    ]
    with bb.function("main", [x, *zeros]):
        with bb.dataflow():
            cur = x
            for z in zeros:
                cur = bb.emit(relax.op.add(cur, z))
            bb.emit_output(cur)
        bb.emit_func_output(cur)
    return bb.finalize()


def _build_mul_mod(m: int, n: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorStructInfo((m, n), DTYPE))
    b = relax.Var("b", relax.TensorStructInfo((m, n), DTYPE))
    with bb.function("main", [a, b]):
        with bb.dataflow():
            out = bb.emit(relax.op.multiply(a, b))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _build_residual_mlp_mod(m: int, d_model: int, hidden: int) -> tvm.IRModule:
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((m, d_model), DTYPE))
    w1 = relax.Var("w1", relax.TensorStructInfo((d_model, hidden), DTYPE))
    b1 = relax.Var("b1", relax.TensorStructInfo((hidden,), DTYPE))
    w2 = relax.Var("w2", relax.TensorStructInfo((hidden, d_model), DTYPE))
    b2 = relax.Var("b2", relax.TensorStructInfo((d_model,), DTYPE))
    with bb.function("main", [x, w1, b1, w2, b2]):
        with bb.dataflow():
            ff1 = bb.emit(relax.op.matmul(x, w1))
            ff1b = bb.emit(relax.op.add(ff1, b1))
            act = bb.emit(relax.op.nn.relu(ff1b))
            ff2 = bb.emit(relax.op.matmul(act, w2))
            ff2b = bb.emit(relax.op.add(ff2, b2))
            out = bb.emit(relax.op.add(ff2b, x))
            bb.emit_output(out)
        bb.emit_func_output(out)
    return bb.finalize()


def _normal(rng: np.random.Generator, shape: Sequence[int], scale: float = 0.1) -> np.ndarray:
    return (scale * rng.standard_normal(shape)).astype(DTYPE)


def _ref_matmul(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    return (inputs["x"].astype(np.float32) @ inputs["w"].astype(np.float32)).astype(DTYPE)


def _ref_matmul_bias(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    y = inputs["x"].astype(np.float32) @ inputs["w"].astype(np.float32)
    y = y + inputs["b"].astype(np.float32)[None, :]
    return y.astype(DTYPE)


def _ref_matmul_bias_relu(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    y = inputs["x"].astype(np.float32) @ inputs["w"].astype(np.float32)
    y = y + inputs["b"].astype(np.float32)[None, :]
    y = np.maximum(y, 0.0)
    return y.astype(DTYPE)


def _ref_add(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    return (inputs["a"].astype(np.float32) + inputs["b"].astype(np.float32)).astype(DTYPE)


def _ref_add_chain_identity(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    y = inputs["x"].astype(np.float32)
    for name, value in inputs.items():
        if name == "x":
            continue
        y = y + value.astype(np.float32)
    return y.astype(DTYPE)


def _ref_mul(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    return (inputs["a"].astype(np.float32) * inputs["b"].astype(np.float32)).astype(DTYPE)


def _ref_residual_mlp(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["x"].astype(np.float32)
    ff1 = x @ inputs["w1"].astype(np.float32) + inputs["b1"].astype(np.float32)[None, :]
    act = np.maximum(ff1, 0.0)
    ff2 = act @ inputs["w2"].astype(np.float32) + inputs["b2"].astype(np.float32)[None, :]
    return (ff2 + x).astype(DTYPE)


def _task_template_catalog() -> List[Dict[str, object]]:
    return [
        {
            "name": "cna_dpu",
            "role": "matmul / conv without extra operand DMA",
            "regcmd_count": int(hardware.REGCMD_COUNT),
            "regcfg_amount": int(hardware.REGCFG_AMOUNT),
            "enable_mask_hex": f"0x{int(hardware.TASK_ENABLE_MASK):04x}",
        },
        {
            "name": "cna_dpu_dpu_rdma",
            "role": "matmul / conv with bias or other extra operand stream",
            "regcmd_count": int(hardware.REGCMD_COUNT_MODE6),
            "regcfg_amount": int(hardware.REGCFG_AMOUNT_MODE6),
            "enable_mask_hex": f"0x{int(hardware.TASK_ENABLE_MASK_MODE6):04x}",
        },
        {
            "name": "ew",
            "role": "elementwise add / mul / relu style task",
            "regcmd_count": int(hardware.REGCMD_COUNT_EW),
            "regcfg_amount": int(hardware.REGCMD_COUNT_EW - 4),
            "enable_mask_hex": f"0x{int(hardware.TASK_EW_ENABLE_MASK):04x}",
        },
        {
            "name": "ppu",
            "role": "pooling / planar post-processing task",
            "regcmd_count": int(hardware.REGCMD_COUNT_PPU),
            "regcfg_amount": int(hardware.REGCMD_COUNT_PPU - 4),
            "enable_mask_hex": f"0x{int(hardware.TASK_PPU_ENABLE_MASK):04x}",
        },
        {
            "name": "lut_combined",
            "role": "combined LUT upload + eval task",
            "regcmd_count": int(hardware.REGCMD_COUNT_LUT_COMBINED),
            "regcfg_amount": int(hardware.REGCFG_AMOUNT_LUT_COMBINED),
            "enable_mask_hex": f"0x{int(hardware.TASK_EW_ENABLE_MASK):04x}",
        },
    ]


def _build_cases(m_small: int, m_large: int, d_model: int, hidden: int) -> List[Case]:
    small = {
        "m": m_small,
        "k": d_model,
        "n": hidden,
        "d_model": d_model,
        "hidden": hidden,
    }
    large = {
        "m": m_large,
        "k": d_model,
        "n": hidden,
        "d_model": d_model,
        "hidden": hidden,
    }

    def matmul_case(size_tag: str, cfg: Dict[str, int]) -> Case:
        m, k, n = cfg["m"], cfg["k"], cfg["n"]
        return Case(
            name=f"matmul_{size_tag}",
            family="matmul",
            size_tag=size_tag,
            ordered_input_names=("x", "w"),
            build_mod=lambda m=m, k=k, n=n: _build_matmul_mod(m, k, n),
            make_inputs=lambda rng, m=m, k=k, n=n: {
                "x": _normal(rng, (m, k)),
                "w": _normal(rng, (k, n)),
            },
            ref=_ref_matmul,
            metrics=lambda m=m, k=k, n=n: {
                "shape": f"[{m}x{k}] x [{k}x{n}] -> [{m}x{n}]",
                "macs": float(m * k * n),
            },
            env_overrides={},
        )

    def matmul_bias_case(size_tag: str, cfg: Dict[str, int]) -> Case:
        m, k, n = cfg["m"], cfg["k"], cfg["n"]
        return Case(
            name=f"matmul_bias_{size_tag}",
            family="matmul_bias",
            size_tag=size_tag,
            ordered_input_names=("x", "w", "b"),
            build_mod=lambda m=m, k=k, n=n: _build_matmul_bias_mod(m, k, n),
            make_inputs=lambda rng, m=m, k=k, n=n: {
                "x": _normal(rng, (m, k)),
                "w": _normal(rng, (k, n)),
                "b": _normal(rng, (n,)),
            },
            ref=_ref_matmul_bias,
            metrics=lambda m=m, k=k, n=n: {
                "shape": f"[{m}x{k}] x [{k}x{n}] + [{n}] -> [{m}x{n}]",
                "macs": float(m * k * n),
            },
            env_overrides={},
        )

    def matmul_bias_relu_case(
        size_tag: str,
        cfg: Dict[str, int],
        overrides: Dict[str, str],
        name_suffix: str = "",
        comparison_mode: str = "base",
    ) -> Case:
        m, k, n = cfg["m"], cfg["k"], cfg["n"]
        label = f"matmul_bias_relu_{size_tag}{name_suffix}"
        return Case(
            name=label,
            family="matmul_bias_relu" if not name_suffix else f"matmul_bias_relu{name_suffix}",
            size_tag=size_tag,
            ordered_input_names=("x", "w", "b"),
            build_mod=lambda m=m, k=k, n=n: _build_matmul_bias_relu_mod(m, k, n),
            make_inputs=lambda rng, m=m, k=k, n=n: {
                "x": _normal(rng, (m, k)),
                "w": _normal(rng, (k, n)),
                "b": _normal(rng, (n,)),
            },
            ref=_ref_matmul_bias_relu,
            metrics=lambda m=m, k=k, n=n: {
                "shape": f"relu([{m}x{k}] x [{k}x{n}] + [{n}]) -> [{m}x{n}]",
                "macs": float(m * k * n),
            },
            env_overrides=overrides,
            comparison_group="matmul_bias_relu",
            comparison_mode=comparison_mode,
        )

    def add_case(size_tag: str, cfg: Dict[str, int]) -> Case:
        m, n = cfg["m"], cfg["k"]
        return Case(
            name=f"add_{size_tag}",
            family="add",
            size_tag=size_tag,
            ordered_input_names=("a", "b"),
            build_mod=lambda m=m, n=n: _build_add_mod(m, n),
            make_inputs=lambda rng, m=m, n=n: {
                "a": _normal(rng, (m, n)),
                "b": _normal(rng, (m, n)),
            },
            ref=_ref_add,
            metrics=lambda m=m, n=n: {
                "shape": f"[{m}x{n}] + [{m}x{n}] -> [{m}x{n}]",
                "elements": float(m * n),
            },
            env_overrides={},
        )

    def add_chain_identity_case(
        size_tag: str,
        cfg: Dict[str, int],
        depth: int,
        overrides: Dict[str, str],
        name_suffix: str,
        comparison_mode: str,
    ) -> Case:
        m, n = cfg["m"], cfg["k"]
        ordered_input_names = ("x",) + tuple(f"z{i + 1}" for i in range(depth))
        return Case(
            name=f"add_identity_chain_{size_tag}{name_suffix}",
            family="add_identity_chain" if not name_suffix else f"add_identity_chain{name_suffix}",
            size_tag=size_tag,
            ordered_input_names=ordered_input_names,
            build_mod=lambda m=m, n=n, depth=depth: _build_add_chain_identity_mod(m, n, depth),
            make_inputs=lambda rng, m=m, n=n, depth=depth: {
                "x": _normal(rng, (m, n)),
                **{f"z{i + 1}": np.zeros((m, n), dtype=DTYPE) for i in range(depth)},
            },
            ref=_ref_add_chain_identity,
            metrics=lambda m=m, n=n, depth=depth: {
                "shape": f"{depth}x add-zero chain over [{m}x{n}]",
                "elements": float(m * n),
                "chain_depth": int(depth),
            },
            env_overrides=overrides,
            comparison_group="ew_add_identity_chain",
            comparison_mode=comparison_mode,
        )

    def mul_case(size_tag: str, cfg: Dict[str, int]) -> Case:
        m, n = cfg["m"], cfg["k"]
        return Case(
            name=f"mul_{size_tag}",
            family="mul",
            size_tag=size_tag,
            ordered_input_names=("a", "b"),
            build_mod=lambda m=m, n=n: _build_mul_mod(m, n),
            make_inputs=lambda rng, m=m, n=n: {
                "a": _normal(rng, (m, n)),
                "b": _normal(rng, (m, n)),
            },
            ref=_ref_mul,
            metrics=lambda m=m, n=n: {
                "shape": f"[{m}x{n}] * [{m}x{n}] -> [{m}x{n}]",
                "elements": float(m * n),
            },
            env_overrides={},
        )

    def residual_mlp_case(
        size_tag: str,
        cfg: Dict[str, int],
        overrides: Dict[str, str],
        name_suffix: str = "",
        comparison_mode: str = "base",
    ) -> Case:
        m, d_model, hidden = cfg["m"], cfg["d_model"], cfg["hidden"]
        return Case(
            name=f"residual_mlp_relu_{size_tag}{name_suffix}",
            family="residual_mlp_relu" if not name_suffix else f"residual_mlp_relu{name_suffix}",
            size_tag=size_tag,
            ordered_input_names=("x", "w1", "b1", "w2", "b2"),
            build_mod=lambda m=m, d_model=d_model, hidden=hidden: _build_residual_mlp_mod(m, d_model, hidden),
            make_inputs=lambda rng, m=m, d_model=d_model, hidden=hidden: {
                "x": _normal(rng, (m, d_model)),
                "w1": _normal(rng, (d_model, hidden)),
                "b1": _normal(rng, (hidden,)),
                "w2": _normal(rng, (hidden, d_model)),
                "b2": _normal(rng, (d_model,)),
            },
            ref=_ref_residual_mlp,
            metrics=lambda m=m, d_model=d_model, hidden=hidden: {
                "shape": (
                    f"relu([{m}x{d_model}] x [{d_model}x{hidden}] + [{hidden}]); "
                    f"[{m}x{hidden}] x [{hidden}x{d_model}] + [{d_model}] + residual"
                ),
                "macs": float(2 * m * d_model * hidden),
            },
            env_overrides=overrides,
            comparison_group="residual_mlp_relu",
            comparison_mode=comparison_mode,
        )

    return [
        matmul_case("small", small),
        matmul_case("large", large),
        matmul_bias_case("small", small),
        matmul_bias_case("large", large),
        matmul_bias_relu_case("small", small, {}, comparison_mode="fused"),
        matmul_bias_relu_case("large", large, {}, comparison_mode="fused"),
        add_case("small", small),
        add_case("large", large),
        add_chain_identity_case(
            "small",
            small,
            1,
            {},
            name_suffix="",
            comparison_mode="baseline",
        ),
        add_chain_identity_case(
            "large",
            large,
            1,
            {},
            name_suffix="",
            comparison_mode="baseline",
        ),
        add_chain_identity_case(
            "small",
            small,
            3,
            {},
            name_suffix="_same_submit_tasks3",
            comparison_mode="same_submit_extra_tasks",
        ),
        add_chain_identity_case(
            "large",
            large,
            3,
            {},
            name_suffix="_same_submit_tasks3",
            comparison_mode="same_submit_extra_tasks",
        ),
        add_chain_identity_case(
            "small",
            small,
            3,
            {"TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1"},
            name_suffix="_multi_submit_tasks3",
            comparison_mode="multi_submit_extra_tasks",
        ),
        add_chain_identity_case(
            "large",
            large,
            3,
            {"TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1"},
            name_suffix="_multi_submit_tasks3",
            comparison_mode="multi_submit_extra_tasks",
        ),
        mul_case("small", small),
        mul_case("large", large),
        residual_mlp_case("small", small, {}, comparison_mode="fused"),
        residual_mlp_case("large", large, {}, comparison_mode="fused"),
        matmul_bias_relu_case(
            "small",
            small,
            {"TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1"},
            name_suffix="_split_tasks",
            comparison_mode="split_tasks",
        ),
        matmul_bias_relu_case(
            "large",
            large,
            {"TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1"},
            name_suffix="_split_tasks",
            comparison_mode="split_tasks",
        ),
        matmul_bias_relu_case(
            "small",
            small,
            {
                "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1",
                "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1",
            },
            name_suffix="_split_submits",
            comparison_mode="split_submits",
        ),
        matmul_bias_relu_case(
            "large",
            large,
            {
                "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1",
                "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1",
            },
            name_suffix="_split_submits",
            comparison_mode="split_submits",
        ),
        residual_mlp_case(
            "small",
            small,
            {"TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1"},
            name_suffix="_split_tasks",
            comparison_mode="split_tasks",
        ),
        residual_mlp_case(
            "large",
            large,
            {"TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1"},
            name_suffix="_split_tasks",
            comparison_mode="split_tasks",
        ),
        residual_mlp_case(
            "small",
            small,
            {
                "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1",
                "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1",
            },
            name_suffix="_split_submits",
            comparison_mode="split_submits",
        ),
        residual_mlp_case(
            "large",
            large,
            {
                "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1",
                "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1",
            },
            name_suffix="_split_submits",
            comparison_mode="split_submits",
        ),
    ]


def _group_by_family(records: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        key = f"{record.get('host', 'unknown')}::{record['family']}"
        out.setdefault(key, []).append(record)
    return out


def _fusion_penalty_section(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    indexed = {
        (
            str(record.get("host", "unknown")),
            str(record.get("comparison_group", "")),
            str(record.get("comparison_mode", "")),
            str(record["size_tag"]),
        ): record
        for record in records
        if str(record.get("comparison_group", ""))
    }
    out: List[Dict[str, object]] = []
    hosts = sorted({str(record.get("host", "unknown")) for record in records})
    comparison_groups = sorted(
        {
            str(record.get("comparison_group", ""))
            for record in records
            if str(record.get("comparison_group", ""))
        }
    )
    for host in hosts:
        for comparison_group in comparison_groups:
            for size_tag in ("small", "large"):
                required = ["fused", "split_tasks", "split_submits"]
                if any((host, comparison_group, mode, size_tag) not in indexed for mode in required):
                    continue
                fused = indexed[(host, comparison_group, "fused", size_tag)]
                split_tasks = indexed[(host, comparison_group, "split_tasks", size_tag)]
                split_submits = indexed[(host, comparison_group, "split_submits", size_tag)]
                fused_ns = float(fused["latency_ns_median"])
                split_tasks_ns = float(split_tasks["latency_ns_median"])
                split_submits_ns = float(split_submits["latency_ns_median"])
                out.append(
                    {
                        "host": host,
                        "comparison_group": comparison_group,
                        "size_tag": size_tag,
                        "fused": fused,
                        "split_tasks": split_tasks,
                        "split_submits": split_submits,
                        "ratios_vs_fused": {
                            "split_tasks": split_tasks_ns / fused_ns if fused_ns > 0 else math.inf,
                            "split_submits": split_submits_ns / fused_ns if fused_ns > 0 else math.inf,
                        },
                        "delta_ns_vs_fused": {
                            "split_tasks": int(split_tasks_ns - fused_ns),
                            "split_submits": int(split_submits_ns - fused_ns),
                        },
                    }
                )
    return out


def _overhead_penalty_section(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    indexed = {
        (
            str(record.get("host", "unknown")),
            str(record.get("comparison_group", "")),
            str(record.get("comparison_mode", "")),
            str(record["size_tag"]),
        ): record
        for record in records
        if str(record.get("comparison_group", "")) == "ew_add_identity_chain"
    }
    out: List[Dict[str, object]] = []
    for host in sorted({key[0] for key in indexed.keys()}):
        for size_tag in ("small", "large"):
            required = ["baseline", "same_submit_extra_tasks", "multi_submit_extra_tasks"]
            if any((host, "ew_add_identity_chain", mode, size_tag) not in indexed for mode in required):
                continue
            baseline = indexed[(host, "ew_add_identity_chain", "baseline", size_tag)]
            same_submit = indexed[(host, "ew_add_identity_chain", "same_submit_extra_tasks", size_tag)]
            multi_submit = indexed[(host, "ew_add_identity_chain", "multi_submit_extra_tasks", size_tag)]
            base_ns = float(baseline["latency_ns_median"])
            same_ns = float(same_submit["latency_ns_median"])
            multi_ns = float(multi_submit["latency_ns_median"])
            # Three-task chains add two extra tasks and, in the multi-submit case, two extra submit boundaries.
            extra_task_ns = (same_ns - base_ns) / 2.0
            extra_submit_ns = (multi_ns - same_ns) / 2.0
            out.append(
                {
                    "host": host,
                    "group": "ew_add_identity_chain",
                    "size_tag": size_tag,
                    "baseline": baseline,
                    "same_submit": same_submit,
                    "multi_submit": multi_submit,
                    "ratios_vs_baseline": {
                        "same_submit": same_ns / base_ns if base_ns > 0 else math.inf,
                        "multi_submit": multi_ns / base_ns if base_ns > 0 else math.inf,
                    },
                    "approx_extra_task_ns": int(extra_task_ns),
                    "approx_extra_submit_ns": int(extra_submit_ns),
                }
            )
    return out


def _practical_peaks(records: List[Dict[str, object]]) -> Dict[str, object]:
    best_tmac = None
    best_tflops = None
    best_gelem = None
    for record in records:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        tmac = metrics.get("effective_tmac_s")
        tflops = metrics.get("effective_tflop_s")
        gelem = metrics.get("effective_gelem_s")
        if isinstance(tmac, (int, float)):
            if best_tmac is None or float(tmac) > float(best_tmac["value"]):
                best_tmac = {
                    "case": record["name"],
                    "host": record.get("host", "unknown"),
                    "value": float(tmac),
                }
        if isinstance(tflops, (int, float)):
            if best_tflops is None or float(tflops) > float(best_tflops["value"]):
                best_tflops = {
                    "case": record["name"],
                    "host": record.get("host", "unknown"),
                    "value": float(tflops),
                }
        if isinstance(gelem, (int, float)):
            if best_gelem is None or float(gelem) > float(best_gelem["value"]):
                best_gelem = {
                    "case": record["name"],
                    "host": record.get("host", "unknown"),
                    "value": float(gelem),
                }
    return {
        "best_effective_tmac_s": best_tmac,
        "best_effective_tflop_s": best_tflops,
        "best_effective_gelem_s": best_gelem,
    }


def _markdown_report(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    config = payload["config"]
    lines.append("# RK3588 NPU Performance Reference")
    lines.append("")
    lines.append("This file is generated from `tools/rknpu_performance_reference.py`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"- Measured on the current 2-D clean subset with `M in {{{config['m_small']}, {config['m_large']}}}`, "
        f"`D={config['d_model']}`, `H={config['hidden']}`."
    )
    lines.append(f"- Host harness mode: `{config['host']}`.")
    lines.append(f"- Suite selection: `{config['suite']}` with comparison group `{config['comparison_group']}`.")
    lines.append(
        f"- Primary latency is `{config['primary_latency_kind']}` after warmup mode "
        f"`{config['warmup_mode']}` (minimum warmup `{config['warmup']}` iteration(s))."
    )
    lines.append(f"- Runtime DMA cache mode: `{config['bridge_cache_dma']}`.")
    lines.append(
        "- Persistent DMA cache can materially change split-submit versus same-submit comparisons; "
        "use `--bridge-cache-dma off` for raw schedule studies and compare both modes when in doubt."
    )
    if config["warmup_mode"] == "steady":
        lines.append(
            f"- Steady state is declared when the last `{config['steady_state_window']}` hardware-time "
            f"samples fit within `{config['steady_state_rtol']:.3f}` relative span, with up to "
            f"`{config['steady_state_max_extra_warmup']}` extra warmup iterations."
        )
    lines.append(f"- Driver probe capture: `{config['capture_driver_probe']}`.")
    lines.append("- Theoretical per-block TOPS are intentionally not claimed here unless they come directly from the TRM.")
    lines.append("")
    driver_probe = payload.get("driver_probe_initial", {})
    if config["capture_driver_probe"] and isinstance(driver_probe, dict) and driver_probe.get("ok"):
        action_metrics = driver_probe.get("action_metrics", {})
        devfreq_probe = driver_probe.get("devfreq", {})
        lines.append("## Kernel Snapshot")
        lines.append("")
        lines.append(f"- Device: `{driver_probe.get('device_path', '')}`")
        drm_version = driver_probe.get("drm_version", {})
        if isinstance(drm_version, dict):
            lines.append(
                f"- DRM driver: `{drm_version.get('name', '')}` "
                f"`{drm_version.get('version_major', '')}.{drm_version.get('version_minor', '')}.{drm_version.get('version_patchlevel', '')}` "
                f"(date `{drm_version.get('date', '')}`)"
            )
        if isinstance(action_metrics, dict):
            for label, key in (
                ("freq_hz", "freq_hz"),
                ("volt_uv", "volt_uv"),
                ("iommu_enabled", "iommu_enabled"),
                ("iommu_domain_id", "iommu_domain_id"),
                ("total_sram_bytes", "total_sram_bytes"),
                ("free_sram_bytes", "free_sram_bytes"),
            ):
                probe = action_metrics.get(key, {})
                if isinstance(probe, dict) and probe.get("ok"):
                    lines.append(f"- Driver {label}: `{probe.get('value')}`")
        if isinstance(devfreq_probe, dict) and devfreq_probe.get("present"):
            lines.append(
                f"- Devfreq: governor `{devfreq_probe.get('governor')}`, "
                f"cur `{devfreq_probe.get('cur_freq')}`, target `{devfreq_probe.get('target_freq')}`, "
                f"load `{devfreq_probe.get('load')}`"
            )
        lines.append("")
    lines.append("## Task Templates")
    lines.append("")
    lines.append("| Name | Role | RegCmds | Data RegCfg | Enable Mask |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for item in payload["task_templates"]:
        lines.append(
            f"| `{item['name']}` | {item['role']} | {item['regcmd_count']} | "
            f"{item['regcfg_amount']} | `{item['enable_mask_hex']}` |"
        )
    lines.append("")
    lines.append("## Latency Atlas")
    lines.append("")
    lines.append(
        "| Host | Family | Size | Shape | Cold Runtime (ns) | Tail Runtime (ns) | Tail HW (ns) | Tail Wall (ns) | Submits | Tasks | Stage IDs |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for record in payload["records"]:
        if str(record["family"]).endswith("split_tasks") or str(record["family"]).endswith("split_submits"):
            continue
        metrics = record["metrics"]
        shape = metrics.get("shape", "")
        sched = record["schedule"]
        lines.append(
            f"| `{record.get('host', 'unknown')}` | `{record['family']}` | `{record['size_tag']}` | `{shape}` | "
            f"{record['cold_start']['runtime_total_ns']} | {record['runtime_total_ns_tail_median']} | "
            f"{record['runtime_hw_ns_tail_median']} | {record['wall_ns_tail_median']} | "
            f"{sched['num_submits']} | {sched['total_tasks']} | "
            f"`{sched['submit_stage_ids']}` |"
        )
    lines.append("")
    lines.append("## Fusion Penalty")
    lines.append("")
    lines.append(
        "| Host | Group | Size | Mode | Cold Runtime (ns) | Tail Runtime (ns) | Tail HW (ns) | Tail Wall (ns) | Ratio vs fused | Submits | Tasks | Stage IDs |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in payload["fusion_penalty"]:
        fused = item["fused"]
        split_tasks = item["split_tasks"]
        split_submits = item["split_submits"]
        lines.append(
            f"| `{item['host']}` | `{item['comparison_group']}` | `{item['size_tag']}` | `fused task` | {fused['cold_start']['runtime_total_ns']} | "
            f"{fused['runtime_total_ns_tail_median']} | {fused['runtime_hw_ns_tail_median']} | {fused['wall_ns_tail_median']} | 1.000 | "
            f"{fused['schedule']['num_submits']} | {fused['schedule']['total_tasks']} | "
            f"`{fused['schedule']['submit_stage_ids']}` |"
        )
        lines.append(
            f"| `{item['host']}` | `{item['comparison_group']}` | `{item['size_tag']}` | `one-submit non-fused` | {split_tasks['cold_start']['runtime_total_ns']} | "
            f"{split_tasks['runtime_total_ns_tail_median']} | {split_tasks['runtime_hw_ns_tail_median']} | {split_tasks['wall_ns_tail_median']} | "
            f"{item['ratios_vs_fused']['split_tasks']:.3f} | {split_tasks['schedule']['num_submits']} | "
            f"{split_tasks['schedule']['total_tasks']} | `{split_tasks['schedule']['submit_stage_ids']}` |"
        )
        lines.append(
            f"| `{item['host']}` | `{item['comparison_group']}` | `{item['size_tag']}` | `multi-submit non-fused` | {split_submits['cold_start']['runtime_total_ns']} | "
            f"{split_submits['runtime_total_ns_tail_median']} | {split_submits['runtime_hw_ns_tail_median']} | {split_submits['wall_ns_tail_median']} | "
            f"{item['ratios_vs_fused']['split_submits']:.3f} | {split_submits['schedule']['num_submits']} | "
            f"{split_submits['schedule']['total_tasks']} | `{split_submits['schedule']['submit_stage_ids']}` |"
        )
    if payload["overhead_penalty"]:
        lines.append("")
        lines.append("## Submit/Task Overhead")
        lines.append("")
        lines.append(
            "| Host | Size | Baseline Runtime (ns) | Same-submit 3-task Runtime (ns) | Multi-submit 3-task Runtime (ns) | same/base | multi/base | Approx extra task (ns) | Approx extra submit (ns) |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for item in payload["overhead_penalty"]:
            baseline = item["baseline"]
            same_submit = item["same_submit"]
            multi_submit = item["multi_submit"]
            lines.append(
                f"| `{item['host']}` | `{item['size_tag']}` | "
                f"{baseline['runtime_total_ns_tail_median']} | "
                f"{same_submit['runtime_total_ns_tail_median']} | "
                f"{multi_submit['runtime_total_ns_tail_median']} | "
                f"{item['ratios_vs_baseline']['same_submit']:.3f} | "
                f"{item['ratios_vs_baseline']['multi_submit']:.3f} | "
                f"{item['approx_extra_task_ns']} | "
                f"{item['approx_extra_submit_ns']} |"
            )
    lines.append("")
    lines.append("## Effective Throughput")
    lines.append("")
    peaks = payload["practical_peaks"]
    if peaks["best_effective_tflop_s"] is not None:
        lines.append(
            f"- Best measured effective TFLOP/s: `{peaks['best_effective_tflop_s']['value']:.6f}` "
            f"from `{peaks['best_effective_tflop_s']['host']}::{peaks['best_effective_tflop_s']['case']}`."
        )
    if peaks["best_effective_tmac_s"] is not None:
        lines.append(
            f"- Best measured effective TMAC/s: `{peaks['best_effective_tmac_s']['value']:.6f}` "
            f"from `{peaks['best_effective_tmac_s']['host']}::{peaks['best_effective_tmac_s']['case']}`."
        )
    if peaks["best_effective_gelem_s"] is not None:
        lines.append(
            f"- Best measured effective Gelem/s: `{peaks['best_effective_gelem_s']['value']:.6f}` "
            f"from `{peaks['best_effective_gelem_s']['host']}::{peaks['best_effective_gelem_s']['case']}`."
        )
    lines.append("")
    lines.append("## Useful Additional Fields To Catalog")
    lines.append("")
    lines.append("- actual NPU core clock during the run")
    lines.append("- 1-core vs 3-core scaling for the same task template")
    lines.append("- LUT and PPU throughput once those paths are fully signed off")
    lines.append("- materialized-temp bytes and DRAM traffic proxies")
    lines.append("- thermal state and clock governor settings")
    lines.append("- shape cliffs where tiling or layout changes cause step-function latency jumps")
    lines.append("")
    lines.append("## Unknowns")
    lines.append("")
    lines.append("- The public TRM does not give precise per-block TOPS numbers for `DPU.BS`, `DPU.BN`, `DPU.EW`, `DPU.LUT`, or `PPU`.")
    lines.append("- `MIPS` is not a useful metric for this accelerator; task latency and effective throughput are more actionable.")
    return "\n".join(lines) + "\n"


def _build_payload(
    args: argparse.Namespace,
    devfreq_dir: Path | None,
    devfreq_initial: Dict[str, object],
    devfreq_final: Dict[str, object],
    driver_probe_initial: Dict[str, object],
    driver_probe_final: Dict[str, object],
    records: List[Dict[str, object]],
    isolate_cases: bool,
) -> Dict[str, object]:
    return {
        "config": {
            "host": args.host,
            "suite": args.suite,
            "comparison_group": args.comparison_group,
            "m_small": args.m_small,
            "m_large": args.m_large,
            "d_model": args.d_model,
            "hidden": args.hidden,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "primary_latency_kind": "runtime_total_tail",
            "bridge_cache_dma": args.bridge_cache_dma,
            "warmup_mode": args.warmup_mode,
            "steady_state_window": args.steady_state_window,
            "steady_state_rtol": args.steady_state_rtol,
            "steady_state_max_extra_warmup": args.steady_state_max_extra_warmup,
            "capture_devfreq": bool(args.capture_devfreq),
            "capture_driver_probe": bool(args.capture_driver_probe),
            "devfreq_dir": str(devfreq_dir) if devfreq_dir is not None else "",
            "driver_probe_device": str(args.driver_probe_device),
            "driver_probe_debugfs_root": str(args.driver_probe_debugfs_root),
            "driver_probe_procfs_root": str(args.driver_probe_procfs_root),
            "driver_probe_devfreq_dir": str(args.driver_probe_devfreq_dir),
            "driver_probe_include_unsupported": bool(args.driver_probe_include_unsupported),
            "isolate_cases": bool(isolate_cases),
        },
        "devfreq_initial": devfreq_initial,
        "devfreq_final": devfreq_final,
        "driver_probe_initial": driver_probe_initial,
        "driver_probe_final": driver_probe_final,
        "task_templates": _task_template_catalog(),
        "records": records,
        "records_by_family": _group_by_family(records),
        "fusion_penalty": _fusion_penalty_section(records),
        "overhead_penalty": _overhead_penalty_section(records),
        "practical_peaks": _practical_peaks(records),
    }


def _measure_case_in_subprocess(
    args: argparse.Namespace, case_name: str, host: str
) -> Dict[str, object]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--host",
        host,
        "--suite",
        str(args.suite),
        "--comparison-group",
        str(args.comparison_group),
        "--case",
        case_name,
        "--m-small",
        str(args.m_small),
        "--m-large",
        str(args.m_large),
        "--d-model",
        str(args.d_model),
        "--hidden",
        str(args.hidden),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
        "--bridge-cache-dma",
        str(args.bridge_cache_dma),
        "--warmup-mode",
        str(args.warmup_mode),
        "--steady-state-window",
        str(args.steady_state_window),
        "--steady-state-rtol",
        str(args.steady_state_rtol),
        "--steady-state-max-extra-warmup",
        str(args.steady_state_max_extra_warmup),
        "--no-isolate-cases",
    ]
    if args.capture_devfreq:
        cmd.extend(["--capture-devfreq", "--devfreq-dir", str(args.devfreq_dir)])
    if args.capture_driver_probe:
        cmd.append("--capture-driver-probe")
        if args.driver_probe_device:
            cmd.extend(["--driver-probe-device", str(args.driver_probe_device)])
        cmd.extend(
            [
                "--driver-probe-debugfs-root",
                str(args.driver_probe_debugfs_root),
                "--driver-probe-procfs-root",
                str(args.driver_probe_procfs_root),
                "--driver-probe-devfreq-dir",
                str(args.driver_probe_devfreq_dir),
            ]
        )
        if args.driver_probe_include_unsupported:
            cmd.append("--driver-probe-include-unsupported")
    with tempfile.NamedTemporaryFile(prefix=f"rknpu_perf_ref_{case_name}_", suffix=".json") as tmp:
        cmd.extend(["--json-out", tmp.name])
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout, end="")
        if proc.returncode != 0:
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            raise RuntimeError(f"subprocess case failed: {case_name} rc={proc.returncode}")
        payload = json.loads(Path(tmp.name).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if len(records) != 1:
        raise RuntimeError(f"subprocess case payload for {case_name} returned {len(records)} records")
    return records[0]


def _select_cases(
    all_cases: Sequence[Case],
    suite: str,
    comparison_group: str,
    case_name: str,
) -> List[Case]:
    case_by_name = {case.name: case for case in all_cases}
    if case_name:
        if case_name not in case_by_name:
            raise RuntimeError(f"unknown case: {case_name}")
        return [case_by_name[case_name]]
    selected = list(all_cases)
    if suite == "fusion_matrix":
        selected = [case for case in selected if case.comparison_group]
        selected = [case for case in selected if case.comparison_group != "ew_add_identity_chain"]
    if suite == "overhead_matrix":
        selected = [case for case in selected if case.comparison_group == "ew_add_identity_chain"]
    if comparison_group != "all":
        selected = [case for case in selected if case.comparison_group == comparison_group]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=("python", "cpp", "both"), default="python")
    parser.add_argument("--suite", choices=("full", "fusion_matrix", "overhead_matrix"), default="full")
    parser.add_argument(
        "--comparison-group",
        choices=("all", "matmul_bias_relu", "residual_mlp_relu", "ew_add_identity_chain"),
        default="all",
    )
    parser.add_argument("--m-small", type=int, default=1)
    parser.add_argument("--m-large", type=int, default=1500)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bridge-cache-dma", choices=("default", "on", "off"), default="on")
    parser.add_argument("--warmup-mode", choices=("fixed", "steady"), default="fixed")
    parser.add_argument("--steady-state-window", type=int, default=4)
    parser.add_argument("--steady-state-rtol", type=float, default=0.10)
    parser.add_argument("--steady-state-max-extra-warmup", type=int, default=24)
    parser.add_argument("--capture-devfreq", action="store_true")
    parser.add_argument("--devfreq-dir", type=str, default=DEFAULT_DEVFREQ_DIR)
    parser.add_argument("--capture-driver-probe", action="store_true")
    parser.add_argument("--driver-probe-device", type=str, default="")
    parser.add_argument("--driver-probe-debugfs-root", type=str, default="/sys/kernel/debug/rknpu")
    parser.add_argument("--driver-probe-procfs-root", type=str, default="/proc/rknpu")
    parser.add_argument("--driver-probe-devfreq-dir", type=str, default=DEFAULT_DEVFREQ_DIR)
    parser.add_argument("--driver-probe-include-unsupported", action="store_true")
    parser.add_argument("--case", type=str, default="")
    parser.add_argument("--no-isolate-cases", action="store_true")
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--markdown-out", type=str, default="")
    args = parser.parse_args()

    devfreq_dir = None
    if args.capture_devfreq:
        candidate = Path(args.devfreq_dir)
        if candidate.is_dir():
            devfreq_dir = candidate
        else:
            raise RuntimeError(f"devfreq dir does not exist: {candidate}")

    global_env_overrides: Dict[str, str] = {}
    if args.bridge_cache_dma == "on":
        global_env_overrides["TVM_RKNPU_BRIDGE_CACHE_DMA"] = "1"
    elif args.bridge_cache_dma == "off":
        global_env_overrides["TVM_RKNPU_BRIDGE_CACHE_DMA"] = "0"

    all_cases = _build_cases(args.m_small, args.m_large, args.d_model, args.hidden)
    selected_cases = _select_cases(all_cases, args.suite, args.comparison_group, args.case)

    records: List[Dict[str, object]] = []
    devfreq_initial = _read_devfreq_snapshot(devfreq_dir) if devfreq_dir is not None else {}
    driver_probe_initial = _read_driver_probe_snapshot(
        enabled=args.capture_driver_probe,
        device=args.driver_probe_device,
        debugfs_root=args.driver_probe_debugfs_root,
        procfs_root=args.driver_probe_procfs_root,
        devfreq_dir=args.driver_probe_devfreq_dir,
        include_unsupported=args.driver_probe_include_unsupported,
    )
    isolate_cases = (not args.no_isolate_cases) and not bool(args.case)
    hosts = ["python", "cpp"] if args.host == "both" else [args.host]
    for case in selected_cases:
        for host in hosts:
            if isolate_cases:
                record = _measure_case_in_subprocess(args, case.name, host)
            else:
                if host == "python":
                    record = _measure_case(
                        case,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        devfreq_dir=devfreq_dir,
                        capture_driver_probe=args.capture_driver_probe,
                        driver_probe_device=args.driver_probe_device,
                        driver_probe_debugfs_root=args.driver_probe_debugfs_root,
                        driver_probe_procfs_root=args.driver_probe_procfs_root,
                        driver_probe_devfreq_dir=args.driver_probe_devfreq_dir,
                        driver_probe_include_unsupported=args.driver_probe_include_unsupported,
                        global_env_overrides=global_env_overrides,
                        warmup_mode=args.warmup_mode,
                        steady_state_window=args.steady_state_window,
                        steady_state_rtol=args.steady_state_rtol,
                        steady_state_max_extra_warmup=args.steady_state_max_extra_warmup,
                    )
                else:
                    record = _measure_case_cpp(
                        case,
                        warmup=args.warmup,
                        iters=args.iters,
                        seed=args.seed,
                        devfreq_dir=devfreq_dir,
                        capture_driver_probe=args.capture_driver_probe,
                        driver_probe_device=args.driver_probe_device,
                        driver_probe_debugfs_root=args.driver_probe_debugfs_root,
                        driver_probe_procfs_root=args.driver_probe_procfs_root,
                        driver_probe_devfreq_dir=args.driver_probe_devfreq_dir,
                        driver_probe_include_unsupported=args.driver_probe_include_unsupported,
                        global_env_overrides=global_env_overrides,
                        warmup_mode=args.warmup_mode,
                        steady_state_window=args.steady_state_window,
                        steady_state_rtol=args.steady_state_rtol,
                        steady_state_max_extra_warmup=args.steady_state_max_extra_warmup,
                    )
                print(
                    "PERF_REF "
                    f"host={record['host']} "
                    f"name={record['name']} "
                    f"latency_kind={record['latency_kind']} "
                    f"latency_ns={record['latency_ns_median']} "
                    f"cold_runtime_ns={record['cold_start']['runtime_total_ns']} "
                    f"tail_hw_ns={record['runtime_hw_ns_tail_median']} "
                    f"submits={record['schedule']['num_submits']} "
                    f"tasks={record['schedule']['total_tasks']} "
                    f"stage_ids={record['schedule']['submit_stage_ids']}"
                )
            records.append(record)

    payload = _build_payload(
        args,
        devfreq_dir,
        devfreq_initial,
        _read_devfreq_snapshot(devfreq_dir) if devfreq_dir is not None else {},
        driver_probe_initial,
        _read_driver_probe_snapshot(
            enabled=args.capture_driver_probe,
            device=args.driver_probe_device,
            debugfs_root=args.driver_probe_debugfs_root,
            procfs_root=args.driver_probe_procfs_root,
            devfreq_dir=args.driver_probe_devfreq_dir,
            include_unsupported=args.driver_probe_include_unsupported,
        ),
        records,
        isolate_cases,
    )

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(_markdown_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
