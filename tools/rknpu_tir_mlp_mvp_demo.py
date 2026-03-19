#!/usr/bin/env python3
"""Run the residual 2-D MLP MVP block on the RKNPU TIR chain path."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")
sys.modules.setdefault("torch", None)

import tvm
from tvm import relax

import tvm.relax.backend.contrib.rknpu as rknpu


REPO_ROOT = Path(__file__).resolve().parent.parent
CPP_RUNNER_SRC = REPO_ROOT / "tools" / "rknpu_vm_cpp_runner.cc"
CPP_RUNNER_BIN = REPO_ROOT / "build" / "rknpu_vm_cpp_runner"
EXPECTED_STAGE_IDS = [[6, 11, 2]]
DEFAULT_MAX_ERR = 1e-3


def _build_residual_mlp_mod(m: int, d_model: int, hidden: int) -> tvm.ir.IRModule:
    bb = relax.BlockBuilder()

    x = relax.Var("x", relax.TensorStructInfo((m, d_model), "float16"))
    w1 = relax.Var("w1", relax.TensorStructInfo((d_model, hidden), "float16"))
    b1 = relax.Var("b1", relax.TensorStructInfo((hidden,), "float16"))
    w2 = relax.Var("w2", relax.TensorStructInfo((hidden, d_model), "float16"))
    b2 = relax.Var("b2", relax.TensorStructInfo((d_model,), "float16"))

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


def _bind_residual_mlp_params(mod: tvm.ir.IRModule, inputs: Dict[str, np.ndarray]) -> tvm.ir.IRModule:
    return relax.transform.BindParams(
        "main",
        {
            "w1": inputs["w1"],
            "b1": inputs["b1"],
            "w2": inputs["w2"],
            "b2": inputs["b2"],
        },
    )(mod)


def _numpy_ref(inputs: Dict[str, np.ndarray]) -> np.ndarray:
    x = inputs["x"].astype(np.float32)
    ff1 = x @ inputs["w1"].astype(np.float32) + inputs["b1"].astype(np.float32)[None, :]
    act = np.maximum(ff1, 0.0)
    ff2 = act @ inputs["w2"].astype(np.float32) + inputs["b2"].astype(np.float32)[None, :]
    return (ff2 + x).astype(np.float16)


def _make_inputs(
    m: int, d_model: int, hidden: int, seed: int
) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    inputs = {
        "x": (0.1 * rng.standard_normal((m, d_model))).astype("float16"),
        "w1": (0.1 * rng.standard_normal((d_model, hidden))).astype("float16"),
        "b1": (0.1 * rng.standard_normal((hidden,))).astype("float16"),
        "w2": (0.1 * rng.standard_normal((hidden, d_model))).astype("float16"),
        "b2": (0.1 * rng.standard_normal((d_model,))).astype("float16"),
    }
    return [inputs["x"]], inputs


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def _max_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))


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


def _export_cpp_bundle(
    bundle_dir: Path,
    executable,
    runtime_inputs: Dict[str, np.ndarray],
    y_ref: np.ndarray,
    out: Dict[str, object],
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    executable.export_library(str(bundle_dir / "exec.so"))
    for name, arr in runtime_inputs.items():
        (bundle_dir / f"{name}.bin").write_bytes(arr.tobytes(order="C"))
    (bundle_dir / "expected_y.bin").write_bytes(y_ref.tobytes(order="C"))
    (bundle_dir / "pretty_schedule.txt").write_text(str(out["pretty_schedule"]), encoding="utf-8")
    manifest_lines = [
        "entry=main",
        "dtype=float16",
        f"num_inputs={len(runtime_inputs)}",
    ]
    for idx, (name, arr) in enumerate(runtime_inputs.items()):
        manifest_lines.extend(
            [
                f"input{idx}_name={name}",
                f"input{idx}_file={name}.bin",
                f"input{idx}_shape={_shape_text(arr.shape)}",
            ]
        )
    manifest_lines.extend(
        [
            "num_outputs=1",
            "output0_name=y",
            "output0_expected_file=expected_y.bin",
            f"output0_shape={out['config']['m']}x{out['config']['d_model']}",
        ]
    )
    (bundle_dir / "bundle.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    meta = dict(out)
    meta["bundle"] = {
        "format": "rknpu_mlp_mvp_bundle_v2",
        "files": [
            "exec.so",
            "bundle.txt",
            "meta.json",
            "pretty_schedule.txt",
            "x.bin",
            "expected_y.bin",
        ],
        "runtime_inputs": ["x"],
        "bound_params": ["w1", "b1", "w2", "b2"],
    }
    with open(bundle_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _aggregate_runtime_stats(sample_stats: Sequence[Dict[str, object]]) -> Dict[str, int]:
    aggregate: Dict[str, int] = {}
    for stats in sample_stats:
        for key, value in stats.items():
            if isinstance(value, (int, bool)):
                aggregate[key] = int(aggregate.get(key, 0)) + int(value)
    return aggregate


def _runner_perf_summary(payload: Dict[str, object]) -> Dict[str, object]:
    samples = payload.get("timed_samples", [])
    if not isinstance(samples, list) or not samples:
        raise RuntimeError("cpp runner produced no timed samples")
    runtime_total = np.array([int(sample["runtime_total_ns"]) for sample in samples], dtype=np.int64)
    runtime_submit = np.array([int(sample["runtime_submit_ns"]) for sample in samples], dtype=np.int64)
    runtime_hw = np.array([int(sample["runtime_hw_ns"]) for sample in samples], dtype=np.int64)
    wall = np.array([int(sample["wall_ns"]) for sample in samples], dtype=np.int64)
    tail_count = max(1, len(samples) // 2)
    return {
        "tail_count": int(tail_count),
        "runtime_total_ns_tail_median": int(np.median(runtime_total[-tail_count:])),
        "runtime_submit_ns_tail_median": int(np.median(runtime_submit[-tail_count:])),
        "runtime_hw_ns_tail_median": int(np.median(runtime_hw[-tail_count:])),
        "wall_ns_tail_median": int(np.median(wall[-tail_count:])),
        "runtime_total_ms_tail_median": float(np.median(runtime_total[-tail_count:]) / 1e6),
        "runtime_submit_ms_tail_median": float(np.median(runtime_submit[-tail_count:]) / 1e6),
        "runtime_hw_ms_tail_median": float(np.median(runtime_hw[-tail_count:]) / 1e6),
        "wall_ms_tail_median": float(np.median(wall[-tail_count:]) / 1e6),
    }


def _run_cpp_bundle_benchmark(bundle_dir: Path, warmup: int, iters: int) -> Dict[str, object]:
    runner = _ensure_cpp_runner_built()
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
    env = os.environ.copy()
    ld_paths = [str(REPO_ROOT / "build"), str(REPO_ROOT / "build" / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
    env.setdefault("TVM_LIBRARY_PATH", str(REPO_ROOT / "build"))
    env.setdefault("TVM_RKNPU_BRIDGE_REAL_SUBMIT", "1")
    env.setdefault("TVM_RKNPU_BRIDGE_USE_RELOCS", "1")
    env.setdefault("TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK", "1")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    with open(runner_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _mlp_pretty_symbols(m: int, d_model: int, hidden: int, mode: str) -> Dict[str, object]:
    locations = [
        {
            "location": "buf.x",
            "kind": "materialized",
            "name": "x",
            "dtype": "float16",
            "shape": [m, d_model],
            "bytes": m * d_model * 2,
        },
        {
            "location": "const.w1",
            "kind": "materialized",
            "name": "w1",
            "dtype": "float16",
            "shape": [d_model, hidden],
            "bytes": d_model * hidden * 2,
        },
        {
            "location": "const.b1",
            "kind": "materialized",
            "name": "b1",
            "dtype": "float16",
            "shape": [hidden],
            "bytes": hidden * 2,
        },
        {
            "location": "const.w2",
            "kind": "materialized",
            "name": "w2",
            "dtype": "float16",
            "shape": [hidden, d_model],
            "bytes": hidden * d_model * 2,
        },
        {
            "location": "const.b2",
            "kind": "materialized",
            "name": "b2",
            "dtype": "float16",
            "shape": [d_model],
            "bytes": d_model * 2,
        },
        {
            "location": "buf.y",
            "kind": "materialized",
            "name": "y",
            "dtype": "float16",
            "shape": [m, d_model],
            "bytes": m * d_model * 2,
        },
    ]
    if mode == "fused":
        locations.extend(
            [
                {
                    "location": "chain.h1",
                    "kind": "internal",
                    "name": "h1",
                    "dtype": "float16",
                    "shape": [m, hidden],
                },
                {
                    "location": "chain.h2",
                    "kind": "internal",
                    "name": "h2",
                    "dtype": "float16",
                    "shape": [m, d_model],
                },
            ]
        )
        stage_bindings = {
            (0, 0): {
                "logical_op": "op.matmul_bias_relu",
                "computes": "h1 = relu(x @ w1 + b1)",
                "reads": [
                    {"name": "x", "location": "buf.x"},
                    {"name": "w1", "location": "const.w1"},
                    {"name": "b1", "location": "const.b1"},
                ],
                "writes": [{"name": "h1", "location": "chain.h1"}],
            },
            (0, 1): {
                "logical_op": "op.matmul_bias",
                "computes": "h2 = h1 @ w2 + b2",
                "reads": [
                    {"name": "h1", "location": "chain.h1"},
                    {"name": "w2", "location": "const.w2"},
                    {"name": "b2", "location": "const.b2"},
                ],
                "writes": [{"name": "h2", "location": "chain.h2"}],
            },
            (0, 2): {
                "logical_op": "op.add",
                "computes": "y = h2 + x",
                "reads": [
                    {"name": "h2", "location": "chain.h2"},
                    {"name": "x", "location": "buf.x"},
                ],
                "writes": [{"name": "y", "location": "buf.y"}],
            },
        }
        return {"locations": locations, "stage_bindings": stage_bindings}

    temp_prefix = "chain" if mode == "split_tasks" else "tmp"
    temp_kind = "internal" if mode == "split_tasks" else "materialized"
    temp_specs = [
        ("ff1m", [m, hidden]),
        ("ff1b", [m, hidden]),
        ("act", [m, hidden]),
        ("ff2m", [m, d_model]),
        ("ff2b", [m, d_model]),
    ]
    for name, shape in temp_specs:
        item = {
            "location": f"{temp_prefix}.{name}",
            "kind": temp_kind,
            "name": name,
            "dtype": "float16",
            "shape": shape,
        }
        if temp_kind == "materialized":
            item["bytes"] = int(np.prod(shape)) * 2
        locations.append(item)

    bindings = {
        (0, 0): {
            "logical_op": "op.matmul",
            "computes": "ff1m = x @ w1",
            "reads": [
                {"name": "x", "location": "buf.x"},
                {"name": "w1", "location": "const.w1"},
            ],
            "writes": [{"name": "ff1m", "location": f"{temp_prefix}.ff1m"}],
        },
        (0, 1): {
            "logical_op": "op.add",
            "computes": "ff1b = ff1m + b1",
            "reads": [
                {"name": "ff1m", "location": f"{temp_prefix}.ff1m"},
                {"name": "b1", "location": "const.b1"},
            ],
            "writes": [{"name": "ff1b", "location": f"{temp_prefix}.ff1b"}],
        },
        (0, 2): {
            "logical_op": "op.relu",
            "computes": "act = relu(ff1b)",
            "reads": [{"name": "ff1b", "location": f"{temp_prefix}.ff1b"}],
            "writes": [{"name": "act", "location": f"{temp_prefix}.act"}],
        },
        (0, 3): {
            "logical_op": "op.matmul",
            "computes": "ff2m = act @ w2",
            "reads": [
                {"name": "act", "location": f"{temp_prefix}.act"},
                {"name": "w2", "location": "const.w2"},
            ],
            "writes": [{"name": "ff2m", "location": f"{temp_prefix}.ff2m"}],
        },
        (0, 4): {
            "logical_op": "op.add",
            "computes": "ff2b = ff2m + b2",
            "reads": [
                {"name": "ff2m", "location": f"{temp_prefix}.ff2m"},
                {"name": "b2", "location": "const.b2"},
            ],
            "writes": [{"name": "ff2b", "location": f"{temp_prefix}.ff2b"}],
        },
        (0, 5): {
            "logical_op": "op.add",
            "computes": "y = ff2b + x",
            "reads": [
                {"name": "ff2b", "location": f"{temp_prefix}.ff2b"},
                {"name": "x", "location": "buf.x"},
            ],
            "writes": [{"name": "y", "location": "buf.y"}],
        },
    }
    if mode == "split_submits":
        stage_bindings = {(submit_index, 0): bindings[(0, submit_index)] for submit_index in range(6)}
    else:
        stage_bindings = bindings
    return {"locations": locations, "stage_bindings": stage_bindings}


def _fail(msg: str) -> None:
    raise RuntimeError(msg)


@contextlib.contextmanager
def _temp_env(overrides: Dict[str, str], clear_keys: Sequence[str] = ()):
    saved = {key: os.environ.get(key) for key in set(overrides).union(clear_keys)}
    for key in clear_keys:
        os.environ.pop(key, None)
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _variant_config(mode: str) -> Dict[str, object]:
    if mode == "fused":
        return {
            "mode": mode,
            "label": "fused",
            "env_overrides": {},
            "expected_stage_ids": [[6, 11, 2]],
            "expected_num_submits": 1,
            "expected_blocked_boundaries": 0,
        }
    if mode == "split_tasks":
        return {
            "mode": mode,
            "label": "one-submit non-fused",
            "env_overrides": {"TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1"},
            "expected_stage_ids": [[1, 2, 3, 1, 2, 2]],
            "expected_num_submits": 1,
            "expected_blocked_boundaries": 0,
        }
    if mode == "split_submits":
        return {
            "mode": mode,
            "label": "multi-submit non-fused",
            "env_overrides": {
                "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION": "1",
                "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES": "1",
            },
            "expected_stage_ids": [[1], [2], [3], [1], [2], [2]],
            "expected_num_submits": 6,
            "expected_blocked_boundaries": 0,
        }
    _fail(f"unsupported variant mode: {mode}")


def _enforce_mvp_invariants(
    *,
    report: Dict[str, object],
    runtime_stats: Dict[str, int],
    max_err: float,
    non_finite: int,
    iters: int,
    max_err_limit: float,
    chain_blob_source: str,
    expected_stage_ids: List[List[int]],
    expected_num_submits: int,
    expected_blocked_boundaries: int,
) -> None:
    submit_stage_ids = report.get("submit_stage_ids")
    if submit_stage_ids != expected_stage_ids:
        _fail(f"submit_stage_ids={submit_stage_ids} expected={expected_stage_ids}")
    if int(report.get("num_submits", 0)) != expected_num_submits:
        _fail(f"num_submits={report.get('num_submits')} expected={expected_num_submits}")
    blocked = int(report.get("chain_compatibility", {}).get("blocked_boundary_count", 0))
    if blocked != expected_blocked_boundaries:
        _fail(
            f"blocked_boundary_count={blocked} expected={expected_blocked_boundaries}"
        )
    if chain_blob_source != "embedded":
        _fail(f"chain_blob_source={chain_blob_source} expected=embedded")
    if not np.isfinite(max_err) or max_err > max_err_limit:
        _fail(f"max_err={max_err:.6g} exceeds {max_err_limit:.6g}")
    if non_finite != 0:
        _fail(f"output_non_finite={non_finite} expected=0")
    for key in (
        "touch_fallback",
        "real_submit_fail",
        "reloc_submit_fallbacks",
        "reloc_semantic_mismatch",
        "reloc_range_mismatch",
    ):
        if int(runtime_stats.get(key, 0)) != 0:
            _fail(f"{key}={runtime_stats.get(key)} expected=0")
    expected_submit_calls = int(report.get("num_submits", 0)) * int(max(iters, 1))
    expected_task_calls = int(report.get("total_tasks", 0)) * int(max(iters, 1))
    if int(runtime_stats.get("real_submit_ok", -1)) != expected_submit_calls:
        _fail(
            f"real_submit_ok={runtime_stats.get('real_submit_ok')} expected={expected_submit_calls}"
        )
    if int(runtime_stats.get("submitted_tasks", -1)) != expected_task_calls:
        _fail(
            f"submitted_tasks={runtime_stats.get('submitted_tasks')} expected={expected_task_calls}"
        )


def _run_variant(args: argparse.Namespace, mode: str, bundle_dir_path: Path | None = None) -> Dict[str, object]:
    variant = _variant_config(mode)
    runtime_inputs, all_inputs = _make_inputs(args.m, args.d_model, args.hidden, args.seed)
    base_env = {
        "TVM_RKNPU_BRIDGE_REAL_SUBMIT": "1",
        "TVM_RKNPU_BRIDGE_USE_RELOCS": "1",
        "TVM_RKNPU_BRIDGE_FAIL_ON_FALLBACK": "1",
    }
    if args.bridge_cache_dma != "default":
        base_env["TVM_RKNPU_BRIDGE_CACHE_DMA"] = "1" if args.bridge_cache_dma == "on" else "0"
    if args.bridge_debug_checks:
        base_env["TVM_RKNPU_BRIDGE_VALIDATE_RELOCS"] = "1"
    with _temp_env(
        {**base_env, **variant["env_overrides"]},
        clear_keys=(
            "TVM_RKNPU_PC_CHAIN_ENABLE_MATMUL_BIAS_FUSION",
            "TVM_RKNPU_PC_CHAIN_DISABLE_FUSION",
            "TVM_RKNPU_PC_CHAIN_SPLIT_STAGES",
            "TVM_RKNPU_PC_CHAIN_SINGLETON_ALLOWLIST",
        ),
    ):
        mod = _bind_residual_mlp_params(
            _build_residual_mlp_mod(args.m, args.d_model, args.hidden), all_inputs
        )
        tir_mod = rknpu.plan_rknpu_tir_memory(rknpu.lower_to_rknpu_tir_with_pc_chain(mod))
        report = rknpu.get_rknpu_schedule_report(tir_mod)
        executable = rknpu.build_with_runtime_bridge(tir_mod, target="llvm")
        y_ref = _numpy_ref(all_inputs)
        pretty_schedule = rknpu.format_rknpu_schedule_report(
            report,
            symbols=_mlp_pretty_symbols(args.m, args.d_model, args.hidden, mode),
        )

        out = {
            "variant": {
                "mode": variant["mode"],
                "label": variant["label"],
                "env_overrides": dict(variant["env_overrides"]),
            },
            "config": {
                "m": args.m,
                "d_model": args.d_model,
                "hidden": args.hidden,
                "seed": args.seed,
                "warmup": args.warmup,
                "iters": args.iters,
                "real_submit": True,
                "bridge_debug_checks": bool(args.bridge_debug_checks),
                "bridge_cache_dma": args.bridge_cache_dma,
                "runtime_inputs": ["x"],
                "bound_params": ["w1", "b1", "w2", "b2"],
            },
            "expected_schedule": {
                "submit_stage_ids": variant["expected_stage_ids"],
                "num_submits": variant["expected_num_submits"],
                "blocked_boundary_count": variant["expected_blocked_boundaries"],
            },
            "pretty_schedule": pretty_schedule,
            "tir_schedule_report": report,
        }

        with tempfile.TemporaryDirectory(prefix=f"rknpu_mlp_mvp_{mode}_") as tmp_dir:
            active_bundle_dir = bundle_dir_path or Path(tmp_dir)
            _export_cpp_bundle(
                active_bundle_dir,
                executable,
                {"x": runtime_inputs[0]},
                y_ref,
                out,
            )
            runner_payload = _run_cpp_bundle_benchmark(
                active_bundle_dir, args.warmup, args.iters
            )

    timed_samples = runner_payload.get("timed_samples", [])
    sample_stats = [
        sample["runtime_bridge_stats"]
        for sample in timed_samples
        if isinstance(sample, dict) and isinstance(sample.get("runtime_bridge_stats"), dict)
    ]
    aggregated_stats = _aggregate_runtime_stats(sample_stats)
    max_err = float(runner_payload.get("max_err", float("inf")))
    output_non_finite = int(runner_payload.get("output_non_finite", -1))
    _enforce_mvp_invariants(
        report=report,
        runtime_stats=aggregated_stats,
        max_err=max_err,
        non_finite=output_non_finite,
        iters=args.iters,
        max_err_limit=args.max_err,
        chain_blob_source=str(runner_payload.get("chain_blob_source", "")),
        expected_stage_ids=variant["expected_stage_ids"],
        expected_num_submits=variant["expected_num_submits"],
        expected_blocked_boundaries=variant["expected_blocked_boundaries"],
    )

    out["correctness"] = {
        "tir_max_err": max_err,
        "tir_non_finite": output_non_finite,
    }
    out["publishable_perf"] = _runner_perf_summary(runner_payload)
    out["runtime_bridge_stats"] = aggregated_stats
    out["cpp_runner"] = {
        "chain_blob_source": str(runner_payload.get("chain_blob_source", "")),
        "warmup": runner_payload.get("warmup", {}),
        "timed_samples": timed_samples,
    }
    if sample_stats and isinstance(sample_stats[-1].get("host_dma_debug"), list):
        out["runtime_bridge_debug_last"] = {"host_dma_debug": sample_stats[-1]["host_dma_debug"]}
    if bundle_dir_path is not None:
        out["bundle"] = {
            "dir": str(bundle_dir_path),
            "format": "rknpu_mlp_mvp_bundle_v2",
        }
        with open(bundle_dir_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    return out


def _print_variant(result: Dict[str, object], pretty: bool) -> None:
    report = result["tir_schedule_report"]
    perf = result["publishable_perf"]
    if pretty:
        print(result["pretty_schedule"])
    print(
        "MLP_MVP "
        f"variant={result['variant']['mode']} "
        f"label=\"{result['variant']['label']}\" "
        f"m={result['config']['m']} d_model={result['config']['d_model']} hidden={result['config']['hidden']} "
        f"cache_dma={result['config']['bridge_cache_dma']} "
        f"tir_max_err={result['correctness']['tir_max_err']:.6f} "
        f"tir_non_finite={result['correctness']['tir_non_finite']} "
        f"runtime_total_ms_tail={perf['runtime_total_ms_tail_median']:.3f} "
        f"runtime_hw_ms_tail={perf['runtime_hw_ms_tail_median']:.3f} "
        f"submits={report.get('num_submits')} tasks={report.get('total_tasks')} "
        f"stage_ids={report.get('submit_stage_ids')} "
        f"chain_blob={result['cpp_runner']['chain_blob_source']}"
    )


def _print_compare_summary(results: Sequence[Dict[str, object]]) -> None:
    indexed = {result["variant"]["mode"]: result for result in results}
    fused_ms = float(indexed["fused"]["publishable_perf"]["runtime_total_ms_tail_median"])
    print("")
    print("MLP_MVP_COMPARE")
    for mode in ("split_submits", "split_tasks", "fused"):
        result = indexed[mode]
        perf = result["publishable_perf"]
        stats = result["runtime_bridge_stats"]
        runtime_ms = float(perf["runtime_total_ms_tail_median"])
        ratio = runtime_ms / fused_ms if fused_ms > 0 else float("inf")
        iters = int(result["config"]["iters"])
        sync_to = int(stats.get("data_sync_to_device_bytes", 0)) // max(iters, 1)
        sync_from = int(stats.get("data_sync_from_device_bytes", 0)) // max(iters, 1)
        chain_reuse = int(stats.get("chain_reuse_bytes", 0)) // max(iters, 1)
        print(
            "  "
            f"variant={mode} label=\"{result['variant']['label']}\" "
            f"runtime_total_ms_tail={runtime_ms:.3f} "
            f"runtime_hw_ms_tail={float(perf['runtime_hw_ms_tail_median']):.3f} "
            f"ratio_vs_fused={ratio:.3f} "
            f"submits={result['tir_schedule_report']['num_submits']} "
            f"tasks={result['tir_schedule_report']['total_tasks']} "
            f"sync_to_device_bytes_per_iter={sync_to} "
            f"sync_from_device_bytes_per_iter={sync_from} "
            f"chain_reuse_bytes_per_iter={chain_reuse} "
            f"stage_ids={result['tir_schedule_report']['submit_stage_ids']}"
        )


def _format_markdown_command(args: argparse.Namespace) -> str:
    cmd = [
        "PYTHONPATH=python:.",
        "TVM_LIBRARY_PATH=build",
        "TVM_FFI_DISABLE_TORCH_C_DLPACK=1",
        "python3",
        "tools/rknpu_tir_mlp_mvp_demo.py",
        "--m",
        str(args.m),
        "--d-model",
        str(args.d_model),
        "--hidden",
        str(args.hidden),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--real-submit",
        "--compare-variants",
        "--pretty",
        "--bridge-cache-dma",
        args.bridge_cache_dma,
    ]
    if args.bridge_debug_checks:
        cmd.append("--bridge-debug-checks")
    return " ".join(shlex.quote(part) for part in cmd)


def _variant_perf_row(result: Dict[str, object], fused_ms: float) -> str:
    perf = result["publishable_perf"]
    stats = result["runtime_bridge_stats"]
    report = result["tir_schedule_report"]
    iters = max(int(result["config"]["iters"]), 1)
    runtime_ms = float(perf["runtime_total_ms_tail_median"])
    ratio = runtime_ms / fused_ms if fused_ms > 0 else float("inf")
    sync_to = int(stats.get("data_sync_to_device_bytes", 0)) // iters
    sync_from = int(stats.get("data_sync_from_device_bytes", 0)) // iters
    chain_reuse = int(stats.get("chain_reuse_bytes", 0)) // iters
    return (
        f"| `{result['variant']['mode']}` | `{result['variant']['label']}` | "
        f"`{report['submit_stage_ids']}` | {int(report['num_submits'])} | {int(report['total_tasks'])} | "
        f"{runtime_ms:.3f} | {float(perf['runtime_hw_ms_tail_median']):.3f} | "
        f"{sync_to} | {sync_from} | {chain_reuse} | {ratio:.3f} |"
    )


def _render_compare_markdown(args: argparse.Namespace, results: Sequence[Dict[str, object]]) -> str:
    indexed = {result["variant"]["mode"]: result for result in results}
    fused = indexed["fused"]
    fused_ms = float(fused["publishable_perf"]["runtime_total_ms_tail_median"])
    lines = [
        "# RKNPU MVP Hero Demo",
        "",
        "This file is generated from `tools/rknpu_tir_mlp_mvp_demo.py`.",
        "",
        "## Workload",
        "",
        "- block: residual 2-D MLP with ReLU",
        f"- shape: `M={args.m}, D={args.d_model}, H={args.hidden}`",
        f"- cache mode: `{args.bridge_cache_dma}`",
        f"- warmup: `{args.warmup}`",
        f"- iters: `{args.iters}`",
        "",
        "Computation:",
        "",
        "- `ff1 = matmul(x, w1) + b1`",
        "- `act = relu(ff1)`",
        "- `ff2 = matmul(act, w2) + b2`",
        "- `out = ff2 + x`",
        "",
        "The compare run holds the math fixed and lowers it three ways:",
        "",
        "- `split_submits`: multi-submit non-fused",
        "- `split_tasks`: one-submit non-fused",
        "- `fused`: one-submit fused",
        "",
        "## Reproduce",
        "",
        "```bash",
        _format_markdown_command(args),
        "```",
        "",
        "## Strict Invariants",
        "",
        "Every variant run in this artifact enforces:",
        "",
        "- exact expected `submit_stage_ids`",
        "- exact expected `num_submits`",
        "- `blocked_boundary_count == 0`",
        "- zero fallback and reloc mismatch counters",
        "- embedded chain blob path",
        f"- `max_err <= {args.max_err:g}` and zero non-finite outputs",
        "",
        "## Summary",
        "",
        "| Variant | Meaning | Stage IDs | Submits | Tasks | Runtime ms tail | HW ms tail | Sync To Bytes/Iter | Sync From Bytes/Iter | Chain Reuse Bytes/Iter | Ratio vs Fused |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("split_submits", "split_tasks", "fused"):
        lines.append(_variant_perf_row(indexed[mode], fused_ms))
    lines.extend(
        [
            "",
            "The architectural result for this hero shape is:",
            "",
            "- `fused < one-submit non-fused < multi-submit non-fused`",
            "- fewer submit boundaries reduce traffic",
            "- fusion reduces both traffic and hardware work further",
            "",
            "## Variant Details",
            "",
        ]
    )
    for mode in ("split_submits", "split_tasks", "fused"):
        result = indexed[mode]
        perf = result["publishable_perf"]
        stats = result["runtime_bridge_stats"]
        report = result["tir_schedule_report"]
        lines.extend(
            [
                f"### `{mode}`",
                "",
                f"- label: `{result['variant']['label']}`",
                f"- `submit_stage_ids = {report['submit_stage_ids']}`",
                f"- `num_submits = {int(report['num_submits'])}`",
                f"- `total_tasks = {int(report['total_tasks'])}`",
                f"- `runtime_total_ms_tail = {float(perf['runtime_total_ms_tail_median']):.3f}`",
                f"- `runtime_hw_ms_tail = {float(perf['runtime_hw_ms_tail_median']):.3f}`",
                f"- `data_sync_to_device_bytes = {int(stats.get('data_sync_to_device_bytes', 0))}`",
                f"- `data_sync_from_device_bytes = {int(stats.get('data_sync_from_device_bytes', 0))}`",
                f"- `chain_reuse_bytes = {int(stats.get('chain_reuse_bytes', 0))}`",
                f"- `tir_max_err = {float(result['correctness']['tir_max_err']):.6f}`",
                "",
                "```text",
                str(result["pretty_schedule"]).rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--real-submit", action="store_true")
    parser.add_argument("--bridge-debug-checks", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--compare-variants", action="store_true")
    parser.add_argument(
        "--bridge-cache-dma",
        choices=("off", "on", "default"),
        default="off",
    )
    parser.add_argument(
        "--variant",
        choices=("fused", "split_tasks", "split_submits"),
        default="fused",
    )
    parser.add_argument("--bundle-dir", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--max-err", type=float, default=DEFAULT_MAX_ERR)
    args = parser.parse_args()

    if not args.real_submit:
        _fail("the MVP demo requires --real-submit")

    if args.compare_variants and args.bundle_dir:
        _fail("--bundle-dir is only supported for a single variant run")

    if args.compare_variants:
        results = [
            _run_variant(args, mode)
            for mode in ("split_submits", "split_tasks", "fused")
        ]
        for result in results:
            _print_variant(result, args.pretty)
            if args.pretty:
                print("")
        _print_compare_summary(results)
        out = {
            "comparison": {
                "shape": {
                    "m": args.m,
                    "d_model": args.d_model,
                    "hidden": args.hidden,
                },
                "variants": results,
            }
        }
        markdown = _render_compare_markdown(args, results)
    else:
        bundle_dir_path = Path(args.bundle_dir) if args.bundle_dir else None
        out = _run_variant(args, args.variant, bundle_dir_path=bundle_dir_path)
        _print_variant(out, args.pretty)
        markdown = ""

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    if args.markdown_out:
        if not markdown:
            _fail("--markdown-out is only supported with --compare-variants")
        with open(args.markdown_out, "w", encoding="utf-8") as f:
            f.write(markdown)


if __name__ == "__main__":
    main()
