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
"""Benchmark XNNPACK Relax BYOC against normal TVM CPU lowering.

The default model is intentionally small and in-tree so the benchmark is
reproducible without downloading model files.
"""

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R


@tvm.script.ir_module
class TinyCNNModule:
    @R.function
    def main(
        x: R.Tensor((1, 8, 8, 3), "float32"),
        residual: R.Tensor((1, 3, 3, 4), "float32"),
        w: R.Tensor((4, 3, 3, 3), "float32"),
        b: R.Tensor((4,), "float32"),
    ):
        with R.dataflow():
            conv = relax.op.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            biased = relax.op.add(conv, b)
            relu = relax.op.nn.relu(biased)
            pooled = relax.op.nn.max_pool2d(
                relu,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            added = relax.op.add(pooled, residual)
            z = relax.op.tanh(added)
            R.output(z)
        return z


@tvm.script.ir_module
class StaticQS8TinyCNNModule:
    @R.function
    def main(
        x: R.Tensor((1, 4, 4, 2), "int8"), y: R.Tensor((1, 4, 4, 2), "int8")
    ) -> R.Tensor((1, 2, 8, 2), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y_f = R.dequantize(
                y,
                R.const(0.25, "float32"),
                R.const(0, "int8"),
                axis=-1,
                out_dtype="float32",
            )
            added = relax.op.add(x_f, y_f)
            clipped = relax.op.clip(added, 0, 6)
            added_q = R.quantize(
                clipped, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            added_f = R.dequantize(
                added_q,
                R.const(0.25, "float32"),
                R.const(0, "int8"),
                axis=-1,
                out_dtype="float32",
            )
            reshaped = relax.op.reshape(added_f, [1, 2, 8, 2])
            z = R.quantize(
                reshaped, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


@tvm.script.ir_module
class LargeCNNModule:
    @R.function
    def main(
        x: R.Tensor((1, 32, 32, 8), "float32"),
        residual: R.Tensor((1, 16, 16, 16), "float32"),
        w1: R.Tensor((16, 3, 3, 8), "float32"),
        b1: R.Tensor((16,), "float32"),
        w2: R.Tensor((16, 3, 3, 16), "float32"),
        b2: R.Tensor((16,), "float32"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.conv2d(
                x,
                w1,
                strides=[1, 1],
                padding=[1, 1, 1, 1],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            bias1 = relax.op.add(conv1, b1)
            relu1 = relax.op.nn.relu(bias1)
            pool1 = relax.op.nn.max_pool2d(
                relu1,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            conv2 = relax.op.nn.conv2d(
                pool1,
                w2,
                strides=[1, 1],
                padding=[1, 1, 1, 1],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
            )
            bias2 = relax.op.add(conv2, b2)
            relu2 = relax.op.nn.relu(bias2)
            added = relax.op.add(relu2, residual)
            z = relax.op.tanh(added)
            R.output(z)
        return z


@tvm.script.ir_module
class LargeMLPModule:
    @R.function
    def main(
        x: R.Tensor((16, 64), "float32"),
        residual: R.Tensor((16, 128), "float32"),
        w1: R.Tensor((64, 128), "float32"),
        b1: R.Tensor((128,), "float32"),
        w2: R.Tensor((128, 128), "float32"),
        b2: R.Tensor((128,), "float32"),
    ):
        with R.dataflow():
            fc1 = R.matmul(x, w1)
            bias1 = R.add(fc1, b1)
            gelu = R.nn.gelu(bias1)
            added = R.add(gelu, residual)
            fc2 = R.matmul(added, w2)
            bias2 = R.add(fc2, b2)
            approx_gelu = R.nn.gelu_tanh(bias2)
            z = R.nn.softmax(approx_gelu, axis=-1)
            R.output(z)
        return z


@tvm.script.ir_module
class LargeStaticQS8CNNModule:
    @R.function
    def main(
        x: R.Tensor((1, 16, 16, 8), "int8"), y: R.Tensor((1, 16, 16, 8), "int8")
    ) -> R.Tensor((1, 8, 8, 8), "int8"):
        with R.dataflow():
            x_f = R.dequantize(
                x, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            y_f = R.dequantize(
                y, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="float32"
            )
            added = relax.op.add(x_f, y_f)
            clipped = relax.op.clip(added, 0, 6)
            added_q = R.quantize(
                clipped, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            added_f = R.dequantize(
                added_q,
                R.const(0.25, "float32"),
                R.const(0, "int8"),
                axis=-1,
                out_dtype="float32",
            )
            pooled = relax.op.nn.max_pool2d(
                added_f,
                pool_size=[2, 2],
                strides=[2, 2],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                ceil_mode=False,
                layout="NHWC",
                out_layout="NHWC",
            )
            z = R.quantize(
                pooled, R.const(0.25, "float32"), R.const(0, "int8"), axis=-1, out_dtype="int8"
            )
            R.output(z)
        return z


IN_TREE_MODELS = (
    "xnnpack_tiny_cnn",
    "xnnpack_static_qs8_tiny_cnn",
    "xnnpack_large_cnn_fp32",
    "xnnpack_large_mlp_fp32",
    "xnnpack_large_qs8_cnn",
)
TORCHVISION_MODELS = ("mobilenet_v2", "mobilenet_v3_small", "resnet18")


def has_xnnpack_enabled() -> bool:
    return (
        tvm.get_global_func("relax.ext.xnnpack", allow_missing=True) is not None
        and tvm.get_global_func("runtime.XNNPACKJSONRuntimeCreate", allow_missing=True) is not None
    )


def get_xnnpack_capabilities() -> Dict[str, int]:
    func = tvm.get_global_func("runtime.XNNPACKJSONRuntimeGetCapabilities", allow_missing=True)
    if func is None:
        return {}
    return {str(key): int(value) for key, value in func().items()}


def get_memory_kib() -> int:
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss // 1024
        return rss
    except Exception:  # pylint: disable=broad-except
        return -1


def count_xnnpack_partitions(mod: tvm.IRModule) -> int:
    count = 0

    for func in mod.functions.values():
        if (
            isinstance(func, relax.Function)
            and func.attrs
            and func.attrs.get("Codegen") == "xnnpack"
        ):
            count += 1

    return count


def bind_tiny_cnn_params() -> tvm.IRModule:
    weight = np.linspace(-0.2, 0.2, num=4 * 3 * 3 * 3, dtype="float32").reshape(4, 3, 3, 3)
    bias = np.array([0.15, -0.05, 0.25, -0.10], dtype="float32")
    return relax.transform.BindParams("main", {"w": weight, "b": bias})(TinyCNNModule)


def make_tiny_cnn_inputs(seed: int) -> List[tvm.runtime.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-1.0, 1.0, size=(1, 8, 8, 3)).astype("float32")
    residual_np = rng.uniform(-0.5, 0.5, size=(1, 3, 3, 4)).astype("float32")
    return [tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)]


def load_tiny_cnn(seed: int) -> Tuple[tvm.IRModule, List[tvm.runtime.Tensor], str]:
    return bind_tiny_cnn_params(), make_tiny_cnn_inputs(seed), "xnnpack_tiny_cnn"


def make_static_qs8_tiny_cnn_inputs(seed: int) -> List[tvm.runtime.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = rng.integers(-8, 8, size=(1, 4, 4, 2), dtype=np.int8)
    y_np = rng.integers(-4, 4, size=(1, 4, 4, 2), dtype=np.int8)
    return [tvm.runtime.tensor(x_np), tvm.runtime.tensor(y_np)]


def load_static_qs8_tiny_cnn(seed: int) -> Tuple[tvm.IRModule, List[tvm.runtime.Tensor], str]:
    return StaticQS8TinyCNNModule, make_static_qs8_tiny_cnn_inputs(seed), "xnnpack_static_qs8_tiny_cnn"


def bind_large_cnn_params() -> tvm.IRModule:
    w1 = np.linspace(-0.2, 0.2, num=16 * 3 * 3 * 8, dtype="float32").reshape(16, 3, 3, 8)
    b1 = np.linspace(-0.1, 0.1, num=16, dtype="float32")
    w2 = np.linspace(0.15, -0.15, num=16 * 3 * 3 * 16, dtype="float32").reshape(16, 3, 3, 16)
    b2 = np.linspace(0.05, -0.05, num=16, dtype="float32")
    return relax.transform.BindParams("main", {"w1": w1, "b1": b1, "w2": w2, "b2": b2})(
        LargeCNNModule
    )


def make_large_cnn_inputs(seed: int) -> List[tvm.runtime.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-1.0, 1.0, size=(1, 32, 32, 8)).astype("float32")
    residual_np = rng.uniform(-0.25, 0.25, size=(1, 16, 16, 16)).astype("float32")
    return [tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)]


def load_large_cnn(seed: int) -> Tuple[tvm.IRModule, List[tvm.runtime.Tensor], str]:
    return bind_large_cnn_params(), make_large_cnn_inputs(seed), "xnnpack_large_cnn_fp32"


def bind_large_mlp_params() -> tvm.IRModule:
    w1 = np.linspace(-0.25, 0.25, num=64 * 128, dtype="float32").reshape(64, 128)
    b1 = np.linspace(-0.1, 0.1, num=128, dtype="float32")
    w2 = np.linspace(0.2, -0.2, num=128 * 128, dtype="float32").reshape(128, 128)
    b2 = np.linspace(0.05, -0.05, num=128, dtype="float32")
    return relax.transform.BindParams("main", {"w1": w1, "b1": b1, "w2": w2, "b2": b2})(
        LargeMLPModule
    )


def make_large_mlp_inputs(seed: int) -> List[tvm.runtime.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = rng.uniform(-1.0, 1.0, size=(16, 64)).astype("float32")
    residual_np = rng.uniform(-0.25, 0.25, size=(16, 128)).astype("float32")
    return [tvm.runtime.tensor(x_np), tvm.runtime.tensor(residual_np)]


def load_large_mlp(seed: int) -> Tuple[tvm.IRModule, List[tvm.runtime.Tensor], str]:
    return bind_large_mlp_params(), make_large_mlp_inputs(seed), "xnnpack_large_mlp_fp32"


def make_large_static_qs8_cnn_inputs(seed: int) -> List[tvm.runtime.Tensor]:
    rng = np.random.default_rng(seed)
    x_np = rng.integers(-8, 8, size=(1, 16, 16, 8), dtype=np.int8)
    y_np = rng.integers(-8, 8, size=(1, 16, 16, 8), dtype=np.int8)
    return [tvm.runtime.tensor(x_np), tvm.runtime.tensor(y_np)]


def load_large_static_qs8_cnn(seed: int) -> Tuple[tvm.IRModule, List[tvm.runtime.Tensor], str]:
    return (
        LargeStaticQS8CNNModule,
        make_large_static_qs8_cnn_inputs(seed),
        "xnnpack_large_qs8_cnn",
    )


def load_torchvision_model(model_name: str, input_shape: Tuple[int, ...]):
    torch_spec = importlib.util.find_spec("torch")
    torchvision_spec = importlib.util.find_spec("torchvision")
    if torch_spec is None or torchvision_spec is None:
        raise RuntimeError("torch and torchvision are required for torchvision:* models")

    import torch
    import torchvision
    from torch.export import export
    from tvm.relax.frontend.torch import from_exported_program

    if not hasattr(torchvision.models, model_name):
        raise RuntimeError(f"torchvision.models has no model named {model_name!r}")
    if model_name not in TORCHVISION_MODELS:
        raise RuntimeError(
            "supported torchvision models are "
            + ", ".join(f"torchvision:{name}" for name in TORCHVISION_MODELS)
        )

    model = getattr(torchvision.models, model_name)(weights=None).eval()
    example_input = torch.zeros(input_shape, dtype=torch.float32)
    with torch.no_grad():
        exported = export(model, (example_input,))
        mod = from_exported_program(exported, keep_params_as_input=False)

    input_np = np.zeros(input_shape, dtype="float32")
    return mod, [tvm.runtime.tensor(input_np)], f"torchvision:{model_name}"


def model_family(model_name: str) -> str:
    if model_name.startswith("torchvision:"):
        return "torchvision"
    if "mlp" in model_name:
        return "mlp"
    if "qs8" in model_name:
        return "static_qs8"
    return "cnn"


def fixture_size(model_name: str) -> str:
    if "large" in model_name:
        return "large"
    if "medium" in model_name:
        return "medium"
    return "small"


def tensor_shape_list(tensor: tvm.runtime.Tensor) -> List[int]:
    return [int(dim) for dim in tensor.shape]


def estimate_parameter_count(mod: tvm.IRModule) -> int:
    const_map = mod.attrs.get("const_name_to_constant", {}) if mod.attrs else {}
    total = 0
    for const in const_map.values():
        total += int(np.prod(const.data.shape))
    if total > 0:
        return total
    def visit(expr):
        nonlocal total
        if isinstance(expr, relax.Constant):
            total += int(np.prod(expr.data.shape))

    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(func, visit)
    return total


def estimate_op_count(mod: tvm.IRModule) -> int:
    count = 0

    def visit(expr):
        nonlocal count
        if isinstance(expr, relax.Call):
            count += 1

    for func in mod.functions.values():
        if isinstance(func, relax.Function):
            relax.analysis.post_order_visit(func, visit)
    return count


def model_metadata(mod: tvm.IRModule, inputs: List[tvm.runtime.Tensor], model_name: str):
    return {
        "model_family": model_family(model_name),
        "fixture_size": fixture_size(model_name),
        "input_shapes": [tensor_shape_list(tensor) for tensor in inputs],
        "parameter_count_estimate": estimate_parameter_count(mod),
        "op_count_estimate": estimate_op_count(mod),
    }


def resolve_model_name(model: str, quantization_mode: str, model_size: str) -> str:
    if model in ("xnnpack_cnn_fp32", "cnn"):
        return "xnnpack_large_cnn_fp32" if model_size in ("medium", "large") else "xnnpack_tiny_cnn"
    if model in ("xnnpack_mlp_fp32", "mlp"):
        return "xnnpack_large_mlp_fp32"
    if model in ("xnnpack_qs8_cnn", "qs8_cnn"):
        return (
            "xnnpack_large_qs8_cnn"
            if model_size in ("medium", "large")
            else "xnnpack_static_qs8_tiny_cnn"
        )
    if quantization_mode == "static_qs8" and model == "xnnpack_tiny_cnn":
        return "xnnpack_static_qs8_tiny_cnn"
    return model


def load_model(args: argparse.Namespace, model_override: str | None = None):
    model = resolve_model_name(model_override or args.model, args.quantization_mode, args.model_size)
    if args.quantization_mode == "static_qs8" and model.startswith("torchvision:"):
        raise RuntimeError("torchvision models are only supported with --quantization-mode fp32")
    if model == "xnnpack_static_qs8_tiny_cnn" or (
        args.quantization_mode == "static_qs8" and model == "xnnpack_tiny_cnn"
    ):
        return load_static_qs8_tiny_cnn(args.seed)
    if model == "xnnpack_large_qs8_cnn":
        return load_large_static_qs8_cnn(args.seed)
    if model == "xnnpack_tiny_cnn":
        return load_tiny_cnn(args.seed)
    if model == "xnnpack_large_cnn_fp32":
        return load_large_cnn(args.seed)
    if model == "xnnpack_large_mlp_fp32":
        return load_large_mlp(args.seed)
    if model.startswith("torchvision:"):
        return load_torchvision_model(model.split(":", 1)[1], args.input_shape)
    raise RuntimeError(
        "supported models are "
        + ", ".join(IN_TREE_MODELS)
        + ", xnnpack_cnn_fp32, xnnpack_mlp_fp32, xnnpack_qs8_cnn, "
        + "and torchvision:<name>"
    )


def partition_for_xnnpack(mod: tvm.IRModule, args: argparse.Namespace):
    from tvm.relax.backend.xnnpack import partition_for_xnnpack as partition

    return partition(
        mod,
        precision=args.precision,
        partition_policy=args.partition_policy,
        layout=args.layout,
        min_subgraph_size=args.min_subgraph_size,
        min_compute_to_copy_ratio=args.min_compute_to_copy_ratio,
        allow_isolated_elementwise=args.allow_isolated_elementwise,
        allow_layout_rewrite=not args.disable_layout_rewrite,
        allow_cast_boundary=args.allow_cast_boundary,
        report_partition_decisions=args.report_partition_decisions,
    )


def summarize_partition_report(report: List[Dict[str, object]]) -> Dict[str, object]:
    accepted = sum(1 for entry in report if entry["accepted"])
    rejected = len(report) - accepted
    reasons: Dict[str, int] = {}
    totals = {
        "estimated_flops": 0.0,
        "copy_bytes": 0,
        "padded_copy_bytes": 0,
        "layout_transform_bytes": 0,
        "cast_bytes": 0,
    }
    for entry in report:
        reason = str(entry["reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
        for key in totals:
            totals[key] += entry.get(key, 0) or 0
    accepted_flops = sum(
        float(entry.get("estimated_flops", 0.0) or 0.0) for entry in report if entry["accepted"]
    )
    total_flops = float(totals["estimated_flops"])
    return {
        "candidates": len(report),
        "accepted": accepted,
        "rejected": rejected,
        "reasons": reasons,
        "totals": totals,
        "accepted_candidate_ratio": float(accepted) / float(len(report)) if report else 0.0,
        "accepted_flop_ratio": accepted_flops / total_flops if total_flops > 0 else 0.0,
    }


def platform_info() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def compile_vm(mod: tvm.IRModule, target: str) -> relax.VirtualMachine:
    executable = tvm.compile(mod, target=target)
    return relax.VirtualMachine(executable, tvm.cpu())


def benchmark_vm(
    vm: relax.VirtualMachine, args: List[tvm.runtime.Tensor], number: int, repeat: int
):
    vm["main"](*args)
    evaluator = vm.time_evaluator("main", tvm.cpu(), number=number, repeat=repeat)
    return evaluator(*args)


def format_result(result) -> Dict[str, object]:
    results = [float(x) for x in result.results]
    steady_state = results[1:] if len(results) > 1 else results
    return {
        "mean_ms": float(np.mean(results) * 1000.0),
        "median_ms": float(np.percentile(results, 50) * 1000.0),
        "p50_ms": float(np.percentile(results, 50) * 1000.0),
        "p90_ms": float(np.percentile(results, 90) * 1000.0),
        "p99_ms": float(np.percentile(results, 99) * 1000.0),
        "steady_state_mean_ms": float(np.mean(steady_state) * 1000.0),
        "raw_ms": [x * 1000.0 for x in results],
    }


def correctness_tolerance(precision: str, quantization_mode: str) -> Tuple[float, float]:
    if quantization_mode == "static_qs8":
        return 0.0, 1.0
    if precision == "fp32":
        return 1e-5, 1e-5
    return 5e-2, 5e-2


def summarize_profile_json(profile_json: str) -> Dict[str, object]:
    if not profile_json:
        return {"available": False}
    try:
        parsed = json.loads(profile_json)
    except json.JSONDecodeError:
        return {"available": True, "raw": profile_json}
    if isinstance(parsed, list):
        operators = parsed
    elif isinstance(parsed, dict):
        operators = parsed.get("operators", [])
    else:
        operators = []
    total_us = sum(float(op.get("time_us", 0.0) or 0.0) for op in operators)
    return {"available": True, "operator_count": len(operators), "total_time_us": total_us}


def parse_shape(shape: str) -> Tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape.replace("x", ",").split(",") if dim)
    if not dims:
        raise argparse.ArgumentTypeError("input shape must contain at least one dimension")
    return dims


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xnnpack_tiny_cnn")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--model-size",
        choices=("small", "medium", "large"),
        default="small",
        help="Size selector for model aliases such as xnnpack_cnn_fp32 and xnnpack_qs8_cnn.",
    )
    parser.add_argument(
        "--compare-models",
        default="",
        help="Comma-separated model names to benchmark sequentially.",
    )
    parser.add_argument("--dump-partition-report-json", default="")
    parser.add_argument("--target", default="llvm")
    parser.add_argument(
        "--quantization-mode",
        choices=("fp32", "static_qs8"),
        default="fp32",
        help="Benchmark graph family. Runtime precision remains controlled by --precision.",
    )
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input-shape", type=parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--xnnpack-prefix-info", default="")
    parser.add_argument("--use-weights-cache", action="store_true")
    parser.add_argument("--use-workspace", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dont-spin-workers", action="store_true")
    parser.add_argument("--transient-indirection-buffer", action="store_true")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--partition-policy",
        choices=("greedy", "cost", "debug_all_supported"),
        default="greedy",
    )
    parser.add_argument("--layout", choices=("auto", "NHWC", "preserve"), default="auto")
    parser.add_argument("--min-subgraph-size", type=int, default=2)
    parser.add_argument("--min-compute-to-copy-ratio", type=float, default=8.0)
    parser.add_argument("--allow-isolated-elementwise", action="store_true")
    parser.add_argument("--disable-layout-rewrite", action="store_true")
    parser.add_argument("--allow-cast-boundary", action="store_true")
    parser.add_argument("--report-partition-decisions", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16_hint", "fp16_force"),
        default="fp32",
        help="XNNPACK runtime precision policy. Does not rewrite TVM IR dtypes.",
    )
    return parser.parse_args(argv)


def available_models() -> List[str]:
    return [
        *IN_TREE_MODELS,
        "xnnpack_cnn_fp32",
        "xnnpack_mlp_fp32",
        "xnnpack_qs8_cnn",
        *(f"torchvision:{name}" for name in TORCHVISION_MODELS),
    ]


def run_benchmark(args: argparse.Namespace, model_override: str | None = None) -> Dict[str, Any]:
    xnnpack_enabled = has_xnnpack_enabled()
    xnnpack_options = {
        "use_weights_cache": args.use_weights_cache,
        "use_workspace": args.use_workspace,
        "profile": args.profile,
        "dont_spin_workers": args.dont_spin_workers,
        "transient_indirection_buffer": args.transient_indirection_buffer,
        "num_threads": args.num_threads,
        "precision": args.precision,
    }
    capabilities = get_xnnpack_capabilities()

    load_error = None
    metadata = {}
    try:
        mod, inputs, model_name = load_model(args, model_override)
        metadata = model_metadata(mod, inputs, model_name)
    except Exception as err:  # pylint: disable=broad-except
        mod, inputs, model_name = None, [], model_override or args.model
        load_error = str(err)
    effective_quantization_mode = (
        "static_qs8" if metadata.get("model_family") == "static_qs8" else args.quantization_mode
    )

    partition_count = 0
    correctness = "not run"
    baseline_status = "not run"
    xnnpack_status = "not run"
    baseline_timing = None
    byoc_timing = None
    baseline_error = None
    byoc_error = None
    byoc_first_run_ms = None
    byoc_compile_ms = None
    partition_report_summary = None
    profile_summary = None
    memory_before_kib = get_memory_kib()
    memory_after_kib = -1

    if mod is not None:
        try:
            baseline_vm = compile_vm(mod, args.target)
            baseline_output = baseline_vm["main"](*inputs)
            baseline_timing = format_result(
                benchmark_vm(baseline_vm, inputs, args.number, args.repeat)
            )
            baseline_status = "passed"
        except Exception as err:  # pylint: disable=broad-except
            baseline_error = str(err)
            baseline_output = None
            baseline_status = "failed"

        byoc_mod = None
        try:
            byoc_result = partition_for_xnnpack(mod, args)
            if args.report_partition_decisions:
                byoc_mod, partition_report = byoc_result
                partition_report_summary = summarize_partition_report(partition_report)
            else:
                byoc_mod = byoc_result
            partition_count = count_xnnpack_partitions(byoc_mod)
        except Exception as err:  # pylint: disable=broad-except
            byoc_error = str(err)
            correctness = "failed"
            xnnpack_status = "partition failed"

        if xnnpack_enabled and baseline_output is not None and byoc_mod is not None:
            try:
                if partition_count > 0:
                    compile_start = time.perf_counter()
                    byoc_mod = relax.transform.RunCodegen({"xnnpack": xnnpack_options})(byoc_mod)
                    byoc_vm = compile_vm(byoc_mod, args.target)
                    byoc_compile_ms = (time.perf_counter() - compile_start) * 1000.0
                    first_run_start = time.perf_counter()
                    byoc_output = byoc_vm["main"](*inputs)
                    byoc_first_run_ms = (time.perf_counter() - first_run_start) * 1000.0
                    rtol, atol = correctness_tolerance(args.precision, effective_quantization_mode)
                    tvm.testing.assert_allclose(
                        byoc_output.numpy(), baseline_output.numpy(), rtol=rtol, atol=atol
                    )
                    correctness = "passed"
                    xnnpack_status = "passed"
                    byoc_timing = format_result(
                        benchmark_vm(byoc_vm, inputs, args.number, args.repeat)
                    )
                    if args.profile and byoc_mod.attrs and "external_mods" in byoc_mod.attrs:
                        profile_entries = []
                        for ext_mod in byoc_mod.attrs["external_mods"]:
                            get_profile = ext_mod.get_function("get_profile_json", query_imports=True)
                            if get_profile is not None:
                                profile_entries.append(summarize_profile_json(get_profile()))
                        profile_summary = profile_entries
                else:
                    correctness = "not run: no XNNPACK partitions"
                    xnnpack_status = "no partitions"
            except Exception as err:  # pylint: disable=broad-except
                byoc_error = str(err)
                correctness = "failed"
                xnnpack_status = "failed"
        elif xnnpack_status != "partition failed":
            correctness = "not run: XNNPACK is not enabled"
            xnnpack_status = "disabled" if not xnnpack_enabled else "not run"
    memory_after_kib = get_memory_kib()

    result = {
        "model": model_name,
        "metadata": metadata,
        "platform": platform_info(),
        "architecture": platform.machine(),
        "target": args.target,
        "tvm_target": args.target,
        "precision": args.precision,
        "quantization_mode": effective_quantization_mode,
        "xnnpack_enabled": xnnpack_enabled,
        "xnnpack_capabilities": capabilities if capabilities else "not available",
        "xnnpack_runtime_options": xnnpack_options,
        "xnnpack_partition_policy": args.partition_policy,
        "xnnpack_partition_report": partition_report_summary or "not requested",
        "xnnpack_prefix_info": args.xnnpack_prefix_info or "not provided",
        "xnnpack_partitions": partition_count,
        "baseline_status": baseline_status,
        "xnnpack_status": xnnpack_status,
        "correctness": correctness,
        "load_error": load_error,
        "baseline_error": baseline_error,
        "byoc_error": byoc_error,
        "xnnpack_compile_and_codegen_ms": byoc_compile_ms,
        "xnnpack_first_run_ms": byoc_first_run_ms,
        "max_rss_delta_kib": (
            memory_after_kib - memory_before_kib
            if memory_before_kib >= 0 and memory_after_kib >= 0
            else "not available"
        ),
        "baseline_latency": baseline_timing or "not measured",
        "xnnpack_byoc_latency": byoc_timing or "not measured",
        "xnnpack_profile_summary": profile_summary or "not requested",
    }
    if baseline_timing is not None and byoc_timing is not None:
        result["speedup_vs_baseline_mean"] = (
            baseline_timing["mean_ms"] / byoc_timing["mean_ms"]
        )
    return result


def print_result(result: Dict[str, Any]) -> None:
    print(f"model: {result['model']}")
    metadata = result.get("metadata") or {}
    print(f"model_family: {metadata.get('model_family', 'unknown')}")
    print(f"fixture_size: {metadata.get('fixture_size', 'unknown')}")
    print(f"input_shapes: {metadata.get('input_shapes', 'unknown')}")
    print(f"parameter_count_estimate: {metadata.get('parameter_count_estimate', 'unknown')}")
    print(f"op_count_estimate: {metadata.get('op_count_estimate', 'unknown')}")
    print(f"platform: {result['platform']}")
    print(f"architecture: {result['architecture']}")
    print(f"target: {result['target']}")
    print(f"tvm_target: {result['tvm_target']}")
    print(f"precision: {result['precision']}")
    print(f"quantization_mode: {result['quantization_mode']}")
    print(f"xnnpack_enabled: {result['xnnpack_enabled']}")
    print(f"xnnpack_capabilities: {result['xnnpack_capabilities']}")
    print(f"xnnpack_runtime_options: {result['xnnpack_runtime_options']}")
    print(f"xnnpack_partition_policy: {result['xnnpack_partition_policy']}")
    print(f"xnnpack_partition_report: {result['xnnpack_partition_report']}")
    print(f"xnnpack_prefix_info: {result['xnnpack_prefix_info']}")
    print(f"xnnpack_partitions: {result['xnnpack_partitions']}")
    print(f"baseline_status: {result['baseline_status']}")
    print(f"xnnpack_status: {result['xnnpack_status']}")
    threading = (
        "threadpool=nullptr / caller-thread"
        if result["xnnpack_runtime_options"]["num_threads"] <= 1
        else f"private pthreadpool / {result['xnnpack_runtime_options']['num_threads']} threads"
    )
    print(f"threading: {threading}")
    print("layout_policy: NHWC only, no inserted transposes")
    print(f"correctness: {result['correctness']}")
    if result.get("load_error"):
        print(f"load_error: {result['load_error']}")
    if result.get("baseline_error"):
        print(f"baseline_error: {result['baseline_error']}")
    if result.get("byoc_error"):
        print(f"byoc_error: {result['byoc_error']}")
    print(f"xnnpack_compile_and_codegen_ms: {result['xnnpack_compile_and_codegen_ms'] or 'not measured'}")
    print(f"xnnpack_first_run_ms: {result['xnnpack_first_run_ms'] or 'not measured'}")
    print(f"max_rss_delta_kib: {result['max_rss_delta_kib']}")
    print(f"baseline_latency: {result['baseline_latency']}")
    print(f"xnnpack_byoc_latency: {result['xnnpack_byoc_latency']}")
    print(f"xnnpack_profile_summary: {result['xnnpack_profile_summary']}")
    if "speedup_vs_baseline_mean" in result:
        print(f"speedup_vs_baseline_mean: {result['speedup_vs_baseline_mean']:.6f}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        for model in available_models():
            print(model)
        return
    models = [model.strip() for model in args.compare_models.split(",") if model.strip()]
    if not models:
        models = [args.model]
    results = [run_benchmark(args, model) for model in models]
    for index, result in enumerate(results):
        if index:
            print("")
        print_result(result)
    if args.dump_partition_report_json:
        Path(args.dump_partition_report_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
