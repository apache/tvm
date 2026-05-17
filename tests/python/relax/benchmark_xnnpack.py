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
import sys
import time
from typing import Dict, List, Tuple

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

    model = getattr(torchvision.models, model_name)(weights=None).eval()
    example_input = torch.zeros(input_shape, dtype=torch.float32)
    with torch.no_grad():
        exported = export(model, (example_input,))
        mod = from_exported_program(exported, keep_params_as_input=False)

    input_np = np.zeros(input_shape, dtype="float32")
    return mod, [tvm.runtime.tensor(input_np)], f"torchvision:{model_name}"


def partition_for_xnnpack(mod: tvm.IRModule) -> tvm.IRModule:
    from tvm.relax.backend.xnnpack import partition_for_xnnpack as partition

    return partition(mod)


def compile_vm(mod: tvm.IRModule, target: str) -> relax.VirtualMachine:
    executable = tvm.compile(mod, target=target)
    return relax.VirtualMachine(executable, tvm.cpu())


def benchmark_vm(vm: relax.VirtualMachine, args: List[tvm.runtime.Tensor], number: int, repeat: int):
    vm["main"](*args)
    evaluator = vm.time_evaluator("main", tvm.cpu(), number=number, repeat=repeat)
    return evaluator(*args)


def format_result(result) -> Dict[str, object]:
    results = [float(x) for x in result.results]
    return {
        "mean_ms": float(np.mean(results) * 1000.0),
        "median_ms": float(np.median(results) * 1000.0),
        "raw_ms": [x * 1000.0 for x in results],
    }


def parse_shape(shape: str) -> Tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape.replace("x", ",").split(",") if dim)
    if not dims:
        raise argparse.ArgumentTypeError("input shape must contain at least one dimension")
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="xnnpack_tiny_cnn")
    parser.add_argument("--target", default="llvm")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xnnpack_enabled = has_xnnpack_enabled()
    xnnpack_options = {
        "use_weights_cache": args.use_weights_cache,
        "use_workspace": args.use_workspace,
        "profile": args.profile,
        "dont_spin_workers": args.dont_spin_workers,
        "transient_indirection_buffer": args.transient_indirection_buffer,
        "num_threads": args.num_threads,
    }
    capabilities = get_xnnpack_capabilities()

    load_error = None
    try:
        if args.model == "xnnpack_tiny_cnn":
            mod, inputs, model_name = load_tiny_cnn(args.seed)
        elif args.model.startswith("torchvision:"):
            model = args.model.split(":", 1)[1]
            mod, inputs, model_name = load_torchvision_model(model, args.input_shape)
        else:
            raise RuntimeError("supported models are xnnpack_tiny_cnn and torchvision:<name>")
    except Exception as err:  # pylint: disable=broad-except
        mod, inputs, model_name = None, [], args.model
        load_error = str(err)

    partition_count = 0
    correctness = "not run"
    baseline_timing = None
    byoc_timing = None
    byoc_error = None
    byoc_first_run_ms = None
    byoc_compile_ms = None
    memory_before_kib = get_memory_kib()
    memory_after_kib = -1

    if mod is not None:
        baseline_vm = compile_vm(mod, args.target)
        baseline_output = baseline_vm["main"](*inputs)
        baseline_timing = format_result(benchmark_vm(baseline_vm, inputs, args.number, args.repeat))

        if xnnpack_enabled:
            try:
                byoc_mod = partition_for_xnnpack(mod)
                partition_count = count_xnnpack_partitions(byoc_mod)
                if partition_count > 0:
                    compile_start = time.perf_counter()
                    byoc_mod = relax.transform.RunCodegen({"xnnpack": xnnpack_options})(byoc_mod)
                    byoc_vm = compile_vm(byoc_mod, args.target)
                    byoc_compile_ms = (time.perf_counter() - compile_start) * 1000.0
                    first_run_start = time.perf_counter()
                    byoc_output = byoc_vm["main"](*inputs)
                    byoc_first_run_ms = (time.perf_counter() - first_run_start) * 1000.0
                    tvm.testing.assert_allclose(
                        byoc_output.numpy(), baseline_output.numpy(), rtol=1e-5, atol=1e-5
                    )
                    correctness = "passed"
                    byoc_timing = format_result(
                        benchmark_vm(byoc_vm, inputs, args.number, args.repeat)
                    )
                else:
                    correctness = "not run: no XNNPACK partitions"
            except Exception as err:  # pylint: disable=broad-except
                byoc_error = str(err)
                correctness = "failed"
        else:
            correctness = "not run: XNNPACK is not enabled"
    memory_after_kib = get_memory_kib()

    print(f"model: {model_name}")
    print(f"target: {args.target}")
    print(f"xnnpack_enabled: {xnnpack_enabled}")
    print(f"xnnpack_capabilities: {capabilities if capabilities else 'not available'}")
    print(f"xnnpack_runtime_options: {xnnpack_options}")
    print(f"xnnpack_prefix_info: {args.xnnpack_prefix_info or 'not provided'}")
    print(f"xnnpack_partitions: {partition_count}")
    threading = (
        "threadpool=nullptr / caller-thread"
        if args.num_threads <= 1
        else f"private pthreadpool / {args.num_threads} threads"
    )
    print(f"threading: {threading}")
    print("layout_policy: NHWC only, no inserted transposes")
    print(f"correctness: {correctness}")
    if load_error:
        print(f"load_error: {load_error}")
    if byoc_error:
        print(f"byoc_error: {byoc_error}")
    print(f"xnnpack_compile_and_codegen_ms: {byoc_compile_ms if byoc_compile_ms is not None else 'not measured'}")
    print(f"xnnpack_first_run_ms: {byoc_first_run_ms if byoc_first_run_ms is not None else 'not measured'}")
    if memory_before_kib >= 0 and memory_after_kib >= 0:
        print(f"max_rss_delta_kib: {memory_after_kib - memory_before_kib}")
    else:
        print("max_rss_delta_kib: not available")
    print(f"baseline_latency: {baseline_timing if baseline_timing is not None else 'not measured'}")
    print(f"xnnpack_byoc_latency: {byoc_timing if byoc_timing is not None else 'not measured'}")
    if baseline_timing is not None and byoc_timing is not None:
        speedup = baseline_timing["mean_ms"] / byoc_timing["mean_ms"]
        print(f"speedup_vs_baseline_mean: {speedup:.6f}")


if __name__ == "__main__":
    main()
