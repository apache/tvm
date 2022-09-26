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
"""
This script is for benchmarking TVM performance on models from TorchBench.
It uses the TorchDynamo as the frontend to ingest models into TVM, and it also
leverages the benchmark util from TorchDynamo.

TorchDynamo (https://github.com/pytorch/torchdynamo) and TorchBench
(https://github.com/pytorch/benchmark) need to be in the parent directory of TVM.
We need a local clone of these repos because torchbench and the benchmark runner
in TorchDynamo isn't designed to be used as a Python package.

To setup the environment, run the following commands in the parent directory of TVM and with
the appropriate Python environment:
```bash
# torchdynamo requires nightly pytorch. If it fails to find the specified version, try
# installing the latest nightly pytorch.
pip3 install --pre \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu116 \
    torch==1.13.0.dev20220926 \
    torchvision==0.14.0.dev20220926 \
    torchtext==0.14.0.dev20220926

git clone https://github.com/pytorch/torchdynamo
pushd torchdynamo
git checkout c537639f9712621dc04ca09908796dbbe86c354b
pip install -e .
popd

sudo apt install git-lfs  # git lfs is used for TorchBench
git clone https://github.com/pytorch/benchmark
pushd benchmark
python install.py --continue_on_fail  # fambench_xlmr might fail to install
popd
```

To run a benchmark, the script can be run under 'tune' mode by
```bash
python python/tvm/meta_schedule/testing/torchbench/run.py \
    --mode tune \
    --model resnet50 \
    --target "nvidia/geforce-rtx-3070" \
    --work-dir /path/to/work/dir/ \
    --num-trials 20000 \
    --rpc-host <rpc tracker host for tuning> \
    --rpc-port <rpc tracker port for tuning> \
    --rpc-key <rpc key> \
```

All available target tags (like nvidia/geforce-rtx-3070) can be found at
https://github.com/apache/tvm/blob/main/src/target/tag.cc

Then the script can be run under 'eval' mode to actual benchmark the performance,
using the tuning database under the work directory. This can be executed on a different
machine than the one executes tuning (the database json files need to be inside
of the work directory).
```bash
python python/tvm/meta_schedule/testing/torchbench/run.py \
    --mode eval \
    --model resnet50 \
    --target "nvidia/geforce-rtx-3070" \
    --work-dir /path/to/work/dir/ \
    --num-trials 0
```

Alternatively, both tuning and evaluation can be done in a single run on the same machine,
by
```bash
python python/tvm/meta_schedule/testing/torchbench/run.py \
    --mode all \
    --model resnet50 \
    --target "llvm -num-cores 6" \
    --work-dir /path/to/work/dir/ \
    --num-trials 0
```
"""
# pylint: disable=logging-format-interpolation
import argparse
import functools
import logging
import warnings
from enum import Enum
from typing import Callable, List, Tuple

import numpy as np  # type: ignore
import torch  # type: ignore
import tvm
import tvm.relay
from scipy.stats import ttest_ind  # type: ignore
from tvm import meta_schedule as ms
from tvm.contrib.graph_executor import GraphModule
from tvm.meta_schedule.testing.torchbench.utils import (
    load_torchdynamo_benchmark_runner,
    same,
    timed,
)
from tvm.runtime.vm import VirtualMachine
from tvm.support import describe

# Needs to be imported after the .utils is executed
import torchdynamo  # type: ignore  # isort: skip, pylint: disable=wrong-import-order


class RunMode(Enum):
    """
    The running mode of this script. Available values are:
    - tune: Only tune the model and create the tuning database.
    - eval: Only benchmark model using pre-existing tuning database.
    - all: Run both tuning and benchmark
    """

    ALL = "all"
    TUNE = "tune"
    EVAL = "eval"

    @property
    def should_tune(self):
        """
        Returns whether it should tune the model.
        """
        return self != RunMode.EVAL

    @property
    def should_eval(self):
        """
        Returns whether it should actually benchmark the model.
        """
        return self != RunMode.TUNE


class ResultComparisonMetric(Enum):
    """
    This changes how it compares the results with the expected value during
    accuracy check.
    - cosine: Use the cosine similarity. It should be greater than 0.99.
    - allclose-1e-4: Use the max elementwise absolute difference. It should be less than 1e-4.
    """

    COSINE = "cosine"
    ALLCLOSE = "allclose-1e-4"


def parse_args():
    """
    Parse arguments
    """
    args = argparse.ArgumentParser()

    args.add_argument(
        "--mode",
        type=RunMode,
        default=RunMode.ALL,
        help=RunMode.__doc__,
    )
    args.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="The batch size of model input. Use TorchBench's default value if not specified.",
    )
    args.add_argument(
        "--result-metric",
        type=ResultComparisonMetric,
        default=ResultComparisonMetric.ALLCLOSE,
        help=ResultComparisonMetric.__doc__,
    )
    args.add_argument(
        "--benchmark-repeat",
        type=int,
        default=10,
        help="The number of times to repeat the benchmark measurement.",
    )
    args.add_argument(
        "--benchmark-warmup-rounds",
        type=int,
        default=5,
        help="The number of rounds to warmup before starting to measure the performance.",
    )

    # Model selection
    args.add_argument(
        "--model",
        type=str,
        required=True,
        help="""
        The name of model to run. It should a directory name under 
        https://github.com/pytorch/benchmark/tree/main/torchbenchmark/models.
        """,
    )

    # Tuning-related config
    args.add_argument(
        "--target",
        type=tvm.target.Target,
        required=True,
        help="The target to tune and run benchmark for.",
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="""
        The working directory to save intermediate results and store databases for compilation.
        """,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="The max number of trials to run MetaSchedule.",
    )
    args.add_argument(
        "--max-trials-per-task",
        type=int,
        default=None,
        help="""
        The max number of trials to run per task extracted in MetaSchedule. 
        By default it's the same as --num-trials.
        """,
    )
    args.add_argument(
        "--backend",
        type=str,
        choices=["graph", "vm"],
        default="graph",
        help="The backend to use for relay compilation(graph / vm).",
    )
    # TODO(@yelite): Add a layout arg to transform the network after
    # ingesting into Relay and before feeding into MetaSchedule.

    # Evaluator-related config
    args.add_argument(
        "--number",
        type=int,
        default=3,
        help="The number of times to run the model for taking average in a single measurement.",
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="The number of times to repeat the measurement.",
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
        help="""
        Minimum repeat time in ms. The number of runs will be increased if the actual
        repeat time is lowered than this.
        """,
    )
    args.add_argument(
        "--adaptive-training",
        action="store_true",
        help="Whether to use adaptive training for cost model.",
    )
    args.add_argument(
        "--cpu-flush",
        action="store_true",
        help="Whether to perform CPU cache flush.",
    )

    # RPC-related args
    args.add_argument(
        "--rpc-host",
        type=str,
        help="Host of the RPC Tracker for tuning. Use LocalRunner if not provided",
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        help="Port of the RPC Tracker for tuning",
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        help="Key of the RPC Tracker for tuning",
    )

    parsed = args.parse_args()
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = parse_args()
IS_CUDA = ARGS.target.kind.name == "cuda"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)


runner = load_torchdynamo_benchmark_runner(  # pylint: disable=invalid-name
    IS_CUDA,
    cosine_similarity=ARGS.result_metric == ResultComparisonMetric.COSINE,
)


def get_meta_schedule_runner() -> ms.runner.PyRunner:
    """
    Get the Runner for MetaSchedule.

    It returns RPCRunner if --rpc-host is given, otherwise it returns LocalRunner
    """
    if ARGS.rpc_host is not None:
        assert ARGS.rpc_port is not None, "Missing rpc_port"
        assert ARGS.rpc_key is not None, "Missing rpc_key"
        return ms.runner.RPCRunner(
            rpc_config=ms.runner.RPCConfig(
                tracker_host=ARGS.rpc_host,
                tracker_port=ARGS.rpc_port,
                tracker_key=ARGS.rpc_key,
                session_timeout_sec=600,
            ),
            evaluator_config=ms.runner.EvaluatorConfig(
                number=ARGS.number,
                repeat=ARGS.repeat,
                min_repeat_ms=ARGS.min_repeat_ms,
                enable_cpu_cache_flush=ARGS.cpu_flush,
            ),
            alloc_repeat=1,
        )
    else:
        warnings.warn("Falling back to MetaSchedule LocalRunner because --rpc-host isn't provided.")
        return ms.runner.LocalRunner()


def get_graph_executor_forward(mod: GraphModule, device: tvm.runtime.Device) -> Callable:
    """
    Get the forward function for graph executor, in order to integrate with TorchDynamo.
    """

    def forward(*args):
        if IS_CUDA:
            torch.cuda.synchronize()
        args = tuple(arg.contiguous() for arg in args)
        for idx, arg in enumerate(args, 0):
            mod.set_input(
                f"inp_{idx}",
                tvm.nd.from_dlpack(arg),
            )
        mod.run()
        device.sync()
        result = [torch.from_dlpack(mod.get_output(i)) for i in range(mod.get_num_outputs())]
        return result

    return forward


def get_vm_forward(virtual_machine: VirtualMachine, device: tvm.runtime.Device) -> Callable:
    """
    Get the forward function for VM, in order to integrate with TorchDynamo.
    """

    def forward(*args):
        if IS_CUDA:
            torch.cuda.synchronize()
        args = tuple(tvm.nd.from_dlpack(arg.contiguous()) for arg in args)
        result = virtual_machine.invoke("main", *args)
        device.sync()

        if isinstance(result, tvm.nd.NDArray):
            result = [result]
        return [torch.from_dlpack(m) for m in result]

    return forward


def create_tvm_task_collection_backend(tasks: List[ms.ExtractedTask]) -> Callable:
    """
    This torchdynamo backend only collects the extracted tasks from MetaSchedule.
    It doesn't tune the model.
    """

    def backend(graph_module, example_inputs):
        jit_mod = torch.jit.trace(graph_module, example_inputs)
        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
        ir_mod, params = tvm.relay.frontend.from_pytorch(jit_mod, shape_list)

        extracted_tasks = ms.relay_integration.extract_tasks(
            mod=ir_mod,
            target=ARGS.target,
            params=params,
        )
        logger.info("Extracted %d tasks", len(extracted_tasks))
        tasks.extend(extracted_tasks)

        return graph_module.forward

    return backend


def create_tvm_compilation_backend(database: ms.database.Database) -> Callable:
    """
    This torchdynamo backend compiles the model using history best record from the
    MetaSchedule database.
    """

    def backend(graph_module, example_inputs):
        jit_mod = torch.jit.trace(graph_module, example_inputs)
        shape_list = [(f"inp_{idx}", i.shape) for idx, i in enumerate(example_inputs)]
        ir_mod, params = tvm.relay.frontend.from_pytorch(jit_mod, shape_list)

        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=ir_mod,
            target=ARGS.target,
            params=params,
            backend=ARGS.backend,
        )
        device = tvm.cuda(0) if IS_CUDA else tvm.cpu(0)

        if ARGS.backend == "graph":
            mod = GraphModule(lib["default"](device))
            return get_graph_executor_forward(mod, device)
        elif ARGS.backend == "vm":
            vm = VirtualMachine(lib, device)  # pylint: disable=invalid-name
            return get_vm_forward(vm, device)
        else:
            raise RuntimeError(f"Unknown backend {ARGS.backend}")

    return backend


def format_time(seconds: float) -> str:
    """
    Format elapsed time based on its value.
    """
    if seconds > 1:
        return f"{seconds:.3g}s"
    else:
        return f"{seconds * 1000:.3g}ms"


def is_output_correct(output: torch.Tensor, expected: torch.Tensor) -> bool:
    """
    Check whether the output is correct.
    """
    comparison_metric = ARGS.result_metric
    if comparison_metric == ResultComparisonMetric.COSINE:
        return same(expected, output, cosine_similarity=True)
    elif comparison_metric == ResultComparisonMetric.ALLCLOSE:
        return same(expected, output, tol=1e-4)
    else:
        raise RuntimeError(f"Unknown comparison metric {comparison_metric}")


def performance_experiment(
    model_iter_fn: Callable,
    model: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor],
) -> str:
    """
    Performs the actual benchmarking
    Simplified from https://github.com/pytorch/torchdynamo/blob/c537639f9712621dc04ca09908796dbbe86c354b/benchmarks/common.py#L494 pylint: disable=line-too-long
    """
    timings = np.zeros((ARGS.benchmark_repeat, 2), np.float64)

    is_correct = True

    frozen_model_iter_fn = torchdynamo.run(model_iter_fn)

    for _ in range(ARGS.benchmark_warmup_rounds):
        frozen_model_iter_fn(model, example_inputs)
        model_iter_fn(model, example_inputs)

    for rep in range(ARGS.benchmark_repeat):
        # interleave the runs to handle frequency scaling and load changes
        timings[rep, 0], expected_output = timed(
            model, model_iter_fn, example_inputs, return_result=True
        )
        timings[rep, 1], actual_output = timed(
            model, frozen_model_iter_fn, example_inputs, return_result=True
        )
        is_correct = is_correct and is_output_correct(expected_output, actual_output)

    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    logger.info(
        f"eager:{format_time(median[0])} "
        f"optimized:{format_time(median[1])} "
        f"speedup:{speedup:.3f}x p:{pvalue:.3f}"
    )
    if not is_correct:
        logger.error("Result is incorrect.")
        logger.error(f"Expected (PyTorch eager): {expected_output}")
        logger.error(f"Actual (Optimized): {actual_output}")

    return ""


def get_torch_device_type(target: tvm.target.Target) -> str:
    if target.kind.name == "llvm":
        return "cpu"
    elif target.kind.name == "cuda":
        return "cuda"
    else:
        raise RuntimeError(f"Unsupported target {target}")


def main():
    """
    Entry point of the benchmark
    """
    describe()

    database = ms.database.JSONDatabase(work_dir=ARGS.work_dir)
    if not ARGS.mode.should_tune:
        if len(database) == 0:
            raise RuntimeError(
                "Script is running in eval mode while the tuning database is empty. "
                "Please tune the model first."
            )

    if IS_CUDA and ARGS.cpu_flush:
        warnings.warn(
            "Benchmark is running on CUDA, while --cpu-flush is turned on. "
            "This flag will have no effect on CUDA."
        )
        ARGS.cpu_flush = False

    try:
        _, name, model, example_inputs, batch_size = runner.load_model(
            get_torch_device_type(ARGS.target),
            ARGS.model,
            batch_size=ARGS.batch_size,
        )
        logger.info(
            f"batch size: {batch_size} input shape: {[input.shape for input in example_inputs]}"
        )
    except NotImplementedError:
        logging.exception(f"{ARGS.model} failed to load")
        return

    if ARGS.mode.should_tune:
        extracted_tasks: List[ms.ExtractedTask] = []
        task_collect_ctx = torchdynamo.optimize(create_tvm_task_collection_backend(extracted_tasks))
        task_collect_ctx(runner.model_iter_fn)(model, example_inputs)
        tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
            extracted_tasks=extracted_tasks,
            work_dir=ARGS.work_dir,
        )
        database = ms.tune.tune_tasks(
            tasks=tasks,
            task_weights=task_weights,
            work_dir=ARGS.work_dir,
            max_trials_global=ARGS.num_trials,
            max_trials_per_task=ARGS.num_trials_per_task,
            runner=get_meta_schedule_runner(),  # type: ignore
            database=database,
            cost_model=ms.cost_model.XGBModel(  # type: ignore
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=ARGS.adaptive_training,
            ),
        )

    if ARGS.mode.should_eval:
        torchdynamo.reset()
        model_compile_ctx = torchdynamo.optimize(create_tvm_compilation_backend(database))
        experiment = functools.partial(performance_experiment, runner.model_iter_fn)
        runner.run_one_model(name, model, example_inputs, model_compile_ctx, experiment)


if __name__ == "__main__":
    main()
