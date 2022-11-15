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
"""JSON Database validation script"""
import argparse
import logging
import warnings
from distutils.util import strtobool
from typing import Callable, Tuple, Union, List, Any

import numpy as np  # type: ignore

import tvm
from tvm import meta_schedule as ms
from tvm._ffi import get_global_func, register_func
from tvm.ir import IRModule
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.tune_utils import create_calculator, generate_input_data
from tvm.support import describe
from tvm.target import Target
from tvm.tir import Schedule
from tvm.tir.schedule import Trace
from tvm.tir.tensor_intrin import cuda, x86  # type: ignore # pylint: disable=unused-import

DELIMITOR = "\n" + "-" * 30 + "\n"


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="The path to the work directory containing database files.",
    )
    args.add_argument(
        "--target",
        type=Target,
        required=True,
    )
    args.add_argument(
        "--baseline-target",
        type=Target,
        default="llvm -num-cores=1",
        required=False,
        help="The baseline target to compile the original module.",
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
    )
    args.add_argument(
        "--number",
        type=int,
        default=3,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    args.add_argument(
        "--top-k",
        type=int,
        default=10**9,
        help="The number of top-k tuning records to validate for each unique original workload.",
    )
    args.add_argument(
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=600,
    )
    if parsed.cpu_flush and parsed.target.kind.name != "llvm":
        warnings.warn("cpu_flush is only supported on llvm target")
    return parsed


# logging
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

# arg parser
ARGS = _parse_args()


@register_func("tvm.meta_schedule.testing.default_input_generator")
def default_input_generator(mod: IRModule) -> List[tvm.nd.NDArray]:
    args_info = ms.arg_info.TensorInfo.from_prim_func(mod["main"])
    inputs = [
        tvm.nd.array(generate_input_data(input_shape=arg_info.shape, input_dtype=arg_info.dtype))
        for arg_info in args_info
    ]
    return inputs


@register_func("tvm.meta_schedule.testing.default_check_metric")
def default_check_metric(a: List[tvm.nd.NDArray], b: List[tvm.nd.NDArray]) -> bool:
    assert len(a) == len(b), "Different number of outputs from two modules"
    for i, _ in enumerate(a):
        if not np.allclose(a[i].numpy(), b[i].numpy(), rtol=1e-3, atol=2e-3):
            return False
    return True


def to_numpy(a: List[tvm.nd.NDArray]) -> List[np.ndarray]:
    """Convert a list of TVM NDArray to a list of numpy array"""
    assert a is not None, "Empty result cannot be converted to numpy"
    return [x.numpy() for x in a]


def to_tvm_ndarray(a: List[np.ndarray]) -> List[tvm.nd.NDArray]:
    """Convert a list of numpy array to a list of TVM NDArray"""
    assert a is not None, "Empty result cannot be converted to TVM NDArray"
    return [tvm.nd.array(x) for x in a]


def is_failed_record(record: ms.database.TuningRecord) -> bool:
    """Check if a tuning record is failed."""
    return len(record.run_secs) == 1 and record.run_secs[0] == 1e9


def print_result(
    counter: int,
    total: int,
    result: str,
    time: float,
    *,
    original_mod: IRModule = None,
    scheduled_mod: IRModule = None,
    inputs: List[np.ndarray] = None,
    original_res: List[np.ndarray] = None,
    scheduled_res: List[np.ndarray] = None,
    exception: Exception = None,
    trace: Trace = None,
) -> None:
    """Print the validation result."""
    output = [
        f"Progress {counter: 6d} / {total: 6d} checked, "
        f"used {float(time): 3.3f} sec. Result: {result}"
    ]
    if result not in ["pass", "skip"]:
        output.extend(
            [
                "Original IRModule:" + DELIMITOR + original_mod.script(),
                "Scheduled IRModule:" + DELIMITOR + scheduled_mod.script(),
                "Trace" + DELIMITOR + str(trace),
                "Input:" + DELIMITOR + str(inputs),
            ]
        )
        if result == "wrong answer":
            output.extend(
                [
                    "Original Result:" + DELIMITOR + str(original_res),
                    "Scheduled Result:" + DELIMITOR + str(scheduled_res),
                    "Max Diff:"
                    + DELIMITOR
                    + str(
                        [
                            np.max(np.abs(original_res[i] - scheduled_res[i]))
                            for i in range(len(original_res))
                        ]
                    ),
                ]
            )
        elif result == "exception":
            output.extend(
                [
                    "Exception:" + DELIMITOR + str(exception),
                ]
            )
        else:
            raise ValueError(f"Unknown result: {result}")
    print("\n\n".join(output))


def check_and_run(func: Union[str, Callable], *args, **kwargs) -> Any:
    """Check if the function is a string or a callable, and run it."""
    if isinstance(func, str):
        func = get_global_func(func)
    return func(*args, **kwargs)  # type: ignore


def build_and_run(
    mod: IRModule,
    target: Target,
    rpc_config: ms.runner.RPCConfig,
    dev_type: str,
    inputs: List[np.ndarray],
) -> List[np.ndarray]:
    """Build and run the module on the target device."""
    rt_mod = tvm.build(mod, target=target)
    return to_numpy(
        run_module_via_rpc(
            rpc_config=rpc_config,
            lib=rt_mod,
            dev_type=dev_type,
            args={i: v for i, v in enumerate(inputs)},  # pylint: disable=unnecessary-comprehension
            continuation=create_calculator(backend="tir"),
            backend="tir",
        )
    )


def validate_correctness(
    original_mod: IRModule,  # compiled for "baseline_target"
    scheduled_mod: IRModule,  # compiled for "target"
    *,
    baseline_target: Target,
    target: Target,
    dev_type: str,
    rpc_config: ms.runner.RPCConfig,
    inputs: List[np.ndarray] = None,  # for input reuse
    original_res: List[np.ndarray] = None,  # for original mod results reuse
    f_input_generator: Union[
        str, Callable[[IRModule], List[tvm.nd.NDArray]]
    ] = default_input_generator,
    f_check_metric: Union[
        str, Callable[[tvm.nd.NDArray, tvm.nd.NDArray], bool]
    ] = default_check_metric,
) -> Tuple[bool, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Function to validate the correctness of a scheduled module.

    Parameters
    ----------
    original_mod : IRModule
        The original module to be compiled.
    scheduled_mod : IRModule
        The scheduled module to be compiled.
    baseline_target : Target
        The baseline target to compile the original module.
    target : Target
        The target to compile the scheduled module.
    dev_type : str
        The device type to run the module via rpc.
    rpc_config : RPCConfig
        The RPCConfig to run the scheduled module.
    inputs : List[np.ndarray]
        The input data to be reused, if None, generate new inputs.
    original_res : List[np.ndarray]
        The original module results to be reused, if None, run the original module.
    f_input_generator : Union[str, Callable]
        The function to generate the input data.
    f_check_metric : Union[str, Callable]
        The function to check the metric.

    Returns
    -------
    passed: bool
        Whether the validation passed.
    inputs: List[np.ndarray]
        The input data used for validation in numpy array.
    original_res: List[np.ndarray]
        The original module results in numpy array.
    scheduled_res: List[np.ndarray]
        The scheduled module results in numpy array.
    """

    # fetch input function & prepare inputs
    if inputs is None:
        inputs = to_numpy(check_and_run(f_input_generator, original_mod))

    # build & run original result
    if original_res is None:
        original_res = build_and_run(
            original_mod,
            target=baseline_target,
            rpc_config=rpc_config,
            dev_type="cpu",
            inputs=inputs,
        )
    scheduled_res = build_and_run(
        scheduled_mod, target=target, rpc_config=rpc_config, dev_type=dev_type, inputs=inputs
    )

    # fetch comparison function
    validation_res = check_and_run(f_check_metric, original_res, scheduled_res)

    # check metric
    return (
        validation_res,
        inputs,
        original_res,
        scheduled_res,
    )


def main():
    """Main function"""
    describe()
    target = ARGS.target
    database = ms.database.create(work_dir=ARGS.work_dir)

    # determine target kind
    if target.kind.name == "llvm":
        dev_type = "cpu"
    elif target.kind.name == "cuda":
        dev_type = "cuda"
    else:
        raise RuntimeError(f"Unsupported target kind: {target.kind.name}")

    # start profiling
    with ms.Profiler() as profiler:
        # collect records
        with profiler.timeit("collect records"):
            records = database.get_all_tuning_records()
            print(
                f"Total {len(records)} records to be validated. "
                f"Collected in {float(profiler.get()['collect records']): 3.3f} sec."
            )

        # collect unique original TIR
        with profiler.timeit("deduplicate records"):
            workloads = dict()
            for record in records:
                mod = record.workload.mod
                s_hash = tvm.ir.structural_hash(mod)
                if s_hash not in workloads:
                    workloads[s_hash] = [mod]
                else:
                    duplicate = False
                    for previous_mod in workloads[hash]:
                        if tvm.ir.structural_equal(mod, previous_mod):
                            duplicate = True
                            break
                    if not duplicate:
                        workloads[hash].append(mod)
            # put the workload into a list
            unique_workloads = []
            for _, mods in workloads.items():
                unique_workloads.extend(mods)
            print(
                f"Total {len(unique_workloads)} unique original TIR to be validated. "
                f"Deduplicated in {float(profiler.get()['deduplicate records']): 3.3f} sec."
            )

        # validate correctness
        counter = 0
        for original_mod in unique_workloads:
            records = database.get_top_k(workload=database.commit_workload(mod), top_k=10**9)
            inputs = None
            original_res = None
            for record in records:
                counter += 1
                scope_name = f"validate #{counter}"
                if is_failed_record(record):
                    # skip failed records where run_secs is 1e9
                    # these records are only negative samples for cost model
                    print_result(counter + 1, total=len(records), result="skip", time=0.0)
                    continue
                try:
                    with profiler.timeit(scope_name):
                        # prepare scheduled module
                        sch = Schedule(original_mod)
                        record.trace.apply_to_schedule(sch=sch, remove_postproc=False)
                        scheduled_mod = sch.mod
                        # validate correctness
                        passed, inputs, original_res, scheduled_res = validate_correctness(
                            original_mod=original_mod,
                            scheduled_mod=scheduled_mod,
                            target=target,
                            baseline_target=ARGS.baseline_target,
                            dev_type=dev_type,
                            rpc_config=ARGS.rpc_config,
                            inputs=inputs,
                            original_res=original_res,
                        )
                    # validation finished
                    print_result(
                        counter + 1,
                        total=len(records),
                        result="pass" if passed else "wrong answer",
                        time=profiler.get()[scope_name],
                        original_mod=original_mod,
                        scheduled_mod=scheduled_mod,
                        trace=record.trace,
                        inputs=inputs,
                        original_res=original_res,
                        scheduled_res=scheduled_res,
                    )
                except Exception as e:  # pylint: disable=broad-except, invalid-name
                    # validation failed with exception
                    print_result(
                        counter + 1,
                        total=len(records),
                        result="exception",
                        time=profiler.get()[scope_name],
                        original_mod=original_mod,
                        scheduled_mod=scheduled_mod,
                        trace=record.trace,
                        exception=e,
                    )
        print("Validation finished!")
        print(f"Total time spent: {float(profiler.get()['Total']): 3.3f} sec.")


if __name__ == "__main__":
    main()
