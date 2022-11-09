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
from typing import Union, Callable, List
from distutils.util import strtobool
import argparse
import logging
import warnings
import numpy as np  # type: ignore

import tvm
from tvm.target import Target
from tvm.ir import IRModule
from tvm.tir import Schedule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.tune_utils import create_calculator, generate_input_data
from tvm._ffi import get_global_func, register_func
from tvm.support import describe

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


def validate_correctness(
    original_mod: IRModule,  # compiled for "baseline_target"
    scheduled_mod: IRModule,  # compiled for "target"
    *,
    baseline_target: Target,
    target: Target,
    dev_type: str,
    rpc_config: ms.runner.RPCConfig,
    f_input_generator: Union[
        str, Callable[[IRModule], List[tvm.nd.NDArray]]
    ] = default_input_generator,
    f_check_metric: Union[
        str, Callable[[tvm.nd.NDArray, tvm.nd.NDArray], bool]
    ] = default_check_metric,
) -> bool:
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
    f_input_generator : Union[str, Callable]
        The function to generate the input data.
    f_check_metric : Union[str, Callable]
        The function to check the metric.

    Returns
    -------
    result : bool
        The result of the validation.
    """

    def to_numpy(a: List[tvm.nd.NDArray]) -> List[np.ndarray]:
        """Convert a list of TVM NDArray to a list of numpy array"""
        assert a is not None, "Empty result cannot be converted to numpy"
        return [x.numpy() for x in a]

    def to_tvm_ndarray(a: List[np.ndarray]) -> List[tvm.nd.NDArray]:
        """Convert a list of numpy array to a list of TVM NDArray"""
        assert a is not None, "Empty result cannot be converted to TVM NDArray"
        return [tvm.nd.array(x) for x in a]

    def build_and_run(mod: IRModule, target: Target, dev_type: str) -> np.ndarray:
        """Build and run the module on the target device."""
        rt_mod = tvm.build(mod, target=target)
        return run_module_via_rpc(
            rpc_config=rpc_config,
            lib=rt_mod,
            dev_type=dev_type,
            args={i: v for i, v in enumerate(inputs)},  # pylint: disable=unnecessary-comprehension
            continuation=create_calculator(backend="tir"),
            backend="tir",
        )

    # fetch functions & prepare inputs
    if isinstance(f_input_generator, str):
        f_input_generator = get_global_func(f_input_generator)
    if isinstance(f_check_metric, str):
        f_check_metric = get_global_func(f_check_metric)
    inputs = to_numpy(f_input_generator(original_mod))  # type: ignore
    # build & run original result
    original_res = to_numpy(build_and_run(original_mod, target=baseline_target, dev_type="cpu"))
    scheduled_res = to_numpy(build_and_run(scheduled_mod, target=target, dev_type=dev_type))
    # check metric
    if f_check_metric(to_tvm_ndarray(original_res), to_tvm_ndarray(scheduled_res)):  # type: ignore
        return True
    else:
        print(
            ("\n\n").join(
                [
                    "Validation failed!",
                    "Original Result:" + DELIMITOR + str(original_res),
                    "Scheduled Result:" + DELIMITOR + str(scheduled_res),
                    "Input:" + DELIMITOR + str(inputs),
                    "Original IRModule:" + DELIMITOR + original_mod.script(),
                    "Scheduled IRModule:" + DELIMITOR + scheduled_mod.script(),
                ]
            )
        )
        return False


def main():
    """Main function"""
    describe()
    database = ms.database.create(work_dir=ARGS.work_dir)
    target = ARGS.target
    if target.kind.name == "llvm":
        dev_type = "cpu"
    elif target.kind.name == "cuda":
        dev_type = "cuda"
    else:
        raise RuntimeError(f"Unsupported target kind: {target.kind.name}")
    records = database.get_all_tuning_records()
    with ms.Profiler() as profiler:
        for i, record in enumerate(records):
            scope_name = f"validate #{i}"
            with profiler.timeit(scope_name):
                original_mod = record.workload.mod
                sch = Schedule(original_mod)
                record.trace.apply_to_schedule(sch=sch, remove_postproc=False)
                scheduled_mod = sch.mod
                is_success = False
                try:
                    is_success = validate_correctness(
                        original_mod=original_mod,
                        scheduled_mod=scheduled_mod,
                        target=target,
                        baseline_target=ARGS.baseline_target,
                        dev_type=dev_type,
                        rpc_config=ARGS.rpc_config,
                    )
                except Exception as e:  # pylint: disable=broad-except, invalid-name
                    print(
                        ("\n\n").join(
                            [
                                "Validation failed!",
                                "Original IRModule:" + DELIMITOR + original_mod.script(),
                                "Scheduled IRModule:" + DELIMITOR + scheduled_mod.script(),
                                "Exception" + DELIMITOR + str(e),
                            ]
                        )
                    )
            if is_success:
                print(
                    f"Progress {i+1: 6d} / {len(records): 6d} checked,"
                    f" used {float(profiler.get()[scope_name]): 3.3f} sec."
                )
            else:
                return

    print("Validation passed!")
    print(f"Total time spent: {float(profiler.get()['Total']): 3.3f} sec.")


if __name__ == "__main__":
    main()
