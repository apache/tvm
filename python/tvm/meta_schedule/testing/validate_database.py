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
import numpy as np

import tvm
from tvm.target import Target
from tvm.ir import IRModule
from tvm.tir import Schedule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.tune_utils import create_computer, generate_input_data
from tvm._ffi import get_global_func, register_func
from tvm.support import describe


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--path-workload",
        type=str,
        required=True,
        help="The path to the database workload file.",
    )
    args.add_argument(
        "--path-tuning-record",
        type=str,
        required=True,
        help="The path to the database tuning record file.",
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--baseline-target",
        type=str,
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
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)

# arg parser
ARGS = _parse_args()


@register_func("tvm.meta_schedule.testing.default_input_generator")
def default_input_generator(mod: IRModule) -> List[np.ndarray]:
    args_info = ms.arg_info.TensorInfo.from_prim_func(mod["main"])
    inputs = [
        generate_input_data(input_shape=arg_info.shape, input_dtype=arg_info.dtype)
        for arg_info in args_info
    ]
    return inputs


@register_func("tvm.meta_schedule.testing.default_check_metric")
def default_check_metric(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, rtol=1e-3, atol=2e-3)


def validate_correctness(
    original_mod: IRModule,  # compiled for "baseline_target"
    scheduled_mod: IRModule,  # compiled for "target"
    *,
    baseline_target: Union[str, Target],
    target: Union[str, Target],
    dev_type: str,
    rpc_config: ms.runner.RPCConfig,
    f_input_generator: Union[str, Callable] = "tvm.meta_schedule.testing.default_input_generator",
    f_check_metric: Union[str, Callable] = "tvm.meta_schedule.testing.default_check_metric",
) -> bool:
    """Function to validate the correctness of a scheduled module.

    Parameters
    ----------
    original_mod : IRModule
        The original module to be compiled.
    scheduled_mod : IRModule
        The scheduled module to be compiled.
    target : Target
        The target to compile the scheduled module.
    rpc_config : RPCConfig
        The RPCConfig to run the scheduled module.
    f_input_generator : Union[str, Callable]
        The function to generate the input data.
    f_check_metric : Union[str, Callable]
        The function to check the metric.

    Returns
    -------
    result : ...
        The result of the validation.
    """

    def build_and_run(mod: IRModule, target: Target, dev_type: str) -> np.ndarray:
        """Build and run the module on the target device."""
        rt_mod = tvm.build(mod, target=target)
        args = {i: arg for i, arg in enumerate(inputs)}
        return run_module_via_rpc(
            rpc_config=rpc_config,
            lib=rt_mod,
            dev_type=dev_type,
            args=args,
            continuation=create_computer(backend="tir"),
            backend="tir",
        )

    # make targets
    target = Target(target)
    baseline_target = Target(baseline_target)
    # fetch functions & prepare inputs
    if isinstance(f_input_generator, str):
        input_generator_func = get_global_func(f_input_generator)
    if isinstance(f_check_metric, str):
        check_metric_func = get_global_func(f_check_metric)
    inputs = input_generator_func(original_mod)
    # build & run original result
    original_res = build_and_run(original_mod, target=baseline_target, dev_type="cpu")
    scheduled_res = build_and_run(scheduled_mod, target=target, dev_type=dev_type)
    # check metric
    if not check_metric_func(original_res, scheduled_res):
        return True
    else:
        print(
            ("\n\n").join(
                [
                    "Validation failed!",
                    "Original Result:\n" + "-" * 10 + str(original_res),
                    "Scheduled Result:\n" + "-" * 10 + str(scheduled_res),
                    "Input:\n" + "-" * 10 + str(inputs),
                    "Original IRModule:\n" + "-" * 10 + original_mod.script(),
                    "Scheduled IRModule:\n" + "-" * 10 + scheduled_mod.script(),
                ]
            )
        )
        return False


def main():
    """Main function"""
    describe()
    database = ms.database.JSONDatabase(
        path_workload=ARGS.path_workload, path_tuning_record=ARGS.path_tuning_record
    )
    assert Target(ARGS.target).kind.name in ["llvm", "cuda"]
    dev_type = "cpu" if Target(ARGS.target).kind.name == "llvm" else "cuda"
    records = database.get_all_tuning_records()
    for i, record in enumerate(records):
        original_mod = record.workload.mod
        sch = Schedule(original_mod)
        scheduled_mod = record.trace.apply_to_schedule(sch=sch, remove_postproc=False)
        try:
            flag = validate_correctness(
                original_mod=original_mod,
                scheduled_mod=scheduled_mod,
                target=ARGS.target,
                baseline_target=ARGS.baseline_target,
                dev_type=dev_type,
                rpc_config=ARGS.rpc_config,
            )
        except Exception as e:  # pylint: disable=broad-except, invalid-name
            print(
                ("\n\n").join(
                    [
                        "Validation failed!",
                        "Original IRModule:\n" + "-" * 10 + original_mod.script(),
                        "Scheduled IRModule:\n" + "-" * 10 + scheduled_mod.script(),
                        "Exception\n" + "-" * 10 + str(e),
                    ]
                )
            )
        if flag:
            print(f"Progress {i+1: 6d} / {len(records): 6d} checked.")
        else:
            return

    print("Validation passed!")


if __name__ == "__main__":
    main()
