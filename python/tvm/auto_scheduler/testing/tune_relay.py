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
# pylint: disable=missing-docstring
import argparse
import json
import os
from distutils.util import strtobool

import tvm
from tvm import auto_scheduler
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import create_timer, generate_input_data
from tvm.meta_schedule.utils import cpu_count
from tvm.support import describe


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
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
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--layout",
        type=str,
        default=None,
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=None,
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
        "--adaptive-training",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        default=True,
    )
    args.add_argument(
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    args.add_argument(
        "--backend",
        type=str,
        choices=["graph", "vm"],
        help="example: graph / vm",
        required=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=600,
    )
    return parsed


ARGS = _parse_args()


def main():
    log_file = os.path.join(ARGS.work_dir, f"{ARGS.workload}.json")

    runner = auto_scheduler.RPCRunner(
        key=ARGS.rpc_key,
        host=ARGS.rpc_host,
        port=ARGS.rpc_port,
        n_parallel=cpu_count(logical=True),
        number=ARGS.number,
        repeat=ARGS.repeat,
        min_repeat_ms=ARGS.min_repeat_ms,
        enable_cpu_cache_flush=ARGS.cpu_flush,
        timeout=ARGS.rpc_config.session_timeout_sec,
    )

    if ARGS.target.kind.name == "llvm":
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=int(ARGS.target.attrs["num-cores"]),
            target=ARGS.target,
        )
    elif ARGS.target.kind.name == "cuda":
        hardware_params = auto_scheduler.HardwareParams(
            num_cores=-1,
            vector_unit_bytes=16,
            cache_line_bytes=64,
            max_shared_memory_per_block=int(ARGS.target.attrs["max_shared_memory_per_block"]),
            max_threads_per_block=int(ARGS.target.attrs["max_threads_per_block"]),
            # The value `max_local_memory_per_block` is not used in AutoScheduler,
            # but is required by the API.
            max_local_memory_per_block=12345678,
            max_vthread_extent=8,
            warp_size=32,
        )
    else:
        raise NotImplementedError(f"Unsupported target {ARGS.target}")

    describe()
    print(f"Workload: {ARGS.workload}")
    mod, params, (input_name, input_shape, input_dtype) = get_network(
        ARGS.workload,
        ARGS.input_shape,
        layout=ARGS.layout,
        cache_dir=ARGS.cache_dir,
    )
    input_info = [
        {
            "name": input_name,
            "shape": input_shape,
            "dtype": input_dtype,
        },
    ]
    input_data = {
        item["name"]: generate_input_data(item["shape"], item["dtype"]) for item in input_info
    }
    for item in input_info:
        print(f"  input_name : {item['name']}")
        print(f"  input_shape: {item['shape']}")
        print(f"  input_dtype: {item['dtype']}")

    with ms.Profiler() as profiler:
        with ms.Profiler.timeit("TaskExtraction"):
            tasks, task_weights = auto_scheduler.extract_tasks(
                mod["main"],
                params,
                target=ARGS.target,
                hardware_params=hardware_params,
            )
            for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
                print(
                    f"==== Task {idx}: {task.desc} "
                    f"(weight {task_weight} key: {task.workload_key}) ====="
                )
                print(task.compute_dag)

        with ms.Profiler.timeit("Tuning"):
            if ARGS.num_trials > 0:
                tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
                tuner.tune(
                    auto_scheduler.TuningOptions(
                        num_measure_trials=ARGS.num_trials,
                        runner=runner,
                        measure_callbacks=[
                            auto_scheduler.RecordToFile(log_file),
                        ],
                    ),
                    adaptive_training=ARGS.adaptive_training,
                )

        relay_build = {"graph": relay.build, "vm": relay.vm.compile}[ARGS.backend]
        with ms.Profiler.timeit("PostTuningCompilation"):
            with auto_scheduler.ApplyHistoryBest(log_file):
                with tvm.transform.PassContext(
                    opt_level=3,
                    config={"relay.backend.use_auto_scheduler": True},
                ):
                    lib = relay_build(
                        mod,
                        target=ARGS.target,
                        params=params,
                    )
    print("Tuning Time:")
    print(profiler.table())

    run_module_via_rpc(
        rpc_config=ARGS.rpc_config,
        lib=lib,
        dev_type=ARGS.target.kind.name,
        args=input_data,
        continuation=create_timer(ARGS.backend),
        backend=ARGS.backend,
    )


if __name__ == "__main__":
    main()
