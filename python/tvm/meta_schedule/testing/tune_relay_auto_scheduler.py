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

import numpy as np  # type: ignore
import tvm
from tvm import auto_scheduler
from tvm import meta_schedule as ms
from tvm import relay
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.relay_workload import get_network


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
        "--rpc-workers",
        type=int,
        required=True,
    )
    args.add_argument(
        "--log-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=None,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    parsed.rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=3600,
    )
    return parsed


ARGS = _parse_args()


def main():
    log_file = os.path.join(ARGS.log_dir, f"{ARGS.workload}.json")

    runner = auto_scheduler.RPCRunner(
        key=ARGS.rpc_key,
        host=ARGS.rpc_host,
        port=ARGS.rpc_port,
        n_parallel=ARGS.rpc_workers,
        number=3,
        repeat=1,
        min_repeat_ms=100,  # TODO
        enable_cpu_cache_flush=False,  # TODO
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
    mod, params, (input_name, input_shape, input_dtype) = get_network(
        ARGS.workload,
        ARGS.input_shape,
        cache_dir=ARGS.cache_dir,
    )
    print(f"Workload: {ARGS.workload}")
    print(f"  input_name: {input_name}")
    print(f"  input_shape: {input_shape}")
    print(f"  input_dtype: {input_dtype}")
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod["main"],
        params,
        target=ARGS.target,
        hardware_params=hardware_params,
    )
    for idx, (task, task_weight) in enumerate(zip(tasks, task_weights)):
        print(f"==== Task {idx}: {task.desc} (weight {task_weight} key: {task.workload_key}) =====")
        print(task.compute_dag)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tuner.tune(
        auto_scheduler.TuningOptions(
            num_measure_trials=ARGS.num_trials,
            runner=runner,
            measure_callbacks=[
                auto_scheduler.RecordToFile(log_file),
            ],
        )
    )

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_auto_scheduler": True},
        ):
            lib = relay.build(
                mod,
                target=ARGS.target,
                params=params,
            )
    graph, rt_mod, params = lib.graph_json, lib.lib, lib.params
    if input_dtype.startswith("float"):
        input_data = np.random.uniform(size=input_shape).astype(input_dtype)
    else:
        input_data = np.random.randint(low=0, high=10000, size=input_shape, dtype=input_dtype)

    def f_timer(rt_mod, dev, input_data):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.graph_executor import GraphModule

        # pylint: enable=import-outside-toplevel

        mod = GraphModule(rt_mod["default"](dev))
        mod.set_input(input_name, input_data)
        ftimer = mod.module.time_evaluator(
            "run",
            dev,
            min_repeat_ms=500,
            repeat=3,
        )
        results = list(np.array(ftimer().results) * 1000.0)  # type: ignore
        print("Running time in time_evaluator: ", results)

    run_module_via_rpc(
        rpc_config=ARGS.rpc_config,
        lib=lib,
        dev_type=ARGS.target.kind.name,
        args=[input_data],
        continuation=f_timer,
    )

    def f_per_layer(rt_mod, dev, input_data):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.debugger.debug_executor import create

        # pylint: enable=import-outside-toplevel
        mod = create(graph, rt_mod, dev)
        mod.set_input(input_name, input_data)
        graph_nodes = [n["name"] for n in json.loads(graph)["nodes"]]
        graph_time = mod.run_individual(number=10, repeat=1, min_repeat_ms=5000)
        print("|graph_nodes| = ", len(graph_nodes))
        print("|graph_time| = ", len(graph_time))
        graph_nodes_time = {k: float(v) for k, v in zip(graph_nodes, graph_time)}
        for k, v in graph_nodes_time.items():
            print(f"{k} : {v:.3f}")

    run_module_via_rpc(
        rpc_config=ARGS.rpc_config,
        lib=rt_mod,
        dev_type=ARGS.target.kind.name,
        args=[input_data],
        continuation=f_per_layer,
    )


if __name__ == "__main__":
    main()
