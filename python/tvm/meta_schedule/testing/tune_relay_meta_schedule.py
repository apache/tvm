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
import logging
import os

import numpy as np  # type: ignore
import tvm
from tvm import meta_schedule as ms
from tvm.ir.transform import PassContext
from tvm.meta_schedule.integration import extract_task_from_relay
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.relay import build as relay_build


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
        "--work-dir",
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


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def tune_each_task(
    mod,
    target,
    config,
    runner,
    work_dir,
    params,
):
    extracted_tasks = extract_task_from_relay(mod, target, params)
    database = ms.database.JSONDatabase(
        path_workload=os.path.join(work_dir, "default_database_workload.json"),
        path_tuning_record=os.path.join(work_dir, "default_database_tuning_record.json"),
    )
    for task in extracted_tasks:
        # pylint: disable=protected-access
        tune_context = ms.tune.Parse._tune_context(
            tune_context=None,
            mod=ms.tune.Parse._mod(task.dispatched[0]),
            target=target,
            config=config,
            task_name=task.task_name,
            space_generator=None,
            sch_rules=None,
            postprocs=None,
            mutator_probs=None,
            num_threads=os.cpu_count(),
        )
        task_scheduler = ms.tune.Parse._task_scheduler(
            None,
            [tune_context],
            builder=ms.tune.Parse._builder(None),
            runner=ms.tune.Parse._runner(runner),
            database=database,
            cost_model=ms.tune.Parse._cost_model(None),
            measure_callbacks=ms.tune.Parse._callbacks(None),
        )
        # pylint: enable=protected-access
        task_scheduler.tune()
    with target, ms.integration.ApplyHistoryBest(database):
        with PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            return relay_build(mod, target=target, params=params)


def main():
    mod, params, (input_name, input_shape, input_dtype) = get_network(
        ARGS.workload,
        ARGS.input_shape,
        cache_dir=ARGS.cache_dir,
    )
    print(f"Workload: {ARGS.workload}")
    print(f"  input_name: {input_name}")
    print(f"  input_shape: {input_shape}")
    print(f"  input_dtype: {input_dtype}")
    alloc_repeat = 1
    runner = ms.runner.RPCRunner(
        rpc_config=ARGS.rpc_config,
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        ),
        alloc_repeat=alloc_repeat,
        max_workers=ARGS.rpc_workers,
    )
    lib = tune_each_task(  # or ms.tune_relay
        mod=mod,
        target=ARGS.target,
        config=ms.EvolutionarySearchConfig(
            num_trials_per_iter=64,
            num_trials_total=ARGS.num_trials,
            init_min_unmeasured=50,
        ),
        runner=runner,  # type: ignore
        work_dir=ARGS.work_dir,
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
