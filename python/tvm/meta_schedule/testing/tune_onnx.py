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

from distutils.util import strtobool
import numpy as np  # type: ignore
import onnx  # type: ignore
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relay.frontend import from_onnx
from tvm.support import describe


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model-name",
        type=str,
        required=True,
    )
    args.add_argument(
        "--onnx-path",
        type=str,
        required=True,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help='example: `[{"name": "input1", "dtype": "int64", "shape": [1, 1, 8]}]',
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
        required=True,
        help="example: `True / False",
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


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)
ARGS = _parse_args()


def main():
    describe()
    print(f"Workload: {ARGS.model_name}")
    onnx_model = onnx.load(ARGS.onnx_path)
    shape_dict = {}
    for item in ARGS.input_shape:
        print(f"  input_name: {item['name']}")
        print(f"  input_shape: {item['shape']}")
        print(f"  input_dtype: {item['dtype']}")
        shape_dict[item["name"]] = item["shape"]
    mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)
    runner = ms.runner.RPCRunner(
        rpc_config=ARGS.rpc_config,
        evaluator_config=ms.runner.EvaluatorConfig(
            number=ARGS.number,
            repeat=ARGS.repeat,
            min_repeat_ms=ARGS.min_repeat_ms,
            enable_cpu_cache_flush=ARGS.cpu_flush,
        ),
        alloc_repeat=1,
    )
    with ms.Profiler() as profiler:
        lib = ms.tune_relay(
            mod=mod,
            target=ARGS.target,
            config=ms.TuneConfig(
                strategy="evolutionary",
                num_trials_per_iter=64,
                max_trials_per_task=ARGS.num_trials,
                max_trials_global=ARGS.num_trials,
            ),
            runner=runner,  # type: ignore
            work_dir=ARGS.work_dir,
            params=params,
        )
    print("Tuning Time:")
    print(profiler.table())
    graph, rt_mod, params = lib.graph_json, lib.lib, lib.params
    input_data = {}
    for item in ARGS.input_shape:
        input_name, input_shape, input_dtype = item["name"], item["shape"], item["dtype"]
        if input_dtype.startswith("float"):
            input_data[input_name] = np.random.uniform(size=input_shape).astype(input_dtype)
        else:
            input_data[input_name] = np.random.randint(
                low=0, high=10000, size=input_shape, dtype=input_dtype
            )

    def f_timer(rt_mod, dev, input_data):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.graph_executor import GraphModule

        # pylint: enable=import-outside-toplevel

        mod = GraphModule(rt_mod["default"](dev))
        for input_name, input_value in input_data.items():
            mod.set_input(input_name, input_value)
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
        args=input_data,
        continuation=f_timer,
    )

    def f_per_layer(rt_mod, dev, input_data):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.debugger.debug_executor import create

        # pylint: enable=import-outside-toplevel
        mod = create(graph, rt_mod, dev)
        for input_name, input_value in input_data.items():
            mod.set_input(input_name, input_value)
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
        args=input_data,
        continuation=f_per_layer,
    )


if __name__ == "__main__":
    main()
