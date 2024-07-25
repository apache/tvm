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

import onnx  # type: ignore
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relay.frontend import from_onnx
from tvm.support import describe
from tvm.testing.utils import strtobool

from .tune_utils import create_timer, generate_input_data


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


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


def main():
    describe()
    print(f"Workload: {ARGS.model_name}")

    onnx_model = onnx.load(ARGS.onnx_path)
    shape_dict = {}
    for item in ARGS.input_shape:
        print(f"  input_name : {item['name']}")
        print(f"  input_shape: {item['shape']}")
        print(f"  input_dtype: {item['dtype']}")
        shape_dict[item["name"]] = item["shape"]
    mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)
    input_data = {
        item["name"]: generate_input_data(item["shape"], item["dtype"]) for item in ARGS.input_shape
    }

    with ms.Profiler() as profiler:
        database = ms.relay_integration.tune_relay(
            mod=mod,
            target=ARGS.target,
            params=params,
            work_dir=ARGS.work_dir,
            max_trials_global=ARGS.num_trials,
            num_trials_per_iter=64,
            runner=ms.runner.RPCRunner(  # type: ignore
                rpc_config=ARGS.rpc_config,
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=ARGS.number,
                    repeat=ARGS.repeat,
                    min_repeat_ms=ARGS.min_repeat_ms,
                    enable_cpu_cache_flush=ARGS.cpu_flush,
                ),
                alloc_repeat=1,
            ),
            cost_model=ms.cost_model.XGBModel(  # type: ignore
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=ARGS.adaptive_training,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
        )
        lib = ms.relay_integration.compile_relay(
            database=database,
            mod=mod,
            target=ARGS.target,
            params=params,
            backend=ARGS.backend,
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
