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
import warnings
from distutils.util import strtobool

import tvm
from tvm import autotvm
from tvm import meta_schedule as ms
from tvm import relay
from tvm.autotvm.graph_tuner import DPTuner
from tvm.autotvm.tuner import XGBTuner
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.tune_utils import create_timer, generate_input_data
from tvm.support import describe


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
        help="The name of the workload to tune. Supported models: "
        "https://github.com/apache/tvm/blob/main/python/tvm/meta_schedule/testing/relay_workload.py#L303-L322",  # pylint: disable=line-too-long
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
        help="The input shape of the workload. Example: '[1, 3, 224, 224]'",
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
        help="The target device to tune. "
        "Example: 'aws/cpu/c5.9xlarge', 'nvidia/nvidia-v100', 'nvidia/geforce-rtx-3090'",
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
        help="The number of trials per kernel. Example: 800",
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        required=True,
        help="The host address of the RPC tracker. Example: 192.168.6.66",
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        required=True,
        help="The port of the RPC tracker. Example: 4445",
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        required=True,
        help="The key of the RPC tracker. Example: '3090ti'",
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help="The working directory to store the tuning logs. Example: '/tmp/tune_relay'",
    )
    args.add_argument(
        "--layout",
        type=str,
        default=None,
        help="The layout of the workload. Example: 'NCHW', 'NHWC'",
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
        "--cpu-flush",
        type=lambda x: bool(strtobool(x)),
        help="example: True / False",
        required=True,
    )
    args.add_argument(
        "--graph-tuner",
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
    if ARGS.target.kind.name != "llvm" and ARGS.graph_tuner:
        raise ValueError("GraphTuner only supports llvm target")
    if ARGS.target.kind.name != "llvm" and ARGS.cpu_flush:
        raise ValueError("cpu_flush only supports llvm target")
    if ARGS.target.kind.name == "llvm" and not ARGS.cpu_flush:
        warnings.warn("cpu_flush is not enabled for llvm target")

    log_file = os.path.join(ARGS.work_dir, f"{ARGS.workload}.json")
    graph_opt_sch_file = os.path.join(ARGS.work_dir, f"{ARGS.workload}_graph_opt.log")
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            key=ARGS.rpc_key,
            host=ARGS.rpc_host,
            port=ARGS.rpc_port,
            number=ARGS.number,
            repeat=ARGS.repeat,
            min_repeat_ms=ARGS.min_repeat_ms,
            enable_cpu_cache_flush=ARGS.cpu_flush,
        ),
    )
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
            # extract workloads from relay program
            tasks = autotvm.task.extract_from_program(
                mod["main"],
                target=ARGS.target,
                params=params,
                ops=(
                    relay.op.get("nn.conv2d"),
                    relay.op.get("nn.conv3d"),
                    relay.op.get("nn.conv2d_transpose"),
                    relay.op.get("nn.dense"),
                    relay.op.get("nn.batch_matmul"),
                ),
            )
            for i, task in enumerate(tasks):
                print(f"Task {i} {task.name}: {task}")

        with ms.Profiler.timeit("Tuning"):
            if ARGS.num_trials > 0:
                for i, task in enumerate(tasks):
                    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
                    tuner_obj = XGBTuner(task, loss_type="rank")
                    n_trial = min(len(task.config_space), ARGS.num_trials)
                    tuner_obj.tune(
                        n_trial=n_trial,
                        early_stopping=800,
                        measure_option=measure_option,
                        callbacks=[
                            autotvm.callback.progress_bar(n_trial, prefix=prefix),
                            autotvm.callback.log_to_file(log_file),
                        ],
                    )
                if ARGS.graph_tuner:
                    executor = DPTuner(
                        graph=mod["main"],
                        input_shapes={input_name: input_shape},
                        records=log_file,
                        target_ops=[
                            relay.op.get("nn.conv2d"),
                        ],
                        target=ARGS.target,
                    )
                    executor.benchmark_layout_transform(min_exec_num=1000)
                    executor.run()
                    executor.write_opt_sch2record_file(graph_opt_sch_file)

        relay_build = {"graph": relay.build, "vm": relay.vm.compile}[ARGS.backend]
        with ms.Profiler.timeit("PostTuningCompilation"):
            if ARGS.graph_tuner:
                ctx = autotvm.apply_graph_best(graph_opt_sch_file)
            else:
                ctx = autotvm.apply_history_best(log_file)
            with ctx:
                print("compile...")
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay_build(mod, target=ARGS.target, params=params)
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
