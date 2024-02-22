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
"""Benchmark script for various models on Adreno GPU.
"""
import argparse

import numpy as np

import os
import sys
import tvm
from tvm import te
from tvm.relay import testing
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_executor as runtime
from tvm import relay
from tvm import autotvm
from tvm.contrib import utils, ndk


def get_network(name, batch_size, dtype="float32"):
    """Get the symbol definition and random weight of a network

    Parameters
    ----------
    name: str
        The name of the network, can be 'resnet-18', 'resnet-50', 'vgg-16', 'inception_v3', 'mobilenet', ...
    batch_size: int
        batch size
    dtype: str
        Data type

    Returns
    -------
    net: tvm.IRModule
        The relay function of network definition
    params: dict
        The random parameters for benchmark
    input_shape: tuple
        The shape of input tensor
    output_shape: tuple
        The shape of output tensor
    """
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if name == "mobilenet":
        net, params = testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299)
        net, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif "resnet" in name:
        n_layer = int(name.split("-")[1])
        net, params = testing.resnet.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        net, params = testing.vgg.get_workload(
            num_layers=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "densenet" in name:
        n_layer = int(name.split("-")[1])
        net, params = testing.densenet.get_workload(
            densenet_size=n_layer, batch_size=batch_size, dtype=dtype
        )
    elif "squeezenet" in name:
        version = name.split("_v")[1]
        net, params = testing.squeezenet.get_workload(
            batch_size=batch_size, version=version, dtype=dtype
        )
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape


def print_progress(msg):
    """print progress message

    Parameters
    ----------
    msg: str
        The message to print
    """
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1024,
    early_stopping=None,
    log_filename="tuning.log",
):
    from tvm.autotvm.tuner import XGBTuner

    tmp_log_file = log_filename + ".tmp"

    for i, tsk in enumerate(reversed(tasks)):
        print("Task: ", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

        autotvm.record.pick_best(tmp_log_file, log_filename)


def evaluate_network(network, target, target_host, dtype, repeat):
    print_progress(network)
    net, params, input_shape, output_shape = get_network(network, batch_size=1, dtype=dtype)

    # Auto Tuning
    tune_log = "adreno-" + network + "-" + dtype + ".log"
    tuning_options = {
        "log_filename": tune_log,
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
            runner=autotvm.RPCRunner(
                args.rpc_key,
                host=args.host,
                port=args.port,
                number=3,
                timeout=600,
            ),
        ),
    }
    if args.tune:
        tasks = autotvm.task.extract_from_program(
            net, target=target, target_host=target_host, params=params
        )
        tune_tasks(tasks, **tuning_options)

    print_progress("%-20s building..." % network)

    # Build the tuning log
    if os.path.exists(tune_log):
        with autotvm.apply_history_best(tune_log):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(
                    net, target=tvm.target.Target(target, host=target_host), params=params
                )
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(
                net, target=tvm.target.Target(target, host=target_host), params=params
            )

    tmp = tempdir()

    filename = "%s.so" % network
    lib.export_library(tmp.relpath(filename), fcompile=ndk.create_shared)

    # upload library and params
    print_progress("%-20s uploading..." % network)

    # connect to remote device
    tracker = tvm.rpc.connect_tracker(args.host, args.port)
    remote = tracker.request(args.rpc_key)

    dev = remote.device(str(target), 0)
    remote.upload(tmp.relpath(filename))

    rlib = remote.load_module(filename)
    module = runtime.GraphModule(rlib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # evaluate
    print_progress("%-20s evaluating..." % network)
    ftimer = module.module.time_evaluator("run", dev, number=1, repeat=repeat)
    prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
    print(
        "%-20s %-19s (%s)"
        % (network + "-" + dtype, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )
    return (np.mean(prof_res), np.std(prof_res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        choices=[
            "resnet-18",
            "resnet-34",
            "resnet-50",
            "vgg-16",
            "vgg-19",
            "densenet-121",
            "inception_v3",
            "mobilenet",
            "squeezenet_v1.0",
            "squeezenet_v1.1",
        ],
        help="The name of neural network",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--rpc-key", type=str, default="android")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--tune", type=bool, default=False)
    args = parser.parse_args()

    if args.network is None:
        networks = [
            "resnet-18",
            "resnet-34",
            "resnet-50",
            "vgg-16",
            "vgg-19",
            "densenet-121",
            "inception_v3",
            "mobilenet",
            "squeezenet_v1.0",
            "squeezenet_v1.1",
        ]
    else:
        networks = [args.network]

    target = "opencl -device=adreno"
    target_host = "llvm -mtriple=arm64-linux-android"

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")

    results = {}

    for network in networks:
        ftime = evaluate_network(network, target, target_host, "float32", args.repeat)
        results[network + "-float32"] = ftime
        ftime = evaluate_network(network, target, target_host, "float16", args.repeat)
        results[network + "-float16"] = ftime

    print("----------------------------------------------------------------------")
    print("%-30s %-30s" % ("Network Name", "Mean Inference Time        (std dev)"))
    print("----------------------------------------------------------------------")
    for key, val in results.items():
        print("%-30s %-30s (%s)" % (key, "%.2f ms" % val[0], "%.2f ms" % val[1]))
