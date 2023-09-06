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
"""Benchmark script for ImageNet models on ARM CPU.
see README.md for the usage and results of this script.
"""
import argparse

import numpy as np

import tvm
from tvm import te
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_executor as runtime
from tvm import relay

from util import get_network, print_progress


def evaluate_network(network, target, target_host, repeat):
    # connect to remote device
    tracker = tvm.rpc.connect_tracker(args.host, args.port)
    remote = tracker.request(args.rpc_key)

    print_progress(network)
    net, params, input_shape, output_shape = get_network(network, batch_size=1)

    print_progress("%-20s building..." % network)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(net, target=tvm.target.Target(target, host=target_host), params=params)

    tmp = tempdir()
    if "android" in str(target):
        from tvm.contrib import ndk

        filename = "%s.so" % network
        lib.export_library(tmp.relpath(filename), fcompile=ndk.create_shared)
    else:
        filename = "%s.tar" % network
        lib.export_library(tmp.relpath(filename))

    # upload library and params
    print_progress("%-20s uploading..." % network)
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
        "%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res))
    )


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
    parser.add_argument(
        "--model",
        type=str,
        choices=["rk3399", "mate10", "mate10pro", "p20", "p20pro", "pixel2", "rasp3b", "pynq"],
        default="rk3399",
        help="The model of the test device. If your device is not listed in "
        "the choices list, pick the most similar one as argument.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--rpc-key", type=str, required=True)
    parser.add_argument("--repeat", type=int, default=10)
    args = parser.parse_args()

    dtype = "float32"

    if args.network is None:
        networks = ["squeezenet_v1.1", "mobilenet", "resnet-18", "vgg-16"]
    else:
        networks = [args.network]

    target = tvm.target.arm_cpu(model=args.model)
    target_host = None

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        evaluate_network(network, target, target_host, args.repeat)
