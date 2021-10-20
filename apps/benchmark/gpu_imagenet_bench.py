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
"""Benchmark script for ImageNet models on GPU.
see README.md for the usage and results of this script.
"""
import argparse
import threading

import numpy as np

import tvm
from tvm import te
import tvm.contrib.graph_executor as runtime
from tvm import relay

from util import get_network


def benchmark(network, target):
    net, params, input_shape, output_shape = get_network(network, batch_size=1)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(net, target=target, params=params)

    # create runtime
    dev = tvm.device(str(target), 0)
    module = runtime.GraphModule(lib["default"](dev))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input("data", data_tvm)

    # evaluate
    ftimer = module.module.time_evaluator("run", dev, number=1, repeat=args.repeat)
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
        "--device",
        type=str,
        choices=["amd_apu"],
        default="amd_apu",
        help="The name of the test device. If your device is not listed in "
        "the choices list, pick the most similar one as argument.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["1080ti", "titanx", "tx2", "gfx900", "v1000"],
        default="1080ti",
        help="The model of the test device. If your device is not listed in "
        "the choices list, pick the most similar one as argument.",
    )
    parser.add_argument("--repeat", type=int, default=600)
    parser.add_argument(
        "--target",
        type=str,
        choices=["cuda", "opencl", "rocm", "nvptx", "metal", "vulkan"],
        default="cuda",
        help="The tvm compilation target",
    )
    parser.add_argument("--thread", type=int, default=1, help="The number of threads to be run.")
    args = parser.parse_args()

    dtype = "float32"

    if args.network is None:
        networks = ["resnet-50", "mobilenet", "vgg-19", "inception_v3"]
    else:
        networks = [args.network]

    target = tvm.target.Target("%s -device=%s -model=%s" % (args.target, args.device, args.model))

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        if args.thread == 1:
            benchmark(network, target)
        else:
            threads = list()
            for n in range(args.thread):
                thread = threading.Thread(
                    target=benchmark, args=([network, target]), name="thread%d" % n
                )
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()
