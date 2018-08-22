"""Benchmark script for ARM CPU.
see README.md for the usage and results of this script.
"""
import argparse

import numpy as np

import tvm
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
import nnvm.compiler
import nnvm.testing

from util import get_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, choices=
                        ['resnet-18', 'resnet-34', 'vgg-16', 'mobilenet', 'squeezenet v1.1', ])
    parser.add_argument("--device", type=str, required=True, choices=
                        ['rk3399', 'mate10', 'mate10pro', 'p20', 'p20pro',
                         'pixel2', 'rasp3b', 'pynq'])
    parser.add_argument("--host", type=str, default='localhost')
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--rpc-key", type=str, required=True)
    parser.add_argument("--number", type=int, default=6)
    args = parser.parse_args()

    dtype = 'float32'

    if args.network is None:
        networks = ['squeezenet_v1.1', 'mobilenet', 'resnet-18', 'vgg-16']
    else:
        networks = [args.network]

    target = tvm.target.arm_cpu(model=args.device)

    # connect to remote device
    tracker = tvm.rpc.connect_tracker(args.host, args.port)
    remote = tracker.request(args.rpc_key)

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        net, params, input_shape, output_shape = get_network(network, batch_size=1)

        with nnvm.compiler.build_config(opt_level=2, add_pass=['AlterOpLayout']):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, shape={'data': input_shape}, params=params, dtype=dtype)

        tmp = tempdir()
        if 'android' in str(target):
            from tvm.contrib import ndk
            filename = "%s.so" % network
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "%s.tar" % network
            lib.export_library(tmp.relpath(filename))

        # upload library and params
        ctx = remote.context(str(target), 0)
        remote.upload(tmp.relpath(filename))
        rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

        rlib = remote.load_module(filename)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**rparams)

        # evaluate
        ftimer = module.module.time_evaluator("run", ctx, number=args.number, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
        print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
