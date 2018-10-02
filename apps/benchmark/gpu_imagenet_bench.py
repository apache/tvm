"""Benchmark script for ImageNet models on GPU.
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
                        ['resnet-18', 'resnet-34', 'resnet-50',
                         'vgg-16', 'vgg-19', 'densenet-121', 'inception_v3',
                         'mobilenet', 'mobilenet_v2', 'squeezenet_v1.0', 'squeezenet_v1.1'],
                        help='The name of neural network')
    parser.add_argument("--model", type=str,
                        choices=['1080ti', 'titanx', 'gfx900'], default='1080ti',
                        help="The model of the test device. If your device is not listed in "
                             "the choices list, pick the most similar one as argument.")
    parser.add_argument("--number", type=int, default=500)
    parser.add_argument("--target", type=str,
                        choices=['cuda', 'opencl', 'rocm', 'nvptx', 'metal'], default='cuda',
                        help="The tvm compilation target")
    args = parser.parse_args()

    dtype = 'float32'

    if args.network is None:
        networks = ['resnet-50', 'mobilenet', 'vgg-19', 'inception_v3']
    else:
        networks = [args.network]

    target = tvm.target.create('%s -model=%s' % (args.target, args.model))

    print("--------------------------------------------------")
    print("%-20s %-20s" % ("Network Name", "Mean Inference Time (std dev)"))
    print("--------------------------------------------------")
    for network in networks:
        net, params, input_shape, output_shape = get_network(network, batch_size=1)

        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, shape={'data': input_shape}, params=params, dtype=dtype)

        # create runtime
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        ftimer = module.module.time_evaluator("run", ctx, number=args.number, repeat=3)
        prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
        print("%-20s %-19s (%s)" % (network, "%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
