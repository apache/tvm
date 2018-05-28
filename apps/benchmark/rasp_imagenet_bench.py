""" Benchmark script for performance on Raspberry Pi. For example, run the file with:
`python rasp_imagenet_bench.py --model='modbilenet' --host='rasp0' --port=9090`. For
more details about how to set up the inference environment on Raspberry Pi, Please
refer to NNVM Tutorial: Deploy the Pretrained Model on Raspberry Pi """
import time
import argparse
import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util, rpc
from tvm.contrib import graph_runtime as runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['resnet', 'mobilenet'],
        help="The model type.")
    parser.add_argument('--host', type=str, required=True, help="The host address of your Raspberry Pi.")
    parser.add_argument('--port', type=int, required=True, help="The port number of your Raspberry Pi.")
    parser.add_argument('--opt-level', type=int, default=1, help="Level of optimization.")
    parser.add_argument('--num-iter', type=int, default=50, help="Number of iteration during benchmark.")
    args = parser.parse_args()

    opt_level = args.opt_level

    num_iter = args.num_iter
    batch_size = 1
    num_classes = 1000
    image_shape = (3, 224, 224)

    data_shape = (batch_size,) + image_shape
    out_shape = (batch_size, num_classes)
    if args.model == 'resnet':
        net, params = nnvm.testing.resnet.get_workload(
            batch_size=1, image_shape=image_shape)
    elif args.model == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(
            batch_size=1, image_shape=image_shape)
    else:
        raise ValueError('no benchmark prepared for {}.'.format(args.model))


    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            net, tvm.target.rasp(), shape={"data": data_shape}, params=params)

    tmp = util.tempdir()
    lib_fname = tmp.relpath('net.o')
    lib.save(lib_fname)

    remote = rpc.connect(args.host, args.port)
    remote.upload(lib_fname)

    ctx = remote.cpu(0)
    rlib = remote.load_module('net.o')
    rparams = {k: tvm.nd.array(v, ctx) for k, v in params.items()}

    module = runtime.create(graph, rlib, ctx)
    module.set_input('data', tvm.nd.array(np.random.uniform(size=(data_shape)).astype("float32")))
    module.set_input(**rparams)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape, ctx=ctx))
    out.asnumpy()

    print('benchmark args: {}'.format(args))
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(3):
        prof_res = ftimer()
        print(prof_res)
        # sleep for avoiding cpu overheat
        time.sleep(45)


if __name__ == '__main__':
    main()
