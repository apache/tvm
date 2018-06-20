""" Benchmark script for performance on GPUs.

For example, run the file with:
`python gpu_imagenet_bench.py --model=mobilenet --target=cuda`.
For more details about how to set up the inference environment on GPUs,
please refer to NNVM Tutorial: ImageNet Inference on the GPU
"""
import time
import argparse
import numpy as np
import tvm
import nnvm.compiler
import nnvm.testing
from tvm.contrib import util, nvcc
from tvm.contrib import graph_runtime as runtime

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['resnet', 'mobilenet'],
                        help="The model type.")
    parser.add_argument('--target', type=str, required=True,
                        choices=['cuda', 'rocm', 'opencl', 'metal'],
                        help="Compilation target.")
    parser.add_argument('--opt-level', type=int, default=1, help="Level of optimization.")
    parser.add_argument('--num-iter', type=int, default=1000, help="Number of iteration during benchmark.")
    parser.add_argument('--repeat', type=int, default=1, help="Number of repeative times.")
    args = parser.parse_args()
    opt_level = args.opt_level
    num_iter = args.num_iter
    ctx = tvm.context(args.target, 0)
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

    if args.target == "cuda":
        unroll = 1400
    else:
        unroll = 128
    with nnvm.compiler.build_config(opt_level=opt_level):
        with tvm.build_config(auto_unroll_max_step=unroll,
                              unroll_explicit=(args.target != "cuda")):
            graph, lib, params = nnvm.compiler.build(
                net, args.target, shape={"data": data_shape}, params=params)

    data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
    module = runtime.create(graph, lib, ctx)
    module.set_input(**params)
    module.set_input("data", data)
    module.run()
    out = module.get_output(0, tvm.nd.empty(out_shape))
    out.asnumpy()

    print('benchmark args: {}'.format(args))
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for i in range(args.repeat):
        prof_res = ftimer()
        print(prof_res)
        # sleep for avoiding device overheat
        if i + 1 != args.repeat:
            time.sleep(45)

if __name__ == '__main__':
    main()
