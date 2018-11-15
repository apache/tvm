import time
import numpy as np
import argparse
import nnvm
import tvm
from tvm.contrib import graph_runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch

batch_size = 1

models = ['resnet18_v1',
          'resnet34_v1',
          'resnet50_v1',
          'resnet101_v1',
          'resnet152_v1',
          'resnet18_v2',
          'resnet34_v2',
          'resnet50_v2',
          'resnet101_v2',
          'resnet152_v2',
          'vgg11',
          'vgg13',
          'vgg16',
          'vgg19',
          'vgg11_bn',
          'vgg13_bn',
          'vgg16_bn',
          'vgg19_bn',
          'alexnet',
          'densenet121',
          'densenet161',
          'densenet169',
          'densenet201',
          'squeezenet1.0',
          'squeezenet1.1',
          'inceptionv3',
          'mobilenet1.0',
          'mobilenet0.75',
          'mobilenet0.5',
          'mobilenet0.25',
          'mobilenetv2_1.0',
          'mobilenetv2_0.75',
          'mobilenetv2_0.5',
          'mobilenetv2_0.25']


def get_data_shape(model_name):
    if model_name.startswith('inception'):
        return (batch_size, 3, 299, 299)
    else:
        return (batch_size, 3, 224, 224)


def get_tvm_workload(network, **kwargs):
    from nnvm.frontend import from_mxnet
    from mxnet.gluon.model_zoo.vision import get_model
    block = get_model(network, **kwargs)
    if network.startswith('resnet152'):
        import sys
        sys.setrecursionlimit(10000)
    sym, params = from_mxnet(block)
    sym = nnvm.sym.softmax(sym)
    return sym, params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark TVM')
    parser.add_argument('--ext-accel', type=str, default='none', choices=['none', 'tensorrt'])
    parser.add_argument('--network', type=str, required=True, choices=models)
    parser.add_argument('--cuda-arch', type=str, required=True, choices=['sm_37', 'sm_70', 'sm_53', 'sm_62'])
    parser.add_argument('--target-host', type=str, required=True, choices=['x86_64-linux-gnu', 'aarch64-linux-gnu'])
    parser.add_argument('--compile', dest='compile', action='store_true')
    parser.add_argument('--run', dest='run', action='store_true')
    parser.set_defaults(compile=False)
    parser.set_defaults(run=False)
    args = parser.parse_args()

    network = args.network
    num_classes = 1000
    data_shape = get_data_shape(network)
    ext_accel = None if args.ext_accel == 'none' else args.ext_accel
    cuda_arch = args.cuda_arch
    set_cuda_target_arch(cuda_arch)
    target_host = 'llvm -target=%s' % args.target_host

    if args.compile:
        net, params = get_tvm_workload(network, pretrained=True)
        net = nnvm.graph.create(net)
        print("===========Saving graph for model %s" % network)
        with open('%s.json' % network, "w") as fo:
            fo.write(net.json())
        opt_level = 3
        target = tvm.target.cuda()
        print("===========Start to compile %s graph with params, external accelerator: %s" % (network, ext_accel))
        start = time.time()
        with nnvm.compiler.build_config(opt_level=opt_level, ext_accel=ext_accel):
            graph, lib, params = nnvm.compiler.build(
                net, target, shape={"data": data_shape}, params=params, target_host=target_host)
        print("===========Compiling model %s took %.3fs" % (network, time.time() - start))

        print("===========Saving lowered graph for model %s" % network)
        with open('%s_ext_accel_%s_%s.json' % (network, ext_accel, cuda_arch), "w") as fo:
            fo.write(graph.json())
        print("===========Saving module for model %s" % network)
        if lib.is_empty():
            print("lib is empty")
        else:
            print("lib is not empty")
        lib.export_library('%s_ext_accel_%s_%s.tar' % (network, ext_accel, cuda_arch))
        print("===========Saving params for model %s" % network)
        with open('%s_ext_accel_%s_%s.params' % (network, ext_accel, cuda_arch), 'wb') as fo:
            fo.write(nnvm.compiler.save_param_dict(params))

    if args.run:
        print("===========Starting to load model %s" % network)
        loaded_json = open('%s_ext_accel_%s_%s.json' % (network, ext_accel, cuda_arch)).read()
        loaded_lib = tvm.module.load('%s_ext_accel_%s_%s.tar' % (network, ext_accel, cuda_arch))
        loaded_params = bytearray(open('%s_ext_accel_%s_%s.params' % (network, ext_accel, cuda_arch), 'rb').read())
        ctx = tvm.gpu()
        np.random.seed(3342902)
        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        data = tvm.nd.array(data)
        # create module
        module = graph_runtime.create(loaded_json, loaded_lib, ctx)
        module.load_params(loaded_params)
        # set input and parameters
        module.set_input("data", data)
        repeat = 100
        print("===========Building TensorRT inference engine...")
        s = time.time()
        module.run()
        e = time.time() - s
        print("===========Building TensorRT inference engine took %.3f seconds" % e)
        print("===========Warming up inference engine...")
        for i in range(repeat):
            module.run(data=data)

        print("===========Starting to time inference...")
        repeat = 1000
        start = time.time()
        for i in range(repeat):
            module.run(data=data)
        total_elapse = time.time() - start
        avg_time = total_elapse / repeat * 1000.0
        import resource
        print("peak memory usage (bytes on OS X, kilobytes on Linux) {}"
              .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print("%s, ext_accel=%s, average time cost/forward: %.3fms" % (network, ext_accel, avg_time))
