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

import time
import numpy as np
import argparse
import nnvm
import tvm
from tvm.contrib import graph_runtime
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
import json

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark TVM')
    parser.add_argument('--ext-accel', type=str, default='none', choices=['none', 'tensorrt'])
    parser.add_argument('--network', type=str, required=True, choices=models)
    parser.add_argument('--cuda-arch', type=str, required=True, choices=['sm_37', 'sm_70', 'sm_53', 'sm_62'])
    parser.add_argument('--target-host', type=str, required=True, choices=['x86_64-linux-gnu', 'aarch64-linux-gnu'])
    args = parser.parse_args()

    network = args.network
    num_classes = 1000
    data_shape = get_data_shape(network)
    ext_accel = None if args.ext_accel == 'none' else args.ext_accel
    cuda_arch = args.cuda_arch

    print("===========Loading model %s" % network)
    loaded_json = open('%s.json' % network).read()
    loaded_params = bytearray(open('%s.params' % network, 'rb').read())
    net = nnvm.graph.load_json(loaded_json)
    params = nnvm.compiler.load_param_dict(loaded_params)
    opt_level = 3
    target = tvm.target.cuda()
    set_cuda_target_arch(cuda_arch)
    target_host = 'llvm -target=%s' % args.target_host
    print("===========Start to compile %s graph with params, external accelerator: %s" % (network, ext_accel))
    start = time.time()
    with nnvm.compiler.build_config(opt_level=opt_level, ext_accel=ext_accel):
        graph, lib, params = nnvm.compiler.build(
            net, target, shape={"data": data_shape}, params=params, target_host=target_host)

    # Verify that TRT subgraphs are partitioned
    def check_trt_used(graph):
        graph = json.loads(graph.json())
        num_trt_subgraphs = sum([1 for n in graph['nodes'] if n['op'] == '_tensorrt_subgraph_op'])
        assert num_trt_subgraphs >= 1
    check_trt_used(graph)

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
