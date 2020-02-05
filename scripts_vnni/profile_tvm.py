
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

import argparse
import logging
import os
import time
import numpy as np
import statistics

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.relay import expr as _expr
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime


def profile(data, symbol_file, num_inference_images, sym, devs, label_name):
    debug = False
    import tvm
    from tvm.contrib import graph_runtime
    from tvm.contrib.debugger import debug_runtime as debug_runtime

    base = '/home/ubuntu/mxnet_compiled_models/tvm_' + symbol_file.split('/')[-1].replace('.json','')

    path_lib = base + '_deploy_lib.tar'
    path_graph =  base + '_deploy_graph.json'
    path_params = base + '_deploy_params.params'

    graph = open(path_graph).read()
    lib = tvm.module.load(path_lib)
    params = bytearray(open(path_params, 'rb').read())

    if debug:
        rt_mod = debug_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.load_params(params)
        rt_mod.run()
        return

    rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    rt_mod.load_params(params)

    # warm up
    warm_up = 0
    for i in range(0, 50):
        rt_mod.run()
        warm_up += 1
        if warm_up == 50:
            break


    counter = 0
    time_tvm = list()
    for i in range(0, num_inference_images):
        time0 = time.time()
        rt_mod.run()
        time1 = time.time()
        time_tvm.append(time1 - time0)
        counter += 1
        if counter == num_inference_images:
            break

    avg = lambda x : round(1000*sum(x)/len(x), 6)
    std = lambda x: round(statistics.stdev(x), 6)


    total_tvm = avg(time_tvm)
    sec_tvm = total_tvm/1000
    std_tvm = std(time_tvm)
    min_tvm = round(min(time_tvm), 6)
    min_tvm_ms = round(min(time_tvm)*1000, 6)
    deviation_from_min_tvm = round(sec_tvm/min_tvm*100 - 100, 6)
    deviation_from_std_tvm = round(std_tvm/sec_tvm*100, 6)

    net_name = symbol_file.split('/')[-1].replace('.json','')
    print("Perf", "Tvm", net_name, total_tvm, min_tvm_ms, std_tvm, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--ctx', type=str, default='gpu')
    parser.add_argument('--benchmark', type=bool, default=False, help='dummy data benchmark')
    parser.add_argument('--score_tvm', type=bool, default=False, help='score tvm')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=False, help='param file path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--dataset', type=str, required=False, help='dataset path')
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--rgb-std', type=str, default='1,1,1')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--data-layer-type', type=str, default="float32",
                        choices=['float32', 'int8', 'uint8'],
                        help='data type for data layer')

    args = parser.parse_args()

    ctx = None

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    rgb_mean = args.rgb_mean
    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}
    rgb_std = args.rgb_std
    logger.info('rgb_std = %s' % rgb_std)
    rgb_std = [float(i) for i in rgb_std.split(',')]
    std_args = {'std_r': rgb_std[0], 'std_g': rgb_std[1], 'std_b': rgb_std[2]}
    combine_mean_std = {}
    combine_mean_std.update(mean_args)
    combine_mean_std.update(std_args)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    image_shape = args.image_shape
    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    data_layer_type = args.data_layer_type
    num_inference_images = args.num_inference_batches * batch_size
    logger.info('Running model %s for inference' % symbol_file)
    sym = None
    data = None
    profile(data, symbol_file, num_inference_images, sym, [ctx], label_name)
