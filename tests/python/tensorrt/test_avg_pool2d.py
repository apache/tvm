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

import numpy as np
import mxnet as mx
from mxnet import gluon
import nnvm
import tvm
from tvm.contrib import graph_runtime
import json


def test_avg_pool2d():

    # Generate the data
    np.random.seed(0)
    input_shape = [1, 1, 28, 28]
    output_shape = [1, 1, 28, 28]
    data = np.random.random(input_shape).astype('float32')
    
    # Baseline model in MXNet
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
    net.hybridize()
    baseline_input = mx.nd.array(data, ctx=mx.cpu())
    baseline_output = net(baseline_input).asnumpy()
    
    # Compiled model
    sym, params = nnvm.frontend.from_mxnet(net)
    target = tvm.target.cuda()
    with nnvm.compiler.build_config(opt_level=3, ext_accel='tensorrt'):
        graph, lib, params = nnvm.compiler.build(sym, target,
                                                 shape={'data': input_shape},
                                                 params=params)

    # Verify that TRT subgraphs are partitioned
    def check_trt_used(graph):
        graph = json.loads(graph.json())
        num_trt_subgraphs = sum([1 for n in graph['nodes'] if n['op'] == '_tensorrt_subgraph_op'])
        assert num_trt_subgraphs == 1
    check_trt_used(graph)

    # Execute
    if not tvm.module.enabled("gpu"):
        return
    compiled_model = graph_runtime.create(graph, lib, tvm.gpu())
    compiled_input = tvm.nd.array(data, ctx=tvm.gpu())
    compiled_model.set_input('data', compiled_input)
    compiled_model.set_input(**params)
    compiled_model.run()
    compiled_output = compiled_model.get_output(0, tvm.nd.empty(output_shape)).asnumpy()
    
    # Compare outputs
    np.testing.assert_almost_equal(baseline_output, compiled_output, decimal=3)


if __name__ == '__main__':
    test_avg_pool2d()
