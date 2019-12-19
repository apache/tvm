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

import random
import logging
logging.basicConfig(level=logging.INFO)
import numpy as np
import json

import nnvm.compiler
import nnvm.testing
import nnvm.subgraph
import nnvm.graph
import nnvm.compiler.graph_util
import tvm
from tvm.contrib import graph_runtime


def test_tensorrt_image_classification_models():
    def compile_model(graph, params, data_shapes, **kwargs):
        _, output_shapes = nnvm.compiler.graph_util.infer_shape(graph, **data_shapes)
        assert len(output_shapes) == 1
        target = tvm.target.cuda()
        with nnvm.compiler.build_config(opt_level=3, **kwargs):
            graph, lib, params = nnvm.compiler.build(
                graph, target, shape=data_shapes, params=params)
        return graph, lib, params, output_shapes[0]

    def get_output(module, data, params, output_shape):
        module.set_input("data", data)
        module.set_input(**params)
        module.run()
        return module.get_output(0).asnumpy()
        out = module.get_output(0, tvm.nd.empty(output_shape))
        return out.asnumpy()

    def copy_params(params):
        new_params = {}
        for k, v in params.items():
            new_params[k] = tvm.nd.array(v)
        return new_params

    def check_trt_model(baseline_module, baseline_params, graph, params, data_shape,
                        subgraph_backend=None, op_names=None, **kwargs):
        trt_graph, trt_lib, trt_params, output_shape = compile_model(graph, params, {'data': data_shape},
                                                                     **kwargs)
        # Verify that TRT subgraphs are partitioned
        def check_trt_used(graph):
            graph = json.loads(graph.json())
            num_trt_subgraphs = sum([1 for n in graph['nodes'] if n['op'] == '_tensorrt_subgraph_op'])
            assert num_trt_subgraphs >= 1
        check_trt_used(trt_graph)

        if not tvm.module.enabled("gpu"):
            return
        data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
        baseline_out = get_output(baseline_module, data, baseline_params, output_shape)
        trt_module = graph_runtime.create(trt_graph, trt_lib, tvm.gpu())
        trt_out = get_output(trt_module, data, trt_params, output_shape)
        np.testing.assert_almost_equal(baseline_out, trt_out, decimal=5)

    workload_dict = {'resnet': nnvm.testing.resnet.get_workload,
                     'inception_v3': nnvm.testing.inception_v3.get_workload,
                     'mobilenet': nnvm.testing.mobilenet.get_workload,
                     'mobilenet_v2': nnvm.testing.mobilenet_v2.get_workload,
                     'squeezenet': nnvm.testing.squeezenet.get_workload,
                     'vgg': nnvm.testing.vgg.get_workload,
                     'densenet': nnvm.testing.densenet.get_workload}
    for model_name, get_workload in workload_dict.items():
        logging.info('Testing TensorRT for model %s' % model_name)
        flags = {'batch_size': 1,
                 'image_shape': (3, 224, 224),
                 'num_classes': 100}
        if model_name == 'inception_v3':
            flags['image_shape'] = (3, 299, 299)
        if model_name.startswith('resnet'):
            flags['num_layers'] = 18
        data_shape = (flags['batch_size'],) + flags['image_shape']
        if model_name == 'mobilenet_v2' or model_name == 'densenet':
            flags.pop('image_shape')
        net, params = get_workload(**flags)
        graph_json_str = nnvm.graph.create(net).json()
        with nnvm.compiler.build_config(opt_level=3):
            baseline_graph, baseline_lib, baseline_params = nnvm.compiler.build(
                nnvm.graph.load_json(graph_json_str), tvm.target.cuda(),
                shape={'data': data_shape}, params=copy_params(params))
        baseline_module = graph_runtime.create(baseline_graph, baseline_lib, tvm.gpu())

        # Test whole graph run using tensorrt. nnvm.compiler.build_config has
        # graph partitioning turned on when ext_accel='tensorrt'.
        check_trt_model(baseline_module, baseline_params, nnvm.graph.load_json(graph_json_str),
                        copy_params(params), data_shape, ext_accel='tensorrt')


if __name__ == '__main__':
    test_tensorrt_image_classification_models()
