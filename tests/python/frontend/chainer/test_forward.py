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
# pylint: disable=import-self, invalid-name, unused-argument
"""Unit tests for various models and operators"""
import numpy as np
import chainer
from chainer import function
import chainer.functions as F
import chainer.links as L

import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list

def verify_model(model, input_data, ctxl=ctx_list()):
    """Assert that the output of a compiled model matches with that of its
    baseline."""
    # Form input data
    shape_dict = {}
    dtype_dict = {}
    input_vars = []
    input_names = []
    for idx, data in enumerate(input_data):
        input_vars.append(chainer.Variable(data))
        input_name = "{}_{}".format("input", idx)
        shape_dict[input_name] = data.shape
        dtype_dict[input_name] = data.dtype
        input_names.append(input_name)

    # Run Chainer Model for the input
    with function.force_backprop_mode(), chainer.using_config('train', False):
        if len(input_vars) > 1:
            baseline_outputs = model(input_vars).data
        else:
            baseline_outputs = model(*input_vars).data

    # Convert Chainer model to TVM and get the output
    mod, params = relay.frontend.from_chainer(model, shape_dict, dtype_dict)

    with relay.build_config(opt_level=3):
        for target, ctx in ctxl:
            relay_graph, relay_lib, relay_params = relay.build(mod, target=target, params=params)
            relay_model = graph_runtime.create(relay_graph, relay_lib, ctx)
            relay_model.set_input(**relay_params)
            for name, x in zip(input_names, input_data):
                relay_model.set_input(name, tvm.nd.array(x.astype(dtype_dict[name])))

            relay_model.run()

            compiled_output = relay_model.get_output(0).asnumpy()
            #TODO: Make multi-output compatible
            tvm.testing.assert_allclose(baseline_outputs, compiled_output,
                                        rtol=1e-3, atol=1e-3)

# Single operator tests
def test_forward_relu():
    class Link(chainer.Chain):
        def __call__(self, x):
            return F.relu(x)
    input_data = np.random.uniform(-1, 1, (1, 3, 7, 7)).astype(np.float32)
    verify_model(Link(), [input_data])

def test_forward_concat():
    class Link_0(chainer.Chain):
        def __call__(self, x):
            return F.concat(x, axis=0)

    class Link_1(chainer.Chain):
        def __call__(self, x):
            return F.concat(x, axis=1)

    input_data_0 = np.random.uniform(-1, 1, (1, 3, 7, 7)).astype(np.float32)
    input_data_1 = np.random.uniform(-1, 1, (1, 3, 7, 7)).astype(np.float32)

    verify_model(Link_0(), [input_data_0, input_data_1])
    verify_model(Link_1(), [input_data_0, input_data_1])

def test_forward_conv():
    class Link(chainer.Chain):
        def __init__(self, args, kwargs):
            super(Link, self).__init__()
            with self.init_scope():
                self.l1 = L.Convolution2D(*args, **kwargs)
        def forward(self, x):
            return self.l1(x)

    # Convolution2D(in_channels, out_channels, ksize, stride, pad, groups, dilation)
    test_sets = [{'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
                  'args': [None, 3, 3, 1, 1],
                  'kwargs': {}}, #TestCase-1
                 {'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
                  'args': [None, 3, 3, 1, 2, True],
                  'kwargs': {}}, #TestCase-2
                 {'in_shape': (1, 3, 5, 5), 'in_type': np.float32,
                  'args': [None, 3, 3, 1, 1],
                  'kwargs': {'groups': 3}}] #TestCase-3

    for test in test_sets:
        input_data = np.random.uniform(-1, 1, test['in_shape']).astype(test['in_type'])
        verify_model(Link(test['args'], test['kwargs']), [input_data])

def test_forward_reshape():
    class Link(chainer.Chain):
        def __init__(self, args):
            super(Link, self).__init__()
            self.args = args
        def forward(self, x):
            return F.reshape(x, **self.args)

    # reshape(input, new_shape)
    test_sets = [{'in_shape': (1, 6), 'in_type': np.float32,
                  'args': {'shape': (1, 2, 1, 3)}}, #TestCase-1
                 {'in_shape': (1, 3, 7, 7), 'in_type': np.float32,
                  'args': {'shape': (1, -1)}}] #TestCase-2

    for test in test_sets:
        input_data = np.random.uniform(-1, 1, test['in_shape']).astype(test['in_type'])
        verify_model(Link(test['args']), [input_data])

if __name__ == "__main__":
    # Single operator tests
    test_forward_relu()
    test_forward_concat()
    test_forward_conv()
    test_forward_reshape()
