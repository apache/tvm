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
import logging
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
import tvm
import tvm.relay as relay
import topi
from tvm import te
from tvm.contrib import graph_runtime

# mxnet exp custom layer


class exp(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(exp, self).__init__(**kwargs)

    def forward(self, x):
        return x.exp()

# mxnet tanh custom layer


class tanh(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(tanh, self).__init__(**kwargs)

    def forward(self, x):
        return x.tanh()


def test_fast_exp():
    numpy_input = np.arange(-88, 88, 0.01).astype("float32")

    # make mxnet unit test net
    unit_test_net = nn.HybridSequential()
    unit_test_net.add(exp())
    unit_test_net.initialize()
    mxnet_input = nd.array(numpy_input)
    mxnet_output = unit_test_net(mxnet_input)

    # Compile the Graph
    shape_dict = {'data': numpy_input.shape}
    target = 'llvm'
    ctx = tvm.cpu(0)
    dtype = 'float32'

    mod, params = relay.frontend.from_mxnet(unit_test_net, shape_dict,)

    logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion
    with relay.build_config(opt_level=3, required_pass=['FastMath']):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(numpy_input.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)

    tvm.testing.assert_allclose(tvm_output.asnumpy(), mxnet_output.asnumpy(),
                                rtol=1e-5, atol=1e-5)


def test_fast_tanh():
    numpy_input = np.arange(-88, 88, 0.01).astype("float32")

    # make mxnet unit test net
    unit_test_net = nn.HybridSequential()
    unit_test_net.add(tanh())
    unit_test_net.initialize()

    mxnet_input = nd.array(numpy_input)
    mxnet_output = unit_test_net(mxnet_input)

    shape_dict = {'data': numpy_input.shape}
    target = 'llvm'
    ctx = tvm.cpu(0)
    dtype = 'float32'

    mod, params = relay.frontend.from_mxnet(unit_test_net, shape_dict,)
    ftr12
    logging.basicConfig(level=logging.DEBUG)  # to dump TVM IR after fusion
    with relay.build_config(opt_level=3, required_pass=['FastMath']):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(numpy_input.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0)

    tvm.testing.assert_allclose(tvm_output.asnumpy(), mxnet_output.asnumpy(),
                                rtol=1e-9, atol=4e-5)


if __name__ == "__main__":
    test_fast_exp()
    test_fast_tanh()
