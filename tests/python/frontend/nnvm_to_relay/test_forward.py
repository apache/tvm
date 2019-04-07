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

import tvm
import nnvm
import nnvm.testing
from nnvm.to_relay import to_relay
from tvm import relay
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime

def verify_nnvm_to_relay(nnvm_sym, params, data_shape=(1, 3, 224, 224)):
    def get_nnvm_output(sym, x, params, target, ctx, dtype='float32'):
        shape_dict = {'data': x.shape}
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        return m.get_output(0).asnumpy()

    def get_relay_output(sym, x, params, target, ctx, dtype='float32'):
        shape_dict = {'data': x.shape}
        func, params = to_relay(sym, shape_dict, dtype, params)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target=target, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        return m.get_output(0).asnumpy()

    x = np.random.uniform(size=data_shape)
    for target, ctx in ctx_list():
        nnvm_out = get_nnvm_output(nnvm_sym, x, params, target, ctx)
        relay_out = get_relay_output(nnvm_sym, x, params, target, ctx)
        tvm.testing.assert_allclose(nnvm_out, relay_out, rtol=1e-5, atol=1e-5)


def test_forward_mlp():
    model, params = nnvm.testing.mlp.get_workload(1)
    verify_nnvm_to_relay(model, params)


def test_forward_vgg():
    model, params = nnvm.testing.vgg.get_workload(1)
    verify_nnvm_to_relay(model, params)


def test_forward_resnet():
    model, params = nnvm.testing.resnet.get_workload(1)
    verify_nnvm_to_relay(model, params)


def test_forward_squeezenet():
    model, params = nnvm.testing.squeezenet.get_workload(1)
    verify_nnvm_to_relay(model, params)


def test_forward_inception_v3():
    model, params = nnvm.testing.inception_v3.get_workload(1)
    verify_nnvm_to_relay(model, params, data_shape=(1, 3, 299, 299))


def test_forward_densenet():
    model, params = nnvm.testing.squeezenet.get_workload(1)
    verify_nnvm_to_relay(model, params)


def test_forward_dqn():
    model, params = nnvm.testing.dqn.get_workload(1)
    verify_nnvm_to_relay(model, params, data_shape=(1, 4, 84, 84))


def test_forward_split_concatenate():
    shape = (2, 16)

    tensor = nnvm.sym.Variable("data", shape=shape)

    splited = nnvm.sym.split(tensor, indices_or_sections=2, axis=1)

    concatenated = nnvm.sym.concatenate(*splited, axis=1)

    params = {}

    verify_nnvm_to_relay(splited[0], params, data_shape=shape)
    verify_nnvm_to_relay(splited[1], params, data_shape=shape)
    verify_nnvm_to_relay(splited, params, data_shape=shape)
    verify_nnvm_to_relay(concatenated, params, data_shape=shape)


if __name__ == '__main__':
    test_forward_mlp()
    test_forward_vgg()
    test_forward_resnet()
    test_forward_squeezenet()
    test_forward_inception_v3()
    test_forward_densenet()
    test_forward_dqn()
    test_forward_split_concatenate()
