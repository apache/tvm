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

import topi
import tvm
from tvm.contrib import graph_runtime
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
from nnvm import frontend
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import model_zoo


def verify_mxnet_frontend_impl(mx_symbol, data_shape=(1, 3, 224, 224), out_shape=(1, 1000),
                               gluon_impl=False, name=None, dtype='float32'):
    """Use name different from test to avoid let nose pick it up"""
    if gluon_impl:
        def get_gluon_output(name, x):
            net = vision.get_model(name)
            net.collect_params().initialize(mx.init.Xavier())
            net_sym = gluon.nn.SymbolBlock(outputs=net(mx.sym.var('data')),
                                           inputs=mx.sym.var('data'),
                                           params=net.collect_params())
            out = net_sym(mx.nd.array(x.astype(dtype))).asnumpy()
            return out, net_sym
    else:
        def get_mxnet_output(symbol, x, dtype='float32'):
            from collections import namedtuple
            Batch = namedtuple('Batch', ['data'])
            mod = mx.mod.Module(symbol, label_names=None)
            mod.bind(data_shapes=[('data', x.shape)], for_training=False)
            mod.init_params()
            mod.forward(Batch([mx.nd.array(x.astype(dtype))]))
            out = mod.get_outputs()[0].asnumpy()
            args, auxs = mod.get_params()
            return out, args, auxs

    def get_tvm_output(symbol, x, args, auxs, target, ctx, dtype='float32'):
        if gluon_impl:
            new_sym, params = frontend.from_mxnet(symbol)
        else:
            new_sym, params = frontend.from_mxnet(symbol, args, auxs)

        dshape = x.shape
        shape_dict = {'data': dshape}
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.set_input(**params)
        m.run()
        # get outputs
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.asnumpy()

    # random input
    x = np.random.uniform(size=data_shape)
    if gluon_impl:
        gluon_out, gluon_sym = get_gluon_output(name, x)
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(gluon_sym, x, None, None, target, ctx, dtype)
            tvm.testing.assert_allclose(gluon_out, tvm_out, rtol=1e-5, atol=1e-5)
    else:
        mx_out, args, auxs = get_mxnet_output(mx_symbol, x, dtype)
        assert "data" not in args
        for target, ctx in ctx_list():
            tvm_out = get_tvm_output(mx_symbol, x, args, auxs, target, ctx, dtype)
            tvm.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_mlp():
    mlp = model_zoo.mx_mlp
    verify_mxnet_frontend_impl(mlp)

def test_forward_vgg():
    for n in [11]:
        mx_sym = model_zoo.mx_vgg[n]
        verify_mxnet_frontend_impl(mx_sym)

def test_forward_resnet():
    for n in [18]:
        mx_sym = model_zoo.mx_resnet[n]
        verify_mxnet_frontend_impl(mx_sym)

def test_forward_elu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type='elu')
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_rrelu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type='rrelu', lower_bound=0.3, upper_bound=0.7)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_prelu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type='prelu')
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_softrelu():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.Activation(data, act_type='softrelu')
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_fc_flatten():
    # test flatten=True option in mxnet 0.11.1
    data = mx.sym.var('data')
    try:
        mx_sym = mx.sym.FullyConnected(data, num_hidden=100, flatten=True)
        verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 100))
        mx_sym = mx.sym.FullyConnected(mx.sym.Flatten(data), num_hidden=100, flatten=False)
        verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 100))
    except:
        pass

def test_forward_clip():
    data = mx.sym.var('data')
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.clip(data, a_min=0, a_max=1)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))

def test_forward_split():
    data = mx.sym.var('data')
    mx_sym = mx.sym.split(data, axis=1, num_outputs=4, squeeze_axis=False)
    verify_mxnet_frontend_impl(mx_sym, (1, 4, 2, 1), (1, 1, 2, 1))

def test_forward_split_squeeze():
    data = mx.sym.var('data')
    mx_sym = mx.sym.split(data, axis=1, num_outputs=4, squeeze_axis=True)
    verify_mxnet_frontend_impl(mx_sym, (1, 4, 2, 1), (1, 2, 1))

def test_forward_expand_dims():
    data = mx.sym.var('data')
    mx_sym = mx.sym.expand_dims(data, axis=1)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 1, 3, 4))

def test_forward_pooling():
    data = mx.sym.var('data')
    mx_sym = mx.sym.Pooling(data, kernel=(3, 3), pad=(1, 1), pool_type='avg')
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 8, 8))

    mx_sym = mx.sym.Pooling(data, kernel=(3, 3), pad=(1, 1), pool_type='max')
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 8, 8))

def test_forward_lrn():
    data = mx.sym.var('data')
    mx_sym = mx.sym.LRN(data, alpha=2, beta=2, knorm=1, nsize=5)
    verify_mxnet_frontend_impl(mx_sym, (1, 10, 24, 24), (1, 10, 24, 24))

def test_forward_ones():
    data = mx.sym.var('data')
    ones = mx.sym.ones(shape=(2, 3, 4), dtype='float32')
    mx_sym = mx.sym.elemwise_add(data, ones)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_zeros():
    data = mx.sym.var('data')
    zeros = mx.sym.zeros(shape=(2, 3, 4), dtype='float32')
    mx_sym = mx.sym.elemwise_add(data, zeros)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_ones_like():
    data = mx.sym.var('data')
    mx_sym = mx.sym.ones_like(data, dtype='float32')
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_zeros_like():
    data = mx.sym.var('data')
    mx_sym = mx.sym.zeros_like(data, dtype='float32')
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))

def test_forward_argmax():
    data = mx.sym.var('data')
    mx_sym = mx.sym.argmax(data, axis=1)
    verify_mxnet_frontend_impl(mx_sym, (5, 3), (5,))

def test_forward_argmin():
    data = mx.sym.var('data')
    mx_sym = mx.sym.argmin(data, axis=0)
    verify_mxnet_frontend_impl(mx_sym, (5, 4), (4,))

def test_forward_where():
    cond = mx.sym.var('cond')
    x = mx.sym.var('x')
    y = mx.sym.var('y')
    dshape = (2, 2)
    dtype = 'float32'
    mx_sym = mx.sym.where(cond, x, y)
    np_cond = np.array([[0, 1], [-1, 0]]).astype(dtype)
    np_x = np.random.uniform(size=dshape).astype(dtype)
    np_y = np.random.uniform(size=dshape).astype(dtype)
    mx_cond = mx.nd.array(np_cond)
    mx_x = mx.nd.array(np_x)
    mx_y = mx.nd.array(np_y)
    mod = mx.mod.Module(mx_sym, label_names=None, data_names=['cond', 'x', 'y'])
    mod.bind(data_shapes=[('cond', dshape), ('x', dshape), ('y', dshape)], for_training=False)
    mod.init_params()
    args, auxs = mod.get_params()
    mx_out = mx.nd.where(mx_cond, mx_x, mx_y).asnumpy()
    out_shape = dshape
    new_sym, params = frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'cond': dshape, 'x': dshape, 'y': dshape}
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("cond", tvm.nd.array(np_cond))
        m.set_input("x", tvm.nd.array(np_x))
        m.set_input("y", tvm.nd.array(np_y))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        tvm.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_slice():
    data = mx.sym.var('data')
    mx_sym = mx.sym.slice(data, begin=(0, 1), end=(2, 4))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 3))
    mx_sym = mx.sym.slice(data, begin=(-1, 1), end=(-3, 4), step=(-1, 2))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 2))

def test_forward_maximum():
    a = mx.sym.var('a')
    b = mx.sym.var('b')
    dshape = (10, 20)
    dtype = 'float32'
    mx_sym = mx.sym._internal._maximum(a, b)
    np_a = np.random.uniform(size=dshape).astype(dtype)
    np_b = np.random.uniform(size=dshape).astype(dtype)
    mx_a = mx.nd.array(np_a)
    mx_b = mx.nd.array(np_b)
    mod = mx.mod.Module(mx_sym, label_names=None, data_names=['a', 'b'])
    mod.bind(data_shapes=[('a', dshape), ('b', dshape)], for_training=False)
    mod.init_params()
    args, auxs = mod.get_params()
    mx_out = mx.nd._internal._maximum(mx_a, mx_b).asnumpy()
    out_shape = dshape
    new_sym, params = frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'a': dshape, 'b': dshape}
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("a", tvm.nd.array(np_a))
        m.set_input("b", tvm.nd.array(np_b))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        tvm.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5, atol=1e-5)

def test_forward_minimum():
    a = mx.sym.var('a')
    b = mx.sym.var('b')
    dshape = (10, 20)
    dtype = 'float32'
    mx_sym = mx.sym._internal._minimum(a, b)
    np_a = np.random.uniform(size=dshape).astype(dtype)
    np_b = np.random.uniform(size=dshape).astype(dtype)
    mx_a = mx.nd.array(np_a)
    mx_b = mx.nd.array(np_b)
    mod = mx.mod.Module(mx_sym, label_names=None, data_names=['a', 'b'])
    mod.bind(data_shapes=[('a', dshape), ('b', dshape)], for_training=False)
    mod.init_params()
    args, auxs = mod.get_params()
    mx_out = mx.nd._internal._minimum(mx_a, mx_b).asnumpy()
    out_shape = dshape
    new_sym, params = frontend.from_mxnet(mx_sym, args, auxs)
    shape_dict = {'a': dshape, 'b': dshape}
    for target, ctx in ctx_list():
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        m.set_input("a", tvm.nd.array(np_a))
        m.set_input("b", tvm.nd.array(np_b))
        m.set_input(**params)
        m.run()
        # get outputs
        tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()
        tvm.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    test_forward_mlp()
    test_forward_vgg()
    test_forward_resnet()
    test_forward_elu()
    test_forward_rrelu()
    test_forward_prelu()
    test_forward_softrelu()
    test_forward_fc_flatten()
    test_forward_clip()
    test_forward_split()
    test_forward_split_squeeze()
    test_forward_expand_dims()
    test_forward_pooling()
    test_forward_lrn()
    test_forward_ones()
    test_forward_zeros()
    test_forward_ones_like()
    test_forward_zeros_like()
    test_forward_argmax()
    test_forward_argmin()
    test_forward_where()
    test_forward_slice()
    test_forward_maximum()
    test_forward_minimum()
