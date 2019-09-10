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
import operator

import tvm
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm import relay
import mxnet as mx

from mxnet import gluon
from mxnet.gluon.model_zoo import vision
import model_zoo


def verify_mxnet_frontend_impl(mx_symbol,
                               data_shape=(1, 3, 224, 224),
                               out_shape=(1, 1000),
                               gluon_impl=False,
                               name=None,
                               dtype='float32'):
    """Use name different from test to avoid pytest picking it up"""
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
        shape_dict = {"data": x.shape}
        if gluon_impl:
            mod, params = relay.frontend.from_mxnet(symbol, shape_dict)
        else:
            mod, params = relay.frontend.from_mxnet(symbol,
                                                    shape_dict,
                                                    arg_params=args,
                                                    aux_params=auxs)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target, params=params)
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
    mlp = model_zoo.mx_mlp()
    verify_mxnet_frontend_impl(mlp,
                               data_shape=(1, 1, 28, 28),
                               out_shape=(1, 10))

def test_forward_vgg():
    for n in [11]:
        mx_sym = model_zoo.mx_vgg(n)
        verify_mxnet_frontend_impl(mx_sym)

def test_forward_resnet():
    for n in [18]:
        mx_sym = model_zoo.mx_resnet(18)
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

def test_forward_adaptive_pooling():
    data = mx.sym.var('data')
    mx_sym = mx.sym.contrib.AdaptiveAvgPooling2D(data, output_size=(1,))
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 1, 1))

    mx_sym = mx.sym.contrib.AdaptiveAvgPooling2D(data, output_size=(3, 3))
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 3, 3))

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

def test_forward_slice():
    data = mx.sym.var('data')
    mx_sym = mx.sym.slice(data, begin=(0, 1), end=(2, 4))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 3))
    mx_sym = mx.sym.slice(data, begin=(-1, 1), end=(-3, 4), step=(-1, 2))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 2))

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
    shapes = {'cond': dshape, 'x': dshape, 'y': dshape}
    mod = mx.mod.Module(mx_sym, label_names=None, data_names=['cond', 'x', 'y'])
    mod.bind(data_shapes=shapes.items(), for_training=False)
    mod.init_params()
    args, auxs = mod.get_params()
    mx_out = mx.nd.where(mx_cond, mx_x, mx_y).asnumpy()

    mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, args, auxs)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(np_cond, np_x, np_y)
            tvm.testing.assert_allclose(op_res.asnumpy(), mx_out)


def test_forward_arange():
    def _mx_symbol(F, start, stop, step):
        if start is None and step is None:
            sym = F.arange(stop)
        elif start is None:
            sym = F.arange(stop, step=step)
        elif step is None:
            sym = F.arange(start, stop)
        else:
            sym = F.arange(start, stop, step)
        return sym

    def verify(start, stop, step):
        ref_res = _mx_symbol(mx.nd, start, stop, step).asnumpy()
        mx_sym = _mx_symbol(mx.sym, start, stop, step)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()()
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)
    verify(0, 20, None)
    verify(0, 20, 2)
    verify(1, 20, None)
    verify(1, 20, 2)
    verify(1, 20, 1.5)
    verify(1, 20.5, None)
    verify(1, 20, 3)
    verify(20, 1, -1)
    verify(20, 1, -1.5)

def _mx_symbol(F, op_name, inputs):
    op = getattr(F, op_name)
    return op(*inputs)

def test_forward_broadcast_ops():
    for op in ["broadcast_add", "broadcast_sub", "broadcast_mul",
               "broadcast_div", "broadcast_mod", "broadcast_maximum",
               "broadcast_minimum", "broadcast_equal", "broadcast_not_equal",
               "broadcast_greater", "broadcast_greater_equal",
               "broadcast_lesser", "broadcast_lesser_equal"]:
        a_shape = (3, 4, 5)
        b_shape = (4, 5)
        if op == "broadcast_mod":
            dtype = 'int32'
            a_np = np.random.randint(1, 100, size=a_shape).astype(dtype)
            b_np = np.random.randint(1, 100, size=b_shape).astype(dtype)
        else:
            dtype = 'float32'
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            b_np = np.random.uniform(size=b_shape).astype(dtype)
        mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var('a'), mx.sym.var('b')])
        ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np), mx.nd.array(b_np)])
        shapes = {'a': a_shape, 'b': b_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(a_np, b_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())

def test_forward_elemwise_ops():
    for op in ["elemwise_add", "elemwise_sub", "elemwise_mul",
               "elemwise_div", "maximum", "minimum"]:
        shape = (3, 4, 5)
        dtype = 'float32'
        a_np = np.random.uniform(size=shape).astype(dtype)
        b_np = np.random.uniform(size=shape).astype(dtype)
        mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var('a'), mx.sym.var('b')])
        ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np), mx.nd.array(b_np)])
        shapes = {'a': shape, 'b': shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(a_np, b_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())

def test_forward_scalar_ops():
    for op in [operator.add, operator.sub, operator.mul, operator.truediv,
               operator.pow, operator.lt, operator.le, operator.eq,
               operator.ne, operator.gt, operator.ge]:
        dtype='float32'
        a_shape = (3, 4, 5)
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_scalar = 2.3
        mx_sym = op(mx.sym.var('a'), b_scalar)
        ref_res = op(mx.nd.array(a_np), b_scalar)
        shapes = {'a': a_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(a_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    for op in ["maximum", "minimum"]:
        dtype='float32'
        a_shape = (3, 4, 5)
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_scalar = 2.3
        mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var('a'), b_scalar])
        ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np), b_scalar])
        shapes = {'a': a_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(a_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())

def test_forward_slice_axis():
    def verify(shape, axis, begin, end):
        data_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.slice_axis(mx.nd.array(data_np), axis, begin, end)
        mx_sym = mx.sym.slice_axis(mx.sym.var("data"), axis, begin, end)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(data_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((3, 4), 0, 1, 2)
    verify((3, 4), 0, 1, None)
    verify((3, 4), 1, 0, 2)
    verify((3, 4), 1, -3, -1)
    verify((3, 4), -1, -3, -1)

def test_forward_slice_like():
    def verify(x_shape, y_shape, axes):
        x_np = np.random.uniform(size=x_shape).astype("float32")
        y_np = np.random.uniform(size=y_shape).astype("float32")
        if axes is None:
            ref_res = mx.nd.slice_like(mx.nd.array(x_np), mx.nd.array(y_np))
            mx_sym = mx.sym.slice_like(mx.sym.var("x"), mx.sym.var("y"))
        else:
            ref_res = mx.nd.slice_like(mx.nd.array(x_np), mx.nd.array(y_np), axes=axes)
            mx_sym = mx.sym.slice_like(mx.sym.var("x"), mx.sym.var("y"), axes=axes)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": x_shape, "y": y_shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np, y_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((3, 4), (2, 3), None)
    verify((3, 4), (2, 3), (0, 1))
    verify((3, 4), (2, 3), (0))
    verify((3, 4), (2, 3), (-1))

def test_forward_l2_normalize():
    data = mx.sym.var('data')
    mx_sym = mx.sym.L2Normalization(data, mode="channel")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4, 5), (2, 3, 4, 5))

def test_forward_shape_array():
    def verify(shape):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.shape_array(mx.nd.array(x_np))
        mx_sym = mx.sym.shape_array(mx.sym.var("x"))
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, ctx in ctx_list():
            for kind in ["debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((1,))
    verify((3, 4, 5))
    verify((3, 4, 5, 6))

def test_forward_squeeze():
    def verify(shape, axis):
        x_np = np.random.uniform(size=shape).astype("float32")
        if axis is None:
            ref_res = mx.nd.squeeze(mx.nd.array(x_np))
            mx_sym = mx.sym.squeeze(mx.sym.var("x"))
        else:
            ref_res = mx.nd.squeeze(mx.nd.array(x_np), axis=axis)
            mx_sym = mx.sym.squeeze(mx.sym.var("x"), axis=axis)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((1, 3, 1), None)
    verify((1, 3, 1), 0)
    verify((1, 3, 1), 2)
    verify((1, 3, 1), (0, 2))

def test_forward_broadcast_axis():
    def verify(shape, axis, size):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.broadcast_axis(mx.nd.array(x_np), axis=axis, size=size)
        mx_sym = mx.sym.broadcast_axis(mx.sym.var("x"), axis=axis, size=size)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((1, 2, 1), 2, 3)
    verify((1, 2, 1), (0, 2), (2, 3))

def test_forward_full():
    def verify(val, shape, dtype):
        ctx = mx.cpu()
        ref_res = mx.nd.full(shape, val, dtype=dtype)
        mx_sym = mx.sym.full(shape, val, dtype=dtype)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {})
        for target, ctx in ctx_list():
            # Skip testing graph runtime because this op will be optimized out
            # by constant folding.
            for kind in ["debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()()
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify(2, (3, 4), "float32")
    verify(2, (3, 4), "int32")
    verify(3.5, (1, 3, 4), "float32")

def test_forward_embedding():
    def verify(data_shape, weight_shape):
        in_dim, out_dim = weight_shape
        x_np = np.random.randint(0, weight_shape[0], size=data_shape).astype("float32")
        w_np = np.random.uniform(size=weight_shape).astype("float32")
        ref_res = mx.nd.Embedding(mx.nd.array(x_np), mx.nd.array(w_np),
                                  input_dim=in_dim, output_dim=out_dim)
        mx_sym = mx.sym.Embedding(mx.sym.var("x"), mx.sym.var("w"),
                                  input_dim=in_dim, output_dim=out_dim)
        mod, _ = relay.frontend.from_mxnet(
            mx_sym, {"x": data_shape, "w": weight_shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x=x_np, w=w_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((2, 2), (4, 5))
    verify((2, 3, 4), (4, 5))

def test_forward_smooth_l1():
    data = mx.sym.var('data')
    mx_sym = mx.sym.smooth_l1(data)
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (3, 4))
    mx_sym = mx.sym.smooth_l1(data, scalar=1.0)
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (3, 4))

def test_forward_take():
    def verify(shape, indices_src, axis, mode="clip"):
        x_np = np.random.uniform(size=shape).astype("float32")
        indices_np = np.array(indices_src, dtype="float32")
        ref_res = mx.nd.take(mx.nd.array(x_np), mx.nd.array(indices_np), axis, mode)
        mx_sym = mx.sym.take(mx.sym.var("x"), mx.sym.var("y"), axis, mode)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape, "y": indices_np.shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np, indices_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((2,2), [[[1,0],[0,1]]], 0)
    verify((2,2), [[[1,0],[0,1]]], 1)
    verify((4,3,5,6), [[2,1,0,0]], -2)
    verify((3,4), [-1, 5], 0)
    verify((3,4), [-1, 5], 0, mode="wrap")
    verify((3,4), [-1, 5], 1)
    verify((3,4), [-1, 5], 1, mode="wrap")

def test_forward_gather_nd():
    def verify(xshape, yshape, y_data):
        x_data = np.random.uniform(size=xshape).astype("float32")
        ref_res = mx.nd.gather_nd(mx.nd.array(x_data), mx.nd.array(y_data))
        mx_sym = mx.sym.gather_nd(mx.sym.var("x_data"), mx.sym.var("y_data"))
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x_data": xshape, "y_data": yshape}, {"x_data": "float32", "y_data": "int32"})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((2, 2), (2, 3), [[1, 1, 0], [0, 1, 0]])
    verify((2, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify((3, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify((3, 2), (2, 2, 3), [[[0, 1, 2], [2, 0, 1]], [[0, 0, 0], [1, 1, 1]]])

def test_forward_bilinear_resize():
    # add tests including scale_height and scale_width when mxnet is updated to version 1.5
    data = mx.sym.var('data')
    mx_sym = mx.sym.contrib.BilinearResize2D(data, height=5, width=10)
    verify_mxnet_frontend_impl(mx_sym, (1, 2, 3, 4), (1, 2, 5, 10))

def test_forward_rnn_layer():
    def verify(mode, seq_len, input_size, hidden_size, num_layers,
               batch=1, init_states=True, bidirectional=False):
        if mode == "rnn":
            layer = gluon.rnn.RNN(hidden_size, num_layers, bidirectional=bidirectional)
        elif mode == "gru":
            layer = gluon.rnn.GRU(hidden_size, num_layers, bidirectional=bidirectional)
        else: # mode == "lstm"
            layer = gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=bidirectional)
        num_states = 2 if mode == "lstm" else 1
        layer.initialize()
        layer.hybridize()

        dtype = "float32"
        directions = 2 if bidirectional else 1
        data_np = np.random.uniform(size=(seq_len, batch, input_size)).astype(dtype)
        data_mx = mx.nd.array(data_np)

        if init_states:
            shape_dict = {'data0': data_np.shape}
            inputs = {'data0': data_np}
            state_shape = (num_layers*directions, batch, hidden_size)
            states_np = []
            states_mx = []
            for i in range(num_states):
                s = np.random.uniform(size=state_shape).astype(dtype)
                states_np.append(s)
                states_mx.append(mx.nd.array(s))
                shape_dict['data%s' % (i+1)] = s.shape
                inputs['data%s' % (i+1)] = s
            mx_out, mx_states = layer(data_mx, states_mx)
            mx_res = [mx_out] + mx_states
        else:
            shape_dict = {'data': data_np.shape}
            inputs = {'data': data_np}
            mx_res = layer(data_mx)

        mx_sym = layer._cached_graph[1]
        mx_params = {}
        for name, param in layer.collect_params().items():
            mx_params[name] = param._reduce()

        mod, params = relay.frontend.from_mxnet(
            mx_sym, shape=shape_dict, arg_params=mx_params)
        for target, ctx in ctx_list():
            # only test graph runtime because debug runtime is too slow
            for kind in ["graph"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(**inputs, **params)
                if init_states:
                    assert len(op_res) == len(mx_res)
                    for i, val in enumerate(op_res):
                        tvm.testing.assert_allclose(
                            val.asnumpy(), mx_res[i].asnumpy(), rtol=1e-3)
                else:
                    tvm.testing.assert_allclose(
                        op_res.asnumpy(), mx_res.asnumpy(), rtol=1e-3)

    for mode in ["rnn", "gru", "lstm"]:
        verify(mode, 1, 64, 64, 1)
        verify(mode, 10, 64, 64, 2)
        verify(mode, 10, 64, 32, 2)
        verify(mode, 10, 64, 32, 2, batch=2)
        verify(mode, 10, 32, 64, 1, bidirectional=True)
        # The following two codeblocks need to be fixed for mxnet 1.5
        # verify(mode, 10, 64, 64, 3, init_states=False)
        # verify(mode, 10, 64, 64, 3, batch=2, bidirectional=True, init_states=False)

def test_forward_Crop():
    def verify(xshape, yshape, offset=None):
        x_data = np.random.uniform(size=xshape).astype("float32")
        y_data = np.random.uniform(size=yshape).astype("float32")
        if offset is None:
            mx_sym = mx.sym.Crop(mx.sym.var("x"), mx.sym.var("y"))
            ref_res = mx.nd.Crop(mx.nd.array(x_data), mx.nd.array(y_data))
        else:
            mx_sym = mx.sym.Crop(mx.sym.var("x"), mx.sym.var("y"), offset=offset)
            ref_res = mx.nd.Crop(mx.nd.array(x_data), mx.nd.array(y_data), offset=offset)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": xshape, "y": yshape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                if offset is None or offset == (0, 0):
                    op_res = intrp.evaluate()(x_data, y_data)
                else:
                    op_res = intrp.evaluate()(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((1, 3, 40, 40), (1, 3, 20, 20))
    verify((1, 3, 40, 40), (1, 3, 20, 20), (0, 0))
    verify((1, 3, 40, 40), (1, 3, 20, 20), (10, 10))
    verify((5, 32, 40, 40), (5, 32, 25, 25))
    verify((5, 32, 40, 40), (5, 32, 25, 25), (5, 5))

def test_forward_argsort():
    def verify(shape, axis, is_ascend, dtype="float32"):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.argsort(mx.nd.array(x_np), axis=axis, is_ascend=is_ascend, dtype=dtype)
        mx_sym = mx.sym.argsort(mx.sym.var("x"), axis=axis, is_ascend=is_ascend, dtype=dtype)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((2, 3, 4), axis=0, is_ascend=False)
    verify((1, 4, 6), axis=1, is_ascend=True)
    verify((3, 5, 6), axis=-3, is_ascend=False, dtype="int32")

def test_forward_topk():
    def verify(shape, k, axis, ret_type, is_ascend=False, dtype="float32"):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.topk(mx.nd.array(x_np), k=k, axis=axis, ret_typ=ret_type,
                             is_ascend=is_ascend, dtype=dtype)
        mx_sym = mx.sym.topk(mx.sym.var("x"), k=k, axis=axis, ret_typ=ret_type,
                             is_ascend=is_ascend, dtype=dtype)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np)
                if isinstance(ref_res, list):
                    assert len(op_res) == len(ref_res)
                    for i, t in enumerate(op_res):
                        tvm.testing.assert_allclose(t.asnumpy(), ref_res[i].asnumpy())
                else:
                    tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((3, 4), k=1, axis=0, ret_type="both")
    verify((3, 4), k=1, axis=-1, ret_type="indices")
    verify((3, 5, 6), k=2, axis=2, ret_type="value")
    verify((3, 5, 6), k=2, axis=1, ret_type="value", is_ascend=True)
    verify((3, 5, 6), k=0, axis=2, ret_type="both", dtype="int32")

def test_forward_sequence_mask():
    def verify(shape, use_sequence_length, value, axis, dtype, itype):
        data_np = np.random.uniform(size=shape).astype(dtype)
        valid_length_np = np.random.randint(0, shape[axis], size=shape[1-axis]).astype(itype)
        if use_sequence_length:
            ref_res = mx.nd.SequenceMask(mx.nd.array(data_np, dtype=dtype),
                                         sequence_length=mx.nd.array(valid_length_np, dtype=itype),
                                         use_sequence_length=use_sequence_length,
                                         value=value,
                                         axis=axis)
            mx_sym = mx.sym.SequenceMask(mx.sym.var('data'),
                                         sequence_length=mx.sym.var('valid_length'),
                                         use_sequence_length=use_sequence_length,
                                         value=value,
                                         axis=axis)
            mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": shape,
                                                        'valid_length': valid_length_np.shape},
                                               dtype={"data": dtype,
                                                      "valid_length": itype})
        else:
            ref_res = mx.nd.SequenceMask(mx.nd.array(data_np, dtype=dtype),
                                         use_sequence_length=use_sequence_length,
                                         value=value,
                                         axis=axis)
            mx_sym = mx.sym.SequenceMask(mx.sym.var('data'),
                                         use_sequence_length=use_sequence_length,
                                         value=value,
                                         axis=axis)
            mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": shape}, dtype={"data": dtype})
        for target, ctx in ctx_list():
            for kind in ['graph', 'debug']:
                if use_sequence_length is False and kind == 'graph':
                    # Disable the test for 'graph' when it's identity.
                    continue
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                if use_sequence_length:
                    op_res = intrp.evaluate()(data_np, valid_length_np)
                else:
                    op_res = intrp.evaluate()(data_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((5, 10), True, 0.0, 0, 'float32', 'float32')
    verify((5, 4, 3), True, 1.0, 1, 'float32', 'float32')
    verify((5, 4, 3), False, 1.0, 1, 'float64', 'float64')
    verify((5, 4, 3, 2), True, 1.0, 0, 'float32', 'float32')

def test_forward_contrib_div_sqrt_dim():
    def verify(shape):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.contrib.div_sqrt_dim(mx.nd.array(x_np))
        mx_sym = mx.sym.contrib.div_sqrt_dim(mx.sym.var("x"))
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x_np)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy())
    verify((3, 4))
    verify((3, 4, 5))

def test_forward_batch_norm():
    def verify(shape, axis=1, fix_gamma=False):
        x = np.random.uniform(size=shape).astype("float32")
        gamma = np.random.uniform(size=(shape[axis])).astype("float32")
        beta = np.random.uniform(size=(shape[axis])).astype("float32")
        moving_mean = np.random.uniform(size=(shape[axis])).astype("float32")
        moving_var = np.random.uniform(size=(shape[axis])).astype("float32")
        ref_res = mx.nd.BatchNorm(mx.nd.array(x), mx.nd.array(gamma), mx.nd.array(beta),
                                  mx.nd.array(moving_mean), mx.nd.array(moving_var),
                                  axis=axis, use_global_stats=True, fix_gamma=fix_gamma)
        mx_sym = mx.sym.BatchNorm(mx.sym.var("x"), mx.sym.var("gamma"),
                                  mx.sym.var("beta"), mx.sym.var("mean"),
                                  mx.sym.var("var"), axis=axis, use_global_stats=True,
                                  fix_gamma=fix_gamma)

        shape_dict = {"x": x.shape, "gamma": gamma.shape, "beta": beta.shape,
                      "mean": moving_mean.shape, "var": moving_var.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        #print(mod)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x, gamma, beta, moving_mean, moving_var)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy(), rtol=1e-3)
    verify((2, 3, 4, 5))
    verify((2, 3, 4, 5), axis=0)
    verify((2, 3, 4, 5), axis=-1)
    verify((2, 3, 4, 5), fix_gamma=True)


def test_forward_layer_norm():
    def verify(shape, axis=-1):
        x = np.random.uniform(size=shape).astype("float32")
        gamma = np.random.uniform(size=(shape[axis])).astype("float32")
        beta = np.random.uniform(size=(shape[axis])).astype("float32")
        ref_res = mx.nd.LayerNorm(mx.nd.array(x), mx.nd.array(gamma), mx.nd.array(beta),
                                  axis=axis)
        mx_sym = mx.sym.LayerNorm(mx.sym.var("x"), mx.sym.var("gamma"),
                                  mx.sym.var("beta"), axis=axis)
        shape_dict = {"x": x.shape, "gamma": gamma.shape, "beta": beta.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x, gamma, beta)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy(), rtol=1e-3)
    verify((2, 5))
    verify((2, 5), axis=0)
    verify((2, 5, 6))

def test_forward_one_hot():
    def verify(indices_shape, depth, on_value, off_value, dtype):
        x = np.random.randint(0, 5, size=indices_shape)
        ref_res = mx.nd.one_hot(mx.nd.array(x), depth, on_value, off_value, dtype)
        mx_sym = mx.sym.one_hot(mx.sym.var("x"), depth, on_value, off_value, dtype)
        shape_dict = {"x": x.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, mod=mod, ctx=ctx, target=target)
                op_res = intrp.evaluate()(x.astype("float32"))
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res.asnumpy(), rtol=1e-3)
    verify((3,), 3, 1, 0, "int32")
    verify((3,), 3, 1.0, 0.0, "float32")
    verify((2, 2), 5, 2, -2, "int32")
    verify((2, 2), 5, 0.5, -0.5, "float32")
    verify((3, 2, 4, 5), 6, 1, 0, "int32")
    verify((3, 2, 4, 5), 6, 1.0, 0.0, "float32")

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
    test_forward_adaptive_pooling()
    test_forward_lrn()
    test_forward_ones()
    test_forward_zeros()
    test_forward_ones_like()
    test_forward_zeros_like()
    test_forward_argmax()
    test_forward_argmin()
    test_forward_where()
    test_forward_arange()
    test_forward_broadcast_ops()
    test_forward_elemwise_ops()
    test_forward_scalar_ops()
    test_forward_slice_like()
    test_forward_slice_axis()
    test_forward_l2_normalize()
    test_forward_shape_array()
    test_forward_squeeze()
    test_forward_broadcast_axis()
    test_forward_full()
    test_forward_embedding()
    test_forward_smooth_l1()
    test_forward_take()
    test_forward_gather_nd()
    test_forward_bilinear_resize()
    test_forward_rnn_layer()
    test_forward_Crop()
    test_forward_argsort()
    test_forward_topk()
    test_forward_sequence_mask()
    test_forward_contrib_div_sqrt_dim()
    test_forward_batch_norm()
    test_forward_layer_norm()
    test_forward_one_hot()
