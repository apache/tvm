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
import operator
import random

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay, te
from tvm.contrib import graph_executor

import model_zoo
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.model_zoo import vision


def verify_mxnet_frontend_impl(
    mx_symbol,
    data_shape=(1, 3, 224, 224),
    out_shape=(1, 1000),
    gluon_impl=False,
    name=None,
    dtype="float32",
):
    """Use name different from test to avoid pytest picking it up"""
    if gluon_impl:

        def get_gluon_output(name, x):
            net = vision.get_model(name)
            net.collect_params().initialize(mx.init.Xavier())
            net_sym = gluon.nn.SymbolBlock(
                outputs=net(mx.sym.var("data")),
                inputs=mx.sym.var("data"),
                params=net.collect_params(),
            )
            out = net_sym(mx.nd.array(x.astype(dtype))).asnumpy()
            return out, net_sym

    else:

        def get_mxnet_output(symbol, x, dtype="float32"):
            from collections import namedtuple

            Batch = namedtuple("Batch", ["data"])
            mod = mx.mod.Module(symbol, label_names=None)
            mod.bind(data_shapes=[("data", x.shape)], for_training=False)
            mod.init_params()
            mod.forward(Batch([mx.nd.array(x.astype(dtype))]))
            out = mod.get_outputs()[0].asnumpy()
            args, auxs = mod.get_params()
            return out, args, auxs

    def get_tvm_output(symbol, x, args, auxs, target, dev, dtype="float32"):
        shape_dict = {"data": x.shape}
        if gluon_impl:
            mod, params = relay.frontend.from_mxnet(symbol, shape_dict)
        else:
            mod, params = relay.frontend.from_mxnet(
                symbol, shape_dict, arg_params=args, aux_params=auxs
            )
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target, params=params)
        m = graph_executor.GraphModule(lib["default"](dev))
        # set inputs
        m.set_input("data", tvm.nd.array(x.astype(dtype)))
        m.run()
        # get outputs
        out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
        return out.numpy()

    # random input
    x = np.random.uniform(size=data_shape)
    if gluon_impl:
        gluon_out, gluon_sym = get_gluon_output(name, x)
        for target, dev in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output(gluon_sym, x, None, None, target, dev, dtype)
            tvm.testing.assert_allclose(gluon_out, tvm_out, rtol=1e-5, atol=1e-5)
    else:
        mx_out, args, auxs = get_mxnet_output(mx_symbol, x, dtype)
        assert "data" not in args
        for target, dev in tvm.testing.enabled_targets():
            tvm_out = get_tvm_output(mx_symbol, x, args, auxs, target, dev, dtype)
            tvm.testing.assert_allclose(mx_out, tvm_out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_mlp():
    mlp = model_zoo.mx_mlp()
    verify_mxnet_frontend_impl(mlp, data_shape=(1, 1, 28, 28), out_shape=(1, 10))


@tvm.testing.uses_gpu
def test_forward_vgg():
    for n in [11]:
        mx_sym = model_zoo.mx_vgg(n)
        verify_mxnet_frontend_impl(mx_sym)


@tvm.testing.uses_gpu
def test_forward_resnet():
    for n in [18]:
        mx_sym = model_zoo.mx_resnet(18)
        verify_mxnet_frontend_impl(mx_sym)


@tvm.testing.uses_gpu
def test_forward_leaky_relu():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))
    mx_sym = mx.sym.LeakyReLU(data, act_type="leaky")
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_elu():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type="elu")
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_rrelu():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type="rrelu", lower_bound=0.3, upper_bound=0.7)
    verify_mxnet_frontend_impl(mx_sym[0], (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_prelu():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type="prelu")
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_gelu():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.LeakyReLU(data, act_type="gelu")
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_softrelu():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.Activation(data, act_type="softrelu")
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_fc_flatten():
    # test flatten=True option in mxnet 0.11.1
    data = mx.sym.var("data")
    try:
        mx_sym = mx.sym.FullyConnected(data, num_hidden=100, flatten=True)
        verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 100))
        mx_sym = mx.sym.FullyConnected(mx.sym.Flatten(data), num_hidden=100, flatten=False)
        verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 100))
    except:
        pass


@tvm.testing.uses_gpu
def test_forward_clip():
    data = mx.sym.var("data")
    data = mx.sym.concat(data, -data, dim=1)  # negative part explicitly
    mx_sym = mx.sym.clip(data, a_min=0, a_max=1)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 6, 100, 100))


@tvm.testing.uses_gpu
def test_forward_split():
    data = mx.sym.var("data")
    mx_sym = mx.sym.split(data, axis=1, num_outputs=4, squeeze_axis=False)
    verify_mxnet_frontend_impl(mx_sym, (1, 4, 2, 1), (1, 1, 2, 1))


@tvm.testing.uses_gpu
def test_forward_split_squeeze():
    data = mx.sym.var("data")
    mx_sym = mx.sym.split(data, axis=1, num_outputs=4, squeeze_axis=True)
    verify_mxnet_frontend_impl(mx_sym, (1, 4, 2, 1), (1, 2, 1))


@tvm.testing.uses_gpu
def test_forward_expand_dims():
    data = mx.sym.var("data")
    mx_sym = mx.sym.expand_dims(data, axis=1)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 1, 3, 4))


@tvm.testing.uses_gpu
def test_forward_pooling():
    data = mx.sym.var("data")
    mx_sym = mx.sym.Pooling(data, kernel=(3, 3), pad=(1, 1), pool_type="avg")
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 8, 8))

    mx_sym = mx.sym.Pooling(data, kernel=(3, 3), pad=(1, 1), pool_type="max")
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 8, 8))


@tvm.testing.uses_gpu
def test_forward_pooling3d():
    data = mx.sym.var("data")
    mx_sym = mx.sym.Pooling(data, kernel=(3, 3, 3), pad=(1, 1, 1), pool_type="avg")
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8, 8), (1, 20, 8, 8, 8))

    mx_sym = mx.sym.Pooling(data, kernel=(3, 3, 3), pad=(1, 1, 1), pool_type="max")
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8, 8), (1, 20, 8, 8, 8))


@tvm.testing.uses_gpu
def test_forward_adaptive_pooling():
    data = mx.sym.var("data")
    mx_sym = mx.sym.contrib.AdaptiveAvgPooling2D(data, output_size=(1,))
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 1, 1))

    mx_sym = mx.sym.contrib.AdaptiveAvgPooling2D(data, output_size=(3, 3))
    verify_mxnet_frontend_impl(mx_sym, (1, 20, 8, 8), (1, 20, 3, 3))


@tvm.testing.uses_gpu
def test_forward_lrn():
    data = mx.sym.var("data")
    mx_sym = mx.sym.LRN(data, alpha=2, beta=2, knorm=1, nsize=5)
    verify_mxnet_frontend_impl(mx_sym, (1, 10, 24, 24), (1, 10, 24, 24))


@tvm.testing.uses_gpu
def test_forward_ones():
    data = mx.sym.var("data")
    ones = mx.sym.ones(shape=(2, 3, 4), dtype="float32")
    mx_sym = mx.sym.elemwise_add(data, ones)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))


@tvm.testing.uses_gpu
def test_forward_zeros():
    data = mx.sym.var("data")
    zeros = mx.sym.zeros(shape=(2, 3, 4), dtype="float32")
    mx_sym = mx.sym.elemwise_add(data, zeros)
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))


@tvm.testing.uses_gpu
def test_forward_ones_like():
    data = mx.sym.var("data")
    mx_sym = mx.sym.ones_like(data, dtype="float32")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))


@tvm.testing.uses_gpu
def test_forward_make_loss():
    data = mx.sym.var("data")
    ones = mx.sym.ones(shape=(2, 3, 4), dtype="float32")
    mx_sym = mx.sym.make_loss((data - ones) ** 2 / 2, dtype="float32")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))


@tvm.testing.uses_gpu
def test_forward_zeros_like():
    data = mx.sym.var("data")
    mx_sym = mx.sym.zeros_like(data, dtype="float32")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4), (2, 3, 4))


@tvm.testing.uses_gpu
def test_forward_argmax():
    data = mx.sym.var("data")
    mx_sym = mx.sym.argmax(data, axis=1)
    verify_mxnet_frontend_impl(mx_sym, (5, 3), (5,))


@tvm.testing.uses_gpu
def test_forward_argmin():
    data = mx.sym.var("data")
    mx_sym = mx.sym.argmin(data, axis=0)
    verify_mxnet_frontend_impl(mx_sym, (5, 4), (4,))


@tvm.testing.uses_gpu
def test_forward_slice():
    data = mx.sym.var("data")
    mx_sym = mx.sym.slice(data, begin=(0, 1), end=(2, 4))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 3))
    mx_sym = mx.sym.slice(data, begin=(-1, 1), end=(-3, 4), step=(-1, 2))
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (2, 2))


@tvm.testing.uses_gpu
def test_forward_where():
    cond = mx.sym.var("cond")
    x = mx.sym.var("x")
    y = mx.sym.var("y")
    dshape = (2, 2)
    dtype = "float32"
    mx_sym = mx.sym.where(cond, x, y)
    np_cond = np.array([[0, 1], [-1, 0]]).astype(dtype)
    np_x = np.random.uniform(size=dshape).astype(dtype)
    np_y = np.random.uniform(size=dshape).astype(dtype)
    mx_cond = mx.nd.array(np_cond)
    mx_x = mx.nd.array(np_x)
    mx_y = mx.nd.array(np_y)
    shapes = {"cond": dshape, "x": dshape, "y": dshape}
    mod = mx.mod.Module(mx_sym, label_names=None, data_names=["cond", "x", "y"])
    mod.bind(data_shapes=shapes.items(), for_training=False)
    mod.init_params()
    args, auxs = mod.get_params()
    mx_out = mx.nd.where(mx_cond, mx_x, mx_y).asnumpy()

    mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, args, auxs)
    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                np_cond, np_x, np_y
            )
            tvm.testing.assert_allclose(op_res.numpy(), mx_out)


@tvm.testing.uses_gpu
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
        ref_res = _mx_symbol(mx.nd, start, stop, step)
        mx_sym = _mx_symbol(mx.sym, start, stop, step)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(
                    kind, mod=mod, device=dev, target=target
                ).evaluate()()
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

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


@tvm.testing.uses_gpu
def test_forward_broadcast_ops():
    for op in [
        "broadcast_add",
        "broadcast_plus",
        "broadcast_sub",
        "broadcast_minus",
        "broadcast_mul",
        "broadcast_div",
        "broadcast_mod",
        "broadcast_maximum",
        "broadcast_minimum",
        "broadcast_equal",
        "broadcast_not_equal",
        "broadcast_greater",
        "broadcast_greater_equal",
        "broadcast_lesser",
        "broadcast_lesser_equal",
        "broadcast_power",
        "broadcast_logical_or",
        "broadcast_logical_and",
        "broadcast_logical_xor",
    ]:
        a_shape = (3, 4, 5)
        b_shape = (4, 5)
        if op == "broadcast_mod":
            dtype = "int32"
            a_np = np.random.randint(1, 100, size=a_shape).astype(dtype)
            b_np = np.random.randint(1, 100, size=b_shape).astype(dtype)
        else:
            dtype = "float32"
            a_np = np.random.uniform(size=a_shape).astype(dtype)
            b_np = np.random.uniform(size=b_shape).astype(dtype)
        mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var("a"), mx.sym.var("b")])
        ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np), mx.nd.array(b_np)])
        shapes = {"a": a_shape, "b": b_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np, b_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())


@tvm.testing.uses_gpu
def test_forward_elemwise_ops():
    for op in [
        "elemwise_add",
        "elemwise_sub",
        "elemwise_mul",
        "elemwise_div",
        "maximum",
        "minimum",
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
    ]:
        shape = (3, 4, 5)
        dtype = "float32"
        a_np = np.random.uniform(size=shape).astype(dtype)
        b_np = np.random.uniform(size=shape).astype(dtype)
        if type(op) == str:
            mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var("a"), mx.sym.var("b")])
            ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np), mx.nd.array(b_np)])
        else:
            mx_sym = op(mx.sym.var("a"), mx.sym.var("b"))
            ref_res = op(mx.nd.array(a_np), mx.nd.array(b_np))
        shapes = {"a": shape, "b": shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np, b_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())


@tvm.testing.uses_gpu
def test_forward_softmin():
    data = mx.sym.var("data")
    mx_sym = mx.sym.softmin(data)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 3, 100, 100))

    mx_sym = mx.sym.softmin(data, axis=2)
    verify_mxnet_frontend_impl(mx_sym, (1, 3, 100, 100), (1, 3, 100, 100))


@tvm.testing.uses_gpu
def test_forward_unary_ops():
    for op in [
        "abs",
        "sqrt",
        "ceil",
        "floor",
        "round",
        "reciprocal",
        "trunc",
        "softsign",
        "hard_sigmoid",
        "cos",
        "sin",
        "tan",
        "cosh",
        "sinh",
        "tanh",
        "arccos",
        "arcsin",
        "arctan",
        "arccosh",
        "arcsinh",
        "arctanh",
    ]:
        shape = (1, 3, 4, 5)
        dtype = "float32"
        a_np = np.random.uniform(size=shape).astype(dtype)
        mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var("a")])
        ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np)])
        shapes = {"a": shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_forward_scalar_ops():
    for op in [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        operator.pow,
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
    ]:
        dtype = "float32"
        a_shape = (3, 4, 5)
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_scalar = 2.3
        mx_sym = op(mx.sym.var("a"), b_scalar)
        ref_res = op(mx.nd.array(a_np), b_scalar)
        shapes = {"a": a_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())
    for op in ["maximum", "minimum"]:
        dtype = "float32"
        a_shape = (3, 4, 5)
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_scalar = 2.3
        mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var("a"), b_scalar])
        ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(a_np), b_scalar])
        shapes = {"a": a_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())


@tvm.testing.uses_gpu
def test_forward_slice_axis():
    def verify(shape, axis, begin, end):
        data_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.slice_axis(mx.nd.array(data_np), axis, begin, end)
        mx_sym = mx.sym.slice_axis(mx.sym.var("data"), axis, begin, end)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((3, 4), 0, 1, 2)
    verify((3, 4), 0, 1, None)
    verify((3, 4), 1, 0, 2)
    verify((3, 4), 1, -3, -1)
    verify((3, 4), -1, -3, -1)


@tvm.testing.uses_gpu
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
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np, y_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((3, 4), (2, 3), None)
    verify((3, 4), (2, 3), (0, 1))
    verify((3, 4), (2, 3), (0))
    verify((3, 4), (2, 3), (-1))


@tvm.testing.uses_gpu
def test_forward_sequence_reverse():
    def verify(shape, seq_lengths, use_seq_lengths, seq_axis):
        data_np = np.random.uniform(size=shape).astype("float32")

        ref_res_args = [mx.nd.array(data_np), None, use_seq_lengths, seq_axis]
        mx_sym_args = [mx.sym.var("data"), None, use_seq_lengths, seq_axis]
        from_mxnet_args = [{"data": shape}, {"data": "float32"}]
        in_data = [data_np]

        if use_seq_lengths and seq_lengths:
            seq_lengths_np = np.array(seq_lengths).astype("int32")
            ref_res_args[1] = mx.nd.array(seq_lengths_np)
            mx_sym_args[1] = mx.sym.var("seq_lengths")
            from_mxnet_args[0].update({"seq_lengths": seq_lengths_np.shape})
            from_mxnet_args[1].update({"seq_lengths": "int32"})
            in_data.append(seq_lengths_np)

        ref_res = mx.nd.SequenceReverse(*ref_res_args)
        mx_sym = mx.sym.SequenceReverse(*mx_sym_args)
        mod, _ = relay.frontend.from_mxnet(mx_sym, *from_mxnet_args)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    *in_data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((3, 4), [1, 2, 3, 1], True, 0)
    verify((3, 4), None, False, 0)
    verify((3, 5, 5, 6), [1, 2, 3, 1, 3], True, 0)
    # MXNet accepts axis value as 0 only
    # verify((3, 4, 5, 6), None, False, 2)


@tvm.testing.uses_gpu
def test_forward_l2_normalize():
    data = mx.sym.var("data")
    mx_sym = mx.sym.L2Normalization(data, mode="channel")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4, 5), (2, 3, 4, 5))

    mx_sym = mx.sym.L2Normalization(data, mode="instance")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4, 5), (2, 3, 4, 5))

    mx_sym = mx.sym.L2Normalization(data, mode="spatial")
    verify_mxnet_frontend_impl(mx_sym, (2, 3, 4, 5), (2, 3, 4, 5))


@tvm.testing.uses_gpu
def test_forward_logistic_regression_output():
    data_shape = (1, 10)
    dtype = "float32"
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    label_np = np.random.uniform(size=data_shape).astype(dtype)
    mx_sym = mx.symbol.LogisticRegressionOutput(mx.sym.var("data"), mx.sym.var("label"))
    ref_res = mx.nd.LogisticRegressionOutput(mx.nd.array(data_np), mx.nd.array(label_np))
    shapes = {"data": data_shape}
    mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                data_np
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())


@tvm.testing.uses_gpu
def test_forward_dot():
    def verify(a_shape, b_shape, transpose_b=False):
        dtype = "float32"
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        b_np = np.random.uniform(size=b_shape).astype(dtype)
        mx_sym = mx.symbol.dot(mx.sym.var("a"), mx.sym.var("b"), transpose_b=transpose_b)
        ref_res = mx.nd.dot(mx.nd.array(a_np), mx.nd.array(b_np), transpose_b=transpose_b)
        shapes = {"a": a_shape, "b": b_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np, b_np
                )
                tvm.testing.assert_allclose(
                    op_res.numpy(), ref_res.asnumpy(), rtol=1e-05, atol=1e-05
                )

    verify((1, 256), (256, 1))
    verify((1, 256), (1, 256), transpose_b=True)
    verify((5,), (5,))
    verify((3,), (3, 5))
    verify((3,), (5, 3), transpose_b=True)
    verify((3,), (3, 5, 3, 5))
    verify((3,), (5, 5, 3, 3), transpose_b=True)
    verify((10, 1), (1,))
    verify((1, 1), (4, 3, 2, 1), transpose_b=True)
    verify((4, 3, 2, 1), (1,))
    verify((1, 2, 3, 4), (1, 4), transpose_b=True)
    verify((4, 1, 1), (1, 2, 3))
    verify((1, 1, 4), (2, 3, 4), transpose_b=True)


@tvm.testing.uses_gpu
def test_forward_shape_array():
    def verify(shape):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.shape_array(mx.nd.array(x_np))
        mx_sym = mx.sym.shape_array(mx.sym.var("x"))
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((1,))
    verify((3, 4, 5))
    verify((3, 4, 5, 6))


@tvm.testing.uses_gpu
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
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((1, 3, 1), None)
    verify((1, 3, 1), 0)
    verify((1, 3, 1), 2)
    verify((1, 3, 1), (0, 2))


@tvm.testing.uses_gpu
def test_forward_broadcast_axis():
    def verify(shape, axis, size):
        x_np = np.random.uniform(size=shape).astype("float32")
        for op in ["broadcast_axis", "broadcast_axes"]:
            mx_sym = _mx_symbol(mx.sym, op, [mx.sym.var("x"), axis, size])
            ref_res = _mx_symbol(mx.nd, op, [mx.nd.array(x_np), axis, size])
            mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
            for target, dev in tvm.testing.enabled_targets():
                for kind in ["graph", "debug"]:
                    op_res = relay.create_executor(
                        kind, mod=mod, device=dev, target=target
                    ).evaluate()(x_np)
                    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((1, 2, 1), 2, 3)
    verify((1, 2, 1), (0, 2), (2, 3))


@tvm.testing.uses_gpu
def test_forward_broadcast_to():
    def verify(input_shape, shape):
        x_np = np.random.uniform(size=input_shape).astype("float32")
        ref_res = mx.nd.broadcast_to(mx.nd.array(x_np), shape=shape)
        mx_sym = mx.sym.broadcast_to(mx.sym.var("x"), shape=shape)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": input_shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((1, 2, 3), (3, 2, 3))
    verify((4, 1, 32, 32), (4, 8, 32, 32))


@tvm.testing.uses_gpu
def test_forward_broadcast_like():
    def verify(input_shape, like_shape):
        x_np = np.random.uniform(size=input_shape).astype("float32")
        y_np = np.random.uniform(size=like_shape).astype("float32")
        ref_res = mx.nd.broadcast_like(mx.nd.array(x_np), mx.nd.array(y_np))
        mx_sym = mx.sym.broadcast_like(mx.sym.var("x"), mx.sym.var("y"))
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": input_shape, "y": like_shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np, y_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((1, 2, 3), (3, 2, 3))
    verify((4, 1, 32, 32), (4, 8, 32, 32))


@tvm.testing.uses_gpu
def test_forward_logical_not():
    a_shape = (3, 4, 5)
    dtype = "float32"
    a_np = np.random.uniform(size=a_shape).astype(dtype)
    mx_sym = mx.sym.logical_not(mx.sym.var("a"))
    ref_res = mx.nd.logical_not(mx.nd.array(a_np))
    shapes = {"a": a_shape}
    mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)
    for target, dev in tvm.testing.enabled_targets():
        for kind in ["graph", "debug"]:
            op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                a_np
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())


@tvm.testing.uses_gpu
def test_forward_full():
    def verify(val, shape, dtype):
        dev = mx.cpu()
        ref_res = mx.nd.full(shape, val, dtype=dtype)
        mx_sym = mx.sym.full(shape, val, dtype=dtype)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {})
        for target, dev in tvm.testing.enabled_targets():
            # Skip testing graph executor because this op will be optimized out
            # by constant folding.
            for kind in ["debug"]:
                op_res = relay.create_executor(
                    kind, mod=mod, device=dev, target=target
                ).evaluate()()
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify(2, (3, 4), "float32")
    verify(2, (3, 4), "int32")
    verify(3.5, (1, 3, 4), "float32")


@tvm.testing.uses_gpu
def test_forward_embedding():
    def verify(data_shape, weight_shape):
        in_dim, out_dim = weight_shape
        x_np = np.random.randint(0, weight_shape[0], size=data_shape).astype("float32")
        w_np = np.random.uniform(size=weight_shape).astype("float32")
        ref_res = mx.nd.Embedding(
            mx.nd.array(x_np), mx.nd.array(w_np), input_dim=in_dim, output_dim=out_dim
        )
        mx_sym = mx.sym.Embedding(
            mx.sym.var("x"), mx.sym.var("w"), input_dim=in_dim, output_dim=out_dim
        )
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": data_shape, "w": weight_shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x=x_np, w=w_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((2, 2), (4, 5))
    verify((2, 3, 4), (4, 5))


@tvm.testing.uses_gpu
def test_forward_smooth_l1():
    data = mx.sym.var("data")
    mx_sym = mx.sym.smooth_l1(data)
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (3, 4))
    mx_sym = mx.sym.smooth_l1(data, scalar=1.0)
    verify_mxnet_frontend_impl(mx_sym, (3, 4), (3, 4))


@tvm.testing.uses_gpu
def test_forward_take():
    def verify(shape, indices_src, axis, mode="clip"):
        x_np = np.random.uniform(size=shape).astype("float32")
        indices_np = np.array(indices_src, dtype="float32")
        ref_res = mx.nd.take(mx.nd.array(x_np), mx.nd.array(indices_np), axis, mode)
        mx_sym = mx.sym.take(mx.sym.var("x"), mx.sym.var("y"), axis, mode)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape, "y": indices_np.shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np, indices_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((2, 2), [[[1, 0], [0, 1]]], 0)
    verify((2, 2), [[[1, 0], [0, 1]]], 1)
    verify((4, 3, 5, 6), [[2, 1, 0, 0]], -2)
    verify((3, 4), [-1, 5], 0)
    verify((3, 4), [-1, 5], 0, mode="wrap")
    verify((3, 4), [-1, 5], 1)
    verify((3, 4), [-1, 5], 1, mode="wrap")


@tvm.testing.uses_gpu
def test_forward_gather_nd():
    def verify(xshape, yshape, y_data, error=False):
        x_data = np.random.uniform(size=xshape).astype("float32")
        ref_res = mx.nd.gather_nd(mx.nd.array(x_data), mx.nd.array(y_data))
        mx_sym = mx.sym.gather_nd(mx.sym.var("x_data"), mx.sym.var("y_data"))
        mod, _ = relay.frontend.from_mxnet(
            mx_sym, {"x_data": xshape, "y_data": yshape}, {"x_data": "float32", "y_data": "int32"}
        )
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_data, y_data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((2, 2), (2, 3), [[1, 1, 0], [0, 1, 0]])
    verify((2, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify((3, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify((3, 2), (2, 2, 3), [[[0, 1, 2], [2, 0, 1]], [[0, 0, 0], [1, 1, 1]]])
    verify((1, 4), (1, 1), [[0]])


@tvm.testing.uses_gpu
def test_forward_bilinear_resize():
    # add tests including scale_height and scale_width when mxnet is updated to version 1.5
    data = mx.sym.var("data")
    mx_sym = mx.sym.contrib.BilinearResize2D(data, height=5, width=10)
    verify_mxnet_frontend_impl(mx_sym, (1, 2, 3, 4), (1, 2, 5, 10))


@tvm.testing.uses_gpu
def test_forward_grid_generator():
    def verify(shape, transform_type, target_shape):
        x = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.GridGenerator(mx.nd.array(x), transform_type, target_shape)
        mx_sym = mx.sym.GridGenerator(mx.sym.var("x"), transform_type, target_shape)
        shape_dict = {"x": x.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5, atol=1e-5)

    verify((4, 6), "affine", (16, 32))
    verify((4, 2, 16, 16), "warp", None)
    verify((1, 2, 16, 16), "warp", None)


@tvm.testing.uses_gpu
def test_forward_bilinear_sampler():
    def verify(data_shape, grid_shape):
        data = np.random.uniform(size=data_shape).astype("float32")
        grid = np.random.uniform(low=-1.5, high=1.5, size=grid_shape).astype("float32")
        ref_res = mx.nd.BilinearSampler(mx.nd.array(data), mx.nd.array(grid))
        mx_sym = mx.sym.BilinearSampler(mx.sym.var("data"), mx.sym.var("grid"))
        shape_dict = {"data": data.shape, "grid": grid.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data, grid
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5, atol=1e-5)

    verify((4, 4, 16, 32), (4, 2, 8, 8))
    verify((4, 4, 16, 32), (4, 2, 32, 32))


@tvm.testing.uses_gpu
def test_forward_rnn_layer():
    def verify(
        mode,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        batch=1,
        init_states=True,
        bidirectional=False,
    ):
        if mode == "rnn":
            layer = gluon.rnn.RNN(hidden_size, num_layers, bidirectional=bidirectional)
        elif mode == "gru":
            layer = gluon.rnn.GRU(hidden_size, num_layers, bidirectional=bidirectional)
        else:  # mode == "lstm"
            layer = gluon.rnn.LSTM(hidden_size, num_layers, bidirectional=bidirectional)
        num_states = 2 if mode == "lstm" else 1
        layer.initialize()
        layer.hybridize()

        dtype = "float32"
        directions = 2 if bidirectional else 1
        data_np = np.random.uniform(size=(seq_len, batch, input_size)).astype(dtype)
        data_mx = mx.nd.array(data_np)

        if init_states:
            shape_dict = {"data0": data_np.shape}
            inputs = {"data0": data_np}
            state_shape = (num_layers * directions, batch, hidden_size)
            states_np = []
            states_mx = []
            for i in range(num_states):
                s = np.random.uniform(size=state_shape).astype(dtype)
                states_np.append(s)
                states_mx.append(mx.nd.array(s))
                shape_dict["data%s" % (i + 1)] = s.shape
                inputs["data%s" % (i + 1)] = s
            mx_out, mx_states = layer(data_mx, states_mx)
            mx_res = [mx_out] + mx_states
        else:
            shape_dict = {"data": data_np.shape}
            inputs = {"data": data_np}
            mx_res = layer(data_mx)

        mx_sym = layer._cached_graph[1]
        mx_params = {}
        for name, param in layer.collect_params().items():
            mx_params[name] = param._reduce()

        mod, params = relay.frontend.from_mxnet(mx_sym, shape=shape_dict, arg_params=mx_params)
        for target, dev in tvm.testing.enabled_targets():
            # only test graph executor because debug runtime is too slow
            for kind in ["graph"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    **inputs, **params
                )
                if init_states:
                    assert len(op_res) == len(mx_res)
                    for i, val in enumerate(op_res):
                        tvm.testing.assert_allclose(val.numpy(), mx_res[i].asnumpy(), rtol=1e-3)
                else:
                    tvm.testing.assert_allclose(op_res.numpy(), mx_res.asnumpy(), rtol=1e-3)

    for mode in ["rnn", "gru", "lstm"]:
        verify(mode, 1, 64, 64, 1)
        verify(mode, 10, 64, 64, 2)
        verify(mode, 10, 64, 32, 2)
        verify(mode, 10, 64, 32, 2, batch=2)
        verify(mode, 10, 32, 64, 1, bidirectional=True)
        # The following two codeblocks need to be fixed for mxnet 1.5
        # verify(mode, 10, 64, 64, 3, init_states=False)
        # verify(mode, 10, 64, 64, 3, batch=2, bidirectional=True, init_states=False)


@tvm.testing.uses_gpu
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
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                func = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()
                if offset is None or offset == (0, 0):
                    op_res = func(x_data, y_data)
                else:
                    op_res = func(x_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((1, 3, 40, 40), (1, 3, 20, 20))
    verify((1, 3, 40, 40), (1, 3, 20, 20), (0, 0))
    verify((1, 3, 40, 40), (1, 3, 20, 20), (10, 10))
    verify((5, 32, 40, 40), (5, 32, 25, 25))
    verify((5, 32, 40, 40), (5, 32, 25, 25), (5, 5))


@tvm.testing.uses_gpu
def test_forward_argsort():
    def verify(shape, axis, is_ascend, dtype="float32"):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.argsort(mx.nd.array(x_np), axis=axis, is_ascend=is_ascend, dtype=dtype)
        mx_sym = mx.sym.argsort(mx.sym.var("x"), axis=axis, is_ascend=is_ascend, dtype=dtype)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((2, 3, 4), axis=0, is_ascend=False)
    verify((1, 4, 6), axis=1, is_ascend=True)
    verify((3, 5, 6), axis=-3, is_ascend=False, dtype="int32")


@tvm.testing.uses_gpu
def test_forward_topk():
    def verify(shape, k, axis, ret_type, is_ascend=None, dtype="float32"):
        x_np = np.random.uniform(size=shape).astype("float32")
        if is_ascend is None:
            ref_res = mx.nd.topk(mx.nd.array(x_np), k=k, axis=axis, ret_typ=ret_type, dtype=dtype)
            mx_sym = mx.sym.topk(mx.sym.var("x"), k=k, axis=axis, ret_typ=ret_type, dtype=dtype)
        else:
            ref_res = mx.nd.topk(
                mx.nd.array(x_np),
                k=k,
                axis=axis,
                ret_typ=ret_type,
                is_ascend=is_ascend,
                dtype=dtype,
            )
            mx_sym = mx.sym.topk(
                mx.sym.var("x"), k=k, axis=axis, ret_typ=ret_type, is_ascend=is_ascend, dtype=dtype
            )
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np
                )
                if isinstance(ref_res, list):
                    assert len(op_res) == len(ref_res)
                    for i, t in enumerate(op_res):
                        tvm.testing.assert_allclose(t.numpy(), ref_res[i].asnumpy())
                else:
                    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((3, 4), k=1, axis=0, ret_type="both")
    verify((3, 4), k=1, axis=-1, ret_type="indices")
    verify((3, 5, 6), k=2, axis=2, ret_type="value", is_ascend=False)
    verify((3, 5, 6), k=2, axis=1, ret_type="value", is_ascend=True)
    verify((3, 5, 6), k=0, axis=2, ret_type="both", dtype="int32")


@tvm.testing.uses_gpu
def test_forward_sequence_mask():
    def verify(shape, use_sequence_length, value, axis, dtype, itype):
        data_np = np.random.uniform(size=shape).astype(dtype)
        valid_length_np = np.random.randint(0, shape[axis], size=shape[1 - axis]).astype(itype)
        if use_sequence_length:
            ref_res = mx.nd.SequenceMask(
                mx.nd.array(data_np, dtype=dtype),
                sequence_length=mx.nd.array(valid_length_np, dtype=itype),
                use_sequence_length=use_sequence_length,
                value=value,
                axis=axis,
            )
            mx_sym = mx.sym.SequenceMask(
                mx.sym.var("data"),
                sequence_length=mx.sym.var("valid_length"),
                use_sequence_length=use_sequence_length,
                value=value,
                axis=axis,
            )
            mod, _ = relay.frontend.from_mxnet(
                mx_sym,
                {"data": shape, "valid_length": valid_length_np.shape},
                dtype={"data": dtype, "valid_length": itype},
            )
        else:
            ref_res = mx.nd.SequenceMask(
                mx.nd.array(data_np, dtype=dtype),
                use_sequence_length=use_sequence_length,
                value=value,
                axis=axis,
            )
            mx_sym = mx.sym.SequenceMask(
                mx.sym.var("data"), use_sequence_length=use_sequence_length, value=value, axis=axis
            )
            mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": shape}, dtype={"data": dtype})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                if use_sequence_length is False and kind == "graph":
                    # Disable the test for 'graph' when it's identity.
                    continue
                func = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()
                if use_sequence_length:
                    op_res = func(data_np, valid_length_np)
                else:
                    op_res = func(data_np)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((5, 10), True, 0.0, 0, "float32", "float32")
    verify((5, 4, 3), True, 1.0, 1, "float32", "float32")
    verify((5, 4, 3), False, 1.0, 1, "float64", "float64")
    verify((5, 4, 3, 2), True, 1.0, 0, "float32", "float32")


@tvm.testing.uses_gpu
def test_forward_contrib_div_sqrt_dim():
    def verify(shape):
        x_np = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.contrib.div_sqrt_dim(mx.nd.array(x_np))
        mx_sym = mx.sym.contrib.div_sqrt_dim(mx.sym.var("x"))
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"x": shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify((3, 4))
    verify((3, 4, 5))


@tvm.testing.uses_gpu
def test_forward_batch_norm():
    def verify(shape, axis=1, fix_gamma=False):
        x = np.random.uniform(size=shape).astype("float32")
        gamma = np.random.uniform(size=(shape[axis])).astype("float32")
        beta = np.random.uniform(size=(shape[axis])).astype("float32")
        moving_mean = np.random.uniform(size=(shape[axis])).astype("float32")
        moving_var = np.abs(np.random.uniform(size=(shape[axis])).astype("float32")) + 0.5
        ref_res = mx.nd.BatchNorm(
            mx.nd.array(x),
            mx.nd.array(gamma),
            mx.nd.array(beta),
            mx.nd.array(moving_mean),
            mx.nd.array(moving_var),
            axis=axis,
            use_global_stats=True,
            fix_gamma=fix_gamma,
        )
        mx_sym = mx.sym.BatchNorm(
            mx.sym.var("x"),
            mx.sym.var("gamma"),
            mx.sym.var("beta"),
            mx.sym.var("mean"),
            mx.sym.var("var"),
            axis=axis,
            use_global_stats=True,
            fix_gamma=fix_gamma,
        )

        shape_dict = {
            "x": x.shape,
            "gamma": gamma.shape,
            "beta": beta.shape,
            "mean": moving_mean.shape,
            "var": moving_var.shape,
        }
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        # print(mod)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x, gamma, beta, moving_mean, moving_var
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3)

    verify((2, 3, 4, 5))
    verify((2, 3, 4, 5), axis=0)
    verify((2, 3, 4, 5), axis=-1)
    verify((2, 3, 4, 5), fix_gamma=True)


@tvm.testing.uses_gpu
def test_forward_instance_norm():
    def verify(shape, axis=1, epsilon=1e-5):
        x = np.random.uniform(size=shape).astype("float32")
        gamma = np.random.uniform(size=(shape[axis])).astype("float32")
        beta = np.random.uniform(size=(shape[axis])).astype("float32")
        ref_res = mx.nd.InstanceNorm(mx.nd.array(x), mx.nd.array(gamma), mx.nd.array(beta), epsilon)
        mx_sym = mx.sym.InstanceNorm(
            mx.sym.var("x"), mx.sym.var("gamma"), mx.sym.var("beta"), epsilon
        )
        shape_dict = {"x": x.shape, "gamma": gamma.shape, "beta": beta.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x, gamma, beta
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=2e-5, atol=1e-5)

    verify((2, 3, 4, 5))
    verify((32, 64, 80, 64))
    verify((8, 6, 5))
    verify((8, 7, 6, 5, 4))


@tvm.testing.uses_gpu
def test_forward_layer_norm():
    def verify(shape, axis=-1):
        x = np.random.uniform(size=shape).astype("float32")
        gamma = np.random.uniform(size=(shape[axis])).astype("float32")
        beta = np.random.uniform(size=(shape[axis])).astype("float32")
        ref_res = mx.nd.LayerNorm(mx.nd.array(x), mx.nd.array(gamma), mx.nd.array(beta), axis=axis)
        mx_sym = mx.sym.LayerNorm(
            mx.sym.var("x"), mx.sym.var("gamma"), mx.sym.var("beta"), axis=axis
        )
        shape_dict = {"x": x.shape, "gamma": gamma.shape, "beta": beta.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x, gamma, beta
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((2, 5))
    verify((2, 5), axis=0)
    verify((2, 5, 6))


@tvm.testing.uses_gpu
def test_forward_group_norm():
    def verify(shape, num_groups=1):
        x = np.random.uniform(size=shape).astype("float32")
        gamma = np.random.uniform(size=(shape[1])).astype("float32")
        beta = np.random.uniform(size=(shape[1])).astype("float32")
        ref_res = mx.nd.GroupNorm(
            data=mx.nd.array(x),
            gamma=mx.nd.array(gamma),
            beta=mx.nd.array(beta),
            num_groups=num_groups,
        )
        mx_sym = mx.sym.GroupNorm(
            mx.sym.var("x"), mx.sym.var("gamma"), mx.sym.var("beta"), num_groups=num_groups
        )
        shape_dict = {"x": x.shape, "gamma": gamma.shape, "beta": beta.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x, gamma, beta
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((1, 4, 2), num_groups=4)
    # TODO(trevmorr): MXNet GroupNorm implementation is bugged for cases when num_groups != num_channels
    # https://github.com/apache/incubator-mxnet/pull/18199
    # verify((1, 4, 2, 3), num_groups=2)
    # verify((1, 4, 2, 3))


@tvm.testing.uses_gpu
def test_forward_one_hot():
    def verify(indices_shape, depth, on_value, off_value, dtype):
        x = np.random.randint(0, 5, size=indices_shape)
        ref_res = mx.nd.one_hot(mx.nd.array(x), depth, on_value, off_value, dtype)
        mx_sym = mx.sym.one_hot(mx.sym.var("x"), depth, on_value, off_value, dtype)
        shape_dict = {"x": x.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x.astype("float32")
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((3,), 3, 1, 0, "int32")
    verify((3,), 3, 1.0, 0.0, "float32")
    verify((2, 2), 5, 2, -2, "int32")
    verify((2, 2), 5, 0.5, -0.5, "float32")
    verify((3, 2, 4, 5), 6, 1, 0, "int32")
    verify((3, 2, 4, 5), 6, 1.0, 0.0, "float32")


@tvm.testing.uses_gpu
def test_forward_pad():
    def verify(data_shape, out_shape, mode, pad_width, constant_value=0.0):
        data = mx.sym.var("data")
        mx_sym = mx.sym.pad(data, mode=mode, pad_width=pad_width, constant_value=constant_value)
        verify_mxnet_frontend_impl(mx_sym, data_shape=data_shape, out_shape=out_shape)

    verify(
        data_shape=(1, 1, 3, 5),
        out_shape=(1, 1, 6, 12),
        mode="constant",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4),
    )
    verify(
        data_shape=(1, 1, 3, 5),
        out_shape=(1, 1, 6, 12),
        mode="constant",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4),
        constant_value=3.0,
    )
    verify(
        data_shape=(1, 1, 3, 5),
        out_shape=(1, 1, 6, 12),
        mode="edge",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4),
    )
    verify(
        data_shape=(1, 1, 3, 5),
        out_shape=(1, 1, 6, 12),
        mode="reflect",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4),
    )
    verify(
        data_shape=(1, 1, 3, 5, 7),
        out_shape=(1, 1, 6, 12, 18),
        mode="constant",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4, 5, 6),
    )
    verify(
        data_shape=(1, 1, 3, 5, 7),
        out_shape=(1, 1, 6, 12, 18),
        mode="constant",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4, 5, 6),
        constant_value=3.0,
    )
    verify(
        data_shape=(1, 1, 3, 5, 7),
        out_shape=(1, 1, 6, 12, 18),
        mode="edge",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4, 5, 6),
    )
    verify(
        data_shape=(1, 1, 3, 5, 7),
        out_shape=(1, 1, 6, 12, 18),
        mode="reflect",
        pad_width=(0, 0, 0, 0, 1, 2, 3, 4, 5, 6),
    )


@tvm.testing.uses_gpu
def test_forward_slice():
    def verify(data_shape, out_shape, begin, end):
        data = mx.sym.var("data")
        mx_sym = mx.sym.slice(data, begin=begin, end=end)
        verify_mxnet_frontend_impl(mx_sym, data_shape=data_shape, out_shape=out_shape)

    verify(data_shape=(1, 1, 10), out_shape=(1, 1, 8), begin=(0, 0, 2), end=(1, 1, 10))
    verify(
        data_shape=(1, 1, 10), out_shape=(1, 1, 8), begin=(None, None, 2), end=(None, None, None)
    )


@tvm.testing.uses_gpu
def test_forward_convolution():
    def verify(data_shape, kernel_size, stride, pad, num_filter, is_depthwise=False):
        if is_depthwise:
            groups = data_shape[1]
            weight_shape = (
                data_shape[1],
                num_filter // groups,
            ) + kernel_size
        else:
            groups = 1
            weight_shape = (
                num_filter,
                data_shape[1],
            ) + kernel_size
        x = np.random.uniform(size=data_shape).astype("float32")
        weight = np.random.uniform(size=weight_shape).astype("float32")
        bias = np.random.uniform(size=num_filter).astype("float32")
        ref_res = mx.nd.Convolution(
            data=mx.nd.array(x),
            weight=mx.nd.array(weight),
            bias=mx.nd.array(bias),
            kernel=kernel_size,
            stride=stride,
            pad=pad,
            num_filter=num_filter,
            num_group=groups,
        )
        mx_sym = mx.sym.Convolution(
            mx.sym.var("x"),
            mx.sym.var("weight"),
            mx.sym.var("bias"),
            kernel=kernel_size,
            stride=stride,
            pad=pad,
            num_filter=num_filter,
            num_group=groups,
        )
        shape_dict = {"x": x.shape, "weight": weight.shape, "bias": bias.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x, weight, bias
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3)

    verify(data_shape=(1, 1, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(20, 1, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(1, 8, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(20, 8, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(1, 1, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(data_shape=(20, 1, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(data_shape=(1, 8, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(data_shape=(20, 8, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(
        data_shape=(1, 8, 32, 32),
        kernel_size=(3, 3),
        stride=(1, 1),
        pad=(1, 1),
        num_filter=8,
        is_depthwise=True,
    )
    verify(
        data_shape=(1, 1, 16, 16, 16),
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        pad=(1, 1, 1),
        num_filter=2,
    )
    verify(
        data_shape=(20, 1, 16, 16, 16),
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        pad=(1, 1, 1),
        num_filter=2,
    )
    verify(
        data_shape=(1, 8, 16, 16, 16),
        kernel_size=(3, 3, 3),
        stride=(2, 2, 2),
        pad=(1, 1, 1),
        num_filter=2,
    )
    verify(
        data_shape=(20, 8, 16, 16, 16),
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
        pad=(1, 1, 1),
        num_filter=2,
    )


@tvm.testing.uses_gpu
def test_forward_deconvolution():
    def verify(data_shape, kernel_size, stride, pad, num_filter):
        weight_shape = (data_shape[1], num_filter) + kernel_size
        x = np.random.uniform(size=data_shape).astype("float32")
        weight = np.random.uniform(size=weight_shape).astype("float32")
        bias = np.random.uniform(size=num_filter).astype("float32")
        ref_res = mx.nd.Deconvolution(
            data=mx.nd.array(x),
            weight=mx.nd.array(weight),
            bias=mx.nd.array(bias),
            kernel=kernel_size,
            stride=stride,
            pad=pad,
            num_filter=num_filter,
            no_bias=False,
        )
        mx_sym = mx.sym.Deconvolution(
            mx.sym.var("x"),
            mx.sym.var("weight"),
            mx.sym.var("bias"),
            kernel=kernel_size,
            stride=stride,
            pad=pad,
            num_filter=num_filter,
            no_bias=False,
        )
        shape_dict = {"x": x.shape, "weight": weight.shape, "bias": bias.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x, weight, bias
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify(data_shape=(1, 1, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(20, 1, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(1, 8, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(20, 8, 1024 * 16), kernel_size=(17,), stride=(2,), pad=(8,), num_filter=4)
    verify(data_shape=(1, 1, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(data_shape=(20, 1, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(data_shape=(1, 8, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)
    verify(data_shape=(20, 8, 32, 32), kernel_size=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=2)


@tvm.testing.uses_gpu
def test_forward_cond():
    def verify(a_np, b_np):
        a_nd, b_nd = mx.nd.array(a_np), mx.nd.array(b_np)
        pred = a_nd * b_nd < 5
        then_func = lambda: (a_nd + 5) * (b_nd + 5)
        else_func = lambda: (a_nd - 5) * (b_nd - 5)
        ref_res = mx.nd.contrib.cond(pred, then_func, else_func)

        a_sym, b_sym = mx.sym.var("a"), mx.sym.var("b")
        pred = a_sym * b_sym < 5
        then_func = lambda: (a_sym + 5) * (b_sym + 5)
        else_func = lambda: (a_sym - 5) * (b_sym - 5)
        mx_sym = mx.sym.contrib.cond(pred, then_func, else_func)

        shape_dict = {"a": a_np.shape, "b": b_np.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["debug", "vm"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np, b_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3)

    verify(np.asarray([1.0], "float32"), np.asarray([2.0], "float32"))
    verify(np.asarray([4.0], "float32"), np.asarray([3.0], "float32"))


@tvm.testing.uses_gpu
def test_forward_amp_cast():
    def verify(from_dtype, to_dtype):
        from_np = np.random.uniform(size=(1, 3, 18)).astype(from_dtype)
        x_var = mx.sym.var("x", dtype=from_dtype)
        mx_sym = mx.sym.amp_cast(x_var, dtype=to_dtype)
        shape_dict = {"x": (1, 3, 18)}
        dtype_dict = {"x": from_dtype}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "vm", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    from_np
                )
                assert op_res.dtype == to_dtype, op_res.dtype
                tvm.testing.assert_allclose(op_res.numpy(), from_np.astype(to_dtype))

    verify("float32", "float16")
    verify("float16", "float32")


@tvm.testing.uses_gpu
def test_forward_amp_multicast():
    def verify(dtypes, cast_narrow, expected_dtype):
        x_nps = [np.random.uniform(size=(1, 3, 18)).astype(dtype) for dtype in dtypes]
        x_vars = [mx.sym.var(str(i), dtype=dtype) for i, dtype in enumerate(dtypes)]
        mx_sym = mx.sym.amp_multicast(*x_vars, cast_narrow=cast_narrow, num_outputs=len(dtypes))
        shape_dict = {}
        dtype_dict = {}
        for i, dtype in enumerate(dtypes):
            shape_dict[str(i)] = (1, 3, 18)
            dtype_dict[str(i)] = dtype
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "vm", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    *x_nps
                )
                for i, res in enumerate(op_res):
                    assert res.dtype == expected_dtype, res.dtype
                    tvm.testing.assert_allclose(res.numpy(), x_nps[i].astype(expected_dtype))

    verify(["float32", "float16"], False, "float32")
    verify(["float32", "float16"], True, "float16")
    verify(["float32", "float32"], False, "float32")
    verify(["float32", "float32"], True, "float32")
    verify(["float16", "float16"], False, "float16")
    verify(["float16", "float16"], True, "float16")


@tvm.testing.uses_gpu
def test_forward_unravel_index():
    def verify(x, shape, dtype):
        a_np = np.array(x).astype(dtype)
        mx_sym = _mx_symbol(mx.sym, "unravel_index", [mx.sym.var("a"), shape])
        ref_res = _mx_symbol(mx.nd, "unravel_index", [mx.nd.array(a_np), shape])
        shapes = {"a": a_np.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shapes, dtype)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "vm", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    a_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    for dtype in ["int32", "int64"]:
        verify([0, 1, 2, 3], [2, 2], dtype)
        verify([144, 13, 45], [6, 7, 10, 2], dtype)
        verify([456], [6, 7, 10, 2], dtype)

    # In below example, 5 is out of bound for array of size 4.
    # MXNet implementation provides different result than TVM
    # TVM implementation is inline with Tensorflow
    # Ideally error should be thrown just like Numpy
    # verify([0, 1, 2, 5], [2, 2], dtype)


@tvm.testing.uses_gpu
def test_forward_swap_axis():
    def _verify_swap_axis(in_shape, out_shape, dim1, dim2):
        data = mx.sym.var("data")
        mx_sym = mx.sym.swapaxes(data, dim1, dim2)
        verify_mxnet_frontend_impl(mx_sym, in_shape, out_shape)

    _verify_swap_axis((4, 5), (5, 4), 0, 1)
    _verify_swap_axis((2, 4, 4, 5), (2, 5, 4, 4), 1, 3)
    # MXNet errors out when dim1 == dim2
    # _verify_swap_axis((4, 5), (5, 4), 0, 0)


@tvm.testing.uses_gpu
def test_forward_depth_to_space():
    def verify(shape, blocksize=2):
        x = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.depth_to_space(mx.nd.array(x), blocksize)
        mx_sym = mx.sym.depth_to_space(mx.sym.var("x"), blocksize)
        shape_dict = {
            "x": x.shape,
        }
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((1, 18, 3, 3), 3)


@tvm.testing.uses_gpu
def test_forward_space_to_depth():
    def verify(shape, blocksize=2):
        x = np.random.uniform(size=shape).astype("float32")
        ref_res = mx.nd.space_to_depth(mx.nd.array(x), blocksize)
        mx_sym = mx.sym.space_to_depth(mx.sym.var("x"), blocksize)
        shape_dict = {
            "x": x.shape,
        }
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    x
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((1, 1, 9, 9), 3)


@tvm.testing.uses_gpu
def test_forward_correlation():
    def verify(data_shape, kernel_size, max_displacement, stride1, stride2, pad_size, is_multiply):
        data1 = np.random.uniform(size=data_shape).astype("float32")
        data2 = np.random.uniform(size=data_shape).astype("float32")
        ref_res = mx.nd.Correlation(
            data1=mx.nd.array(data1),
            data2=mx.nd.array(data2),
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride1=stride1,
            stride2=stride2,
            pad_size=pad_size,
            is_multiply=is_multiply,
        )
        mx_sym = mx.sym.Correlation(
            data1=mx.sym.var("data1"),
            data2=mx.sym.var("data2"),
            kernel_size=kernel_size,
            max_displacement=max_displacement,
            stride1=stride1,
            stride2=stride2,
            pad_size=pad_size,
            is_multiply=is_multiply,
        )
        shape_dict = {"data1": data1.shape, "data2": data2.shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data1, data2
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify(
        (1, 3, 10, 10),
        kernel_size=1,
        max_displacement=4,
        stride1=1,
        stride2=1,
        pad_size=4,
        is_multiply=False,
    )
    verify(
        (5, 1, 15, 15),
        kernel_size=1,
        max_displacement=5,
        stride1=1,
        stride2=1,
        pad_size=5,
        is_multiply=False,
    )
    verify(
        (5, 1, 15, 15),
        kernel_size=1,
        max_displacement=5,
        stride1=1,
        stride2=1,
        pad_size=5,
        is_multiply=True,
    )
    verify(
        (5, 1, 15, 15),
        kernel_size=1,
        max_displacement=10,
        stride1=1,
        stride2=2,
        pad_size=10,
        is_multiply=True,
    )
    verify(
        (5, 1, 4, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=1,
        stride2=1,
        pad_size=2,
        is_multiply=True,
    )
    verify(
        (5, 1, 4, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=1,
        pad_size=2,
        is_multiply=True,
    )
    verify(
        (5, 1, 4, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=1,
        pad_size=2,
        is_multiply=False,
    )
    verify(
        (5, 1, 6, 4),
        kernel_size=3,
        max_displacement=1,
        stride1=2,
        stride2=1,
        pad_size=2,
        is_multiply=False,
    )
    verify(
        (5, 1, 11, 11),
        kernel_size=5,
        max_displacement=1,
        stride1=1,
        stride2=1,
        pad_size=2,
        is_multiply=False,
    )


@tvm.testing.uses_gpu
def test_forward_arange_like():
    def verify(data_shape, start=None, step=None, axis=None):
        attrs = {}
        if start is not None:
            attrs["start"] = start
        if step is not None:
            attrs["step"] = step
        if axis is not None:
            attrs["axis"] = axis
        data = mx.sym.var("data")
        data_np = np.random.uniform(size=data_shape).astype("float32")
        ref_res = mx.nd.contrib.arange_like(mx.nd.array(data_np), **attrs)

        mx_sym = mx.sym.contrib.arange_like(data, **attrs)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph"]:
                op_res = relay.create_executor(
                    kind, mod=mod, device=dev, target=target
                ).evaluate()()
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy())

    verify(data_shape=(3,), start=0.0, step=1.0)
    verify(data_shape=(3, 4, 5), start=0.0, step=1.0)
    verify(data_shape=(3, 4, 5), start=0.0, step=1.0, axis=-1)
    verify(data_shape=(3, 4, 5), start=2.0, step=3.0, axis=1)


@tvm.testing.uses_gpu
def test_forward_interleaved_matmul_selfatt_qk():
    def verify(batch, seq_length, num_heads, head_dim):
        data_shape = (seq_length, batch, num_heads * head_dim * 3)
        data = mx.sym.var("data")
        data_np = np.random.uniform(size=data_shape).astype("float32")
        ref_res = mx.nd.contrib.interleaved_matmul_selfatt_qk(mx.nd.array(data_np), heads=num_heads)

        mx_sym = mx.sym.contrib.interleaved_matmul_selfatt_qk(data, heads=num_heads)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)

    verify(1, 10, 3, 16)
    verify(3, 10, 6, 8)


@tvm.testing.uses_gpu
def test_forward_interleaved_matmul_selfatt_valatt():
    def verify(batch, seq_length, num_heads, head_dim):
        data_shape = (seq_length, batch, num_heads * head_dim * 3)
        weight_shape = (batch * num_heads, seq_length, seq_length)
        data = mx.sym.var("data")
        weight = mx.sym.var("weight")
        data_np = np.random.uniform(size=data_shape).astype("float32")
        weight_np = np.random.uniform(size=weight_shape).astype("float32")
        ref_res = mx.nd.contrib.interleaved_matmul_selfatt_valatt(
            mx.nd.array(data_np), mx.nd.array(weight_np), heads=num_heads
        )

        mx_sym = mx.sym.contrib.interleaved_matmul_selfatt_valatt(data, weight, heads=num_heads)
        mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape, "weight": weight_shape})
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data=data_np, weight=weight_np
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)

    verify(1, 10, 4, 16)
    verify(3, 10, 6, 8)


@tvm.testing.uses_gpu
def test_forward_box_nms():
    def verify(
        data_shape,
        overlap_thresh=0.5,
        valid_thresh=0,
        topk=1,
        coord_start=2,
        score_index=1,
        id_index=0,
        force_suppress=False,
        in_format="corner",
    ):
        dtype = "float32"
        data = np.random.uniform(low=0, high=1, size=data_shape).astype(dtype)
        ref_res = mx.nd.contrib.box_nms(
            mx.nd.array(data),
            overlap_thresh=overlap_thresh,
            valid_thresh=valid_thresh,
            topk=topk,
            coord_start=coord_start,
            score_index=score_index,
            id_index=id_index,
            force_suppress=force_suppress,
            background_id=-1,
            in_format=in_format,
            out_format=in_format,
        )
        mx_sym = mx.sym.contrib.box_nms(
            mx.sym.var("data"),
            overlap_thresh=overlap_thresh,
            valid_thresh=valid_thresh,
            topk=topk,
            coord_start=coord_start,
            score_index=score_index,
            id_index=id_index,
            force_suppress=force_suppress,
            background_id=-1,
            in_format=in_format,
            out_format=in_format,
        )
        shape_dict = {"data": data_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            if tvm.contrib.thrust.can_use_thrust(
                tvm.target.Target(target + " -libs=thrust"), "tvm.contrib.thrust.sort"
            ):
                target += " -libs=thrust"
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((1, 10, 6))
    # No valid boxes
    verify((1, 10, 6), valid_thresh=1)


@tvm.testing.uses_gpu
def test_forward_box_decode():
    def verify(data_shape, anchor_shape, stds=[1, 1, 1, 1], clip=-1, in_format="corner"):
        dtype = "float32"
        data = np.random.uniform(low=-2, high=2, size=data_shape).astype(dtype)
        anchors = np.random.uniform(low=-2, high=2, size=anchor_shape).astype(dtype)
        ref_res = mx.nd.contrib.box_decode(
            mx.nd.array(data),
            mx.nd.array(anchors),
            stds[0],
            stds[1],
            stds[2],
            stds[3],
            clip,
            in_format,
        )
        mx_sym = mx.sym.contrib.box_decode(
            mx.sym.var("data"),
            mx.sym.var("anchors"),
            stds[0],
            stds[1],
            stds[2],
            stds[3],
            clip,
            in_format,
        )
        shape_dict = {"data": data_shape, "anchors": anchor_shape}
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                    data, anchors
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((1, 10, 4), (1, 10, 4))
    verify((4, 10, 4), (1, 10, 4))
    verify((1, 10, 4), (1, 10, 4), stds=[2, 3, 0.5, 1.5])
    verify((1, 10, 4), (1, 10, 4), clip=1)
    verify((1, 10, 4), (1, 10, 4), in_format="center")


@tvm.testing.uses_gpu
def test_forward_softmax():
    def verify(data_shape, axis, use_length, length):
        dtype = "float32"
        x = np.random.uniform(low=-100, high=100, size=data_shape).astype(dtype)
        if use_length:
            ref_res = mx.nd.softmax(
                data=mx.nd.array(x),
                length=mx.nd.array(length, dtype="int32"),
                axis=axis,
                use_length=use_length,
            )
            mx_sym = mx.symbol.softmax(
                data=mx.sym.var("data"),
                length=mx.sym.var("length"),
                axis=axis,
                use_length=use_length,
            )
            shape_dict = {"data": data_shape, "length": (length.shape)}
            dtype_dict = {"data": dtype, "length": "int32"}
            mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict, dtype_dict)
        else:
            ref_res = mx.nd.softmax(data=mx.nd.array(x), axis=axis)
            mx_sym = mx.symbol.softmax(data=mx.sym.var("data"), axis=axis)
            shape_dict = {"data": data_shape}
            mod, _ = relay.frontend.from_mxnet(mx_sym, shape_dict)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                func = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()
                if use_length:
                    op_res = func(x, length)
                else:
                    op_res = func(x)

                tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-3, atol=1e-5)

    verify((2, 3, 5), -1, False, None)
    verify((2, 3, 5), 2, False, None)
    verify((2, 3), -1, True, np.array([2, 1]).astype("int32"))
    verify((2, 3, 4), -1, True, np.array([[3, 4, 2], [2, 1, 1]]).astype("int32"))
    verify((2, 3, 4), 2, True, np.array([[3, 4, 2], [1, 2, 1]]).astype("int32"))


@pytest.mark.skipif(not hasattr(mx.sym.np, "pad"), reason="mx.sym.np.pad hasn't been publish yet")
@pytest.mark.parametrize(
    "data_shape, pad_width",
    [
        ((1, 1, 3, 5), ((0, 0), (0, 0), (1, 2), (3, 4))),
        ((1, 1, 3, 5, 7), ((0, 0), (0, 0), (1, 2), (3, 4), (5, 6))),
    ],
)
@pytest.mark.parametrize("mode", ["constant", "edge", "reflect"])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
@pytest.mark.parametrize("constant_value", [0.0, 3.0])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_pad(data_shape, pad_width, mode, dtype, constant_value, target, dev, kind):
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    data = mx.sym.var("data")
    if mode == "constant":
        ref_res = np.pad(data_np, mode=mode, pad_width=pad_width, constant_values=constant_value)
        mx_sym = mx.sym.np.pad(
            data.as_np_ndarray(), mode=mode, pad_width=pad_width, constant_values=constant_value
        )
    else:
        ref_res = np.pad(data_np, mode=mode, pad_width=pad_width)
        mx_sym = mx.sym.np.pad(data.as_np_ndarray(), mode=mode, pad_width=pad_width)
    mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape}, dtype=dtype)
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(data_np)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


@pytest.mark.skipif(
    not hasattr(mx.sym.np, "pad"), reason="test'll abort with Mxnet 1.x, skip for now"
)
@pytest.mark.parametrize("data_shape", [(2, 2, 2), (2, 7, 2)])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "bool"])
@pytest.mark.parametrize("axes", [(1, 0, 2), None])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_transpose(data_shape, axes, dtype, target, dev, kind):
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    data = mx.sym.var("data")
    ref_res = mx.np.transpose(mx.np.array(data_np), axes=axes)
    mx_sym = mx.sym.np.transpose(data.as_np_ndarray(), axes=axes)
    mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape}, dtype=dtype)
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(data_np)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize(
    "data_shape1, data_shape2, axis",
    [
        ((2, 2), (2, 2), 1),
        ((2, 4), (2, 3), 1),
        ((1, 3, 2), (1, 3, 5), 2),
        ((1, 3, 3), (1, 3, 3), 1),
        ((1, 3), (1, 3), 0),
    ],
)
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_concatenate(data_shape1, data_shape2, axis, dtype, target, dev, kind):
    data_np1 = np.random.uniform(size=data_shape1).astype(dtype)
    data_np2 = np.random.uniform(size=data_shape2).astype(dtype)
    data1 = mx.sym.var("data1")
    data2 = mx.sym.var("data2")
    ref_res = mx.np.concatenate([mx.np.array(data_np1), mx.np.array(data_np2)], axis=axis)
    mx_sym = mx.sym.np.concatenate([data1.as_np_ndarray(), data2.as_np_ndarray()], axis=axis)
    mod, _ = relay.frontend.from_mxnet(
        mx_sym, shape={"data1": data_shape1, "data2": data_shape2}, dtype=dtype
    )
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
        data_np1, data_np2
    )
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize(
    "data_shape1, data_shape2, axis",
    [
        ((3,), (3,), 0),
        ((3,), (3,), -1),
        ((1, 3, 2), (1, 3, 2), 2),
        ((1, 3, 3), (1, 3, 3), 1),
        ((1, 3), (1, 3), 0),
    ],
)
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_stack(data_shape1, data_shape2, axis, dtype, target, dev, kind):
    data_np1 = np.random.uniform(size=data_shape1).astype(dtype)
    data_np2 = np.random.uniform(size=data_shape2).astype(dtype)
    data1 = mx.sym.var("data1")
    data2 = mx.sym.var("data2")
    ref_res = mx.np.stack([mx.np.array(data_np1), mx.np.array(data_np2)], axis=axis)
    mx_sym = mx.sym.np.stack([data1.as_np_ndarray(), data2.as_np_ndarray()], axis=axis)
    mod, _ = relay.frontend.from_mxnet(
        mx_sym, shape={"data1": data_shape1, "data2": data_shape2}, dtype=dtype
    )
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
        data_np1, data_np2
    )
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize("data_shape", [(2, 2, 2), (2, 7, 2), (2, 2, 2, 1, 2, 3, 1), (1, 8)])
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "bool"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_np_copy(data_shape, dtype, target, dev, kind):
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    data = mx.sym.var("data")
    ref_res = mx.np.copy(mx.np.array(data_np))
    mx_sym = mx.sym.np.copy(data.as_np_ndarray())
    mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape}, dtype=dtype)
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(data_np)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "bool"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
@pytest.mark.parametrize(
    "data_shape,out_shape,reverse",
    [
        ((2, 3, 8), (-2, -2, 2, -1), False),
        ((8, 3, 3, 3, 4, 4), (-6, 2, -1, -4), False),
        ((8, 3, 3, 3, 4, 4), (-5, -4), False),
        ((1, 8, 3, 3, 3, 4, 4), (-3, -5, -4), False),
        ((8, 1, 3, 4), (-2, -3, -1), False),
        ((8, 3, 3, 3, 3, 8), (-4, -5), True),
        ((8, 3, 2, 4, 8), (-4, -1, 2, -6), True),
        ((3, 2, 4, 8, 1, 1), (-4, -1, 2, -6, -5, -3), True),
        ((2, 4, 1, 8), (-4, -3, -1, 2, -6), True),
    ],
)
def test_forward_npx_reshape(data_shape, out_shape, dtype, target, reverse, dev, kind):
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    data = mx.sym.var("data")
    ref_res = mx.npx.reshape(mx.np.array(data_np), newshape=out_shape, reverse=reverse)
    mx_sym = mx.sym.npx.reshape(data.as_np_ndarray(), newshape=out_shape, reverse=reverse)
    mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape}, dtype=dtype)
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(data_np)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize(
    "data_shape", [(2, 2, 2), (2, 7, 2), (2, 2, 2, 1, 2, 3, 1), (1, 8), (2, 2), (1, 3)]
)
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_binary(data_shape, dtype, target, dev, kind):
    ref_ops = [mx.np.power, mx.np.multiply, mx.np.add, mx.np.subtract, mx.np.less]
    mx_ops = [
        mx.sym.np.power,
        mx.sym.np.multiply,
        mx.sym.np.add,
        mx.sym.np.subtract,
        mx.sym.np.less,
    ]
    for i in range(len(ref_ops)):
        ref_op = ref_ops[i]
        mx_op = mx_ops[i]
        # mx.np.power only support float type
        if ref_op == mx.np.power and dtype not in ["float64", "float32"]:
            continue
        data_np1 = np.random.uniform(size=data_shape).astype(dtype)
        data_np2 = np.random.uniform(size=data_shape).astype(dtype)
        data1 = mx.sym.var("lhs")
        data2 = mx.sym.var("rhs")
        ref_res = ref_op(mx.np.array(data_np1), mx.np.array(data_np2))
        mx_sym = mx_op(data1.as_np_ndarray(), data2.as_np_ndarray())
        mod, _ = relay.frontend.from_mxnet(
            mx_sym, shape={"lhs": data_shape, "rhs": data_shape}, dtype=dtype
        )
        op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
            data_np1, data_np2
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize(
    "data_shape", [(2, 2, 2), (2, 7, 2), (2, 2, 2, 1, 2, 3, 1), (1, 8), (2, 2), (1, 3)]
)
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("scalar", [1.0, 2.0, 3.0, 4.0])
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_binary_scalar(data_shape, dtype, scalar, target, dev, kind):
    ref_ops = [mx.np.power, mx.np.multiply, mx.np.add, mx.np.subtract, mx.np.true_divide]
    mx_ops = [
        mx.sym.np.power,
        mx.sym.np.multiply,
        mx.sym.np.add,
        mx.sym.np.subtract,
        mx.sym.np.true_divide,
    ]
    for i in range(len(ref_ops)):
        ref_op = ref_ops[i]
        mx_op = mx_ops[i]
        # mx.np.power only support float type
        if ref_op == mx.np.power and dtype not in ["float64", "float32"]:
            continue
        data_np1 = np.random.uniform(size=data_shape).astype(dtype)
        data1 = mx.sym.var("lhs")
        ref_res = ref_op(mx.np.array(data_np1), scalar)
        mx_sym = mx_op(data1.as_np_ndarray(), scalar)
        mod, _ = relay.frontend.from_mxnet(mx_sym, shape={"lhs": data_shape}, dtype=dtype)
        op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
            data_np1
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize(
    "data_shape", [(2, 2, 2), (2, 7, 2), (2, 2, 2, 1, 2, 3, 1), (1, 8), (2, 2), (1, 3)]
)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_tanh(data_shape, dtype, target, dev, kind):
    data_np1 = np.random.uniform(size=data_shape).astype(dtype)
    data1 = mx.sym.var("data")
    ref_res = mx.np.tanh(mx.np.array(data_np1))
    mx_sym = mx.sym.np.tanh(data1.as_np_ndarray())
    mod, _ = relay.frontend.from_mxnet(mx_sym, shape={"data": data_shape}, dtype=dtype)
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(data_np1)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.skipif(not hasattr(mx.np, "where"), reason="mx.np.where hasn't been publish yet")
@pytest.mark.parametrize(
    "data_shape,cond_shape",
    [[(2, 2, 2), (2, 2, 2)], [(2, 7, 2), (7, 2)], [(2, 2), (1, 2)], [(1, 3), (3, 3)]],
)
@pytest.mark.parametrize("data_dtype", ["float64", "float32", "int64", "int32", "bool"])
@pytest.mark.parametrize("cond_dtype", ["float64", "float32", "int64", "int32", "bool"])
@pytest.mark.parametrize("scalar", [1.0, 2.0])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
def test_forward_npi_where_rscalar(
    data_shape, cond_shape, data_dtype, cond_dtype, scalar, target, dev, kind
):
    if data_dtype == "bool":
        scalar = scalar == 0.0
    cond_np = np.random.uniform(size=cond_shape).astype(cond_dtype)
    data_np = np.random.uniform(size=data_shape).astype(data_dtype)
    cond = mx.sym.var("condition")
    data = mx.sym.var("x")
    ref_res = mx.np.where(mx.np.array(cond_np), mx.np.array(data_np), scalar)
    mx_sym = mx.sym.np.where(cond.as_np_ndarray(), data.as_np_ndarray(), scalar)
    dtypeDic = {}
    dtypeDic["condition"] = cond_dtype
    dtypeDic["x"] = data_dtype
    mod, _ = relay.frontend.from_mxnet(
        mx_sym, shape={"condition": cond_shape, "x": data_shape}, dtype=dtypeDic
    )
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
        cond_np, data_np
    )
    tvm.testing.assert_allclose(op_res.numpy(), ref_res.asnumpy(), rtol=1e-5)


@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "bool"])
@tvm.testing.parametrize_targets
@pytest.mark.parametrize("kind", ["graph", "vm", "debug"])
@pytest.mark.parametrize(
    "data_shape, axis, indices_or_sections, squeeze_axis",
    [
        ((3, 2, 1), 1, 2, False),
        ((3, 2, 1), 0, 3, False),
        ((3, 2, 1), 0, 3, True),
        ((3, 2, 1), 0, (1, 2), False),
    ],
)
def test_forward_split_v2(
    data_shape, axis, dtype, indices_or_sections, squeeze_axis, target, dev, kind
):
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    data = mx.sym.var("data")
    ref_res = mx.ndarray.split_v2(
        mx.nd.array(data_np), indices_or_sections, axis=axis, squeeze_axis=squeeze_axis
    )
    mx_sym = mx.sym.split_v2(
        data.as_nd_ndarray(), indices_or_sections, axis=axis, squeeze_axis=squeeze_axis
    )
    mod, _ = relay.frontend.from_mxnet(mx_sym, {"data": data_shape}, dtype=dtype)
    op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(data_np)
    op_res_ = []
    for arr in op_res:
        op_res_.append(arr.numpy().tolist())
    ref_res_ = []
    for arr in ref_res:
        ref_res_.append(arr.asnumpy().tolist())
    tvm.testing.assert_allclose(op_res_, ref_res_, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
