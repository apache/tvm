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
""" Support level3 operator test cases.
"""
import numpy as np
import pytest
import tvm
from tvm import relay
from tvm.relay import create_executor, transform
from tvm.relay.testing import ctx_list, check_grad

def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def test_zeros_ones():
    for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
        y = op(shape=(124, 50), dtype="float64")
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((124, 50), "float64")
        intrp = create_executor()
        intrp_res = intrp.evaluate(y).asnumpy()
        np.testing.assert_allclose(intrp_res, ref((124, 50), 'float64'))

def test_unary_identity():
    for op, ref in [(relay.zeros_like, np.zeros_like),
               (relay.ones_like, np.ones_like),
               (relay.ceil, np.ceil),
               (relay.floor, np.floor),
               (relay.trunc, np.trunc),
               (relay.round, np.round),
               (relay.abs, np.abs),
               (relay.copy, None), # np.copy
               (relay.negative, np.negative),
               (relay.sign, np.sign)]:
        shape = (8, 9, 4)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = op(x)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(shape, "float32")

        if ref is not None:
            data = np.random.rand(*shape).astype('float32')
            intrp = create_executor()
            op_res = intrp.evaluate(y, { x: relay.const(data) })
            ref_res = ref(data)
            np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)

def test_cast():
    x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
    y = x.astype("int32")
    yy = run_infer_type(y)
    assert "dtype=" in yy.astext()
    assert yy.checked_type == relay.TensorType((8, 9, 4), "int32")

    x = relay.var("x", relay.TensorType((8, 9, 4), "float32"))
    y = relay.cast(x, "int32")
    yy = run_infer_type(y)
    assert "dtype=" in yy.astext()
    assert yy.checked_type == relay.TensorType((8, 9, 4), "int32")


def test_clip():
    a = relay.var("a", relay.TensorType((10, 4), "float32"))
    y = relay.clip(a, 1., 4.)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((10, 4), "float32")

    data = np.random.rand(10, 4).astype('float32')
    intrp = create_executor()
    op_res = intrp.evaluate(y, { a: relay.const(data) })
    ref_res = np.clip(data, 1., 4.)
    np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)


def test_reinterpret():
    a = relay.var("a", relay.TensorType((1000, 4), "float32"))
    y = relay.reinterpret(a, "int32")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000, 4), "int32")

    data = np.random.randn(1000, 4).astype('float32') * 1000
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = data.view("int32")
    np.testing.assert_equal(op_res.asnumpy(), ref_res)


def test_approximate_transcendental():
    def C(x):
        return relay.expr.const(x, "float32")

    def approx_exp(x):
        # An approximation derived from Opus,
        # https://github.com/xiph/opus/blob/c1c247/celt/mathops.h#L147-L165
        x = relay.minimum(relay.maximum(x, C(-88.0)), C(88.0))
        x = C(127.0) + x * C(1.44269504)
        xf = relay.floor(x)
        i = relay.cast(xf, "int32")
        x = x - xf
        Y = C(0.99992522) + x * (C(0.69583354) + x * (C(0.22606716) + x * C(0.078024523)))
        exponent = relay.left_shift(i, relay.expr.const(23, "int32"))
        exponent = relay.reinterpret(exponent, "float32")
        return exponent * Y

    def approximate_sigmoid(x):
        y = approx_exp(x)
        return y / (y + C(1.0))

    def approximate_tanh(x):
        x = x * C(2.0)
        y = approx_exp(x)
        return (y - C(1.0)) / (y + C(1.0))

    a = relay.var("a", relay.TensorType((1000,), "float32"))
    y = approximate_sigmoid(a)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000,), "float32")
    data = np.linspace(-5, 5, 1000).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})

    def reference_sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))
    np.testing.assert_allclose(op_res.asnumpy(), reference_sigmoid(data), atol=2e-5, rtol=1e-9)

    y = approximate_tanh(a)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000,), "float32")
    data = np.linspace(-5, 5, 1000).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})

    def reference_tanh(x):
        return np.tanh(x)
    np.testing.assert_allclose(op_res.asnumpy(), reference_tanh(data), atol=4e-5, rtol=1e-9)


def test_squeeze():
    def verify_squeeze(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        squeeze = relay.squeeze(x, axis=axis)

        np_axis = tuple(axis) if axis is not None else None

        data = np.random.random_sample(shape).astype(dtype)
        intrp = create_executor()
        op_res = intrp.evaluate(squeeze, { x : relay.const(data) })
        ref_res = np.squeeze(data, axis=np_axis)
        np.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=0.01)

    verify_squeeze((1, 3, 2, 5), "float32", None)
    verify_squeeze((1, 3, 1), "float32", [0])
    verify_squeeze((1, 2, 1, 2, 1), "float32", [0, 2])


def test_transpose_infer_type():
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.transpose(x, axes=(1, 0, 2))
    assert "axes=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (t, n, 100), "float32")

    y = relay.transpose(x)
    assert "axes=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (100, t, n), "float32")


def test_transpose():
    def verify_transpose(dshape, axes):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.transpose(x, axes=axes)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.transpose(x_data, axes=axes)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_transpose((2, 3, 4), (0, 2, 1))


def test_squeeze_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(2,))
    assert "axis=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (1, 4), "float32")

    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x)
    assert "axis=" not in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (4,), "float32")

@pytest.mark.xfail(raises=tvm._ffi.base.TVMError)
def test_squeeze_bad_axes_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(1,))
    yy = run_infer_type(y)


def test_reshape_infer_type():
    n, t, d1, d2 = 10, 20, 100, 20
    x = relay.var("x", relay.TensorType((n, t, d1, d2), "float32"))
    y = relay.reshape(x, newshape=(n, t, 2000))
    assert "newshape=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType(
        (n, t, 2000), "float32")

def test_reshape():
    def verify_reshape(shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reshape(x, newshape=newshape)
        zz = run_infer_type(z)
        assert "newshape=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        func = relay.Function([x], z)
        check_grad(func)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_reshape((2, 3, 4), (8, 3), (8, 3))
    verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))
    verify_reshape((2, 3, 4), (4, 0, 2), (4, 3, 2))
    verify_reshape((2, 3, 4), (2, 0, 0), (2, 3, 4))
    verify_reshape((2, 3, 4), (0, -1), (2, 12))
    verify_reshape((2, 3, 4), (-1, 0), (8, 3))
    verify_reshape((2, 3, 4), (2, -2), (2, 3, 4))
    verify_reshape((2, 3, 4), (-2, 1, 1), (2, 3, 4, 1, 1))
    verify_reshape((2, 3, 4), (-3, 4), (6, 4))
    verify_reshape((2, 3, 4, 5), (-3, -3), (6, 20))
    verify_reshape((2, 3, 4), (0, -3), (2, 12))
    verify_reshape((2, 3, 4), (-3, -2), (6, 4))
    verify_reshape((2, 3, 4), (-4, 1, 2, -2), (1, 2, 3, 4))
    verify_reshape((2, 3, 4), (2, -4, -1, 3, -2), (2, 1, 3, 4))


def test_reshape_like_infer_type():
    # concrete shape
    x = relay.var("x", relay.TensorType((1, 2, 3), "float32"))
    y = relay.var("y", relay.TensorType((1,6), "float32"))
    z = relay.reshape_like(x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 6), "float32")

    # symbolic shape
    n, c, h, w = tvm.var("n"), 2, 3, tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.var("y", relay.TensorType((1, 8, 8), "float32"))
    z = relay.reshape_like(x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 8, 8), "float32")


def test_reshape_like():
    def verify_reshape_like(shape, oshape):
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=oshape).astype("float32")
        ref_res = np.reshape(x_data, y_data.shape)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("x", relay.TensorType(oshape, "float32"))
        z = relay.reshape_like(x, y)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.ty.TensorType(ref_res.shape, "float32")

        func = relay.Function([x, y], z)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

    verify_reshape_like((2, 3, 4), (1, 8, 3))
    verify_reshape_like((4, 7), (2, 7, 2))

def test_take_infer_type():
    def verify_take(dshape, indices_shape, oshape, axis=None):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        y = relay.take(x, indices, axis=axis)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(oshape, "float32")

    d1, d2, d3 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3")
    d4, d5, d6 = tvm.var("d4"), tvm.var("d5"), tvm.var("d6")
    verify_take((d1,), (1,), (1,), 0)
    verify_take((4,), (d1, d2), (d1, d2))
    verify_take((3, 3, 3), (1, d2), (1, d2))
    verify_take((d1, d2), (d3, d4, d5), (d3, d4, d5, d2), 0)
    verify_take((d1, d2), (d3, d4, d5), (d1, d3, d4, d5), 1)
    verify_take((d1, d2, d3, d4), (d5, d6), (d1, d2, d5, d6, d4), -2)

def test_take():
    def verify_take(src_shape, indices_src, axis=None, mode="clip"):
        src_dtype = "float32"
        indices_dtype = "int32"
        indices_src = np.array(indices_src, dtype=indices_dtype)
        x = relay.var("x", relay.TensorType(src_shape, src_dtype))
        indices = relay.var("indices", relay.TensorType(indices_src.shape, indices_dtype))
        z = relay.take(x, indices, axis=axis, mode=mode)

        func = relay.Function([x, indices], z)
        x_data = np.random.uniform(low=-1, high=1, size=src_shape).astype(src_dtype)
        np_mode = "raise" if mode == "fast" else mode
        ref_res = np.take(x_data, indices=indices_src, axis=axis, mode=np_mode)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, indices_src)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

    verify_take((4,), [1])
    verify_take((4,), [[0,1,2,3]])
    verify_take((3,3,3), [[11,25]])
    verify_take((4,), [[0,1],[2,3]])
    verify_take((4,), [1], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 0)
    verify_take((2,2), [[[1,0],[0,1]]], 1)
    verify_take((4,3,5,6), [[2,1,0,0]], -2)
    verify_take((3,4), [-5, 20])
    verify_take((3,4), [-5, 20], mode="wrap")
    verify_take((3,4), [-1, 2], axis=0)
    verify_take((3,4), [-1, 2], axis=0, mode="wrap")
    verify_take((3,4), [-1, 2], axis=1)
    verify_take((3,4), [-1, 2], axis=1, mode="wrap")
    verify_take((3,3,3), [[11,25]], mode="fast")
    verify_take((3,4), [0, 2], axis=0, mode="fast")
    verify_take((3,4), [0, 2], axis=1, mode="fast")


def test_split_infer_type():
    def verify_split(dshape, indices_or_sections, ret_type, axis=None):
        x = relay.var("x", relay.ty.TensorType(dshape, "float32"))
        y = relay.split(x, indices_or_sections, axis=axis)
        yy = run_infer_type(y.astuple())
        assert yy.checked_type == ret_type

    idxd = tvm.indexdiv

    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    axis = tvm.var("axis")
    verify_split((5, 5, 2, 2), 5,
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32"),
                     relay.ty.TensorType((5, 1, 2, 2), "float32")])),
                  axis=1)
    verify_split((5, 5, 2, 2), 5,
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((1, 5, 2, 2), "float32"),
                     relay.ty.TensorType((1, 5, 2, 2), "float32"),
                     relay.ty.TensorType((1, 5, 2, 2), "float32"),
                     relay.ty.TensorType((1, 5, 2, 2), "float32"),
                     relay.ty.TensorType((1, 5, 2, 2), "float32")])),
                  axis=0)
    verify_split((d1, d2, d3, d4), 4,
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                     relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                     relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                     relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32")])),
                  axis=2)
    verify_split((d1, d2, d3, d4), 2,
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((idxd(d1, 2), d2, d3, d4), "float32"),
                     relay.ty.TensorType((idxd(d1, 2), d2, d3, d4), "float32")])),
                  axis=0)
    verify_split((d1, d2, d3, d4), (2, 4, 7),
                 relay.ty.TupleType(tvm.convert([
                     relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                     relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                     relay.ty.TensorType((d1, 3, d3, d4), "float32"),
                     relay.ty.TensorType((d1, (d2-7), d3, d4), "float32")])),
                  axis=1)

def test_full_infer_type():
    # default settings: match input dtype
    x = relay.var("x", relay.TensorType((), "int8"))
    y = relay.full(x, ())
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((), "int8")

    # change the shape and dtype
    x = relay.var("x", relay.TensorType((), "float32"))
    y = relay.full(x, (1, 2), "int8")
    "shape=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 2), "int8")


def test_full():
    def verify_full(fill_value, src_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        z = relay.full(x, src_shape, dtype)
        func = relay.Function([x], z)
        ref_res = np.full(src_shape, fill_value)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(np.array(fill_value, dtype))
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_full(4, (1, 3, 4, 4), "int32")
    verify_full(4.0, (1, 4), "float32")


def test_full_like_infer_type():
    # concrete shape
    base = relay.var("base", relay.TensorType((1, 2, 3), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 2, 3), "float32")

    # symbolic shape
    n, c, h, w = tvm.var("n"), 2, 3, tvm.var("w")
    base = relay.var("base", relay.TensorType((n, c, h, w), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w), "float32")


def test_full_like():
    def verify_full_like(base, fill_value, dtype):
        x_data = np.random.uniform(low=-1, high=1, size=base).astype(dtype)
        x = relay.var("x", relay.TensorType(base, dtype))
        y = relay.var("y", relay.scalar_type(dtype))
        z = relay.full_like(x, y)

        func = relay.Function([x, y], z)
        ref_res = np.full_like(x_data, fill_value)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, np.array(fill_value, dtype))
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_full_like((1, 3, 4, 4), 4, "int32")
    verify_full_like((1, 1), 44.0, "float32")


def test_infer_type_leaky_relu():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.nn.leaky_relu(x, alpha=0.1)
    "alpha=0.1" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w), "float32")

    shape = (1, 5, 10, 10)
    dtype = "float32"
    x = relay.var("x", relay.TensorType(shape, dtype))
    z = relay.nn.leaky_relu(x, alpha=0.1)
    assert "alpha=0.1" in z.astext()
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType(shape, dtype)
    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    ref_res = np.where(x_data > 0, x_data, x_data * 0.1)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

def verify_infer_type_prelu(data, alpha, axis, output, dtype="float32"):
    x = relay.var("data", relay.TensorType(data, dtype))
    if alpha:
        y = relay.var("alpha", relay.TensorType(alpha, dtype))
    else:
        y = relay.var("alpha", relay.IncompleteType())
    z = relay.nn.prelu(x, y, axis=axis)
    zz = run_infer_type(z)
    if axis != 1:
        assert "axis" in z.astext()
    assert zz.checked_type == relay.ty.TensorType(output, dtype)
    if not alpha:
        axis = axis if axis else 1
        alpha_shape = (data[axis],)
        assert zz.args[1].checked_type == relay.TensorType(alpha_shape, "float32")

    if all(isinstance(v, tvm.expr.Var) == 1 for v in data) or not alpha:
        return

    func = relay.Function([x, y], z)
    x_data = np.random.uniform(low=-1, high=1, size=data).astype(dtype)
    a_data = np.random.uniform(low=-1, high=1, size=alpha).astype(dtype)

    if axis == 1:
        ref_res = (x_data < 0) * (x_data * a_data.reshape(3, 1, 1)) + (x_data>=0) * x_data
    else:
        ref_res = (x_data < 0) * (x_data * a_data.reshape(1, 1, 3)) + (x_data>=0) * x_data

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data, a_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data, a_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)


def test_infer_type_prelu():
    n, c , h, w = tvm.var("n"), tvm.var("c"), tvm.var("h"), tvm.var("w")
    verify_infer_type_prelu((n, c, h, w), (c,), 1, (n, c, h, w))
    verify_infer_type_prelu((n, h, w, c), (c,), 3, (n, h, w, c))
    verify_infer_type_prelu((n, c, h, w), None, 1, (n, c, h, w))
    verify_infer_type_prelu((n, h, w, c), None, 3, (n, h, w, c))
    verify_infer_type_prelu((1, 3, 2, 2), (3,), 1, (1, 3, 2, 2))
    verify_infer_type_prelu((1, 2, 2, 3), (3,), 3, (1, 2, 2, 3))
    verify_infer_type_prelu((1, 3, 2, 2), None, 1, (1, 3, 2, 2))
    verify_infer_type_prelu((1, 2, 2, 3), None, 3, (1, 2, 2, 3))


def test_arange():
    def verify_arange(start, stop, step):
        dtype = "float32"
        if start is None and step is None:
            x = relay.arange(relay.const(stop, dtype=dtype))
            ref_res = np.arange(stop).astype(dtype)
        elif start is None:
            x = relay.arange(relay.const(stop, dtype=dtype), step=relay.const(step, dtype=dtype))
            ref_res = np.arange(stop, step=step).astype(dtype)
        elif step is None:
            x = relay.arange(relay.const(start, dtype=dtype), relay.const(stop, dtype=dtype))
            ref_res = np.arange(start, stop).astype(dtype)
        else:
            x = relay.arange(
                relay.const(start, dtype=dtype),
                relay.const(stop, dtype=dtype),
                relay.const(step, dtype=dtype))
            ref_res = np.arange(start, stop, step).astype(dtype)

        func = relay.Function([], x)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)()
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_arange(None, 20, None)
    verify_arange(None, 20, 2)
    verify_arange(1, 20, None)
    verify_arange(1, 20, 2)
    # arange doesnt' support floating point right now, see type relation
    # verify_arange(1, 20, 1.5)
    verify_arange(1, 20.5, None)
    verify_arange(1, 20, 3)
    verify_arange(20, 1, -1)
    # arange doesnt' support floating point right now, see type relation
    # verify_arange(20, 1, -1.5)

def test_tile():
    def verify_tile(dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.tile(x, reps=reps)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_tile((2, 3, 4), (3, 2, 1))
    verify_tile((2, 3, 4), (1, 2))
    verify_tile((2, 3), (3, 2, 1))

def test_repeat():
    def verify_repeat(dshape, repeats, axis):
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        func = relay.Function([x], relay.repeat(x, repeats, axis))
        data = np.random.uniform(size=dshape).astype("float32")
        ref_res = np.repeat(data, repeats, axis)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_repeat((3,), 2, 0)
    verify_repeat((3, 10), 2, -1)
    verify_repeat((3, 2, 4), 3, 1)

def test_stack():
    def verify_stack(dshapes, axis):
        y = []
        for shape in dshapes:
            y.append(relay.var("input", relay.TensorType(shape, "float32")))
        x = relay.Tuple(y)
        z = relay.stack(x, axis=axis)

        func = relay.Function(y, z)
        x_data = [np.random.normal(size=shape).astype("float32") for shape in dshapes]
        ref_res = np.stack(x_data, axis=axis)

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(*x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_stack([(2,), (2,), (2,)], -1)
    verify_stack([(2,), (2,), (2,)], 0)
    verify_stack([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1)
    verify_stack([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1)


def test_reverse():
    def verify_reverse(dshape, axis):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.reverse(x, axis=axis)
        zz = run_infer_type(z)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.flip(x_data, axis)
        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_reverse((2, 3, 4), 1)
    verify_reverse((4, 7), 0)
    verify_reverse((2, 3, 4), -1)


def test_gather_nd():
    def verify_gather_nd(xshape, yshape, y_data):
        x = relay.var("x", relay.TensorType(xshape, "float32"))
        y = relay.var("y", relay.TensorType(yshape, "int32"))
        z = relay.gather_nd(x, y)

        func = relay.Function([x, y], z)
        x_data = np.random.uniform(size=xshape).astype("float32")
        ref_res = x_data[tuple(y_data)]

        for target, ctx in ctx_list():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)
    verify_gather_nd((2, 2), (2, 3), [[1, 1, 0], [0, 1, 0]])
    verify_gather_nd((2, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify_gather_nd((3, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify_gather_nd((3, 2), (2, 2, 3), [[[0, 1, 2], [2, 0, 1]], [[0, 0, 0], [1, 1, 1]]])

if __name__ == "__main__":
    test_arange()
    test_cast()
    test_zeros_ones()
    test_unary_identity()
    test_clip()
    test_transpose_infer_type()
    test_transpose()
    test_reshape_infer_type()
    test_reshape()
    test_reshape_like_infer_type()
    test_reshape_like()
    test_take_infer_type()
    test_take()
    test_full_infer_type()
    test_full()
    test_full_like_infer_type()
    test_full_like()
    test_infer_type_leaky_relu()
    test_infer_type_prelu()
    test_squeeze()
    test_squeeze_infer_type()
    test_squeeze_bad_axes_infer_type()
    test_split_infer_type()
    test_arange()
    test_reverse()
    test_stack()
    test_tile()
    test_repeat()
    test_gather_nd()
