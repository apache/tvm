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
from typing import Callable, Optional

import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay, te
from tvm.error import TVMError
from tvm.relay import create_executor, transform
from tvm.relay.testing import check_grad, run_infer_type
from utils import ref_funcs


def test_zeros_ones():
    for op, ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
        y = op(shape=(124, 50), dtype="float64")
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((124, 50), "float64")
        intrp = create_executor()
        intrp_res = intrp.evaluate(y).numpy()
        np.testing.assert_allclose(intrp_res, ref((124, 50), "float64"))


def test_unary_identity():
    for op, ref in [
        (relay.zeros_like, np.zeros_like),
        (relay.ones_like, np.ones_like),
        (relay.ceil, np.ceil),
        (relay.floor, np.floor),
        (relay.trunc, np.trunc),
        (relay.round, np.round),
        (relay.abs, np.abs),
        (relay.copy, None),  # np.copy
        (relay.negative, np.negative),
        (relay.sign, np.sign),
    ]:
        shape = (8, 9, 4)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = op(x)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(shape, "float32")

        if ref is not None:
            data = np.random.rand(*shape).astype("float32")
            intrp = create_executor()
            op_res = intrp.evaluate(y, {x: relay.const(data)})
            ref_res = ref(data)
            np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


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
    y = relay.clip(a, 1.0, 4.0)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((10, 4), "float32")

    data = np.random.rand(10, 4).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = np.clip(data, 1.0, 4.0)
    np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


def test_fixed_point_multiply():
    # Test 23 * 1/16
    # [m,s] = [0.5, -3] = frexp(1/16)
    # M = 0.5*2^31 = 1073741824
    # so M = 1073741824 and s = -3

    a = relay.var("a", relay.TensorType((10, 4), "int32"))
    y = relay.fixed_point_multiply(a, 1073741824, -3)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((10, 4), "int32")

    data = 23 * np.ones((10, 4)).astype("int32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = np.ones((10, 4)).astype("int32")
    np.testing.assert_allclose(op_res.numpy(), ref_res, atol=1)


def test_reinterpret():
    a = relay.var("a", relay.TensorType((1000, 4), "float32"))
    y = relay.reinterpret(a, "int32")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000, 4), "int32")

    data = np.random.randn(1000, 4).astype("float32") * 1000
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})
    ref_res = data.view("int32")
    np.testing.assert_equal(op_res.numpy(), ref_res)


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

    np.testing.assert_allclose(op_res.numpy(), reference_sigmoid(data), atol=2e-5, rtol=1e-9)

    y = approximate_tanh(a)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000,), "float32")
    data = np.linspace(-5, 5, 1000).astype("float32")
    intrp = create_executor()
    op_res = intrp.evaluate(y, {a: relay.const(data)})

    def reference_tanh(x):
        return np.tanh(x)

    np.testing.assert_allclose(op_res.numpy(), reference_tanh(data), atol=4e-5, rtol=1e-9)


def test_squeeze():
    def verify_squeeze(shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        squeeze = relay.squeeze(x, axis=axis)

        np_axis = tuple(axis) if axis is not None else None

        data = np.random.random_sample(shape).astype(dtype)
        intrp = create_executor()
        op_res = intrp.evaluate(squeeze, {x: relay.const(data)})
        ref_res = np.squeeze(data, axis=np_axis)
        np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)

    verify_squeeze((1, 3, 2, 5), "float32", None)
    verify_squeeze((1, 3, 1), "float32", [0])
    verify_squeeze((1, 2, 1, 2, 1), "float32", [0, 2])


def test_transpose_infer_type():
    n, t, d = te.size_var("n"), te.size_var("t"), 100
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.transpose(x, axes=(1, 0, 2))
    assert "axes=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((t, n, 100), "float32")

    y = relay.transpose(x)
    assert "axes=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((100, t, n), "float32")


@tvm.testing.uses_gpu
def test_transpose():
    def verify_transpose(dshape, axes):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.transpose(x, axes=axes)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.transpose(x_data, axes=axes)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_transpose((2, 3, 4), (0, 2, 1))


def test_squeeze_infer_type():
    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x, axis=(2,))
    assert "axis=" in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 4), "float32")

    n, t, d = 1, 4, 1
    x = relay.var("x", relay.TensorType((n, t, d), "float32"))
    y = relay.squeeze(x)
    assert "axis=" not in y.astext()
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((4,), "float32")


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
    assert yy.checked_type == relay.TensorType((n, t, 2000), "float32")


@tvm.testing.uses_gpu
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
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

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


def test_reshape_fail():
    with pytest.raises(TVMError) as reshape_err:
        x = relay.var("x", relay.TensorType([2, 3], "float32"))
        z = relay.reshape(x, [7])
        zz = run_infer_type(z)


def test_reshape_like_infer_type():
    # concrete shape
    x = relay.var("x", relay.TensorType((1, 2, 3), "float32"))
    y = relay.var("y", relay.TensorType((1, 6), "float32"))
    z = relay.reshape_like(x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 6), "float32")

    # symbolic shape
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.var("y", relay.TensorType((1, 8, 8), "float32"))
    z = relay.reshape_like(x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 8, 8), "float32")

    # partial reshaping
    x = relay.var("x", relay.TensorType((1, 2, 3, 4), "float32"))
    y = relay.var("y", relay.TensorType((1, 6, 5), "float32"))
    z = relay.reshape_like(x, y, lhs_begin=1, lhs_end=3, rhs_begin=1, rhs_end=2)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((1, 6, 4), "float32")

    x = relay.var("x", relay.TensorType((1, 2, 3, 4), "float32"))
    y = relay.var("y", relay.TensorType((2, 3, 4, 1, 6), "float32"))
    z = relay.reshape_like(x, y, rhs_end=3)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((2, 3, 4), "float32")
    z = relay.reshape_like(x, y, rhs_begin=2)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((4, 1, 6), "float32")

    # symbolic partial reshaping
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "float32"))
    y = relay.var("y", relay.TensorType((5, 6), "float32"))
    z = relay.var("z", relay.TensorType((4,), "float32"))
    w = relay.reshape_like(x, y, lhs_end=3)
    w = relay.reshape_like(w, z, lhs_begin=2)
    w = run_infer_type(w)
    assert w.checked_type == relay.TensorType((5, 6, 4), "float32")


@tvm.testing.uses_gpu
def test_reshape_like():
    def verify_reshape_like(shape, oshape, shape_like=None, reshape_like_kwargs={}):
        if shape_like is None:
            shape_like = oshape
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        y_data = np.random.uniform(low=-1, high=1, size=shape_like).astype("float32")
        ref_res = np.reshape(x_data, oshape)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = relay.var("x", relay.TensorType(shape_like, "float32"))
        z = relay.reshape_like(x, y, **reshape_like_kwargs)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.ty.TensorType(ref_res.shape, "float32")

        func = relay.Function([x, y], z)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_reshape_like((2, 3, 4), (1, 8, 3))
    verify_reshape_like((4, 7), (2, 7, 2))
    verify_reshape_like(
        (1, 2, 3, 4), (1, 6, 4), (1, 6, 5), dict(lhs_begin=1, lhs_end=3, rhs_begin=1, rhs_end=2)
    )


def test_take_infer_type():
    def verify_take(dshape, indices_shape, oshape, axis=None):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        y = relay.take(x, indices, axis=axis)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(oshape, "float32")

    d1, d2, d3 = te.var("d1"), te.var("d2"), te.var("d3")
    d4, d5, d6 = te.var("d4"), te.var("d5"), te.var("d6")
    verify_take((d1,), (1,), (1,), 0)
    verify_take((4,), (d1, d2), (d1, d2))
    verify_take((3, 3, 3), (1, d2), (1, d2))
    verify_take((d1, d2), (d3, d4, d5), (d3, d4, d5, d2), 0)
    verify_take((d1, d2), (d3, d4, d5), (d1, d3, d4, d5), 1)
    verify_take((d1, d2, d3, d4), (d5, d6), (d1, d2, d5, d6, d4), -2)


@tvm.testing.uses_gpu
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

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data, indices_src)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_take((4,), [1])
    verify_take((4,), [[0, 1, 2, 3]])
    verify_take((3, 3, 3), [[11, 25]])
    verify_take((4,), [[0, 1], [2, 3]])
    verify_take((4,), [1], 0)
    verify_take((2, 2), [[[1, 0], [0, 1]]], 0)
    verify_take((2, 2), [[[1, 0], [0, 1]]], 1)
    verify_take((4, 3, 5, 6), [[2, 1, 0, 0]], -2)
    verify_take((3, 4), [-5, 20])
    verify_take((3, 4), [-5, 20], mode="wrap")
    verify_take((3, 4), [-1, 2], axis=0)
    verify_take((3, 4), [-1, 2], axis=0, mode="wrap")
    verify_take((3, 4), [-1, 2], axis=1)
    verify_take((3, 4), [-1, 2], axis=1, mode="wrap")
    verify_take((3, 3, 3), [[11, 25]], mode="fast")
    verify_take((3, 4), [0, 2], axis=0, mode="fast")
    verify_take((3, 4), [0, 2], axis=1, mode="fast")


def test_split_infer_type():
    def verify_split(dshape, indices_or_sections, ret_type, axis=None):
        x = relay.var("x", relay.ty.TensorType(dshape, "float32"))
        y = relay.split(x, indices_or_sections, axis=axis)
        yy = run_infer_type(y.astuple())
        assert yy.checked_type == ret_type

    idxd = tvm.tir.indexdiv

    d1, d2, d3, d4 = te.var("d1"), te.var("d2"), te.var("d3"), te.var("d4")
    axis = te.var("axis")
    verify_split(
        (5, 5, 2, 2),
        5,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                    relay.ty.TensorType((5, 1, 2, 2), "float32"),
                ]
            )
        ),
        axis=1,
    )
    verify_split(
        (5, 5, 2, 2),
        5,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                    relay.ty.TensorType((1, 5, 2, 2), "float32"),
                ]
            )
        ),
        axis=0,
    )
    verify_split(
        (d1, d2, d3, d4),
        4,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                    relay.ty.TensorType((d1, d2, idxd(d3, 4), d4), "float32"),
                ]
            )
        ),
        axis=2,
    )
    verify_split(
        (d1, d2, d3, d4),
        2,
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((idxd(d1, 2), d2, d3, d4), "float32"),
                    relay.ty.TensorType((idxd(d1, 2), d2, d3, d4), "float32"),
                ]
            )
        ),
        axis=0,
    )
    verify_split(
        (d1, d2, d3, d4),
        (2, 4, 7),
        relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                    relay.ty.TensorType((d1, 2, d3, d4), "float32"),
                    relay.ty.TensorType((d1, 3, d3, d4), "float32"),
                    relay.ty.TensorType((d1, (d2 - 7), d3, d4), "float32"),
                ]
            )
        ),
        axis=1,
    )


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


@tvm.testing.uses_gpu
def test_full():
    def verify_full(fill_value, src_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        z = relay.full(x, src_shape, dtype)
        func = relay.Function([x], z)
        ref_res = np.full(src_shape, fill_value)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(np.array(fill_value, dtype))
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_full(4, (1, 3, 4, 4), "int32")
    # verify_full(4, (1, 3, 4, 4), "int64") # This does not pass, python int32 is not upcast to int64, not sure how to fix it.
    verify_full(4.0, (1, 4), "float32")


def test_full_like_infer_type():
    # concrete shape
    base = relay.var("base", relay.TensorType((1, 2, 3), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1, 2, 3), "float32")

    # symbolic shape
    n, c, h, w = te.size_var("n"), 2, 3, te.size_var("w")
    base = relay.var("base", relay.TensorType((n, c, h, w), "float32"))
    fill = relay.var("fill", relay.TensorType((), "float32"))
    y = relay.full_like(base, fill)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, c, h, w), "float32")


@tvm.testing.uses_gpu
def test_full_like():
    def verify_full_like(base, fill_value, dtype):
        x_data = np.random.uniform(low=-1, high=1, size=base).astype(dtype)
        x = relay.var("x", relay.TensorType(base, dtype))
        y = relay.var("y", relay.scalar_type(dtype))
        z = relay.full_like(x, y)

        func = relay.Function([x, y], z)
        ref_res = np.full_like(x_data, fill_value)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data, np.array(fill_value, dtype))
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_full_like((1, 3, 4, 4), 4, "int32")
    verify_full_like((1, 1), 44.0, "float32")


@tvm.testing.uses_gpu
def test_infer_type_leaky_relu():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
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

    for target, dev in tvm.testing.enabled_targets():
        intrp1 = relay.create_executor("graph", device=dev, target=target)
        intrp2 = relay.create_executor("debug", device=dev, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.numpy(), ref_res, rtol=1e-5)


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

    if all(isinstance(v, tvm.tir.Var) == 1 for v in data) or not alpha:
        return

    func = relay.Function([x, y], z)
    x_data = np.random.uniform(low=-1, high=1, size=data).astype(dtype)
    a_data = np.random.uniform(low=-1, high=1, size=alpha).astype(dtype)

    if axis == 1:
        ref_res = (x_data < 0) * (x_data * a_data.reshape(3, 1, 1)) + (x_data >= 0) * x_data
    else:
        ref_res = (x_data < 0) * (x_data * a_data.reshape(1, 1, 3)) + (x_data >= 0) * x_data

    for target, dev in tvm.testing.enabled_targets():
        intrp1 = relay.create_executor("graph", device=dev, target=target)
        intrp2 = relay.create_executor("debug", device=dev, target=target)
        op_res1 = intrp1.evaluate(func)(x_data, a_data)
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data, a_data)
        tvm.testing.assert_allclose(op_res2.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_infer_type_prelu():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    verify_infer_type_prelu((n, c, h, w), (c,), 1, (n, c, h, w))
    verify_infer_type_prelu((n, h, w, c), (c,), 3, (n, h, w, c))
    verify_infer_type_prelu((n, c, h, w), None, 1, (n, c, h, w))
    verify_infer_type_prelu((n, h, w, c), None, 3, (n, h, w, c))
    verify_infer_type_prelu((1, 3, 2, 2), (3,), 1, (1, 3, 2, 2))
    verify_infer_type_prelu((1, 2, 2, 3), (3,), 3, (1, 2, 2, 3))
    verify_infer_type_prelu((1, 3, 2, 2), None, 1, (1, 3, 2, 2))
    verify_infer_type_prelu((1, 2, 2, 3), None, 3, (1, 2, 2, 3))


@tvm.testing.uses_gpu
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
                relay.const(step, dtype=dtype),
            )
            ref_res = np.arange(start, stop, step).astype(dtype)

        func = relay.Function([], x)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)()
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

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


@tvm.testing.uses_gpu
def test_meshgrid():
    def verify_meshgrid(lengths, indexing="ij"):
        input_vars = []
        input_data = []
        for i, length in enumerate(lengths):
            input_name = "x_{}".format(i)
            if length == 0:
                # Scalar
                input_vars.append(relay.var(input_name, relay.scalar_type("float32")))
                input_data.append(np.array(1, "float32"))
            else:
                input_vars.append(relay.var(input_name, relay.TensorType((length,), "float32")))
                input_data.append(np.arange(length).astype("float32"))

        z = relay.meshgrid(input_vars, indexing=indexing).astuple()
        func = relay.Function(input_vars, z)
        # Get ref
        ref_res = np.meshgrid(*input_data, indexing=indexing)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(*input_data)
                assert len(op_res) == len(ref_res)
                for i in range(len(op_res)):
                    tvm.testing.assert_allclose(op_res[i].numpy(), ref_res[i], rtol=1e-5)

    verify_meshgrid([3, 5])
    verify_meshgrid([4, 2], indexing="xy")
    verify_meshgrid([3, 5, 2])
    verify_meshgrid([3, 1, 5], indexing="xy")
    # Length 0 signifies scalar.
    verify_meshgrid([3, 5, 0])


@tvm.testing.uses_gpu
def test_tile():
    def verify_tile(dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.tile(x, reps=reps)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_tile((2, 3, 4), (3, 2, 1))
    verify_tile((2, 3, 4), (1, 2))
    verify_tile((2, 3), (3, 2, 1))


@tvm.testing.uses_gpu
def test_repeat():
    def verify_repeat(dshape, repeats, axis):
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        func = relay.Function([x], relay.repeat(x, repeats, axis))
        data = np.random.uniform(size=dshape).astype("float32")
        ref_res = np.repeat(data, repeats, axis)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_repeat((3,), 2, 0)
    verify_repeat((3, 10), 2, -1)
    verify_repeat((3, 2, 4), 3, 1)


@tvm.testing.uses_gpu
def test_stack():
    def produce_input_tuple(dshapes):
        y = [relay.var("input", relay.TensorType(shape, "float32")) for shape in dshapes]
        return relay.Tuple(y)

    def ref_stack(inputs, axis):
        return np.stack(inputs, axis=axis)

    def verify_stack(input_expr, relay_args, ref_res, axis):
        z = relay.stack(input_expr, axis=axis)
        inp_vars = relay.analysis.free_vars(z)
        func = relay.Function(inp_vars, z)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(*relay_args)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    def verify_tup_lit_stack(dshapes, axis):
        input_tuple = produce_input_tuple(dshapes)
        input_data = [np.random.normal(size=shape).astype("float32") for shape in dshapes]
        ref_res = ref_stack(input_data, axis)
        verify_stack(input_tuple, input_data, ref_res, axis)

    def verify_list_lit_stack(dshapes, axis):
        input_list = produce_input_tuple(dshapes).fields
        input_data = [np.random.normal(size=shape).astype("float32") for shape in dshapes]
        ref_res = ref_stack(input_data, axis)
        verify_stack(input_list, input_data, ref_res, axis)

    def verify_tup_expr_stack(dshapes, axis):
        input_data = [np.random.normal(size=shape).astype("float32") for shape in dshapes]
        ref_res = ref_stack(input_data, axis)

        # expression that evaluates to a tuple
        # but is not a tuple literal
        x = relay.Var("x")
        input_expr = relay.Let(x, relay.Tuple([relay.const(inp) for inp in input_data]), x)
        verify_stack(input_expr, [], ref_res, axis)

    dshape_axis_combos = [
        ([(2,), (2,), (2,)], -1),
        ([(2,), (2,), (2,)], 0),
        ([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1),
        ([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1),
        ([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], 4),
    ]

    for dshapes, axis in dshape_axis_combos:
        verify_tup_lit_stack(dshapes, axis)
        verify_list_lit_stack(dshapes, axis)
        verify_tup_expr_stack(dshapes, axis)


@tvm.testing.uses_gpu
def test_reverse():
    def verify_reverse(dshape, axis):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.reverse(x, axis=axis)
        zz = run_infer_type(z)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.flip(x_data, axis)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_reverse((2, 3, 4), 1)
    verify_reverse((4, 7), 0)
    verify_reverse((2, 3, 4), -1)


@tvm.testing.uses_gpu
def test_reverse_sequence():
    def verify_reverse_sequence(x_data, seq_lengths, batch_axis, seq_axis, ref_res):
        seq_lengths_data = np.array(seq_lengths).astype("int32")
        x = relay.var("x", relay.TensorType(x_data.shape, str(x_data.dtype)))
        z = relay.reverse_sequence(x, relay.const(seq_lengths_data), seq_axis, batch_axis)
        zz = run_infer_type(z)
        assert zz.checked_type == x.type_annotation
        func = relay.Function([x], z)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = [[0, 5, 10, 15], [4, 1, 6, 11], [8, 9, 2, 7], [12, 13, 14, 3]]
    verify_reverse_sequence(indata, [1, 2, 3, 4], 1, 0, np.array(result))
    verify_reverse_sequence(indata, [1, 2, 3, 4], -1, 0, np.array(result))
    verify_reverse_sequence(
        indata.astype("float32"), [1, 2, 3, 4], 1, 0, np.array(result).astype("float32")
    )

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = [[0, 1, 2, 3], [5, 4, 6, 7], [10, 9, 8, 11], [15, 14, 13, 12]]
    verify_reverse_sequence(indata, [1, 2, 3, 4], 0, 1, np.array(result))
    verify_reverse_sequence(indata, [1, 2, 3, 4], 0, -1, np.array(result))
    verify_reverse_sequence(
        indata.astype("float32"), [1, 2, 3, 4], 0, 1, np.array(result).astype("float32")
    )

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [15, 14, 13, 12]]
    verify_reverse_sequence(indata, [-1, 0, 1, 5], 0, 1, np.array(result))

    indata = np.array(np.arange(0, 54)).reshape([2, 3, 3, 3]).astype("int32")
    result = [
        [
            [[18, 19, 20], [21, 22, 23], [24, 25, 26]],
            [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        ],
        [
            [[45, 46, 47], [48, 49, 50], [51, 52, 53]],
            [[36, 37, 38], [39, 40, 41], [42, 43, 44]],
            [[27, 28, 29], [30, 31, 32], [33, 34, 35]],
        ],
    ]
    verify_reverse_sequence(indata, [3, 3], 0, 1, np.array(result))

    indata = np.array(np.arange(0, 54)).reshape([2, 3, 3, 3]).astype("int32")
    result = [
        [
            [[9, 10, 11], [21, 22, 23], [15, 16, 17]],
            [[0, 1, 2], [12, 13, 14], [6, 7, 8]],
            [[18, 19, 20], [3, 4, 5], [24, 25, 26]],
        ],
        [
            [[36, 37, 38], [48, 49, 50], [42, 43, 44]],
            [[27, 28, 29], [39, 40, 41], [33, 34, 35]],
            [[45, 46, 47], [30, 31, 32], [51, 52, 53]],
        ],
    ]
    verify_reverse_sequence(indata, [2, 3, 2], 2, 1, np.array(result))

    indata = np.array(np.arange(0, 16)).reshape([4, 4]).astype("int32")
    result = []
    with pytest.raises(Exception) as execinfo:
        verify_reverse_sequence(indata, [2, 3, 2, 4, 5], 1, 0, np.array(result))

    assert (
        "For reverse_sequnece seq_lengths size should match with dimension of batch axis,"
        " but got dimension of batch_axis = 4, and seq_length size = 5" in execinfo.value.args[0]
    )


@tvm.testing.uses_gpu
def test_scatter():
    def ref_scatter(data, indices, updates, axis=0):
        idx = np.indices(indices.shape).reshape(indices.ndim, -1)

        updated_idx = np.copy(idx)
        indices = indices.reshape(-1)
        for i in range(len(indices)):
            updated_idx[axis, i] = indices[i]
        scattered = np.copy(data)
        scattered[tuple(updated_idx)] = updates[tuple(idx)]
        return scattered

    def verify_scatter(dshape, ishape, axis=0):
        d = relay.var("d", relay.TensorType(dshape, "float32"))
        i = relay.var("i", relay.TensorType(ishape, "int64"))
        u = relay.var("u", relay.TensorType(ishape, "float32"))
        z = relay.op.scatter(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter(data_np, indices_np, updates_np, axis)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(data_np, indices_np, updates_np)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    def verify_dynamic_scatter(dshape, ishape, axis=0):
        d = relay.var("d", relay.TensorType([relay.Any() for i in range(len(dshape))], "float32"))
        i = relay.var("i", relay.TensorType([relay.Any() for i in range(len(ishape))], "int64"))
        u = relay.var("u", relay.TensorType([relay.Any() for i in range(len(ishape))], "float32"))
        z = relay.op.scatter(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter(data_np, indices_np, updates_np, axis)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["vm", "debug"]:
                mod = tvm.ir.IRModule.from_expr(func)
                intrp = relay.create_executor(kind, mod=mod, device=dev, target=target)
                op_res = intrp.evaluate()(data_np, indices_np, updates_np)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_scatter((10,), (10,), 0)
    verify_scatter((10, 5), (10, 5), -2)
    verify_scatter((10, 5), (10, 5), -1)
    verify_scatter((10, 5), (3, 5), 0)
    verify_scatter((12, 4), (7, 2), 1)
    verify_scatter((2, 3, 4), (1, 3, 4), 0)
    verify_scatter((2, 3, 4), (2, 1, 4), 1)
    verify_scatter((2, 3, 4), (2, 3, 1), 2)
    verify_scatter((4, 2, 1), (1, 1, 1), 0)
    verify_scatter((2, 3, 4, 5), (1, 3, 4, 5), 0)
    verify_scatter((6, 3, 4, 5), (2, 3, 4, 5), 1)
    verify_scatter((2, 3, 8, 5), (2, 3, 1, 1), 2)
    verify_scatter((16, 16, 4, 5), (16, 16, 4, 5), 3)

    verify_dynamic_scatter((10,), (10,), 0)
    verify_dynamic_scatter((10, 5), (10, 5), -2)
    verify_dynamic_scatter((10, 5), (10, 5), -1)
    verify_dynamic_scatter((10, 5), (3, 5), 0)
    verify_dynamic_scatter((12, 4), (7, 2), 1)
    verify_dynamic_scatter((2, 3, 4), (1, 3, 4), 0)
    verify_dynamic_scatter((2, 3, 4), (2, 1, 4), 1)
    verify_dynamic_scatter((2, 3, 4), (2, 3, 1), 2)
    verify_dynamic_scatter((4, 2, 1), (1, 1, 1), 0)
    verify_dynamic_scatter((2, 3, 4, 5), (1, 3, 4, 5), 0)
    verify_dynamic_scatter((6, 3, 4, 5), (2, 3, 4, 5), 1)
    verify_dynamic_scatter((2, 3, 8, 5), (2, 3, 1, 1), 2)
    verify_dynamic_scatter((16, 16, 4, 5), (16, 16, 4, 5), 3)


@tvm.testing.uses_gpu
@pytest.mark.parametrize(
    "dshape, ishape, axis, dtype",
    [
        ((10,), (10,), 0, "int32"),
        ((1000,), (1000,), 0, "int32"),
        ((10, 5), (10, 5), -2, "float32"),
        ((10, 5), (10, 5), -1, "float32"),
        ((10, 5), (3, 5), 0, "float32"),
        ((12, 4), (7, 2), 1, "float32"),
        ((2, 3, 4), (1, 3, 4), 0, "float32"),
        ((2, 3, 4), (2, 1, 4), 1, "float32"),
        ((2, 3, 4), (2, 3, 1), 2, "float32"),
        ((2, 3, 4, 5), (1, 3, 4, 5), 0, "float32"),
        ((6, 3, 4, 5), (2, 3, 4, 5), 1, "float32"),
        ((2, 3, 8, 5), (2, 3, 1, 1), 2, "float32"),
        ((16, 16, 4, 5), (16, 16, 4, 5), 3, "float32"),
    ],
)
def test_scatter_add(dshape, ishape, axis, dtype):
    def ref_scatter_add(data, indices, updates, axis=0):
        output = np.copy(data)
        for index in np.ndindex(*indices.shape):
            new_index = list(index)
            new_index[axis] = indices[index]
            output[tuple(new_index)] += updates[index]
        return output

    def verify_scatter_add(dshape, ishape, axis=0, dtype="float32"):
        d = relay.var("d", relay.TensorType(shape=[relay.Any() for _ in dshape], dtype=dtype))
        i = relay.var("i", relay.TensorType(shape=[relay.Any() for _ in ishape], dtype="int64"))
        u = relay.var("u", relay.TensorType(shape=[relay.Any() for _ in ishape], dtype=dtype))
        z = relay.op.scatter_add(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype(dtype)
        updates_np = np.random.uniform(size=ishape).astype(dtype)
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter_add(data_np, indices_np, updates_np, axis)

        verify_func(
            func,
            [data_np, indices_np, updates_np],
            ref_res,
        )

    verify_scatter_add(dshape, ishape, axis, dtype)


@tvm.testing.uses_gpu
@pytest.mark.parametrize(
    "data, axis, indices, ref_res",
    [
        ([[1, 2], [3, 4]], 1, [[0, 0], [1, 0]], [[1, 1], [4, 3]]),
        ([[1, 2], [3, 4]], -1, [[0, 0], [1, 0]], [[1, 1], [4, 3]]),
        (
            [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
            0,
            [[[1, 0, 1], [1, 1, 0]]],
            [[[6, 1, 8], [9, 10, 5]]],
        ),
        (
            [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]],
            -3,
            [[[1, 0, 1], [1, 1, 0]]],
            [[[6, 1, 8], [9, 10, 5]]],
        ),
        (
            [
                [
                    [-0.2321, -0.2024, -1.7624],
                    [-0.3829, -0.4246, 0.2448],
                    [0.1822, 0.2360, -0.8965],
                    [0.4497, -0.2224, 0.6103],
                ],
                [
                    [0.0408, -0.7667, -0.4303],
                    [-0.3216, 0.7489, -0.1502],
                    [0.0144, -0.4699, -0.0064],
                    [-0.0768, -1.6064, 1.3390],
                ],
            ],
            1,
            [[[2, 2, 0], [1, 0, 3]], [[3, 2, 0], [1, 0, 0]]],
            [
                [[0.1822, 0.2360, -1.7624], [-0.3829, -0.2024, 0.6103]],
                [[-0.0768, -0.4699, -0.4303], [-0.3216, -0.7667, -0.4303]],
            ],
        ),
        (
            [
                [
                    [-0.2321, -0.2024, -1.7624],
                    [-0.3829, -0.4246, 0.2448],
                    [0.1822, 0.2360, -0.8965],
                    [0.4497, -0.2224, 0.6103],
                ],
                [
                    [0.0408, -0.7667, -0.4303],
                    [-0.3216, 0.7489, -0.1502],
                    [0.0144, -0.4699, -0.0064],
                    [-0.0768, -1.6064, 1.3390],
                ],
            ],
            -2,
            [[[2, 2, 0], [1, 0, 3]], [[3, 2, 0], [1, 0, 0]]],
            [
                [[0.1822, 0.2360, -1.7624], [-0.3829, -0.2024, 0.6103]],
                [[-0.0768, -0.4699, -0.4303], [-0.3216, -0.7667, -0.4303]],
            ],
        ),
        (
            [
                [
                    [-0.2321, -0.2024, -1.7624],
                    [-0.3829, -0.4246, 0.2448],
                    [0.1822, 0.2360, -0.8965],
                    [0.4497, -0.2224, 0.6103],
                ],
                [
                    [0.0408, -0.7667, -0.4303],
                    [-0.3216, 0.7489, -0.1502],
                    [0.0144, -0.4699, -0.0064],
                    [-0.0768, -1.6064, 1.3390],
                ],
            ],
            -2,
            [[[2, 2, 0], [1, 0, 3]], [[3, 2, 0], [1, 0, 0]]],
            [
                [[0.1822, 0.2360, -1.7624], [-0.3829, -0.2024, 0.6103]],
                [[-0.0768, -0.4699, -0.4303], [-0.3216, -0.7667, -0.4303]],
            ],
        ),
        (
            [
                [
                    [0.3050, 1.6986, 1.1034],
                    [0.7020, -0.6960, -2.1818],
                    [0.3116, -0.5773, -0.9912],
                    [0.0835, -1.3915, -1.0720],
                ],
                [
                    [0.1694, -0.6091, -0.6539],
                    [-0.5234, -0.1218, 0.5084],
                    [0.2374, -1.9537, -2.0078],
                    [-0.5700, -1.0302, 0.1558],
                ],
            ],
            2,
            [
                [[1, 1, 0, 1], [0, 0, 2, 2], [1, 2, 1, 2], [2, 2, 1, 0]],
                [[0, 0, 1, 2], [2, 2, 1, 0], [1, 2, 0, 0], [0, 2, 0, 2]],
            ],
            [
                [
                    [1.6986, 1.6986, 0.3050, 1.6986],
                    [0.7020, 0.7020, -2.1818, -2.1818],
                    [-0.5773, -0.9912, -0.5773, -0.9912],
                    [-1.0720, -1.0720, -1.3915, 0.0835],
                ],
                [
                    [0.1694, 0.1694, -0.6091, -0.6539],
                    [0.5084, 0.5084, -0.1218, -0.5234],
                    [-1.9537, -2.0078, 0.2374, 0.2374],
                    [-0.5700, 0.1558, -0.5700, 0.1558],
                ],
            ],
        ),
        (
            [
                [
                    [0.3050, 1.6986, 1.1034],
                    [0.7020, -0.6960, -2.1818],
                    [0.3116, -0.5773, -0.9912],
                    [0.0835, -1.3915, -1.0720],
                ],
                [
                    [0.1694, -0.6091, -0.6539],
                    [-0.5234, -0.1218, 0.5084],
                    [0.2374, -1.9537, -2.0078],
                    [-0.5700, -1.0302, 0.1558],
                ],
            ],
            -1,
            [
                [[1, 1, 0, 1], [0, 0, 2, 2], [1, 2, 1, 2], [2, 2, 1, 0]],
                [[0, 0, 1, 2], [2, 2, 1, 0], [1, 2, 0, 0], [0, 2, 0, 2]],
            ],
            [
                [
                    [1.6986, 1.6986, 0.3050, 1.6986],
                    [0.7020, 0.7020, -2.1818, -2.1818],
                    [-0.5773, -0.9912, -0.5773, -0.9912],
                    [-1.0720, -1.0720, -1.3915, 0.0835],
                ],
                [
                    [0.1694, 0.1694, -0.6091, -0.6539],
                    [0.5084, 0.5084, -0.1218, -0.5234],
                    [-1.9537, -2.0078, 0.2374, 0.2374],
                    [-0.5700, 0.1558, -0.5700, 0.1558],
                ],
            ],
        ),
    ],
)
def test_gather(data, axis, indices, ref_res):
    def verify_gather(data, axis, indices, ref_res):
        data = np.asarray(data, dtype="float32")
        indices = np.asarray(indices, dtype="int32")
        ref_res = np.asarray(ref_res)
        d = relay.var("x", relay.TensorType(data.shape, "float32"))
        i = relay.var("y", relay.TensorType(indices.shape, "int32"))
        z = relay.gather(d, axis, i)

        func = relay.Function([d, i], z)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(data, indices)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_gather(data, axis, indices, ref_res)


@tvm.testing.uses_gpu
def test_gather_nd():
    def verify_gather_nd(xshape, yshape, y_data, batch_dims=0):
        x = relay.var("x", relay.TensorType(xshape, "float32"))
        y = relay.var("y", relay.TensorType(yshape, "int32"))
        z = relay.gather_nd(x, y, batch_dims)

        func = relay.Function([x, y], z)

        x_data = np.random.uniform(size=xshape).astype("float32")

        if y_data:
            y_data = np.array(y_data, dtype="int32")
        else:
            y_data = np.random.randint(low=0, high=2, size=yshape, dtype="int32")

        ref_res = ref_funcs.gather_nd(x_data, y_data, batch_dims)

        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_gather_nd((2, 2), (2, 3), [[1, 1, 0], [0, 1, 0]])
    verify_gather_nd((2, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify_gather_nd((3, 2, 2), (2, 2), [[0, 1], [1, 0]])
    verify_gather_nd((3, 2), (2, 2, 3), [[[0, 1, 2], [2, 0, 1]], [[0, 0, 0], [1, 1, 1]]])

    # Examples from tensorflow gather_nd doc
    # https://www.tensorflow.org/api_docs/python/tf/gather_nd
    verify_gather_nd((2, 2, 2), (1, 2), [[1, 0]], 1)
    verify_gather_nd((2, 2, 2), (1, 2, 1), [[[1], [0]]], 1)
    verify_gather_nd((2, 2, 2), (2, 2, 1), [[[1], [0]], [[0], [1]]], 1)

    # Test cases from tensorflow gather_nd tests kernel_tests/array_ops_test.py
    verify_gather_nd((2, 2, 2), (1, 2), None, 1)
    verify_gather_nd((2, 2, 2), (2, 2), None, 1)
    verify_gather_nd((2, 2, 3, 2), (3, 2), None, 1)
    verify_gather_nd((2, 2, 3, 2), (2, 2), None, 1)
    verify_gather_nd((2, 2, 3, 2), (1, 2), None, 1)
    verify_gather_nd((2, 2, 3, 2), (3, 2, 1), None, 1)
    verify_gather_nd((2, 2, 3, 2), (2, 2, 2), None, 1)
    verify_gather_nd((2, 2, 3, 2), (1, 2, 3), None, 1)

    verify_gather_nd((3, 2, 2, 3, 4), (3, 3, 2), None, 2)
    verify_gather_nd((3, 2, 2, 3, 4), (2, 3, 2), None, 2)
    verify_gather_nd((3, 2, 2, 3, 4), (1, 3, 2), None, 2)
    verify_gather_nd((3, 2, 2, 3, 4), (3, 3, 2, 1), None, 2)
    verify_gather_nd((3, 2, 2, 3, 4), (2, 3, 2, 2), None, 2)
    verify_gather_nd((3, 2, 2, 3, 4), (1, 3, 2, 3), None, 2)


def _verify_infiniteness_ops(relay_op, ref_op):
    for dtype in ["float32", "float16", "float16", "int32", "int16"]:
        shape = (2, 8, 8)
        x = relay.var("x", relay.TensorType(shape, dtype))
        y = relay_op(x)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(shape, "bool")

        data = np.random.uniform(size=shape).astype(dtype)
        if dtype.startswith("float"):
            data.ravel()[
                np.random.choice(data.size, int(data.size * 0.5), replace=False)
            ] = np.infty
            data.ravel()[np.random.choice(data.size, int(data.size * 0.5), replace=False)] = np.nan

        intrp = create_executor()
        op_res = intrp.evaluate(y, {x: data})
        ref_res = ref_op(data)
        np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


def test_isfinite():
    _verify_infiniteness_ops(relay.isfinite, np.isfinite)


def test_isinf():
    _verify_infiniteness_ops(relay.isinf, np.isinf)


@tvm.testing.uses_gpu
def test_unravel_index():
    def verify_unravel_index(indices, shape, dtype):
        x_data = np.array(indices).astype(dtype)
        y_data = np.array(shape).astype(dtype)
        x = relay.var("x", relay.TensorType(x_data.shape, dtype))
        y = relay.var("y", relay.TensorType(y_data.shape, dtype))

        z = relay.unravel_index(x, y)
        zz = run_infer_type(z)

        if len(x_data.shape) == 1:
            out_shape = [y_data.shape[0], x_data.shape[0]]
        else:
            out_shape = [y_data.shape[0]]
        assert zz.checked_type == relay.ty.TensorType(out_shape, dtype)

        func = relay.Function([x, y], z)
        ref_res = np.unravel_index(x_data, y_data)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    for dtype in ["int64", "int32"]:
        verify_unravel_index([0, 1, 2, 3], [2, 2], dtype)
        verify_unravel_index([144], [5, 5, 5, 2], dtype)
        verify_unravel_index(144, [5, 5, 5, 2], dtype)
        verify_unravel_index([100, 13, 5], [5, 5, 5, 2], dtype)

        # In below example, 5 is out of bound for array of size 4.
        # Numpy implementation throws error for it
        # TVM implementation does not throw error instead it produces
        # output which is inline with Tensorflow
        # verify_unravel_index([0, 1, 2, 5], [2, 2], dtype)


@tvm.testing.uses_gpu
def test_sparse_to_dense():
    def verify_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape, xpected):
        sparse_indices_data = np.array(sparse_indices)
        sparse_values_data = np.array(sparse_values)
        default_value_data = np.array(default_value)

        a = relay.var(
            "a", relay.TensorType(sparse_indices_data.shape, str(sparse_indices_data.dtype))
        )
        b = relay.var(
            "b", relay.TensorType(sparse_values_data.shape, str(sparse_values_data.dtype))
        )
        if default_value is None:
            args = [a, b]
            d = relay.sparse_to_dense(a, output_shape, b)
        else:
            c = relay.var(
                "c", relay.TensorType(default_value_data.shape, str(default_value_data.dtype))
            )
            args = [a, b, c]
            d = relay.sparse_to_dense(a, output_shape, b, c)

        zz = run_infer_type(d)
        assert zz.checked_type == relay.ty.TensorType(output_shape, str(sparse_values_data.dtype))

        func = relay.Function(args, d)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                if default_value is None:
                    op_res = intrp.evaluate(func)(sparse_indices_data, sparse_values_data)
                else:
                    op_res = intrp.evaluate(func)(
                        sparse_indices_data, sparse_values_data, default_value_data
                    )
                tvm.testing.assert_allclose(op_res.numpy(), xpected, rtol=1e-5)

    verify_sparse_to_dense(1, 3, 0, [5], [0, 3, 0, 0, 0])  # scalar
    verify_sparse_to_dense([0, 1, 4], [3, 3, 3], 0, [5], [3, 3, 0, 0, 3])  # vector
    verify_sparse_to_dense(
        [[0, 0], [1, 2]], [1, 2], 0, [3, 4], [[1, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]]
    )  # nXd
    verify_sparse_to_dense(
        [[0, 0, 0], [1, 2, 3]],
        [1, 2],
        4,
        [2, 3, 4],
        [[[1, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]], [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 2]]],
    )  # nXd
    verify_sparse_to_dense(
        [0, 1, 4], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1]
    )  # floats
    verify_sparse_to_dense(1, 3, None, [5], [0, 3, 0, 0, 0])  # default value not specified

    # negative test cases
    # sparse indices should be ints
    # verify_sparse_to_dense([[0.1, 1.1, 4.1], [0,2,4]], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])
    # sparse_values should be 0d or 1d only
    # verify_sparse_to_dense([[0, 1, 4], [0, 2, 4]], [[[3.1, 3.1, 3.1]]], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])
    # sparse_indices should not be > 2d tensor
    # verify_sparse_to_dense([[[[0, 1, 4], [0, 2, 4]]]], [[[[3.1, 3.1, 3.1]]]], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])


@tvm.testing.uses_gpu
@pytest.mark.parametrize(
    "sparse_indices_np, sparse_values_np, prev_shape_np, new_shape_np",
    [
        (
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 3]], dtype=np.int32),
            np.array([7, 5, 6, 3, 9], dtype=np.int32),
            np.array([2, 3, 6], dtype=np.int32),
            np.array([9, -1], dtype=np.int32),
        ),
        (
            np.array(
                [[0, 0, 0, 0], [0, 0, 1, 2], [0, 1, 0, 3], [1, 0, 0, 4], [1, 2, 3, 6]],
                dtype=np.int64,
            ),
            np.array([7, 5, 6, 3, 9], dtype=np.int64),
            np.array([2, 3, 6, 7], dtype=np.int64),
            np.array([9, -1, 7], dtype=np.int64),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 3],
                    [0, 1, 0, 3, 5],
                    [1, 0, 0, 4, 6],
                    [1, 2, 3, 6, 8],
                ],
                dtype=np.int64,
            ),
            np.array([7, 5, 6, 3, 9], dtype=np.int64),
            np.array([2, 3, 6, 7, 9], dtype=np.int64),
            np.array([9, -1, 7], dtype=np.int64),
        ),
        (
            np.array([[0, 0], [0, 1], [3, 4], [4, 3], [7, 3]], dtype=np.int32),
            np.array([7, 5, 6, 3, 9], dtype=np.int32),
            np.array([9, 4], dtype=np.int32),
            np.array([2, -1, 6], dtype=np.int32),
        ),
        (
            np.array([[0, 0], [0, 1], [3, 4], [4, 3], [7, 3]], dtype=np.int64),
            np.array([7, 5, 6, 3, 9], dtype=np.int64),
            np.array([9, 4], dtype=np.int64),
            np.array([-1], dtype=np.int64),
        ),
        (
            np.array([[0], [5], [10], [20], [24]], dtype=np.int32),
            np.array([7, 5, 6, 3, 9], dtype=np.int32),
            np.array([25], dtype=np.int32),
            np.array([5, 5], dtype=np.int32),
        ),
        (
            np.array([[0, 100], [200, 100], [300, 400], [50, 20], [400, 50]], dtype=np.int64),
            np.array([7, 5, 6, 3, 9], dtype=np.int64),
            np.array([500, 20], dtype=np.int64),
            np.array([500, 20], dtype=np.int64),
        ),
        (
            np.array([[0, 100], [200, 100], [300, 400], [50, 20], [400, 50]], dtype=np.int32),
            np.array([7, 5, 6, 3, 9], dtype=np.int32),
            np.array([500, 20], dtype=np.int32),
            np.array([500, -1], dtype=np.int32),
        ),
        (
            np.array([[0, 100], [200, 100], [300, 400], [50, 20], [400, 50]], dtype=np.int64),
            np.array([7, 5, 6, 3, 9], dtype=np.int64),
            np.array([500, 20], dtype=np.int64),
            np.array([250, 40], dtype=np.int64),
        ),
        (
            np.ones((0, 1), dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([4], dtype=np.int32),
            np.array([2, -1], dtype=np.int32),
        ),
        (
            np.ones((0, 1), dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([4], dtype=np.int64),
            np.array([2, 2], dtype=np.int64),
        ),
        (
            np.ones((0, 2), dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([3, 6], dtype=np.int32),
            np.array([-1, 2], dtype=np.int32),
        ),
    ],
)
@pytest.mark.parametrize("use_dyn", [True, False])
def test_sparse_reshape(sparse_indices_np, sparse_values_np, prev_shape_np, new_shape_np, use_dyn):
    def ref_sparse_reshape(
        sparse_indices: np.ndarray,
        prev_shape: np.ndarray,
        new_shape: np.ndarray,
    ):
        """
        This function calculates the expected output of sparseshape operator given the inputs.
        """

        new_sparse_indices = np.ones(
            (sparse_indices.shape[0], new_shape.shape[0]), dtype=sparse_indices.dtype
        )
        multipliers = np.ones(prev_shape.shape[0])
        dividers = np.ones(new_shape.shape[0])
        total_ele = np.prod(prev_shape)
        division_total_ele = 1
        for i in range(new_shape.shape[0]):
            if new_shape[i] == -1:
                continue
            division_total_ele *= new_shape[i]
        for i in range(prev_shape.shape[0] - 2, -1, -1):
            multipliers[i] = prev_shape[i + 1] * multipliers[i + 1]

        for i in range(len(new_shape)):
            if new_shape[i] == -1:
                new_shape[i] = total_ele // division_total_ele

        if np.array_equal(prev_shape, new_shape):
            return sparse_indices, prev_shape

        for i in range(new_shape.shape[0] - 2, -1, -1):
            dividers[i] = new_shape[i + 1] * dividers[i + 1]

        for row_num, sparse_row in enumerate(sparse_indices):
            flat_idx = 0
            if len(sparse_indices.shape) != 1:
                for i, ele in enumerate(sparse_row):
                    flat_idx += sparse_row[i] * multipliers[i]
            else:
                flat_idx += sparse_row
            if len(new_sparse_indices.shape) != 1:
                for i in range(new_sparse_indices.shape[1]):
                    new_sparse_indices[row_num][i] = flat_idx // dividers[i]
                    flat_idx = flat_idx % dividers[i]
            else:
                new_sparse_indices[row_num] = flat_idx

        return new_sparse_indices, new_shape

    def verify_sparse_reshape(
        sparse_indices_np: np.ndarray,
        sparse_values_np: np.ndarray,
        prev_shape_np: np.ndarray,
        new_shape_np: np.ndarray,
    ):
        """
        This function verifies the relay output of sparse_reshape with its expected output.
        """
        if use_dyn:
            sparse_indices = relay.var(
                "sparse_indices",
                shape=[relay.Any(), relay.Any()],
                dtype=str(sparse_indices_np.dtype),
            )
            prev_shape = relay.var(
                "prev_shape",
                shape=[relay.Any()],
                dtype=str(prev_shape_np.dtype),
            )
            new_shape = relay.var(
                "new_shape",
                shape=[relay.Any()],
                dtype=str(new_shape_np.dtype),
            )
        else:
            sparse_indices = relay.var(
                "sparse_indices",
                relay.TensorType(sparse_indices_np.shape, str(sparse_indices_np.dtype)),
            )
            prev_shape = relay.var(
                "prev_shape", relay.TensorType(prev_shape_np.shape, str(prev_shape_np.dtype))
            )
            new_shape = relay.var(
                "new_shape", relay.TensorType(new_shape_np.shape, str(new_shape_np.dtype))
            )
        z = relay.op.sparse_reshape(sparse_indices, prev_shape, new_shape).astuple()

        func = relay.Function([sparse_indices, prev_shape, new_shape], z)

        ref_res = ref_sparse_reshape(sparse_indices_np, prev_shape_np, new_shape_np)
        outputs = run_infer_type(z)
        new_sparse_indices_infer_type, new_shape_infer_type = (
            outputs.checked_type.fields[0].dtype,
            outputs.checked_type.fields[1].dtype,
        )

        assert new_sparse_indices_infer_type == sparse_indices_np.dtype
        assert new_shape_infer_type == new_shape_np.dtype
        verify_func(
            func,
            [sparse_indices_np, prev_shape_np, new_shape_np],
            ref_res,
        )

    verify_sparse_reshape(
        sparse_indices_np,
        sparse_values_np,
        prev_shape_np,
        new_shape_np,
    )


@tvm.testing.uses_gpu
@pytest.mark.parametrize(
    "data_np, segment_ids_np, num_segments",
    [
        (
            np.array([5, 1, 7, 2, 3, 4], dtype=np.float32),
            np.array([0, 0, 1, 1, 0, 1], dtype=np.int32),
            None,
        ),
        (
            np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float64),
            np.array([0, 0, 1], dtype=np.int32),
            None,
        ),
        (
            np.random.random((6, 4, 5)),
            np.array([2, 0, 1, 0, 3, 2], dtype=np.int64),
            None,
        ),
        (
            np.array([[[1, 7]], [[3, 8]], [[2, 9]]], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.int32),
            None,
        ),
        (
            np.random.random((9, 4, 5, 7)),
            np.array([5, 0, 1, 0, 3, 6, 8, 7, 7], dtype=np.int64),
            9,
        ),
        (
            np.array([[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8]], dtype=np.float64),
            np.array([0, 2], dtype=np.int32),
            4,
        ),
        (
            np.random.random((6, 4, 5)),
            np.array([0, 0, 1, 5, 5], dtype=np.int32),
            100,
        ),
    ],
)
@pytest.mark.parametrize("use_dyn", [True, False])
def test_segment_sum(data_np, segment_ids_np, num_segments, use_dyn):
    def ref_segment_sum(
        data: np.ndarray,
        segment_ids: np.ndarray,
        num_segments: Optional[int] = None,
    ):
        """
        This function calculates the expected output of segment_sum operator given the inputs.
        """
        if not num_segments:
            num_segments = np.unique(segment_ids).shape[0]

        result = np.zeros((num_segments,) + data.shape[1:], data.dtype)
        for i, index in enumerate(segment_ids):
            result[index] += data[i]
        return result

    def verify_segment_sum(
        data_np: np.ndarray, segment_ids_np: np.ndarray, num_segments: Optional[int]
    ):
        """
        This function verifies the relay output of segment_sum with its expected output.
        """
        if use_dyn:
            data = relay.var(
                "data",
                shape=[relay.Any() for _ in data_np.shape],
                dtype=str(data_np.dtype),
            )
            segment_ids = relay.var(
                "segment_ids",
                shape=[relay.Any()],
                dtype=str(segment_ids_np.dtype),
            )
        else:
            data = relay.var(
                "data",
                relay.TensorType(data_np.shape, str(data_np.dtype)),
            )
            segment_ids = relay.var(
                "segment_ids", relay.TensorType(segment_ids_np.shape, str(segment_ids_np.dtype))
            )
        z = relay.op.segment_sum(data, segment_ids, num_segments)

        func = relay.Function([data, segment_ids], z)
        ref_res = ref_segment_sum(data_np, segment_ids_np, num_segments=num_segments)
        segment_sum_result = run_infer_type(z)
        assert segment_sum_result.checked_type.dtype == data_np.dtype
        verify_func(
            func,
            [data_np, segment_ids_np],
            ref_res,
        )

    verify_segment_sum(data_np, segment_ids_np, num_segments)


def verify_func(func, data, ref_res, target_device=tvm.testing.enabled_targets()):
    assert isinstance(data, list)
    for target, dev in target_device:
        for kind in ["vm"]:
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor(kind, mod=mod, device=dev, target=target)
            op_res = intrp.evaluate()(*data)
            if isinstance(op_res, tvm.runtime.container.ADT):
                assert len(op_res) == len(
                    ref_res
                ), "Outputs from TVM and Python implementation must be equal "

                for op_result, ref_result in zip(op_res, ref_res):
                    tvm.testing.assert_allclose(op_result.numpy(), ref_result, rtol=1e-5)
            else:
                tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)
            relay.backend.compile_engine.get().clear()


@tvm.testing.uses_gpu
def test_adv_index():
    def verify_adv_index(data_shape, index_shapes):
        dtype = "float32"
        inputs = [relay.var("data", relay.TensorType(data_shape, dtype))]
        np_data = np.random.uniform(size=data_shape).astype(dtype)
        np_indices = []
        for i, index_shape in enumerate(index_shapes):
            limit = data_shape[i]
            np_indices.append(np.random.uniform(0, limit - 1, size=index_shape).astype("int64"))
            inputs.append(relay.var("index_{}".format(i), relay.TensorType(index_shape, "int64")))
        np_out = np_data[tuple(np_indices)]
        np_args = [np_data] + np_indices
        out = relay.op.adv_index(inputs)

        func = relay.Function(inputs, out)
        for target, dev in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, device=dev, target=target)
                op_res = intrp.evaluate(func)(*np_args)
                tvm.testing.assert_allclose(op_res.numpy(), np_out, rtol=1e-5)

    verify_adv_index((10, 5), [(3, 4), (3, 1)])
    verify_adv_index(
        (10, 5),
        [
            (2,),
        ],
    )
    verify_adv_index((10, 5, 15), [(1, 2, 1), (1, 2, 7)])


# Helper for testing binop functions
scanops_supported = {"cumsum": relay.op.cumsum, "cumprod": relay.op.cumprod}


def run_binop_tests(
    target, dev, binop_type: str, gt_func: Callable[..., np.array], identity_value: int
):
    def assert_relay_scanop(
        data_np: np.array,
        np_out: np.array,
        axis: int = None,
        out_dtype: str = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        exclusive: bool = False,
    ):
        inp = relay.var("data", relay.TensorType(data_np.shape, str(data_np.dtype)))

        if binop_type not in scanops_supported.keys():
            raise ValueError(f"Unknown function {binop_type}. Options: {scanops_supported.keys()}")
        out = scanops_supported[binop_type](inp, axis, out_dtype, exclusive=exclusive)
        func = relay.Function([inp], out)

        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            op_res = intrp.evaluate(func)(data_np)
            tvm.testing.assert_allclose(op_res.numpy(), np_out, rtol=rtol, atol=atol)

    data = np.array([2, 3, 0])
    assert_relay_scanop(data, gt_func(data))
    assert_relay_scanop(data, gt_func(data), out_dtype="int64")

    data = np.random.randn(10, 10)
    assert_relay_scanop(data, gt_func(data))
    assert_relay_scanop(data, gt_func(data, axis=0), axis=0)
    assert_relay_scanop(data, gt_func(data, axis=1), axis=1)

    data = np.random.randn(10, 5, 10).astype("float32")
    assert_relay_scanop(data, gt_func(data), rtol=1e-4, atol=1e-4)
    assert_relay_scanop(data, gt_func(data, axis=0), axis=0, rtol=1e-4, atol=1e-4)
    assert_relay_scanop(data, gt_func(data, axis=1), axis=1, rtol=1e-4, atol=1e-4)
    assert_relay_scanop(data, gt_func(data, axis=-1), axis=-1, rtol=1e-4, atol=1e-4)

    data = np.random.rand(10) > 0.5
    data = data.astype(np.int32)
    assert_relay_scanop(data, gt_func(data, dtype=np.int32))
    assert_relay_scanop(data, gt_func(data, dtype="int64"), out_dtype="int64")

    # Test exclusivity operations
    data = np.random.randint(-100, 100, size=(10, 10)).astype("int64")
    expected_result = np.roll(gt_func(data), 1)
    expected_result[0] = identity_value
    assert_relay_scanop(data, expected_result, exclusive=True)

    expected_result = np.roll(gt_func(data, axis=0), 1, axis=0)
    expected_result[0, :] = identity_value
    assert_relay_scanop(data, expected_result, exclusive=True, axis=0)

    expected_result = np.roll(gt_func(data, axis=1), 1, axis=1)
    expected_result[:, 0] = identity_value
    assert_relay_scanop(data, expected_result, exclusive=True, axis=1)


@tvm.testing.parametrize_targets
def test_cumsum(target, dev):
    run_binop_tests(target, dev, binop_type="cumsum", gt_func=np.cumsum, identity_value=0)


@tvm.testing.parametrize_targets
def test_cumprod(target, dev):
    run_binop_tests(target, dev, binop_type="cumprod", gt_func=np.cumprod, identity_value=1)


@tvm.testing.parametrize_targets
def test_scatter_nd(target, dev):
    def verify_scatter_nd(
        data_np, indices_np, updates_np, ref_res, mode="add", rtol=1e-5, atol=1e-5
    ):
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices = relay.var("indices", shape=indices_np.shape, dtype=str(indices_np.dtype))
        updates = relay.var("updates", shape=updates_np.shape, dtype=str(updates_np.dtype))

        out = relay.op.scatter_nd(data, indices, updates, mode)
        func = relay.Function([data, indices, updates], out)

        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            op_res = intrp.evaluate(func)(data_np, indices_np, updates_np)
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)

    def verify_scatter_nd_with_stack(
        data_np, indices_np, updates_np, ref_res, mode="add", rtol=1e-5, atol=1e-5
    ):
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices_vars = [
            relay.var("ind{i}", shape=v.shape, dtype=str(v.dtype)) for i, v in enumerate(indices_np)
        ]
        updates = relay.var("updates", shape=updates_np.shape, dtype=str(updates_np.dtype))

        # test if scatter_nd works in case indices are prepared by another Relay operator
        indices = relay.op.stack(indices_vars, axis=0)
        out = relay.op.scatter_nd(data, indices, updates, mode)
        func = relay.Function(
            [data, updates] + indices_vars,
            out,
        )

        fargs = [data_np, updates_np]
        for a in indices_np:
            fargs.append(a)
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, device=dev, target=target)
            op_res = intrp.evaluate(func)(*fargs)
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)

    data = np.zeros((2, 2)).astype("int64")
    indices = np.array([[1, 1, 0], [0, 1, 0]])
    updates = np.array([2, 3, 0])
    out = np.array([[0, 0], [2, 3]])
    verify_scatter_nd(data, indices, updates, out)
    verify_scatter_nd_with_stack(data, indices, updates, out)

    data = np.zeros((2, 2, 2, 2)).astype("int64")
    indices = np.array([[0, 1], [1, 1]])
    updates = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    out = np.array([[[[0, 0], [0, 0]], [[1, 2], [3, 4]]], [[[0, 0], [0, 0]], [[5, 6], [7, 8]]]])
    verify_scatter_nd(data, indices, updates, out)
    verify_scatter_nd_with_stack(data, indices, updates, out)

    indices = np.array([[1, 0, 0]])
    updates = np.reshape(np.arange(1560 * 3), (3, 1560)).astype("float32")
    shape = (2, 1560)
    data = np.zeros(shape).astype("float32")
    out = data.copy()
    out[1, :] += updates[0, :]
    out[0, :] += updates[1, :]
    out[0, :] += updates[2, :]
    verify_scatter_nd(data, indices, updates, out)
    verify_scatter_nd_with_stack(data, indices, updates, out)

    for mode in ["add", "update"]:
        indices = np.stack((np.random.randint(2, size=5), np.random.randint(7, size=5))).astype(
            "int64"
        )
        updates = np.ones((5, 3)).astype("float64")
        shape = (2, 7, 3)
        data = np.random.random(shape).astype("float64")
        out = data.copy()
        for i in range(indices.shape[1]):
            for j in range(updates.shape[1]):
                if mode == "add":
                    out[indices[0, i], indices[1, i], j] += updates[i, j]
                elif mode == "update":
                    out[indices[0, i], indices[1, i], j] = updates[i, j]
        verify_scatter_nd(data, indices, updates, out, mode)
        verify_scatter_nd_with_stack(data, indices, updates, out, mode)


def test_unique():
    def calc_numpy_unique(data, is_sorted=False):
        uniq, index, inverse, counts = np.unique(
            data, return_index=True, return_inverse=True, return_counts=True
        )
        num_uniq = np.array([len(uniq)]).astype("int32")
        if not is_sorted:
            order = np.argsort(index)
            reverse_order = np.argsort(order)
            uniq = uniq[order].astype(data.dtype)
            inverse = np.array([reverse_order[i] for i in inverse]).astype("int32")
            counts = counts[order].astype("int32")
            index = np.sort(index)  # In unsorted case, need to sort the index of first occurence
        return [
            uniq.astype(data.dtype),
            index.astype("int32"),
            inverse.astype("int32"),
            num_uniq,
            counts,
        ]

    def verify_unique(n, dtype, is_dyn=False, is_sorted=False, return_counts=False):
        if is_dyn:
            x = relay.var("x", relay.TensorType([relay.Any()], dtype))
        else:
            x = relay.var("x", relay.TensorType([n], dtype))
        outs = relay.unique(x, is_sorted, return_counts)
        outs = outs.astuple()
        func = relay.Function([x], outs)
        x_data = np.random.randint(50, size=n).astype(dtype)

        if is_dyn:
            backends = ["vm", "debug"]
        else:
            backends = ["graph", "debug"]

        for target, dev in tvm.testing.enabled_targets():
            for kind in backends:
                mod = tvm.ir.IRModule.from_expr(func)
                intrp = relay.create_executor(kind, mod=mod, device=dev, target=target)
                tvm_res = intrp.evaluate()(
                    x_data
                )  # unique, indices, inverse_indices, num_unique, (counts)
                np_res = calc_numpy_unique(
                    x_data, is_sorted
                )  # unique, indices, inverse_indices, num_unique, counts
                num_unique = np_res[3][0]

                # num_unique
                assert num_unique == tvm_res[3].numpy()[0]
                # unique
                tvm.testing.assert_allclose(tvm_res[0].numpy()[:num_unique], np_res[0], rtol=1e-5)
                # indices
                tvm.testing.assert_allclose(tvm_res[1].numpy()[:num_unique], np_res[1], rtol=1e-5)
                # inverse_indices
                tvm.testing.assert_allclose(tvm_res[2].numpy(), np_res[2], rtol=1e-5)
                # counts
                if return_counts:
                    tvm.testing.assert_allclose(
                        tvm_res[4].numpy()[:num_unique], np_res[4], rtol=1e-5
                    )

    for dtype in ["int32", "int64"]:
        for i in range(8):
            is_dyn, is_sorted, return_counts = bool(i & 1), bool(i & 2), bool(i & 4)
            verify_unique(10, dtype, is_dyn, is_sorted, return_counts)


if __name__ == "__main__":
    pytest.main([__file__])
