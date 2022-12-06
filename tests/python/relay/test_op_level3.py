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
import sys
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

executor_kind = tvm.testing.parameter("graph", "vm")


class TestZerosOnes:
    config = {"zeros": (relay.zeros, np.zeros), "ones": (relay.ones, np.ones)}
    op, ref = tvm.testing.parameters(*config.values(), ids=config.keys())

    def test_zeros_ones(self, op, ref):
        y = op(shape=(124, 50), dtype="float64")
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((124, 50), "float64")
        intrp_res = create_executor().evaluate(y).numpy()
        np.testing.assert_allclose(intrp_res, ref((124, 50), "float64"))


class TestUnaryIdentity:
    config = {
        "zeros_like": (relay.zeros_like, np.zeros_like),
        "ones_like": (relay.ones_like, np.ones_like),
        "ceil": (relay.ceil, np.ceil),
        "floor": (relay.floor, np.floor),
        "trunc": (relay.trunc, np.trunc),
        "round": (relay.round, np.round),
        "abs": (relay.abs, np.abs),
        "copy": (relay.copy, None),  # np.copy
        "negative": (relay.negative, np.negative),
        "sign": (relay.sign, np.sign),
    }
    op, ref = tvm.testing.parameters(*config.values(), ids=config.keys())

    def test_unary_identity(self, op, ref):
        shape = (8, 9, 4)
        x = relay.var("x", relay.TensorType(shape, "float32"))
        y = op(x)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(shape, "float32")

        if ref is not None:
            data = np.random.rand(*shape).astype("float32")
            op_res = create_executor().evaluate(y, {x: relay.const(data)})
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


def test_sliding_window():
    # Slide a window of shape (3, 4, 5) over the x tensor, beginning with
    # dimension 1, which slides the window over the two subtensors of shape (3,
    # 32, 32).
    x = relay.var("x", relay.TensorType((2, 3, 32, 32), "float32"))
    y = relay.sliding_window(x, 1, [3, 4, 5], [1, 2, 3])

    # The resulting shape still has batch size 2. Each dimension in (1, 15, 10)
    # represents the locations where we were able to form a window; that is, we
    # were able to place the window in one place along the dimension of length
    # 3, 15 places along the dimension of length 32 (when striding by 2), and 10
    # places along the second dimension of length 32 (when striding by 3). The
    # remaining dimensions (3, 4, 5) represent the formed windows.
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((2, 1, 15, 10, 3, 4, 5), "float32")

    data = np.random.rand(2, 3, 32, 32).astype("float32")
    intrp = create_executor()
    result = intrp.evaluate(y, {x: relay.const(data)})
    result_np = result.numpy()
    assert result_np.shape == (2, 1, 15, 10, 3, 4, 5)
    assert np.array_equal(result_np[0, 0, 0, 0, :, :, :], data[0, :, 0:4, 0:5])
    assert np.array_equal(result_np[1, 0, 7, 3, :, :, :], data[1, :, 14:18, 9:14])
    assert np.array_equal(result_np[1, 0, 14, 9, :, :, :], data[1, :, 28:32, 27:32])


def test_clip():
    a = relay.var("a", relay.TensorType((10, 4), "float32"))
    y = relay.clip(a, 1.0, 4.0)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((10, 4), "float32")

    data = np.random.rand(10, 4).astype("float32")
    op_res = create_executor().evaluate(y, {a: relay.const(data)})
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
    op_res = create_executor().evaluate(y, {a: relay.const(data)})
    ref_res = np.ones((10, 4)).astype("int32")
    np.testing.assert_allclose(op_res.numpy(), ref_res, atol=1)


def test_reinterpret():
    a = relay.var("a", relay.TensorType((1000, 4), "float32"))
    y = relay.reinterpret(a, "int32")
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000, 4), "int32")

    data = np.random.randn(1000, 4).astype("float32") * 1000
    op_res = create_executor().evaluate(y, {a: relay.const(data)})
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
    op_res = create_executor().evaluate(y, {a: relay.const(data)})

    def reference_sigmoid(x):
        return np.exp(-np.logaddexp(0, -x))

    np.testing.assert_allclose(op_res.numpy(), reference_sigmoid(data), atol=2e-5, rtol=1e-9)

    y = approximate_tanh(a)
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((1000,), "float32")
    data = np.linspace(-5, 5, 1000).astype("float32")
    op_res = create_executor().evaluate(y, {a: relay.const(data)})

    def reference_tanh(x):
        return np.tanh(x)

    np.testing.assert_allclose(op_res.numpy(), reference_tanh(data), atol=4e-5, rtol=1e-9)


class TestSqueeze:
    shape, dtype, axis = tvm.testing.parameters(
        ((1, 3, 2, 5), "float32", None),
        ((1, 3, 1), "float32", [0]),
        ((1, 2, 1, 2, 1), "float32", [0, 2]),
    )

    def test_squeeze(self, shape, dtype, axis):
        x = relay.var("x", relay.TensorType(shape, dtype))
        squeeze = relay.squeeze(x, axis=axis)

        np_axis = tuple(axis) if axis is not None else None

        data = np.random.random_sample(shape).astype(dtype)
        op_res = create_executor().evaluate(squeeze, {x: relay.const(data)})
        ref_res = np.squeeze(data, axis=np_axis)
        np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


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


def test_transpose(target, dev, executor_kind):
    dshape = (2, 3, 4)
    axes = (0, 2, 1)

    x = relay.var("x", relay.TensorType(dshape, "float32"))
    z = relay.transpose(x, axes=axes)

    func = relay.Function([x], z)
    x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
    ref_res = np.transpose(x_data, axes=axes)

    op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(x_data)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


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


class TestReshape:
    shape, newshape, oshape = tvm.testing.parameters(
        ((2, 3, 4), (8, 3), (8, 3)),
        ((4, 7), (2, 7, 2), (2, 7, 2)),
        ((2, 3, 4), (4, 0, 2), (4, 3, 2)),
        ((2, 3, 4), (2, 0, 0), (2, 3, 4)),
        ((2, 3, 4), (0, -1), (2, 12)),
        ((2, 3, 4), (-1, 0), (8, 3)),
        ((2, 3, 4), (2, -2), (2, 3, 4)),
        ((2, 3, 4), (-2, 1, 1), (2, 3, 4, 1, 1)),
        ((2, 3, 4), (-3, 4), (6, 4)),
        ((2, 3, 4, 5), (-3, -3), (6, 20)),
        ((2, 3, 4), (0, -3), (2, 12)),
        ((2, 3, 4), (-3, -2), (6, 4)),
        ((2, 3, 4), (-4, 1, 2, -2), (1, 2, 3, 4)),
        ((2, 3, 4), (2, -4, -1, 3, -2), (2, 1, 3, 4)),
        ((1,), (), ()),
    )

    def test_reshape(self, target, dev, executor_kind, shape, newshape, oshape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.reshape(x, newshape=newshape)
        zz = run_infer_type(z)
        assert "newshape=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

        func = relay.Function([x], z)
        check_grad(func)
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = np.reshape(x_data, oshape)
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


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


class TestReshapeLike:
    shape, oshape, shape_like, reshape_like_kwargs = tvm.testing.parameters(
        ((2, 3, 4), (1, 8, 3), None, {}),
        ((4, 7), (2, 7, 2), None, {}),
        ((1, 2, 3, 4), (1, 6, 4), (1, 6, 5), dict(lhs_begin=1, lhs_end=3, rhs_begin=1, rhs_end=2)),
    )

    def test_reshape_like(
        self, target, dev, executor_kind, shape, oshape, shape_like=None, reshape_like_kwargs={}
    ):
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

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, y_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestTakeInferType:
    d1, d2, d3 = te.var("d1"), te.var("d2"), te.var("d3")
    d4, d5, d6 = te.var("d4"), te.var("d5"), te.var("d6")
    dshape, indices_shape, oshape, axis = tvm.testing.parameters(
        ((d1,), (1,), (1,), 0),
        ((4,), (d1, d2), (d1, d2), None),
        ((3, 3, 3), (1, d2), (1, d2), None),
        ((d1, d2), (d3, d4, d5), (d3, d4, d5, d2), 0),
        ((d1, d2), (d3, d4, d5), (d1, d3, d4, d5), 1),
        ((d1, d2, d3, d4), (d5, d6), (d1, d2, d5, d6, d4), -2),
    )

    def test_take(self, dshape, indices_shape, oshape, axis):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        indices = relay.var("indices", relay.TensorType(indices_shape, "int32"))
        y = relay.take(x, indices, axis=axis)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(oshape, "float32")


class TestTake:
    src_shape, indices_src, axis, mode, indices_dtype = tvm.testing.parameters(
        ((4,), [1], None, "clip", "int32"),
        ((4,), [[0, 1, 2, 3]], None, "clip", "int32"),
        ((3, 3, 3), [[11, 25]], None, "clip", "int32"),
        ((4,), [[0, 1], [2, 3]], None, "clip", "int32"),
        ((4,), [1], 0, "clip", "int32"),
        ((2, 2), [[[1, 0], [0, 1]]], 0, "clip", "int32"),
        ((2, 2), [[[1, 0], [0, 1]]], 1, "clip", "int32"),
        ((4, 3, 5, 6), [[2, 1, 0, 0]], -2, "clip", "int32"),
        ((3, 4), [-5, 20], None, "clip", "int32"),
        ((3, 4), [-5, 20], None, "wrap", "int32"),
        ((3, 4), [-1, 2], 0, "clip", "int32"),
        ((3, 4), [-1, 2], 0, "wrap", "int32"),
        ((3, 4), [-1, 2], 1, "clip", "int32"),
        ((3, 4), [-1, 2], 1, "wrap", "int32"),
        ((3, 3, 3), [[11, 25]], None, "fast", "int32"),
        ((3, 4), [0, 2], 0, "fast", "int32"),
        ((3, 4), [0, 2], 1, "fast", "int32"),
        ((3, 4), [1, 2], 1, "clip", "uint32"),
        ((3, 4), [1, 2], 1, "wrap", "uint16"),
        ((3, 3, 3), [1, 2], None, "fast", "uint16"),
        ((3, 4), [0, 2], 0, "fast", "uint8"),
    )

    # Incorrect numeric output in some cases on vulkan
    @tvm.testing.known_failing_targets("vulkan")
    def test_take(
        self, target, dev, executor_kind, src_shape, indices_src, axis, mode, indices_dtype
    ):
        src_dtype = "float32"
        indices_src = np.array(indices_src, dtype=indices_dtype)
        x = relay.var("x", relay.TensorType(src_shape, src_dtype))
        indices = relay.var("indices", relay.TensorType(indices_src.shape, indices_dtype))
        z = relay.take(x, indices, axis=axis, mode=mode)

        func = relay.Function([x, indices], z)
        x_data = np.random.uniform(low=-1, high=1, size=src_shape).astype(src_dtype)
        np_mode = "raise" if mode == "fast" else mode

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, indices_src
        )

        # Old versions of numpy has take internally cast inside take which may violate
        # safety rules. We have such version in i386 CI image.
        indices_src = indices_src.astype("int32")
        ref_res = np.take(x_data, indices=indices_src, axis=axis, mode=np_mode)

        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestSplitInferType:
    idxd = tvm.tir.indexdiv

    d1, d2, d3, d4 = te.var("d1"), te.var("d2"), te.var("d3"), te.var("d4")
    axis = te.var("axis")

    dshape, indices_or_sections, ret_type, axis = tvm.testing.parameters(
        (
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
            1,
        ),
        (
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
            0,
        ),
        (
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
            2,
        ),
        (
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
            0,
        ),
        (
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
            1,
        ),
        (
            (d1, d2, d3, d4),
            tuple(np.array([2, 4, 7]).astype(np.int64)),
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
            1,
        ),
    )

    def test_split(self, dshape, indices_or_sections, ret_type, axis):
        x = relay.var("x", relay.ty.TensorType(dshape, "float32"))
        y = relay.split(x, indices_or_sections, axis=axis)
        yy = run_infer_type(y.astuple())
        assert yy.checked_type == ret_type


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


class TestFull:
    fill_value, arr_shape, dtype = tvm.testing.parameters(
        (4, (1, 3, 4, 4), "int32"),
        (4, (1, 3, 4, 4), "int64"),
        (4.0, (1, 4), "float32"),
    )

    def test_full(self, target, dev, executor_kind, fill_value, arr_shape, dtype):
        x = relay.var("x", relay.scalar_type(dtype))
        z = relay.full(x, arr_shape, dtype)
        func = relay.Function([x], z)
        ref_res = np.full(arr_shape, fill_value, dtype=dtype)
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            np.array(fill_value, dtype)
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    def test_full_like(self, target, dev, executor_kind, arr_shape, fill_value, dtype):
        x_data = np.random.uniform(low=-1, high=1, size=arr_shape).astype(dtype)
        x = relay.var("x", relay.TensorType(arr_shape, dtype))
        y = relay.var("y", relay.scalar_type(dtype))
        z = relay.full_like(x, y)

        func = relay.Function([x, y], z)
        ref_res = np.full_like(x_data, fill_value)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, np.array(fill_value, dtype)
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


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


def test_infer_type_leaky_relu(target, dev, executor_kind):
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

    op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(x_data)
    tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestInferTypePrelu:
    dtype = tvm.testing.parameter("float32")

    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    data, alpha, axis, output = tvm.testing.parameters(
        ((n, c, h, w), (c,), 1, (n, c, h, w)),
        ((n, h, w, c), (c,), 3, (n, h, w, c)),
        ((n, c, h, w), None, 1, (n, c, h, w)),
        ((n, h, w, c), None, 3, (n, h, w, c)),
        ((1, 3, 2, 2), (3,), 1, (1, 3, 2, 2)),
        ((1, 2, 2, 3), (3,), 3, (1, 2, 2, 3)),
        ((1, 3, 2, 2), None, 1, (1, 3, 2, 2)),
        ((1, 2, 2, 3), None, 3, (1, 2, 2, 3)),
    )

    def test_infer_type_prelu(self, target, dev, executor_kind, data, alpha, axis, output, dtype):
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

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, a_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestArange:
    dtype = tvm.testing.parameter("float32")

    start, stop, step = tvm.testing.parameters(
        (None, 20, None),
        (None, 20, 2),
        (1, 20, None),
        (1, 20, 2),
        # arange doesnt' support floating point right now, see type relation
        # (1, 20, 1.5),
        (1, 20.5, None),
        (1, 20, 3),
        (20, 1, -1),
        # arange doesnt' support floating point right now, see type relation
        # (20, 1, -1.5),
    )

    def test_arange(self, target, dev, executor_kind, start, stop, step, dtype):
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
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)()
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestMeshgrid:
    lengths, indexing = tvm.testing.parameters(
        ([3, 5], "ij"),
        ([4, 2], "xy"),
        ([3, 5, 2], "ij"),
        ([3, 1, 5], "xy"),
        # Length 0 signifies scalar.
        ([3, 5, 0], "ij"),
    )

    def test_meshgrid(self, target, dev, executor_kind, lengths, indexing="ij"):
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

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            *input_data
        )
        assert len(op_res) == len(ref_res)
        for i in range(len(op_res)):
            tvm.testing.assert_allclose(op_res[i].numpy(), ref_res[i], rtol=1e-5)


class TestTile:
    dshape, reps = tvm.testing.parameters(
        ((2, 3, 4), (3, 2, 1)),
        ((2, 3, 4), (1, 2)),
        ((2, 3), (3, 2, 1)),
    )

    def test_tile(self, target, dev, executor_kind, dshape, reps):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.tile(x, reps=reps)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.tile(x_data, reps=reps)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestRepeat:
    dshape, repeats, axis = tvm.testing.parameters(
        ((3,), 2, 0),
        ((3, 10), 2, -1),
        ((3, 2, 4), 3, 1),
    )

    def test_repeat(self, target, dev, executor_kind, dshape, repeats, axis):
        x = relay.Var("x", relay.TensorType(dshape, "float32"))
        func = relay.Function([x], relay.repeat(x, repeats, axis))
        data = np.random.uniform(size=dshape).astype("float32")
        ref_res = np.repeat(data, repeats, axis)
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestStack:
    dshapes, axis = tvm.testing.parameters(
        ([(2,), (2,), (2,)], -1),
        ([(2,), (2,), (2,)], 0),
        ([(2, 2, 4), (2, 2, 4), (2, 2, 4)], 1),
        ([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], -1),
        ([(2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4), (2, 2, 3, 4)], 4),
    )

    expr_type = tvm.testing.parameter("tuple", "list", "tuple_expr")

    @tvm.testing.fixture
    def ref_data(self, dshapes, axis):
        np_in = [np.random.normal(size=shape).astype("float32") for shape in dshapes]
        np_out = np.stack(np_in, axis=axis)
        return np_in, np_out

    @tvm.testing.fixture
    def input_expr(self, dshapes, axis, expr_type, ref_data):
        input_vars = [relay.var("input", relay.TensorType(shape, "float32")) for shape in dshapes]

        if expr_type == "tuple":
            input_expr = relay.Tuple(input_vars)

        elif expr_type == "list":
            input_expr = input_vars

        elif expr_type == "tuple_expr":
            # expression that evaluates to a tuple
            # but is not a tuple literal
            np_in, np_out = ref_data
            x = relay.Var("x")
            input_expr = relay.Let(x, relay.Tuple([relay.const(inp) for inp in np_in]), x)

        else:
            raise ValueError(f"Unknown expr_type '{expr_type}'")

        return input_expr

    def test_stack(self, target, dev, executor_kind, input_expr, ref_data, axis):
        z = relay.stack(input_expr, axis=axis)
        inp_vars = relay.analysis.free_vars(z)
        func = relay.Function(inp_vars, z)

        np_in, np_out = ref_data
        relay_args = np_in if inp_vars else []

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            *relay_args
        )
        tvm.testing.assert_allclose(op_res.numpy(), np_out, rtol=1e-5)


class TestReverse:
    dshape, axis = tvm.testing.parameters(
        ((2, 3, 4), 1),
        ((4, 7), 0),
        ((2, 3, 4), -1),
    )

    def test_reverse(self, target, dev, executor_kind, dshape, axis):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.reverse(x, axis=axis)
        zz = run_infer_type(z)

        func = relay.Function([x], z)
        x_data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        ref_res = np.flip(x_data, axis)
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


def test_reverse_sequence(target, dev, executor_kind):
    def verify_reverse_sequence(x_data, seq_lengths, batch_axis, seq_axis, ref_res):
        seq_lengths_data = np.array(seq_lengths).astype("int32")
        x = relay.var("x", relay.TensorType(x_data.shape, str(x_data.dtype)))
        z = relay.reverse_sequence(x, relay.const(seq_lengths_data), seq_axis, batch_axis)
        zz = run_infer_type(z)
        assert zz.checked_type == x.type_annotation
        func = relay.Function([x], z)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
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


def ref_scatter(data, indices, updates, axis=0):
    idx = np.indices(indices.shape).reshape(indices.ndim, -1)

    updated_idx = np.copy(idx)
    indices = indices.reshape(-1)
    for i in range(len(indices)):
        updated_idx[axis, i] = indices[i]
    scattered = np.copy(data)
    scattered[tuple(updated_idx)] = updates[tuple(idx)]
    return scattered


def test_scatter(target, dev, executor_kind):
    def verify_scatter(dshape, ishape, axis=0, indices_dtype="int64"):
        d = relay.var("d", relay.TensorType(dshape, "float32"))
        i = relay.var("i", relay.TensorType(ishape, indices_dtype))
        u = relay.var("u", relay.TensorType(ishape, "float32"))
        z = relay.op.scatter(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(0, dshape[axis] - 1, ishape).astype(indices_dtype)

        ref_res = ref_scatter(data_np, indices_np, updates_np, axis)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            data_np, indices_np, updates_np
        )
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
    verify_scatter((16, 16, 4, 5), (16, 16, 4, 5), 3, indices_dtype="uint32")


class TestDynamicScatter:
    dshape, ishape, axis = tvm.testing.parameters(
        ((10,), (10,), 0),
        ((10, 5), (10, 5), -2),
        ((10, 5), (10, 5), -1),
        ((10, 5), (3, 5), 0),
        ((12, 4), (7, 2), 1),
        ((2, 3, 4), (1, 3, 4), 0),
        ((2, 3, 4), (2, 1, 4), 1),
        ((2, 3, 4), (2, 3, 1), 2),
        ((4, 2, 1), (1, 1, 1), 0),
        ((2, 3, 4, 5), (1, 3, 4, 5), 0),
        ((6, 3, 4, 5), (2, 3, 4, 5), 1),
        ((2, 3, 8, 5), (2, 3, 1, 1), 2),
        ((16, 16, 4, 5), (16, 16, 4, 5), 3),
    )

    @pytest.mark.parametrize("executor_kind", ["vm"])
    def test_dynamic_scatter(self, target, dev, executor_kind, dshape, ishape, axis):
        d = relay.var("d", relay.TensorType([relay.Any() for i in range(len(dshape))], "float32"))
        i = relay.var("i", relay.TensorType([relay.Any() for i in range(len(ishape))], "int64"))
        u = relay.var("u", relay.TensorType([relay.Any() for i in range(len(ishape))], "float32"))
        z = relay.op.scatter(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np = np.random.uniform(size=dshape).astype("float32")
        updates_np = np.random.uniform(size=ishape).astype("float32")
        indices_np = np.random.randint(-dshape[axis], dshape[axis] - 1, ishape).astype("int64")

        ref_res = ref_scatter(data_np, indices_np, updates_np, axis)

        mod = tvm.ir.IRModule.from_expr(func)
        op_res = relay.create_executor(
            executor_kind, mod=mod, device=dev, target=target
        ).evaluate()(data_np, indices_np, updates_np)
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)


class TestScatterAdd:
    dshape, ishape, axis, dtype, indice_dtype = tvm.testing.parameters(
        ((10,), (10,), 0, "int32", "int64"),
        ((1000,), (1000,), 0, "int32", "int64"),
        ((10, 5), (10, 5), -2, "float32", "int64"),
        ((10, 5), (10, 5), -1, "float32", "int64"),
        ((10, 5), (3, 5), 0, "float32", "int64"),
        ((12, 4), (7, 2), 1, "float32", "int64"),
        ((2, 3, 4), (1, 3, 4), 0, "float32", "int64"),
        ((2, 3, 4), (2, 1, 4), 1, "float32", "int64"),
        ((2, 3, 4), (2, 3, 1), 2, "float32", "int64"),
        ((2, 3, 4, 5), (1, 3, 4, 5), 0, "float32", "int64"),
        ((6, 3, 4, 5), (2, 3, 4, 5), 1, "float32", "int64"),
        ((2, 3, 8, 5), (2, 3, 1, 1), 2, "float32", "int64"),
        ((16, 16, 4, 5), (16, 16, 4, 5), 3, "float32", "int64"),
        ((16, 16, 4, 5), (16, 16, 4, 5), 3, "float32", "uint32"),
    )

    @tvm.testing.fixture(cache_return_value=True)
    def ref_data(self, dshape, ishape, axis, dtype, indice_dtype):
        data_np = np.random.uniform(size=dshape).astype(dtype)
        updates_np = np.random.uniform(size=ishape).astype(dtype)
        indices_np = np.random.randint(0, dshape[axis] - 1, ishape).astype(indice_dtype)

        out_np = np.copy(data_np)
        for index in np.ndindex(*indices_np.shape):
            new_index = list(index)
            new_index[axis] = indices_np[index]
            out_np[tuple(new_index)] += updates_np[index]
        return data_np, updates_np, indices_np, out_np

    # Optimization can produce tir.atomic_add, not currently supported
    # on vulkan runtime.
    @tvm.testing.known_failing_targets("vulkan")
    def test_scatter_add(self, target, dev, ref_data, dshape, ishape, axis, dtype, indice_dtype):
        d = relay.var("d", relay.TensorType(shape=[relay.Any() for _ in dshape], dtype=dtype))
        i = relay.var(
            "i", relay.TensorType(shape=[relay.Any() for _ in ishape], dtype=indice_dtype)
        )
        u = relay.var("u", relay.TensorType(shape=[relay.Any() for _ in ishape], dtype=dtype))
        z = relay.op.scatter_add(d, i, u, axis)

        func = relay.Function([d, i, u], z)

        data_np, updates_np, indices_np, out_np = ref_data

        verify_func(target, dev, func, [data_np, indices_np, updates_np], out_np)


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
def test_gather(target, dev, executor_kind, data, axis, indices, ref_res):
    def verify_gather(data, axis, indices, ref_res):
        data = np.asarray(data, dtype="float32")
        indices = np.asarray(indices, dtype="int32")
        ref_res = np.asarray(ref_res)
        d = relay.var("x", relay.TensorType(data.shape, "float32"))
        i = relay.var("y", relay.TensorType(indices.shape, "int32"))
        z = relay.gather(d, axis, i)

        func = relay.Function([d, i], z)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            data, indices
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_gather(data, axis, indices, ref_res)


def test_gather_nd(target, dev, executor_kind):
    def verify_gather_nd(xshape, yshape, y_data, batch_dims=0, indices_dtype="int32"):
        x = relay.var("x", relay.TensorType(xshape, "float32"))
        y = relay.var("y", relay.TensorType(yshape, indices_dtype))
        z = relay.gather_nd(x, y, batch_dims)

        func = relay.Function([x, y], z)

        x_data = np.random.uniform(size=xshape).astype("float32")

        if y_data:
            y_data = np.array(y_data, dtype=indices_dtype)
        else:
            y_data = np.random.randint(low=0, high=2, size=yshape, dtype=indices_dtype)

        ref_res = ref_funcs.gather_nd(x_data, y_data, batch_dims)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, y_data
        )
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

    verify_gather_nd((3, 2, 2, 3, 4), (1, 3, 2, 3), None, 2, indices_dtype="uint8")
    verify_gather_nd((2, 2, 2), (2, 2, 1), [[[1], [0]], [[0], [1]]], 1, indices_dtype="uint32")


def _verify_infiniteness_ops(relay_op, ref_op, target="llvm", dev=None):
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

        op_res = create_executor(target=target, device=dev).evaluate(y, {x: data})
        ref_res = ref_op(data)
        np.testing.assert_allclose(op_res.numpy(), ref_res, rtol=0.01)


@tvm.testing.requires_gpu
def test_isfinite():
    for target, dev in tvm.testing.enabled_targets():
        if target not in ["llvm", "cuda"]:
            continue
        _verify_infiniteness_ops(relay.isfinite, np.isfinite, target=target, dev=dev)


@tvm.testing.requires_gpu
def test_isinf():
    for target, dev in tvm.testing.enabled_targets():
        if target not in ["llvm", "cuda"]:
            continue
        _verify_infiniteness_ops(relay.isinf, np.isinf, target=target, dev=dev)


def test_unravel_index(target, dev, executor_kind):
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
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data, y_data
        )
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


def test_sparse_to_dense(target, dev, executor_kind):
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
        f = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)
        if default_value is None:
            op_res = f(sparse_indices_data, sparse_values_data)
        else:
            op_res = f(sparse_indices_data, sparse_values_data, default_value_data)
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
    # default value not specified
    verify_sparse_to_dense(1, 3, None, [5], [0, 3, 0, 0, 0])

    # negative test cases
    # sparse indices should be ints
    # verify_sparse_to_dense([[0.1, 1.1, 4.1], [0,2,4]], [3.1, 3.1, 3.1], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])
    # sparse_values should be 0d or 1d only
    # verify_sparse_to_dense([[0, 1, 4], [0, 2, 4]], [[[3.1, 3.1, 3.1]]], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])
    # sparse_indices should not be > 2d tensor
    # verify_sparse_to_dense([[[[0, 1, 4], [0, 2, 4]]]], [[[[3.1, 3.1, 3.1]]]], 3.5, [5], [3.1, 3.1, 3.5, 3.5, 3.1])


class TestSparseReshape:

    sparse_indices_np, sparse_values_np, prev_shape_np, new_shape_np = tvm.testing.parameters(
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
    )

    use_dyn = tvm.testing.parameter(True, False, ids=["dyn", "static"])

    @tvm.testing.fixture(cache_return_value=True)
    def ref_res(
        self,
        sparse_indices_np: np.ndarray,
        prev_shape_np: np.ndarray,
        new_shape_np: np.ndarray,
    ):
        """
        This function calculates the expected output of sparseshape operator given the inputs.
        """

        new_sparse_indices = np.ones(
            (sparse_indices_np.shape[0], new_shape_np.shape[0]), dtype=sparse_indices_np.dtype
        )
        multipliers = np.ones(prev_shape_np.shape[0])
        dividers = np.ones(new_shape_np.shape[0])
        total_ele = np.prod(prev_shape_np)
        division_total_ele = 1
        for i in range(new_shape_np.shape[0]):
            if new_shape_np[i] == -1:
                continue
            division_total_ele *= new_shape_np[i]
        for i in range(prev_shape_np.shape[0] - 2, -1, -1):
            multipliers[i] = prev_shape_np[i + 1] * multipliers[i + 1]

        for i in range(len(new_shape_np)):
            if new_shape_np[i] == -1:
                new_shape_np[i] = total_ele // division_total_ele

        if np.array_equal(prev_shape_np, new_shape_np):
            return sparse_indices_np, prev_shape_np

        for i in range(new_shape_np.shape[0] - 2, -1, -1):
            dividers[i] = new_shape_np[i + 1] * dividers[i + 1]

        for row_num, sparse_row in enumerate(sparse_indices_np):
            flat_idx = 0
            if len(sparse_indices_np.shape) != 1:
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

        return new_sparse_indices, new_shape_np

    @tvm.testing.known_failing_targets("vulkan")
    def test_sparse_reshape(
        self,
        target,
        dev,
        ref_res,
        sparse_indices_np,
        sparse_values_np,
        prev_shape_np,
        new_shape_np,
        use_dyn,
    ):
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

        outputs = run_infer_type(z)
        new_sparse_indices_infer_type, new_shape_infer_type = (
            outputs.checked_type.fields[0].dtype,
            outputs.checked_type.fields[1].dtype,
        )

        assert new_sparse_indices_infer_type == sparse_indices_np.dtype
        assert new_shape_infer_type == new_shape_np.dtype
        verify_func(
            target,
            dev,
            func,
            [sparse_indices_np, prev_shape_np, new_shape_np],
            ref_res,
        )


class TestSegmentSum:
    data_np, segment_ids_np, num_segments = tvm.testing.parameters(
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
    )

    use_dyn = tvm.testing.parameter(True, False, ids=["dyn", "static"])

    @tvm.testing.fixture(cache_return_value=True)
    def ref_res(
        self,
        data_np: np.ndarray,
        segment_ids_np: np.ndarray,
        num_segments: Optional[int],
    ):
        """
        This function calculates the expected output of segment_sum operator given the inputs.
        """
        if not num_segments:
            num_segments = np.unique(segment_ids_np).shape[0]

        result = np.zeros((num_segments,) + data_np.shape[1:], data_np.dtype)
        for i, index in enumerate(segment_ids_np):
            result[index] += data_np[i]
        return result

    # Optimization can produce tir.atomic_add, not currently supported
    # on vulkan runtime.
    @tvm.testing.known_failing_targets("vulkan")
    def test_segment_sum(
        self,
        target,
        dev,
        ref_res: np.ndarray,
        data_np: np.ndarray,
        segment_ids_np: np.ndarray,
        num_segments: Optional[int],
        use_dyn: bool,
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
        segment_sum_result = run_infer_type(z)
        assert segment_sum_result.checked_type.dtype == data_np.dtype
        verify_func(
            target,
            dev,
            func,
            [data_np, segment_ids_np],
            ref_res,
        )


def verify_func(target, dev, func, data, ref_res, rtol=1e-5, atol=1e-7, kinds=["vm"]):
    assert isinstance(data, list)
    for kind in kinds:
        mod = tvm.ir.IRModule.from_expr(func)
        op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(*data)
        if isinstance(op_res, tvm.runtime.container.ADT):
            assert len(op_res) == len(
                ref_res
            ), "Outputs from TVM and Python implementation must be equal "
            for op_result, ref_result in zip(op_res, ref_res):
                tvm.testing.assert_allclose(op_result.numpy(), ref_result, rtol=rtol, atol=atol)
        else:
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)
        relay.backend.te_compiler.get().clear()


def test_adv_index(target, dev, executor_kind):
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
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            *np_args
        )
        tvm.testing.assert_allclose(op_res.numpy(), np_out, rtol=1e-5)

    verify_adv_index((10, 5), [(3, 4), (3, 1)])
    verify_adv_index((10, 5), [(1, 4), (3, 1)])
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
    target,
    dev,
    executor_kind,
    binop_type: str,
    gt_func: Callable[..., np.array],
    identity_value: int,
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

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            data_np
        )
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
def test_cumsum(target, dev, executor_kind):
    run_binop_tests(
        target, dev, executor_kind, binop_type="cumsum", gt_func=np.cumsum, identity_value=0
    )


@tvm.testing.parametrize_targets
def test_cumprod(target, dev, executor_kind):
    run_binop_tests(
        target, dev, executor_kind, binop_type="cumprod", gt_func=np.cumprod, identity_value=1
    )


@tvm.testing.parametrize_targets
def test_scatter_nd(target, dev, executor_kind):
    def test_scatter_nd_large_shape():
        def before():
            data = relay.const(np.zeros((1, 900, 300), dtype="float32"), dtype="float32")
            indices = relay.const(np.ones((3, 1, 900, 300), dtype="int64"), dtype="int64")
            update = relay.const(np.ones((1, 900, 300), dtype="float32"), dtype="float32")
            b = relay.op.scatter_nd(data, indices, update)
            return relay.Function(relay.analysis.free_vars(b), b)

        passes = tvm.transform.Sequential(
            [
                relay.transform.InferType(),
                relay.transform.FoldConstant(),
            ]
        )
        before_mod = tvm.IRModule.from_expr(before())
        with tvm.transform.PassContext(opt_level=3):
            after_mod = passes(before_mod)

    test_scatter_nd_large_shape()

    def verify_scatter_nd(
        data_np, indices_np, updates_np, ref_res, mode="add", rtol=1e-5, atol=1e-5
    ):
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices = relay.var("indices", shape=indices_np.shape, dtype=str(indices_np.dtype))
        updates = relay.var("updates", shape=updates_np.shape, dtype=str(updates_np.dtype))

        out = relay.op.scatter_nd(data, indices, updates, mode)
        func = relay.Function([data, indices, updates], out)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            data_np, indices_np, updates_np
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)

    def verify_scatter_nd_with_stack(
        data_np, indices_np, updates_np, ref_res, mode="add", rtol=1e-5, atol=1e-5
    ):
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices_vars = [
            relay.var("ind%d" % i, shape=v.shape, dtype=str(v.dtype))
            for i, v in enumerate(indices_np)
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
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            *fargs
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)

    for indice_dtype in ["uint8", "uint16", "uint32"]:
        data = np.zeros((2, 2)).astype("int64")
        indices = np.array([[1, 1, 0], [0, 1, 0]]).astype(indice_dtype)
        updates = np.array([2, 3, 0])
        out = np.array([[0, 0], [2, 3]])
        verify_scatter_nd(data, indices, updates, out)
        verify_scatter_nd_with_stack(data, indices, updates, out)

        data = np.zeros((2, 2, 2, 2)).astype("int64")
        indices = np.array([[0, 1], [1, 1]]).astype(indice_dtype)
        updates = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        out = np.array([[[[0, 0], [0, 0]], [[1, 2], [3, 4]]], [[[0, 0], [0, 0]], [[5, 6], [7, 8]]]])
        verify_scatter_nd(data, indices, updates, out)
        verify_scatter_nd_with_stack(data, indices, updates, out)

        indices = np.array([[1, 0, 0]]).astype(indice_dtype)
        updates = np.reshape(np.arange(1560 * 3), (3, 1560)).astype("float32")
        shape = (2, 1560)
        data = np.zeros(shape).astype("float32")
        out = data.copy()
        out[1, :] += updates[0, :]
        out[0, :] += updates[1, :]
        out[0, :] += updates[2, :]
        verify_scatter_nd(data, indices, updates, out, mode="add")
        verify_scatter_nd_with_stack(data, indices, updates, out)

        for mode in ["add", "update"]:
            indices = np.stack((np.random.randint(2, size=5), np.random.randint(7, size=5))).astype(
                indice_dtype
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


def test_unique(target, dev):
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
            # In unsorted case, need to sort the index of first occurence
            index = np.sort(index)
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
            backend = "vm"
        else:
            backend = "graph"

        mod = tvm.ir.IRModule.from_expr(func)
        tvm_res = relay.create_executor(backend, mod=mod, device=dev, target=target).evaluate()(
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
            tvm.testing.assert_allclose(tvm_res[4].numpy()[:num_unique], np_res[4], rtol=1e-5)

    for dtype in ["int32", "int64"]:
        for i in range(8):
            is_dyn, is_sorted, return_counts = bool(i & 1), bool(i & 2), bool(i & 4)
            verify_unique(10, dtype, is_dyn, is_sorted, return_counts)


class TestSTFT:
    (
        data_np,
        n_fft,
        hop_length,
        win_length,
        window_np,
        normalized,
        onesided,
    ) = tvm.testing.parameters(
        (
            np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32),
            3,
            3,
            3,
            np.array([4, 3, 2], dtype=np.int32),
            False,
            True,
        ),
        (
            np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 5, 7, 8, 5, 6, 7, 3, 2]], dtype=np.float32),
            2,
            1,
            2,
            np.array([1, 3], dtype=np.int32),
            False,
            True,
        ),
        (
            np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 5, 7, 8, 5, 6, 7, 3, 2]], dtype=np.float32),
            2,
            1,
            2,
            np.array([1, 3], dtype=np.int32),
            True,
            True,
        ),
        (
            np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 5, 7, 8, 5, 6, 7, 3, 2]], dtype=np.float32),
            2,
            1,
            2,
            np.array([1, 3], dtype=np.int32),
            False,
            False,
        ),
    )

    @tvm.testing.fixture(cache_return_value=True)
    def ref_res(
        self,
        data_np: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_np,
        normalized,
        onesided,
    ):
        """
        This function calculates the expected output of segment_sum operator given the inputs.
        """

        def pad_window(window_np, n_fft):
            shape = window_np.shape[-1]
            lpad = int((n_fft - shape) // 2)
            lengths = [(0, 0)] * len(window_np.shape)
            lengths[-1] = (lpad, int(n_fft - shape - lpad))
            if lpad < 0:
                print("ERROR Padding")
            return np.pad(window_np, lengths, mode="constant")

        import math

        if not onesided:
            n_rows = n_fft
        else:
            n_rows = n_fft // 2 + 1
        if window_np is None:
            window_np = np.ones(win_length, dtype=np.int32)
        window_np = pad_window(window_np, n_fft)

        n_cols = (data_np.shape[-1] - n_fft) // hop_length + 1
        np_result = np.zeros((data_np.shape[0], n_rows, n_cols, 2))

        for batch in range(data_np.shape[0]):
            for w in range(n_rows):
                for m in range(n_cols):
                    for k in range(n_fft):
                        np_result[batch][w][m][0] += (
                            window_np[k]
                            * data_np[batch][m * hop_length + k]
                            * math.cos(2 * math.pi * w * k / n_fft)
                        )
                        np_result[batch][w][m][1] -= (
                            window_np[k]
                            * data_np[batch][m * hop_length + k]
                            * math.sin(2 * math.pi * w * k / n_fft)
                        )
                    if normalized:
                        np_result[batch][w][m][0] /= math.sqrt(n_fft)
                        np_result[batch][w][m][1] /= math.sqrt(n_fft)
        return np_result

    use_dyn = tvm.testing.parameter(True, False, ids=["dyn", "static"])

    @tvm.testing.parametrize_targets("llvm", "cuda")
    def test_stft(
        self,
        target,
        dev,
        ref_res: np.ndarray,
        data_np: np.ndarray,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_np: np.ndarray,
        normalized: bool,
        onesided: bool,
        use_dyn,
    ):
        if use_dyn:
            data = relay.var(
                "data",
                relay.TensorType([relay.Any(), relay.Any()], str(data_np.dtype)),
            )
            window = relay.var(
                "window",
                relay.TensorType([relay.Any()], str(window_np.dtype)),
            )
            backends = ["vm"]
        else:
            data = relay.var(
                "data",
                relay.TensorType(data_np.shape, str(data_np.dtype)),
            )
            window = relay.var(
                "window",
                relay.TensorType(window_np.shape, str(window_np.dtype)),
            )
            backends = ["graph", "vm"]

        z = relay.op.stft(data, n_fft, hop_length, win_length, window, normalized, onesided)
        func = relay.Function([data, window], z)
        verify_func(
            target, dev, func, [data_np, window_np], ref_res, rtol=1e-3, atol=1e-3, kinds=backends
        )


def test_trilu(target="llvm", dev=tvm.cpu()):
    def verify_trilu(data_shape, upper=True, k=0):
        data = relay.var("data", relay.TensorType(data_shape, "float32"))
        y = relay.trilu(data, k, upper)
        mod = tvm.ir.IRModule.from_expr(y)

        data_np = np.random.normal(size=data_shape).astype("float32")
        tvm_res = (
            relay.create_executor("graph", mod=mod, device=dev, target=target)
            .evaluate()(data_np)
            .numpy()
        )
        if upper:
            np_res = np.triu(data_np, k)
        else:
            np_res = np.tril(data_np, k)
        tvm.testing.assert_allclose(tvm_res, np_res)

    # Test upper and lower triangle
    verify_trilu((3, 3), True, 0)
    verify_trilu((3, 3), False, 0)
    # Test larger matrices with offset.
    verify_trilu((6, 6), True, 1)
    verify_trilu((6, 6), False, 2)
    verify_trilu((6, 6), False, -2)
    # Test batch size
    verify_trilu((8, 6, 6), False, -2)


def test_trilu_shape_i64():
    data_x = np.ones((2, 1), dtype="int32")

    x = relay.var("x", shape=[2, 1], dtype="float32")
    v0 = relay.broadcast_to(x, shape=relay.const([2, 1], dtype="int64"))
    v2 = relay.add(relay.const([[1.0]]), v0)
    v3 = relay.trilu(v0, k=0)

    f = relay.Function([x], relay.Tuple([v2, v3]))
    tvm_res = relay.create_executor("graph", device=tvm.cpu(), target="llvm").evaluate(f)(data_x)

    np_res = (
        np.array([[2.0], [2.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
    )

    tvm.testing.assert_allclose(tvm_res[0].numpy(), np_res[0])
    tvm.testing.assert_allclose(tvm_res[1].numpy(), np_res[1])


def test_trilu_reduce():
    data_i0 = np.ones((2, 2), dtype="int32")
    k = 0

    i0 = relay.var("i0", shape=[2, 2], dtype="int32")
    i1 = relay.var("i1", shape=(), dtype="int64")
    v0 = relay.trilu(i0, i1)
    v1 = relay.argmin(v0, axis=[0])
    f = relay.Function([i0, i1], v1)
    tvm_res = (
        relay.create_executor("graph", device=tvm.cpu(), target="llvm")
        .evaluate(f)(data_i0, k)
        .numpy()
    )

    np_res = np.triu(data_i0, k).argmin(axis=0)
    tvm.testing.assert_allclose(tvm_res, np_res)


if __name__ == "__main__":
    tvm.testing.main()
