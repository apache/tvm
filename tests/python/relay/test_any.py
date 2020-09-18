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
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay.loops import while_loop
from tvm.relay.testing import run_infer_type as infer_type
import tvm.topi.testing


def int32(val):
    return relay.const(val, "int32")


def any_dims(ndim):
    shape = []
    for _ in range(ndim):
        shape.append(relay.Any())
    return tuple(shape)


def check_result(
    args, mod, expected, flatten=False, assert_shape=False, only_vm=False, targets=None
):
    for kind in ["debug", "vm"]:
        targets = targets or tvm.testing.enabled_targets()
        for tgt, ctx in targets:
            if kind == "debug" and (only_vm or ctx.device_type != tvm.cpu().device_type):
                continue
            ex = relay.create_executor(kind, mod=mod, ctx=ctx, target=tgt)
            result = ex.evaluate()(*args)
            result = result.asnumpy()
            if assert_shape:
                assert result.shape == expected, "Shape mismatch: expect %s but got %s." % (
                    str(expected),
                    str(result.shape),
                )
                return

            if flatten:
                result = result.flatten()
                expected = expected.flatten()
            tvm.testing.assert_allclose(result, expected)


def verify_any_broadcast(x_shape, y_shape, x_np_shape, y_np_shape, op, np_op):
    dtype = "float32"
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.var("y", shape=y_shape, dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], op(x, y))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    y_np = np.random.uniform(size=y_np_shape).astype(dtype)
    res_np = np_op(x_np, y_np)
    check_result([x_np, y_np], mod, res_np)


@tvm.testing.uses_gpu
def test_any_broadcast():
    # Test broadcast with 1s
    verify_any_broadcast((relay.Any(),), (3, 2), (1,), (3, 2), relay.add, np.add)
    verify_any_broadcast((relay.Any(), 2), (1, 2), (1, 2), (1, 2), relay.add, np.add)
    verify_any_broadcast((relay.Any(), 2), (1, 2), (3, 2), (1, 2), relay.add, np.add)
    verify_any_broadcast((relay.Any(), 2), (3, 2), (1, 2), (3, 2), relay.add, np.add)
    verify_any_broadcast((relay.Any(), 2), (3, relay.Any()), (1, 2), (3, 1), relay.add, np.add)

    # Test broadcast with values other than 1
    verify_any_broadcast((relay.Any(),), (3, 2), (2,), (3, 2), relay.add, np.add)
    verify_any_broadcast((relay.Any(), 2), (3, 2), (3, 2), (3, 2), relay.add, np.add)


def verify_any_elemwise(x_shape, x_np_shape, op, np_op):
    dtype = "float32"
    x = relay.var("x", shape=x_shape, dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], op(x))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    res_np = np_op(x_np)
    check_result([x_np], mod, res_np)


@tvm.testing.uses_gpu
def test_any_elemwise():
    verify_any_elemwise((relay.Any(),), (3,), relay.sqrt, np.sqrt)
    verify_any_elemwise((relay.Any(), 2), (5, 2), relay.negative, np.negative)
    verify_any_elemwise((relay.Any(), relay.Any()), (5, 4), relay.exp, np.exp)


@tvm.testing.uses_gpu
def test_any_broadcast_fail():
    # Test broadcast with incompatible values at runtime
    def check_fail(x_shape, y_shape, x_np_shape, y_np_shape, op, np_op):
        try:
            verify_any_broadcast(x_shape, y_shape, x_np_shape, y_np_shape, op, np_op)
        except tvm._ffi.base.TVMError:
            pass
        else:
            assert False

    check_fail((relay.Any(),), (3, 2), (1,), (4, 2), relay.add, np.add)
    check_fail((relay.Any(), 2), (3, 2), (4, 2), (4, 2), relay.add, np.add)
    check_fail((relay.Any(), 2), (3, relay.Any()), (1, 2), (4, 1), relay.add, np.add)
    check_fail((relay.Any(), 2), (3, 3), (1, 3), (3, 3), relay.add, np.add)
    check_fail((relay.Any(),), (3, 2), (2), (4, 2), relay.add, np.add)


def verify_any_full_like(x_shape, x_np_shape, relay_op, np_op, dtype="float32"):
    x = relay.var("x", shape=x_shape, dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], relay_op(x))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    res_np = np_op(x_np)
    check_result([x_np], mod, res_np)


@tvm.testing.uses_gpu
def test_any_full_like():
    # zeros_like, ones_like
    verify_any_full_like(any_dims(3), (2, 3, 5), relay.zeros_like, np.zeros_like, "float32")
    verify_any_full_like(any_dims(3), (225, 115, 15), relay.zeros_like, np.zeros_like, "float32")
    verify_any_full_like(
        any_dims(5), (10, 11, 12, 13, 14), relay.zeros_like, np.zeros_like, "int32"
    )
    verify_any_full_like(any_dims(3), (2, 3, 5), relay.ones_like, np.ones_like, "float32")
    verify_any_full_like(any_dims(3), (225, 115, 15), relay.ones_like, np.ones_like, "float32")
    verify_any_full_like(any_dims(5), (10, 11, 12, 13, 14), relay.ones_like, np.ones_like, "int32")


def verify_any_full(x_np_shape, relay_op, np_op, dtype="float32", value=None):
    x = relay.var("x", shape=(len(x_np_shape),), dtype="int32")
    mod = tvm.IRModule()
    out = relay_op(x, dtype) if value is None else relay_op(relay.expr.const(value), x, dtype)
    mod["main"] = relay.Function([x], out)
    res_np = np_op(x_np_shape) if value is None else np_op(x_np_shape, value)
    x_np = np.array(x_np_shape).astype("int32")
    check_result([x_np], mod, res_np)


@tvm.testing.uses_gpu
def test_any_full():
    # zeros, ones, full
    verify_any_full((2, 3, 5), relay.zeros, np.zeros, "float32")
    verify_any_full((225, 115, 15), relay.zeros, np.zeros, "float32")
    verify_any_full((10, 11, 12, 13, 14), relay.zeros, np.zeros, "int32")
    verify_any_full((2, 3, 5), relay.ones, np.ones, "float32")
    verify_any_full((225, 115, 15), relay.ones, np.ones, "float32")
    verify_any_full((10, 11, 12, 13, 14), relay.ones, np.ones, "int32")
    verify_any_full((10, 11, 12, 13, 14), relay.full, np.full, "float32", 2.0)
    verify_any_full((1, 2, 3, 4), relay.full, np.full, "int32", -2)


@tvm.testing.uses_gpu
def test_any_concat():
    x = relay.var("x", shape=(relay.Any(), 2), dtype="float32")
    y = relay.var("y", shape=(1, 2), dtype="float32")
    xx = x - relay.expr.const(3.0)
    yy = y * relay.expr.const(5.0)
    z = relay.op.concatenate([xx, yy], axis=0)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], z)
    x_np = np.random.uniform(size=(3, 2)).astype("float32")
    y_np = np.random.uniform(size=(1, 2)).astype("float32")
    ref = np.concatenate([x_np - 3.0, y_np * 5.0], axis=0)
    check_result([x_np, y_np], mod, ref)


def verify_any_reshape(x_shape, newshape, x_np_shape, out_shape, variable_newshape=False):
    x = relay.var("x", shape=x_shape, dtype="float32")
    relu_x = relay.nn.relu(x)
    data = np.random.uniform(size=x_np_shape).astype("float32")
    params = [x]
    args = [data]

    if variable_newshape:
        newshape_var = relay.var("newshape", shape=(len(newshape),), dtype="int64")
        params.append(newshape_var)
        args.append(np.array(newshape, dtype="int64"))
        newshape = newshape_var

    y = relay.reshape(relu_x, newshape=newshape)
    mod = tvm.IRModule()
    mod["main"] = relay.Function(params, y)
    check_result(args, mod, data, flatten=True)


@tvm.testing.uses_gpu
def test_any_reshape():
    for variable_newshape in [False, True]:
        # Variable newshape only supports that output rank is the same as newshape
        verify_any_reshape(any_dims(3), (1, -1), (2, 3, 4), (1, 24), variable_newshape)
        verify_any_reshape(any_dims(3), (0, -1), (2, 3, 4), (2, 12), variable_newshape)
    verify_any_reshape(any_dims(3), (0, -2), (2, 3, 4), (2, 3, 4))
    verify_any_reshape(any_dims(3), (-4, -1, 2, -3), (6, 3, 4), (3, 2, 12))
    verify_any_reshape(any_dims(3), (-4, 2, -1, -2), (6, 3, 4), (2, 3, 3, 4))


def verify_any_argwhere(x_shape, x_np_shape, dtype="bool"):
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.argwhere(x)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    data = np.random.choice([0, 1, 2, 3], size=x_np_shape).astype(dtype)
    expected = np.argwhere(data)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data).asnumpy()
        assert result.shape == expected.shape
        tvm.testing.assert_allclose(result.flatten(), expected.flatten())

    # TODO(@zhiics) argwhere gpu schedule is currently not avaiable
    # check_result([data], mod, expected, flatten=True)


@tvm.testing.uses_gpu
def test_any_argwhere():
    verify_any_argwhere(any_dims(1), (5,))
    verify_any_argwhere(any_dims(2), (5, 5))
    verify_any_argwhere(any_dims(3), (5, 5, 5))
    verify_any_argwhere(any_dims(4), (5, 5, 5, 5))
    verify_any_argwhere(any_dims(5), (5, 5, 5, 5, 5))
    verify_any_argwhere(any_dims(1), (5,), "int32")
    verify_any_argwhere(any_dims(2), (5, 5), "int32")
    verify_any_argwhere(any_dims(3), (5, 5, 5), "int32")
    verify_any_argwhere(any_dims(4), (5, 5, 5, 5), "int32")
    verify_any_argwhere(any_dims(5), (5, 5, 5, 5, 5), "int32")
    verify_any_argwhere(any_dims(1), (5,), "int8")
    verify_any_argwhere(any_dims(2), (5, 5), "int8")
    verify_any_argwhere(any_dims(3), (5, 5, 5), "int8")
    verify_any_argwhere(any_dims(4), (5, 5, 5, 5), "int8")
    verify_any_argwhere(any_dims(5), (5, 5, 5, 5, 5), "int8")


def verify_any_take(data_shape, indices_shape, axis, data_np_shape, indices_np_shape):
    mod = tvm.IRModule()
    data = relay.var("data", shape=data_shape, dtype="float32")
    indices = relay.var("indices", shape=indices_shape, dtype="int32")
    y = relay.take(data, indices, axis=axis)
    mod["main"] = relay.Function([data, indices], y)
    data_np = np.random.uniform(size=data_np_shape).astype("float32")
    if axis is None:
        max_index = data_np.size
    else:
        max_index = data_np.shape[axis]
    indices_np = np.random.randint(max_index, size=indices_np_shape).astype("int32")
    ref = np.take(data_np, indices_np, axis=axis)
    check_result([data_np, indices_np], mod, ref)


@tvm.testing.uses_gpu
def test_any_take():
    verify_any_take(any_dims(2), (1,), 0, (4, 5), (1,))
    verify_any_take(any_dims(2), (), 0, (4, 5), ())
    verify_any_take(any_dims(2), (), None, (4, 5), ())
    verify_any_take(any_dims(3), any_dims(2), 1, (3, 4, 5), (2, 3))
    verify_any_take(any_dims(2), any_dims(3), None, (4, 5), (2, 3, 4))
    verify_any_take(any_dims(2), any_dims(4), -1, (4, 5), (2, 3, 4, 5))


def verify_any_tile(dshape, reps, np_dshape, np_reps):
    mod = tvm.IRModule()
    x = relay.var("x", shape=dshape, dtype="float32")
    y = relay.tile(x, reps=reps)
    mod["main"] = relay.Function([x], y)
    x_data = np.random.uniform(size=np_dshape).astype("float32")
    ref_res = np.tile(x_data, reps=np_reps)
    check_result([x_data], mod, ref_res)


@tvm.testing.uses_gpu
def test_any_tile():
    verify_any_tile(any_dims(3), (3, 2, 1), (2, 3, 4), (3, 2, 1))
    verify_any_tile(any_dims(3), (1, 2), (2, 3, 4), (1, 2))
    verify_any_tile(any_dims(2), (3, 2, 1), (2, 3), (3, 2, 1))
    verify_any_tile(any_dims(3), (1,), (2, 3, 4), (1,))


@tvm.testing.uses_gpu
def test_any_shape_of():
    x = relay.var("x", shape=any_dims(2), dtype="float32")
    y = relay.shape_of(x)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    data = np.random.uniform(size=(3, 4)).astype("float32")
    check_result([data], mod, np.array([3, 4]).astype("int64"))

    x = relay.var("x", shape=any_dims(3), dtype="float32")
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(1, "int32"))
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y1)
    data = np.random.uniform(size=(2, 3, 4)).astype("float32")
    check_result([data], mod, np.array(3).astype("int64"))


def verify_any_reduce(
    reduce_op, data_shape, axis, exclude, keepdims, static_data_shape, ref_out_shape
):
    mod = tvm.IRModule()
    dtype = "bool" if reduce_op == relay.all else "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = reduce_op(data, axis, keepdims, exclude)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_reduce():
    verify_any_reduce(relay.argmax, any_dims(3), None, False, False, (3, 4, 5), ())
    verify_any_reduce(relay.argmin, any_dims(4), 1, False, True, (3, 4, 5, 6), (3, 1, 5, 6))
    verify_any_reduce(relay.all, any_dims(3), (1, 2), True, False, (3, 4, 5), (4, 5))
    verify_any_reduce(relay.max, any_dims(4), -1, True, True, (3, 4, 5, 6), (1, 1, 1, 6))
    verify_any_reduce(relay.min, any_dims(3), (0, 1), False, False, (4, 5, 6), (6,))
    verify_any_reduce(relay.prod, any_dims(4), 2, True, True, (3, 4, 5, 6), (1, 1, 5, 1))
    verify_any_reduce(relay.mean, any_dims(2), 0, False, False, (1, 2), (2,))
    verify_any_reduce(relay.variance, any_dims(5), (2, 4), False, False, (3, 4, 5, 6, 7), (3, 4, 6))


def verify_any_layout_transform(
    data_shape, src_layout, dst_layout, static_data_shape, ref_out_shape
):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.layout_transform(data, src_layout, dst_layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_layout_transform():
    verify_any_layout_transform(any_dims(4), "NCHW", "NHWC", (3, 4, 5, 6), (3, 5, 6, 4))
    verify_any_layout_transform(
        any_dims(5), "NCHW16c", "NCHW2c", (1, 2, 8, 8, 16), (1, 16, 8, 8, 2)
    )
    verify_any_layout_transform(any_dims(5), "NCHW6n", "NHWC", (3, 4, 5, 6, 6), (18, 5, 6, 4))
    verify_any_layout_transform(any_dims(4), "NCHW", "NCHW4c", (3, 4, 5, 6), (3, 1, 5, 6, 4))
    verify_any_layout_transform((16, 1), "CH", "C4cH", (16, 1), (4, 4, 1))


def verify_any_expand_dims(data_shape, axis, num_newaxis, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.expand_dims(data, axis=axis, num_newaxis=num_newaxis)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_expand_dims():
    verify_any_expand_dims(any_dims(3), 1, 2, (1, 2, 3), (1, 1, 1, 2, 3))
    verify_any_expand_dims(any_dims(3), -1, 2, (1, 2, 3), (1, 2, 3, 1, 1))


def verify_any_transpose(data_shape, axes, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.transpose(data, axes=axes)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.transpose(data_np, axes)
    check_result([data_np], mod, ref_out)


@tvm.testing.uses_gpu
def test_any_transpose():
    verify_any_transpose(any_dims(3), (1, 0, 2), (10, 3, 2))
    verify_any_transpose(any_dims(3), None, (2, 3, 4))
    verify_any_transpose(any_dims(6), (0, 1, 3, 2, 5, 4), (11, 12, 2, 1, 9, 17))
    verify_any_transpose(any_dims(2), (-1, 0), (3, 2))


def verify_any_squeeze(data_shape, axis, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.squeeze(data, axis=axis)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.squeeze(data_np, axis)
    check_result([data_np], mod, ref_out)


@tvm.testing.uses_gpu
def test_any_squeeze():
    verify_any_squeeze((1, relay.Any(), relay.Any()), (0,), (1, 9, 8))
    verify_any_squeeze(
        (1, relay.Any(), relay.Any(), 1, relay.Any(), relay.Any()), (0, 3), (1, 12, 2, 1, 9, 17)
    )


@tvm.testing.uses_gpu
def test_any_reshape_like():
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=(relay.Any(), 3, 10), dtype=dtype)
    shape_like = relay.var("data", shape=(relay.Any(), 5, 6), dtype=dtype)
    y = relay.reshape_like(data, shape_like)
    mod["main"] = relay.Function([data, shape_like], y)
    data_np = np.random.uniform(size=(3, 3, 10)).astype(dtype)
    shape_like_np = np.random.uniform(size=(3, 5, 6)).astype(dtype)
    check_result([data_np, shape_like_np], mod, shape_like_np.shape, assert_shape=True)


def verify_any_conv2d(
    data_shape,
    kernel_shape,
    strides,
    padding,
    dilation,
    static_data_shape,
    ref_out_shape,
):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=dtype)
    y = relay.nn.conv2d(data, kernel, strides, padding, dilation, kernel_size=kernel_shape[2:4])
    mod["main"] = relay.Function([data, kernel], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)
    check_result([data_np, kernel_np], mod, ref_out_shape, assert_shape=True)


# TODO(@kevinthesun): Support dynamic input height and width.
# TODO(@kevinthesun): Support gpu to enable gpu tests.
def test_any_conv2d():
    verify_any_conv2d(
        (relay.Any(), 64, 224, 224),
        (64, 64, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 64, 224, 224),
        (1, 64, 224, 224),
    )
    verify_any_conv2d(
        (relay.Any(), 64, 224, 224),
        (64, 64, 3, 3),
        (1, 1),
        (1, 1),
        (2, 2),
        (2, 64, 224, 224),
        (2, 64, 222, 222),
    )


def verify_any_conv2d_NCHWc(
    data_shape,
    kernel_shape,
    strides,
    padding,
    dilation,
    data_layout,
    kernel_layout,
    out_layout,
    static_data_shape,
    ref_out_shape,
):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=dtype)
    y = relay.nn.contrib_conv2d_nchwc(
        data,
        kernel,
        strides,
        padding,
        dilation,
        kernel_size=kernel_shape[2:4],
        channels=kernel_shape[0] * kernel_shape[-1],
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=out_layout,
    )
    mod["main"] = relay.Function([data, kernel], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)
    check_result([data_np, kernel_np], mod, ref_out_shape, assert_shape=True)


# TODO(@kevinthesun): Support dynamic input height and width.
# TODO(@kevinthesun): Support gpu to enable gpu tests.
def test_any_conv2d_NCHWc():
    verify_any_conv2d_NCHWc(
        (relay.Any(), 8, 224, 224, 8),
        (8, 8, 3, 3, 8, 8),
        (1, 1),
        (1, 1),
        (1, 1),
        "NCHW8c",
        "OIHW8i8o",
        "NCHW8c",
        (1, 8, 224, 224, 8),
        (1, 8, 224, 224, 8),
    )
    verify_any_conv2d_NCHWc(
        (relay.Any(), 8, 224, 224, 8),
        (8, 8, 3, 3, 8, 8),
        (1, 1),
        (1, 1),
        (2, 2),
        "NCHW8c",
        "OIHW8i8o",
        "NCHW8c",
        (2, 8, 224, 224, 8),
        (2, 8, 222, 222, 8),
    )


def verify_any_conv2d_transpose_nchw(
    data_shape,
    kernel_shape,
    strides,
    padding,
    dilation,
    groups,
    static_data_shape,
    ref_out_shape,
    output_padding,
):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=dtype)
    y = relay.nn.conv2d_transpose(
        data,
        kernel,
        strides,
        padding,
        dilation,
        groups,
        kernel_size=kernel_shape[2:4],
        output_padding=output_padding,
    )
    mod["main"] = relay.Function([data, kernel], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)
    check_result(
        [data_np, kernel_np], mod, ref_out_shape, assert_shape=True, targets=[("llvm", tvm.cpu())]
    )


# TODO(@kevinthesun): Support dynamic input height and width.
# TODO(@kevinthesun): Support gpu to enable gpu tests.
def test_any_conv2d_transpose_nchw():
    verify_any_conv2d_transpose_nchw(
        (relay.Any(), 64, 224, 224),
        (64, 192, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        1,
        (2, 64, 224, 224),
        (2, 192, 224, 224),
        (0, 0),
    )
    verify_any_conv2d_transpose_nchw(
        (relay.Any(), 32, 224, 224),
        (32, 64, 3, 3),
        (2, 2),
        (1, 1),
        (1, 1),
        1,
        (1, 32, 224, 224),
        (1, 64, 448, 448),
        (1, 1),
    )


def verify_any_pool2d(
    pool_type, data_shape, pool_size, strides, padding, layout, static_data_shape, ref_out_shape
):
    mod = tvm.IRModule()
    dtype = "float32"
    pool_func = relay.nn.max_pool2d if pool_type == "max" else relay.nn.avg_pool2d
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = pool_func(data, pool_size, strides, padding, layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_pool2d():
    verify_any_pool2d(
        "max",
        (relay.Any(), 3, relay.Any(), relay.Any()),
        (3, 3),
        (1, 1),
        (1, 1),
        "NCHW",
        (2, 3, 220, 220),
        (2, 3, 220, 220),
    )
    verify_any_pool2d(
        "avg",
        (relay.Any(), relay.Any(), relay.Any(), 4),
        (1, 1),
        (2, 2),
        (0, 0),
        "NHWC",
        (3, 220, 220, 4),
        (3, 110, 110, 4),
    )
    verify_any_pool2d(
        "max",
        (relay.Any(), 3, relay.Any(), relay.Any(), 4),
        (3, 3),
        (2, 2),
        (1, 1),
        "NCHW4c",
        (2, 3, 220, 220, 4),
        (2, 3, 110, 110, 4),
    )


def verify_any_global_pool2d(pool_type, data_shape, layout, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    pool_func = relay.nn.global_max_pool2d if pool_type == "max" else relay.nn.global_avg_pool2d
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = pool_func(data, layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_global_pool2d():
    verify_any_global_pool2d(
        "max", (relay.Any(), 3, relay.Any(), relay.Any()), "NCHW", (2, 3, 220, 220), (2, 3, 1, 1)
    )
    verify_any_global_pool2d(
        "avg", (relay.Any(), relay.Any(), relay.Any(), 4), "NHWC", (3, 220, 220, 4), (3, 1, 1, 4)
    )
    verify_any_global_pool2d(
        "max",
        (relay.Any(), 3, relay.Any(), relay.Any(), 4),
        "NCHW4c",
        (2, 3, 220, 220, 4),
        (2, 3, 1, 1, 4),
    )


def verify_any_split(data_shape, indices_or_sections, axis, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.split(data, indices_or_sections, axis)
    mod["main"] = relay.Function([data], y.astuple())
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        for ret, ref_ret in zip(result, ref_out_shape):
            assert ret.asnumpy().shape == ref_ret, "Shape mismatch: expect %s but got %s." % (
                str(ref_ret),
                str(ret.asnumpy().shape),
            )


@tvm.testing.uses_gpu
def test_any_split():
    verify_any_split((relay.Any(), 4), 2, 1, (9, 4), [(9, 2), (9, 2)])
    verify_any_split((relay.Any(), relay.Any()), 2, 1, (9, 4), [(9, 2), (9, 2)])
    verify_any_split((relay.Any(), 12), (1, 4, 8), 1, (7, 12), [(7, 1), (7, 3), (7, 4)])
    verify_any_split((relay.Any(), relay.Any()), (1, 4, 8), 1, (7, 12), [(7, 1), (7, 3), (7, 4)])


@tvm.testing.uses_gpu
def test_any_batch_flatten():
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=any_dims(3), dtype=dtype)
    y = relay.nn.batch_flatten(data)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=(3, 3, 10)).astype(dtype)
    ref_out_shape = (3, 30)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


def verify_any_dense(
    data_shape, weight_shape, units, static_data_shape, static_weight_shape, ref_out_shape
):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("weight", shape=weight_shape, dtype=dtype)
    y = relay.nn.dense(data, weight, units)
    mod["main"] = relay.Function([data, weight], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    weight_np = np.random.uniform(size=static_weight_shape).astype(dtype)
    check_result([data_np, weight_np], mod, ref_out_shape, assert_shape=True)


# TODO(tvm-team) Fix dense schedule
# @tvm.testing.uses_gpu
def test_any_dense():
    verify_any_dense(any_dims(2), any_dims(2), None, (4, 16), (8, 16), (4, 8))
    verify_any_dense(any_dims(2), (50, relay.Any()), 50, (4, 40), (50, 40), (4, 50))


@tvm.testing.uses_gpu
def verify_any_pad(data_shape, pad_width, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.pad(data, pad_width)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.pad(data_np, pad_width)
    check_result([data_np], mod, ref_out)


@tvm.testing.uses_gpu
def test_any_pad():
    verify_any_pad(any_dims(3), ((0, 0), (1, 1), (2, 2)), (1, 2, 3))
    verify_any_pad(any_dims(4), ((1, 0), (1, 3), (0, 2), (9, 0)), (13, 11, 3, 1))


def verify_any_dilate(data_shape, strides, static_data_shape):
    assert len(data_shape) == len(strides)
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.dilate(data, strides)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_shape = tuple(
        (static_data_shape[i] - 1) * strides[i] + 1 for i in range(len(static_data_shape))
    )
    ref_out = np.zeros(shape=ref_shape, dtype=dtype)
    ref_out[tuple(slice(None, None, strides[i]) for i in range(len(data_shape)))] = data_np
    check_result([data_np], mod, ref_out)


@tvm.testing.uses_gpu
def test_any_dilate():
    verify_any_dilate(any_dims(1), (1,), (1,))
    verify_any_dilate(any_dims(1), (1,), (5,))
    verify_any_dilate(any_dims(1), (5,), (5,))
    verify_any_dilate(any_dims(3), (1, 1, 1), (1, 2, 3))
    verify_any_dilate(any_dims(3), (1, 1, 2), (1, 2, 3))
    verify_any_dilate(any_dims(3), (1, 1, 5), (1, 2, 3))
    verify_any_dilate(any_dims(3), (3, 7, 5), (1, 2, 3))
    verify_any_dilate(any_dims(4), (3, 7, 1, 5), (1, 2, 3, 4))


def verify_any_softmax(data_shape, axis, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.softmax(data, axis)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_softmax():
    verify_any_softmax(any_dims(3), -1, (1, 2, 3), (1, 2, 3))
    verify_any_softmax(any_dims(4), 2, (13, 11, 3, 1), (13, 11, 3, 1))


def verify_any_topk(data_shape, kval, np_dshape, dtype, const_k=False):
    mod = tvm.IRModule()
    data = relay.var("data", shape=data_shape, dtype=dtype)
    np_data = np.random.uniform(size=np_dshape).astype(dtype)
    if const_k:
        k = relay.const(kval)
        args = [data]
        in_vals = [np_data]
    else:
        k = relay.var("k", shape=(), dtype="int32")
        args = [data, k]
        in_vals = [np_data, kval]
    out = relay.topk(data, k, ret_type="indices")
    mod["main"] = relay.Function(args, out)

    sorted = np.argsort(-np_data)
    if len(np_dshape) == 2:
        ref_out = sorted[:, 0:kval]
    else:
        ref_out = sorted[0:kval]

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(*in_vals)
        tvm.testing.assert_allclose(result.asnumpy(), ref_out)

    # TODO(@zhiics) Fix topk cuda schedule for dynamic inputs
    # check_result(in_vals, mod, ref_out)


def test_any_topk():
    verify_any_topk(any_dims(1), 5, (10,), "float32")
    verify_any_topk(any_dims(2), 2, (6, 3), "int32")
    verify_any_topk(any_dims(2), 3, (6, 3), "float32", True)


@tvm.testing.uses_gpu
def test_fused_ops():
    x = relay.var("x", shape=(relay.Any(), relay.Any()), dtype="float32")
    y0 = x + relay.const(1.0, "float32")
    y1 = y0 * relay.const(2.0, "float32")
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y1)
    data = np.random.uniform(size=(5, 4)).astype("float32")
    check_result([data], mod, (data + 1) * 2)


@tvm.testing.uses_gpu
def test_arange_with_dynamic_shape():
    # m, n, k = relay.ShapeVar('m'), relay.ShapeVar('n'), relay.ShapeVar('k')
    m, n, k = relay.Any(), relay.Any(), relay.Any()
    x = relay.var("x", shape=(m, n, k), dtype="float32")
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(0, "int32"))
    y2 = relay.op.arange(y1, dtype="int32")
    y3 = y2 + relay.const(1, dtype="int32")
    data = np.random.rand(10, 5, 3).astype("float32")
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y3)
    check_result([data], mod, np.array(range(10)).astype("int32") + 1)


def verify_any_strided_slice(
    data_shape,
    begin_shape,
    end_shape,
    strides_shape,
    data_np_shape,
    slice_mode="end",
    const_attrs=False,
):
    # Generate random numpy input data
    np_data = np.random.uniform(size=data_np_shape).astype("float32")
    np_begin = np.random.randint(2, size=begin_shape, dtype="int32")
    np_end = np.random.randint(5, 10, size=end_shape, dtype="int32")
    np_strides = np.random.randint(
        1, 2 if slice_mode == "size" else 3, size=strides_shape, dtype="int32"
    )
    # target numpy result
    ref_res = tvm.topi.testing.strided_slice_python(
        np_data, np_begin, np_end, np_strides, slice_mode
    )

    # Relay Module
    mod = tvm.IRModule()
    data = relay.var("data", shape=data_shape, dtype="float32")
    if const_attrs:
        data = relay.var("data", shape=data_np_shape, dtype="float32")
        begin = relay.const(np_begin)
        end = relay.const(np_end)
        strides = relay.const(np_strides)
        args = [data]
        np_inputs = [np_data]
    else:
        begin = relay.var("begin", shape=begin_shape, dtype="int32")
        end = relay.var("end", shape=end_shape, dtype="int32")
        strides = relay.var("strides", shape=strides_shape, dtype="int32")
        args = [data, begin, end, strides]
        np_inputs = [np_data, np_begin, np_end, np_strides]

    y = relay.strided_slice(data, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
    mod["main"] = relay.Function(args, y)

    check_result(np_inputs, mod, ref_res)


@tvm.testing.uses_gpu
def test_any_strided_slice():
    verify_any_strided_slice(any_dims(2), (2,), (2,), (2,), (15, 21))
    verify_any_strided_slice(any_dims(3), (3,), (3,), (3,), (15, 17, 21))
    verify_any_strided_slice(any_dims(3), (3,), (3,), (3,), (23, 29, 41))
    verify_any_strided_slice(any_dims(4), (4,), (4,), (4,), (40, 50, 60, 70))
    verify_any_strided_slice(any_dims(3), (3,), (3,), (3,), (15, 17, 21), slice_mode="size")
    verify_any_strided_slice(any_dims(2), (2,), (2,), (2,), (15, 21), const_attrs=True)


@tvm.testing.uses_gpu
def test_recursive_concat():
    """
    fn @concat_loop(%i: int32, %st: (any, 1)) -> (any, 1) {
        if (%i < 10) {
            let %i = reshape(cast(i, "float32"), newshape=(1, ))
            let %new_st = concatenate((st, i), axis=0)
            concat_loop(%i + 1, )
        } else {
            st
        }
    }
    """
    # Initial Values.
    i = relay.var("i", shape=(), dtype="int32")
    st = relay.var("st", shape=(relay.Any(), 1), dtype="int32")

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1, 1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = while_loop(_cond, [i, st], _body)
    start = relay.var("start", shape=(), dtype="int32")
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    mod = tvm.IRModule()
    mod["main"] = func
    data = np.array(0.0, dtype="int32")
    ref = np.array([0] + list(range(10))).reshape((11, 1)).astype("int32")
    check_result([data], mod, ref)


@tvm.testing.uses_gpu
def test_recursive_concat_with_wrong_annotation():
    """
    v0.0.1
    fn (%start: int32) {
        %7 = {
            let %while_loop = fn (%i: int32, %st: Tensor[(1, 1), int32]) {
            %0 = less(%i, 10)
            %1 = min(%0)
            if (%1) {
                %2 = add(%i, 1)
                %3 = reshape(%i, newshape=[1, 1])
                %4 = (%st, %3)
                /* The result of concat should be 1,1 but it is 2, 1. */
                %5 = concatenate(%4)
                %while_loop(%2, %5)
            } else {
                (%i, %st)
            }
        }
        %6 = reshape(0, newshape=[1, 1])
        %while_loop(%start, %6)
    }
    %7.1
    }
    """
    # Initial Values.
    i = relay.var("i", shape=(), dtype="int32")
    st = relay.var("st", shape=(1, 1), dtype="int32")

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1, 1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = while_loop(_cond, [i, st], _body)
    start = relay.var("start", shape=(), dtype="int32")
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    try:
        func = infer_type(func)
        assert False
    except Exception as e:
        assert "in particular dimension 0 conflicts 2 does not match 1" in str(e)


@tvm.testing.uses_gpu
def test_tuple_get_item():
    mod = tvm.IRModule()
    dtype = "float32"
    static_data_shape = (9, 4)
    data_shape = (relay.Any(), 4)
    indices_or_sections = 2
    axis = 1
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.split(data, indices_or_sections, axis)
    y = relay.expr.TupleGetItem(y.astuple(), 0)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out_shape = (9, 2)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_mixed_input_type():
    mod = tvm.IRModule()
    dtype = "float32"
    static_data_shape = (9, 4)
    data_shape = (relay.Any(), 4)
    tensor_type = relay.TensorType(data_shape, dtype)
    tuple_type = relay.TupleType([tensor_type, tensor_type])
    data0 = relay.var("d0", type_annotation=relay.TupleType([tuple_type, tensor_type]))
    data1 = relay.var("d1", shape=(relay.Any(), 4), dtype=dtype)
    data_tuple = relay.expr.TupleWrapper(data0, 2)
    nested_data_tuple = relay.expr.TupleWrapper(data_tuple[0], 2)
    y = nested_data_tuple[1] * data_tuple[1] + data1
    mod["main"] = relay.Function([data0, data1], y)
    data_np0 = np.random.uniform(size=static_data_shape).astype(dtype)
    data_np1 = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out_shape = (9, 4)
    check_result(
        [[[data_np0, data_np0], data_np0], data_np1],
        mod,
        ref_out_shape,
        assert_shape=True,
        only_vm=True,
    )


def verify_any_crop_and_resize(
    data_shape,
    boxes_shape,
    box_indices_shape,
    crop_size,
    layout,
    static_boxes,
    static_box_indices_shape,
    ref_out_shape,
):
    mod = tvm.IRModule()
    dtype = "float32"
    indices_dtype = "int32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    boxes = relay.var("boxes", shape=boxes_shape, dtype=dtype)
    box_indices = relay.var("box_indices", shape=box_indices_shape, dtype=indices_dtype)
    y = relay.image.crop_and_resize(data, boxes, box_indices, crop_size, layout)
    mod["main"] = relay.Function([data, boxes, box_indices], y)
    data_np = np.random.uniform(size=data_shape).astype(dtype)
    boxes_np = np.random.uniform(size=static_boxes).astype(dtype)
    box_indices_np = np.random.uniform(size=static_box_indices_shape).astype(indices_dtype)
    check_result([data_np, boxes_np, box_indices_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_crop_and_resize():
    verify_any_crop_and_resize(
        data_shape=(1, 234, 234, 256),
        boxes_shape=(relay.Any(), 4),
        box_indices_shape=(relay.Any(),),
        crop_size=(14, 14),
        layout="NHWC",
        static_boxes=(128, 4),
        static_box_indices_shape=(128,),
        ref_out_shape=(128, 14, 14, 256),
    )
    verify_any_crop_and_resize(
        data_shape=(1, 256, 234, 234),
        boxes_shape=(relay.Any(), 4),
        box_indices_shape=(relay.Any(),),
        crop_size=(14, 14),
        layout="NCHW",
        static_boxes=(128, 4),
        static_box_indices_shape=(128,),
        ref_out_shape=(128, 256, 14, 14),
    )


def verify_any_mirror_pad(data_shape, pad_width, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.mirror_pad(data, pad_width)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_mirror_pad():
    verify_any_mirror_pad(
        data_shape=(1, 256, 232, 232),
        pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
        static_data_shape=(1, 256, 232, 232),
        ref_out_shape=(1, 256, 234, 234),
    )


def verify_any_ndarray_size(data_np_shape):
    v = relay.var("v", shape=any_dims(len(data_np_shape)), dtype="float32")
    n = relay.ndarray_size(v, dtype="int32")
    mod = tvm.IRModule()
    mod["main"] = relay.Function([v], n)
    np_data = np.zeros(data_np_shape, dtype="float32")
    ref_res = np.size(np_data)
    check_result([np_data], mod, ref_res)


@tvm.testing.uses_gpu
def test_any_ndarray_size():
    verify_any_ndarray_size((2,))
    verify_any_ndarray_size((2, 2))
    verify_any_ndarray_size((1, 2, 3, 4))


def test_any_consecutive_broadcast():
    dtype = "float32"
    data0 = relay.var("data0", shape=any_dims(2), dtype=dtype)
    data1 = relay.var("data1", shape=any_dims(2), dtype=dtype)
    data2 = relay.var("data2", shape=any_dims(2), dtype=dtype)
    data3 = relay.var("data3", shape=any_dims(2), dtype=dtype)

    out0 = data0 + data1
    out1 = data0 * data1
    out2 = out0 - out1

    out3 = data2 + data3
    out4 = data2 * data3
    out5 = out3 - out4

    out6 = out2 * out5

    mod = tvm.IRModule()
    mod["main"] = relay.Function([data0, data1, data2, data3], out6)

    np_data0 = np.random.uniform(size=(1, 4)).astype(dtype)
    np_data1 = np.random.uniform(size=(2, 4)).astype(dtype)
    np_data2 = np.random.uniform(size=(1, 4)).astype(dtype)
    np_data3 = np.random.uniform(size=(2, 4)).astype(dtype)
    ref_res = ((np_data0 + np_data1) - (np_data0 * np_data1)) * (
        (np_data2 + np_data3) - (np_data2 * np_data3)
    )
    check_result([np_data0, np_data1, np_data2, np_data3], mod, ref_res)


def test_reshape_concat():
    dtype = "float32"
    d0 = relay.var("d0", shape=any_dims(2), dtype=dtype)
    d1 = relay.var("d1", shape=any_dims(3), dtype=dtype)
    out = relay.op.concatenate([relay.op.reshape(d0, [-1]), relay.op.reshape(d1, [-1])], axis=0)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d0, d1], out)
    np_data0 = np.random.uniform(size=(4, 5)).astype(dtype)
    np_data1 = np.random.uniform(size=(2, 5, 2)).astype(dtype)
    ref_res = np.concatenate([np.reshape(np_data0, [-1]), np.reshape(np_data1, [-1])], axis=0)
    check_result([np_data0, np_data1], mod, ref_res)

    d0 = relay.var("d0", shape=any_dims(2), dtype=dtype)
    d1 = relay.var("d1", shape=any_dims(2), dtype=dtype)
    s0 = relay.var("s0", shape=any_dims(3), dtype=dtype)
    s1 = relay.var("s1", shape=any_dims(3), dtype=dtype)
    out = relay.op.concatenate(
        [relay.op.reshape_like(d0, s0), relay.op.reshape_like(d1, s1)], axis=0
    )
    mod = tvm.IRModule()
    mod["main"] = relay.Function([d0, d1, s0, s1], out)
    np_data0 = np.random.uniform(size=(4, 5)).astype(dtype)
    np_data1 = np.random.uniform(size=(8, 5)).astype(dtype)
    np_shape_like0 = np.random.uniform(size=(2, 2, 5)).astype(dtype)
    np_shape_like1 = np.random.uniform(size=(4, 2, 5)).astype(dtype)
    ref_res = np.concatenate(
        [np.reshape(np_data0, np_shape_like0.shape), np.reshape(np_data1, np_shape_like1.shape)],
        axis=0,
    )
    check_result([np_data0, np_data1, np_shape_like0, np_shape_like1], mod, ref_res)


def test_any_adv_index():
    data = relay.var("data", shape=(5, relay.Any(), relay.Any()), dtype="float32")
    index0 = relay.var("index0", shape=(1, relay.Any()), dtype="int64")
    index1 = relay.var("index1", shape=(1, relay.Any()), dtype="int64")
    out = relay.adv_index([data, index0, index1])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([data, index0, index1], out)
    np_data_shape = (5, 5, 10)
    np_index_shape = (1, 4)
    np_data = np.random.uniform(size=np_data_shape).astype("float32")
    np_index = np.random.uniform(0, np_data_shape[0], size=np_index_shape).astype("int64")
    ref_res = np_data[tuple([np_index, np_index])]
    check_result([np_data, np_index, np_index], mod, ref_res)


def verify_any_repeat(data_shape, np_dshape, repeats, axis):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.repeat(data, repeats, axis)
    mod["main"] = relay.Function([data], y)
    np_data = np.random.uniform(size=np_dshape).astype(dtype)
    ref_res = np.repeat(np_data, repeats, axis)
    check_result([np_data], mod, ref_res)


@tvm.testing.uses_gpu
def test_any_repeat():
    verify_any_repeat(any_dims(2), (1, 2), 2, 0)
    verify_any_repeat(any_dims(1), (3,), 3, -1)
    verify_any_repeat(any_dims(4), (2, 1, 1, 4), 4, 2)


def verify_any_stack(data_shape, np_dshape, num_data, axis):
    mod = tvm.IRModule()
    dtype = "float32"
    inputs = []
    for i in range(num_data):
        inputs.append(relay.var("data{}".format(i), shape=data_shape, dtype=dtype))
    y = relay.stack(inputs, axis)
    mod["main"] = relay.Function(inputs, y)
    np_inputs = []
    for _ in range(num_data):
        np_inputs.append(np.random.uniform(size=np_dshape).astype(dtype))
    ref_res = np.stack(np_inputs, axis)
    check_result(np_inputs, mod, ref_res)


@tvm.testing.uses_gpu
def test_any_stack():
    verify_any_stack(any_dims(2), (1, 2), 3, 0)
    verify_any_stack(any_dims(1), (3,), 4, -1)
    verify_any_stack(any_dims(4), (2, 1, 1, 4), 2, 2)


if __name__ == "__main__":
    pytest.main([__file__])
