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
import os
import platform

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay, te
from tvm.relay.loops import while_loop
from tvm.relay.testing import run_infer_type as infer_type
from tvm.topi.testing import searchsorted_ref

from utils import ref_funcs
from utils.assert_diagnostic import DiagnosticTesting


def int32(val):
    return relay.const(val, "int32")


def any_dims(ndim):
    shape = []
    for _ in range(ndim):
        shape.append(relay.Any())
    return tuple(shape)


def check_result(
    args,
    mod,
    expected,
    flatten=False,
    assert_shape=False,
    only_vm=False,
    targets=None,
    disable_targets=None,
):
    if not isinstance(expected, list):
        expected = [expected]
    for kind in ["debug", "vm"]:
        targets = targets or tvm.testing.enabled_targets()
        for tgt, dev in targets:
            if disable_targets and tgt in disable_targets:
                continue
            if kind == "debug" and (only_vm or dev.device_type != tvm.cpu().device_type):
                continue
            result = relay.create_executor(kind, mod=mod, device=dev, target=tgt).evaluate()(*args)
            if isinstance(result, tvm.runtime.container.ADT):
                result = [r.numpy() for r in result]
            else:
                result = [result.numpy()]

            for r, e in zip(result, expected):
                if assert_shape:
                    assert r.shape == e, "Shape mismatch: expect %s but got %s." % (
                        str(e),
                        str(r),
                    )
                else:
                    if flatten:
                        r = r.flatten()
                        e = e.flatten()
                    tvm.testing.assert_allclose(r, e, atol=2e-6)


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
    verify_any_elemwise((relay.Any(),), (3,), relay.round, np.round)


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

    num_inputs = 25
    x = [relay.var("x", shape=(relay.Any(),), dtype="float32") for _ in range(num_inputs)]
    z = relay.op.concatenate(x, axis=0)
    mod = tvm.IRModule()
    mod["main"] = relay.Function(x, z)
    x_np = [np.random.uniform(size=(1,)).astype("float32") for _ in range(num_inputs)]
    ref = np.concatenate(x_np, axis=0)
    check_result(x_np, mod, ref)

    def test_oshape(in_vars, axis, oshape):
        z = relay.op.concatenate(in_vars, axis=axis)
        mod = tvm.IRModule()
        mod["main"] = relay.Function(in_vars, z)
        typed_mod = relay.transform.InferType()(mod)
        assert typed_mod["main"].body.checked_type == relay.TensorType(oshape, dtype="float32")

    x = [relay.var("x", shape=(relay.Any(), 3), dtype="float32") for _ in range(3)]
    x.append(relay.var("x", shape=(relay.Any(), relay.Any()), dtype="float32"))

    test_oshape(x, 0, (relay.Any(), 3))
    test_oshape(x, 1, (relay.Any(), relay.Any()))

    # [(1, 3), (1, ?)] -> (2, ?)
    x = [
        relay.var("x", shape=(1, 3), dtype="float32"),
        relay.var("x", shape=(1, relay.Any()), dtype="float32"),
    ]
    test_oshape(x, 0, (2, relay.Any()))
    test_oshape(x, 1, (1, relay.Any()))


def verify_any_reshape(x_shape, newshape, x_np_shape, out_shape, variable_newshape=False):
    x = relay.var("x", shape=x_shape, dtype="float32")
    relu_x = relay.nn.relu(x)
    data = np.random.uniform(size=x_np_shape).astype("float32")
    expected = data.reshape(out_shape)
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
    check_result(args, mod, expected)


@tvm.testing.uses_gpu
def test_any_reshape():
    for variable_newshape in [False, True]:
        # Variable newshape only supports that output rank is the same as newshape
        verify_any_reshape(any_dims(3), (1, -1), (2, 3, 4), (1, 24), variable_newshape)
        verify_any_reshape(any_dims(3), (0, -1), (2, 3, 4), (2, 12), variable_newshape)
    verify_any_reshape(any_dims(3), (0, -2), (2, 3, 4), (2, 3, 4))
    verify_any_reshape(any_dims(3), (-4, -1, 2, -3), (6, 3, 4), (3, 2, 12))
    verify_any_reshape(any_dims(3), (-4, 2, -1, -2), (6, 3, 4), (2, 3, 3, 4))
    verify_any_reshape(any_dims(3), (1, -1, 0), (2, 3, 4), (1, 6, 4))
    verify_any_reshape(any_dims(3), (-1, 1, 0), (2, 3, 4), (6, 1, 4))


def verify_any_one_hot(indices_shape, indices_np_shape, depth, on_value, off_value, axis, dtype):
    indices = relay.var("indices", shape=indices_shape, dtype="int32")
    on_value_const = relay.const(on_value, dtype)
    off_value_const = relay.const(off_value, dtype)
    y = relay.one_hot(indices, on_value_const, off_value_const, depth, axis=axis, dtype=dtype)
    params = [indices]
    mod = tvm.IRModule()
    mod["main"] = relay.Function(params, y)

    indices_npy = np.random.randint(0, depth, size=indices_np_shape).astype("int32")
    out_npy = tvm.topi.testing.one_hot(indices_npy, on_value, off_value, depth, axis, dtype)
    args = [indices_npy]
    check_result(args, mod, out_npy)


@tvm.testing.uses_gpu
def test_any_one_hot():
    verify_any_one_hot(any_dims(1), (3,), 3, 1, 0, -1, "int32")
    verify_any_one_hot(any_dims(2), (2, 2), 5, 0.5, -0.5, 1, "float32")
    verify_any_one_hot(any_dims(4), (3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")


def verify_any_argwhere(x_shape, x_np_shape, dtype="bool"):
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.argwhere(x)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    data = np.random.choice([0, 1, 2, 3], size=x_np_shape).astype(dtype)
    expected = np.argwhere(data)
    check_result([data], mod, expected, flatten=True)


@tvm.testing.uses_gpu
def test_any_argwhere():
    verify_any_argwhere(any_dims(1), (5,))
    verify_any_argwhere(any_dims(2), (5, 5))
    verify_any_argwhere(any_dims(2), (5, 5), "int32")
    verify_any_argwhere(any_dims(2), (5, 5), "int8")
    verify_any_argwhere(any_dims(3), (5, 5, 5))
    verify_any_argwhere(any_dims(4), (5, 5, 5, 5))
    verify_any_argwhere(any_dims(5), (5, 5, 5, 5, 5))
    verify_any_argwhere(any_dims(1), (5,), "int32")
    verify_any_argwhere(any_dims(3), (5, 5, 5), "int32")
    verify_any_argwhere(any_dims(4), (5, 5, 5, 5), "int32")
    verify_any_argwhere(any_dims(5), (5, 5, 5, 5, 5), "int32")
    verify_any_argwhere(any_dims(1), (5,), "int8")
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


class TestAnyReduce:
    config = {
        "argmax": (relay.argmax, any_dims(3), None, False, False, (3, 4, 5), ()),
        "argmin": (relay.argmin, any_dims(4), 1, False, True, (3, 4, 5, 6), (3, 1, 5, 6)),
        "all": (relay.all, any_dims(3), (1, 2), True, False, (3, 4, 5), (4, 5)),
        "max": (relay.max, any_dims(4), -1, True, True, (3, 4, 5, 6), (1, 1, 1, 6)),
        "min": (relay.min, any_dims(3), (0, 1), False, False, (4, 5, 6), (6,)),
        "prod": (relay.prod, any_dims(4), 2, True, True, (3, 4, 5, 6), (1, 1, 5, 1)),
        "mean": (relay.mean, any_dims(2), 0, False, False, (1, 2), (2,)),
        "variance": (relay.variance, any_dims(5), (2, 4), False, False, (3, 4, 5, 6, 7), (3, 4, 6)),
    }

    (
        reduce_op,
        data_shape,
        axis,
        exclude,
        keepdims,
        static_data_shape,
        ref_out_shape,
    ) = tvm.testing.parameters(*config.values(), ids=config.keys())

    def test_any_reduce(
        self,
        target,
        dev,
        reduce_op,
        data_shape,
        axis,
        exclude,
        keepdims,
        static_data_shape,
        ref_out_shape,
    ):
        target = tvm.target.Target(target)
        if target.kind.name == "vulkan" and reduce_op == relay.all:
            pytest.xfail("Known failing test case for vulkan runtime")

        mod = tvm.IRModule()
        dtype = "bool" if reduce_op == relay.all else "float32"
        data = relay.var("data", shape=data_shape, dtype=dtype)
        y = reduce_op(data, axis, keepdims, exclude)
        mod["main"] = relay.Function([data], y)
        data_np = np.random.uniform(size=static_data_shape).astype(dtype)
        check_result([data_np], mod, ref_out_shape, assert_shape=True, targets=[(target, dev)])


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


def test_bilayout_with_any():
    bilayout = tvm.tir.bijective_layout("NCHW", "NHWC")
    assert isinstance(bilayout, tvm.tir.BijectiveLayout)
    dst_shape = bilayout.forward_shape((relay.Any(), 32, 7, relay.Any()))
    assert dst_shape[3] == 32
    src_shape = bilayout.backward_shape(dst_shape)
    assert src_shape[1] == 32


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


def verify_any_squeeze_sqrt(data_shape, axis, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.squeeze(data, axis=axis)
    y = relay.sqrt(y)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.sqrt(np.squeeze(data_np, axis))
    check_result([data_np], mod, ref_out)


@tvm.testing.uses_gpu
def test_any_squeeze():
    verify_any_squeeze((relay.Any(), relay.Any(), relay.Any()), (0,), (1, 9, 8))
    verify_any_squeeze((1, relay.Any(), relay.Any()), (0,), (1, 9, 8))
    verify_any_squeeze(
        (1, relay.Any(), relay.Any(), 1, relay.Any(), relay.Any()), (0, 3), (1, 12, 2, 1, 9, 17)
    )
    verify_any_squeeze_sqrt((1, relay.Any(), 12, 32, 1), (-1,), (1, 100, 12, 32, 1))
    verify_any_squeeze_sqrt((relay.Any(), relay.Any(), relay.Any(), 1), (-1,), (1, 9, 8, 1))


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
    data_layout="NCHW",
    kernel_layout="OIHW",
    use_cudnn=False,
    targets=None,
    disable_targets=None,
):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    kernel = relay.var("kernel", shape=kernel_shape, dtype=dtype)
    y = relay.nn.conv2d(
        data,
        kernel,
        strides,
        padding,
        dilation,
        kernel_size=kernel_shape[2:4] if kernel_layout == "OIHW" else kernel_shape[0:2],
        data_layout=data_layout,
        kernel_layout=kernel_layout,
    )
    mod["main"] = relay.Function([data, kernel], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)

    if use_cudnn and tvm.get_global_func("tvm.contrib.cudnn.conv2d.forward", True):
        targets = [("cuda -libs=cudnn", tvm.cuda(0))]

    check_result(
        [data_np, kernel_np],
        mod,
        ref_out_shape,
        assert_shape=True,
        targets=targets,
        disable_targets=disable_targets,
    )


# TODO(@kevinthesun): Support dynamic input height and width.
@tvm.testing.uses_gpu
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
    verify_any_conv2d(
        (relay.Any(), 64, 224, 224),
        (64, 64, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 64, 224, 224),
        (1, 64, 224, 224),
        use_cudnn=True,
    )
    verify_any_conv2d(
        (relay.Any(), 224, 224, 64),
        (3, 3, 64, 64),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 224, 224, 64),
        (1, 224, 224, 64),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )
    verify_any_conv2d(
        (relay.Any(), 224, 224, 64),
        (3, 3, 64, 64),
        (1, 1),
        (1, 1),
        (2, 2),
        (2, 224, 224, 64),
        (2, 222, 222, 64),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    if platform.machine() == "aarch64":
        pytest.skip(
            reason="Dynamic height and width not supported in arm_cpu. See https://github.com/apache/tvm/issues/16536"
        )

    verify_any_conv2d(
        (relay.Any(), 64, relay.Any(), relay.Any()),
        (64, 64, 3, 3),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 64, 224, 224),
        (1, 64, 224, 224),
        targets=[("llvm", tvm.cpu(0))],
    )
    verify_any_conv2d(
        (relay.Any(), 64, relay.Any(), relay.Any()),
        (64, 64, 1, 1),
        (1, 1),
        (0, 0),
        (1, 1),
        (1, 64, 224, 224),
        (1, 64, 224, 224),
        targets=[("llvm", tvm.cpu(0))],
    )


class TestAnyConv2dNCHWc:
    data_shape = tvm.testing.parameter((relay.Any(), 8, 224, 224, 8))
    kernel_shape = tvm.testing.parameter((8, 8, 3, 3, 8, 8))
    strides = tvm.testing.parameter((1, 1))
    padding = tvm.testing.parameter((1, 1))
    data_layout = tvm.testing.parameter("NCHW8c")
    kernel_layout = tvm.testing.parameter("OIHW8i8o")
    out_layout = tvm.testing.parameter("NCHW8c")

    dilation, static_data_shape, ref_out_shape = tvm.testing.parameters(
        ((1, 1), (1, 8, 224, 224, 8), (1, 8, 224, 224, 8)),
        ((2, 2), (2, 8, 224, 224, 8), (2, 8, 222, 222, 8)),
    )

    @tvm.testing.known_failing_targets("cuda", "vulkan")
    def test_any_conv2d_NCHWc(
        self,
        target,
        dev,
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
        check_result(
            [data_np, kernel_np], mod, ref_out_shape, assert_shape=True, targets=[(target, dev)]
        )


def verify_any_conv1d_transpose_ncw(
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
    y = relay.nn.conv1d_transpose(
        data,
        kernel,
        strides,
        padding,
        dilation,
        groups,
        kernel_size=kernel_shape[2:],
        output_padding=output_padding,
    )
    mod["main"] = relay.Function([data, kernel], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)
    check_result([data_np, kernel_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_conv1d_transpose_ncw():
    verify_any_conv1d_transpose_ncw(
        (relay.Any(), 64, 224),
        (64, 192, 3),
        (1,),
        (1,),
        (1,),
        1,
        (2, 64, 224),
        (2, 192, 224),
        (0, 0),
    )
    verify_any_conv1d_transpose_ncw(
        (relay.Any(), 32, 224),
        (32, 64, 3),
        (2,),
        (1,),
        (1,),
        1,
        (1, 32, 224),
        (1, 64, 448),
        (1, 1),
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
    check_result([data_np, kernel_np], mod, ref_out_shape, assert_shape=True)


# TODO(@kevinthesun): Support dynamic input height and width.
@tvm.testing.uses_gpu
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
    pool_type,
    data_shape,
    pool_size,
    strides,
    dilation,
    padding,
    layout,
    static_data_shape,
    ref_out_shape,
):
    mod = tvm.IRModule()
    dtype = "float32"
    pool_func = relay.nn.max_pool2d if pool_type == "max" else relay.nn.avg_pool2d
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = pool_func(data, pool_size, strides, dilation, padding, layout)
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
        (1, 1),
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
        result = relay.create_executor(kind, mod=mod, device=tvm.cpu(), target="llvm").evaluate()(
            data_np
        )
        for ret, ref_ret in zip(result, ref_out_shape):
            assert ret.numpy().shape == ref_ret, "Shape mismatch: expect %s but got %s." % (
                str(ref_ret),
                str(ret.numpy().shape),
            )


@tvm.testing.uses_gpu
def test_any_split():
    verify_any_split((relay.Any(), 4), 2, -1, (9, 4), [(9, 2), (9, 2)])
    verify_any_split((relay.Any(), 4), 2, 1, (9, 4), [(9, 2), (9, 2)])
    verify_any_split((relay.Any(), relay.Any()), 2, 1, (9, 4), [(9, 2), (9, 2)])
    verify_any_split((relay.Any(), 12), (1, 4, 8), 1, (7, 12), [(7, 1), (7, 3), (7, 4)])
    verify_any_split((relay.Any(), relay.Any()), (1, 4, 8), 1, (7, 12), [(7, 1), (7, 3), (7, 4)])
    verify_any_split((relay.Any(), 12), (8,), 1, (7, 12), [(7, 8), (7, 4)])
    verify_any_split((relay.Any(), relay.Any()), (8,), 1, (7, 12), [(7, 8), (7, 4)])


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


# TODO(tvm-team) Fix dense schedule
@tvm.testing.known_failing_targets("cuda", "vulkan")
class TestAnyDense:
    (
        data_shape,
        weight_shape,
        units,
        static_data_shape,
        static_weight_shape,
        ref_out_shape,
    ) = tvm.testing.parameters(
        (any_dims(2), any_dims(2), None, (4, 16), (8, 16), (4, 8)),
        (any_dims(2), (50, relay.Any()), 50, (4, 40), (50, 40), (4, 50)),
    )

    @tvm.testing.known_failing_targets("cuda", "vulkan")
    def test_any_dense(
        self,
        target,
        dev,
        data_shape,
        weight_shape,
        units,
        static_data_shape,
        static_weight_shape,
        ref_out_shape,
    ):

        if platform.machine() == "aarch64":
            pytest.skip(
                reason="Dynamic height and width not supported in arm_cpu. See https://github.com/apache/tvm/issues/16536"
            )

        mod = tvm.IRModule()
        dtype = "float32"
        data = relay.var("data", shape=data_shape, dtype=dtype)
        weight = relay.var("weight", shape=weight_shape, dtype=dtype)
        y = relay.nn.dense(data, weight, units)
        mod["main"] = relay.Function([data, weight], y)
        data_np = np.random.uniform(size=static_data_shape).astype(dtype)
        weight_np = np.random.uniform(size=static_weight_shape).astype(dtype)

        check_result(
            [data_np, weight_np], mod, ref_out_shape, assert_shape=True, targets=[(target, dev)]
        )

    @tvm.testing.parametrize_targets("cuda -libs=cublas")
    @tvm.testing.known_failing_targets("cuda", "vulkan")
    def test_any_dense_cublas(
        self,
        target,
        dev,
        data_shape,
        weight_shape,
        units,
        static_data_shape,
        static_weight_shape,
        ref_out_shape,
    ):

        self.test_any_dense(
            target,
            dev,
            data_shape,
            weight_shape,
            units,
            static_data_shape,
            static_weight_shape,
            ref_out_shape,
        )


class TestAnyBatchMatmul:
    dtype = tvm.testing.parameter("float32")
    executor_kind = tvm.testing.parameter("vm", "debug")

    (x_shape, y_shape) = tvm.testing.parameters(
        ((1, 16, 32), (1, 32, 16)),
        ((5, 16, 32), (5, 32, 16)),
        ((5, 16, 32), (5, 32, 20)),
        ((30, 16, 32), (30, 32, 20)),
    )

    # any_x = tvm.testing.parameter("none", "batch")
    # any_y = tvm.testing.parameter("none", "batch", "all")

    any_x, any_y = tvm.testing.parameters(
        ("none", "batch"), ("none", "all"), ("batch", "none"), ("batch", "batch"), ("batch", "all")
    )

    transpose_x = tvm.testing.parameter(True, False)
    transpose_y = tvm.testing.parameter(True, False)

    @tvm.testing.fixture
    def x_var_shape(self, x_shape, any_x):
        if any_x == "none":
            return x_shape
        elif any_x == "batch":
            return tuple(relay.Any() if i == 0 else size for i, size in enumerate(x_shape))
        elif any_x == "all":
            return tuple(relay.Any() for _ in x_shape)

    @tvm.testing.fixture
    def y_var_shape(self, y_shape, any_y):
        if any_y == "none":
            return y_shape
        elif any_y == "batch":
            return tuple(relay.Any() if i == 0 else size for i, size in enumerate(y_shape))
        elif any_y == "all":
            return tuple(relay.Any() for _ in y_shape)

    @tvm.testing.known_failing_targets("cuda", "vulkan")
    def test_any_batch_matmul(
        self,
        target,
        dev,
        x_shape,
        y_shape,
        any_x,
        any_y,
        x_var_shape,
        y_var_shape,
        transpose_x,
        transpose_y,
        executor_kind,
        dtype,
    ):
        if transpose_x:
            x_shape = (x_shape[0], x_shape[2], x_shape[1])
            x_var_shape = (x_var_shape[0], x_var_shape[2], x_var_shape[1])

        if transpose_y:
            y_shape = (y_shape[0], y_shape[2], y_shape[1])
            y_var_shape = (y_var_shape[0], y_var_shape[2], y_var_shape[1])

        x = relay.var("x", relay.TensorType(x_var_shape, dtype))
        y = relay.var("y", relay.TensorType(y_var_shape, dtype))
        z = relay.nn.batch_matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)

        func = relay.Function([x, y], z)
        x_np = np.random.uniform(size=x_shape).astype(dtype)
        y_np = np.random.uniform(size=y_shape).astype(dtype)
        z_np = tvm.topi.testing.batch_matmul(x_np, y_np, trans_x=transpose_x, trans_y=transpose_y)

        mod = tvm.ir.IRModule.from_expr(func)
        z = relay.create_executor(executor_kind, mod=mod, device=dev, target=target).evaluate()(
            x_np, y_np
        )
        tvm.testing.assert_allclose(z.numpy(), z_np, rtol=1e-5)


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


def verify_any_dilate(data_shape, strides, static_data_shape, dilation_value=None):
    assert len(data_shape) == len(strides)
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    if dilation_value is None:
        y = relay.nn.dilate(data, strides)
    else:
        y = relay.nn.dilate(data, strides, dilation_value)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_shape = tuple(
        (static_data_shape[i] - 1) * strides[i] + 1 for i in range(len(static_data_shape))
    )
    if dilation_value is None:
        dilation_value = 0.0
    ref_out = np.ones(shape=ref_shape, dtype=dtype)
    ref_out = dilation_value * ref_out
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
    verify_any_dilate(any_dims(4), (3, 7, 1, 5), (1, 2, 3, 4), 1.0)


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


def verify_any_relu(data_shape, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.relu(data)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_relu():
    verify_any_relu(any_dims(3), (1, 2, 3), (1, 2, 3))
    verify_any_relu(any_dims(4), (13, 11, 3, 1), (13, 11, 3, 1))


def verify_any_prelu(data_shape, alpha, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    alpha = relay.const(np.array([alpha]), dtype=dtype)
    y = relay.nn.prelu(data, alpha)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_prelu():
    verify_any_prelu(any_dims(3), 1, (1, 2, 3), (1, 2, 3))
    verify_any_prelu(any_dims(4), 2, (13, 11, 3, 1), (13, 11, 3, 1))


def verify_any_leaky_relu(data_shape, alpha, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.nn.leaky_relu(data, alpha)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_leaky_relu():
    verify_any_leaky_relu(any_dims(3), 0.1, (1, 2, 3), (1, 2, 3))
    verify_any_leaky_relu(any_dims(4), 0.2, (13, 11, 3, 1), (13, 11, 3, 1))


def verify_any_bias_add(data_shape, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    bias = relay.const(np.random.randn(1), dtype=dtype)
    y = relay.nn.bias_add(data, bias)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_bias_add():
    verify_any_bias_add(any_dims(3), (1, 2, 3), (1, 2, 3))
    verify_any_bias_add(any_dims(4), (13, 11, 3, 1), (13, 11, 3, 1))


def verify_any_topk(data_shape, kval, np_dshape, dtype, ret_type="indices", const_k=False):
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
    out = relay.topk(data, k, ret_type=ret_type)
    if ret_type == "both":
        out = out[0]
    mod["main"] = relay.Function(args, out)

    sorted = np.argsort(-np_data)
    if len(np_dshape) == 2:
        ref_out = sorted[:, 0:kval]
    else:
        ref_out = sorted[0:kval]

    check_result(in_vals, mod, ref_out)


@tvm.testing.uses_gpu
def test_any_topk():
    verify_any_topk(any_dims(1), 5, (10,), "float32")
    verify_any_topk(any_dims(2), 2, (6, 3), "int32")
    verify_any_topk(any_dims(2), 3, (6, 3), "float32", const_k=True)
    verify_any_topk(any_dims(1), 0, (0,), "float32", ret_type="both")


def verify_any_get_valid_counts(num_anchor_real, dtype, targets=None):
    mod = tvm.IRModule()
    batch_size = 1
    num_anchor = relay.Any()
    data = relay.var("data", shape=(batch_size, num_anchor, 5), dtype=dtype)
    np_data = np.random.uniform(size=(batch_size, num_anchor_real, 5)).astype(dtype)

    np_out1 = np.zeros(shape=(batch_size,))
    np_out2 = np.zeros(shape=np_data.shape).astype(dtype)
    np_out3 = np.zeros(shape=(batch_size, num_anchor_real))
    score_threshold = 0.95

    for i in range(batch_size):
        np_out1[i] = 0
        inter_idx = 0
        for j in range(num_anchor_real):
            score = np_data[i, j, 0]
            if score > score_threshold:
                for k in range(5):
                    np_out2[i, inter_idx, k] = np_data[i, j, k]
                np_out1[i] += 1
                np_out3[i, inter_idx] = j
                inter_idx += 1
            if j >= np_out1[i]:
                for k in range(5):
                    np_out2[i, j, k] = -1.0
                np_out3[i, j] = -1

    z = relay.vision.get_valid_counts(data, score_threshold, 0, score_index=0)

    mod["main"] = relay.Function([data], z.astuple())

    check_result([np_data], mod, [np_out1, np_out2, np_out3], targets=targets)


@tvm.testing.uses_gpu
def test_any_get_valid_counts():
    verify_any_get_valid_counts(10, "float32")
    # opencl seems to have issues with empty size buffer
    # Check failed: err_code == CL_SUCCESS == false: OpenCL Error,
    # code=-61: CL_INVALID_BUFFER_SIZE
    targets = []
    for tgt, dev in tvm.testing.enabled_targets():
        if "opencl" not in tgt:
            targets.append((tgt, dev))
    verify_any_get_valid_counts(0, "float32", targets=targets)


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


def verify_any_random_strided_slice(
    begin_shape,
    end_shape,
    strides_shape,
    data_shape,
    slice_mode="end",
    const_attrs=False,
):
    # Generate random numpy input data
    np_begin = np.random.randint(2, size=begin_shape, dtype="int32")
    np_end = np.random.randint(5, 10, size=end_shape, dtype="int32")
    np_strides = np.random.randint(
        1, 2 if slice_mode == "size" else 3, size=strides_shape, dtype="int32"
    )

    verify_any_strided_slice(
        np_begin, np_end, np_strides, data_shape, slice_mode=slice_mode, const_attrs=const_attrs
    )


def verify_any_strided_slice(
    np_begin,
    np_end,
    np_strides,
    data_shape,
    axes=None,
    slice_mode="end",
    const_attrs=False,
):
    np_data = np.random.uniform(size=data_shape).astype("float32")
    # target numpy result
    ref_res = tvm.topi.testing.strided_slice_python(
        np_data, np_begin, np_end, np_strides, slice_mode, axes
    )

    # Relay Module
    mod = tvm.IRModule()
    data = relay.var("data", shape=any_dims(len(data_shape)), dtype="float32")
    if const_attrs:
        begin = relay.const(np_begin)
        end = relay.const(np_end)
        strides = relay.const(np_strides)
        args = [data]
        np_inputs = [np_data]
    else:
        begin = relay.var("begin", shape=np_begin.shape, dtype="int32")
        end = relay.var("end", shape=np_end.shape, dtype="int32")
        strides = relay.var("strides", shape=np_strides.shape, dtype="int32")
        args = [data, begin, end, strides]
        np_inputs = [np_data, np_begin, np_end, np_strides]

    y = relay.strided_slice(
        data, begin=begin, end=end, strides=strides, axes=axes, slice_mode=slice_mode
    )
    mod["main"] = relay.Function(args, y)

    check_result(np_inputs, mod, ref_res)


@tvm.testing.uses_gpu
def test_any_strided_slice():
    verify_any_random_strided_slice((2,), (2,), (2,), (15, 21))
    verify_any_random_strided_slice((3,), (3,), (3,), (15, 17, 21))
    verify_any_random_strided_slice((3,), (3,), (3,), (23, 29, 41))
    verify_any_random_strided_slice((4,), (4,), (4,), (40, 50, 60, 70))
    verify_any_random_strided_slice((3,), (3,), (3,), (15, 17, 21), slice_mode="size")
    verify_any_random_strided_slice((2,), (2,), (2,), (15, 21), const_attrs=True)

    begin = np.array([0, 1000000]).astype("int32")
    end = np.array([1000000, -1000000]).astype("int32")
    strides = np.array([1, -1]).astype("int32")
    verify_any_strided_slice(begin, end, strides, (15, 21), const_attrs=False)
    verify_any_strided_slice(begin, end, strides, (15, 21), const_attrs=True)
    verify_any_strided_slice(begin, end, strides, (15, 17, 21), axes=[0, 2], const_attrs=True)


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

    with DiagnosticTesting() as diagnostics:
        diagnostics.assert_message(
            "The Relay type checker is unable to show the following types match:\n"
            "  Tensor[(2, 1), int32]\n"
            "  Tensor[(1, 1), int32]\n"
            "In particular:\n"
            "  dimension 0 conflicts: 2 does not match 1."
        )
        func = infer_type(func)


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


def verify_any_resize2d(data_shape, scale, layout, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    if layout == "NHWC":
        size = (data_shape[1] * scale, data_shape[2] * scale)
    else:
        size = (data_shape[2] * scale, data_shape[3] * scale)
    y = relay.image.resize2d(data, size, None, layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_resize():
    verify_any_resize2d(
        data_shape=(relay.Any(), 4, 4, 4),
        scale=2,
        layout="NHWC",
        static_data_shape=(1, 4, 4, 4),
        ref_out_shape=(1, 8, 8, 4),
    )
    verify_any_resize2d(
        data_shape=(relay.Any(), 8, 17, 20),
        scale=3,
        layout="NCHW",
        static_data_shape=(2, 8, 17, 20),
        ref_out_shape=(2, 8, 51, 60),
    )


def verify_any_grid_sample(data_shape, grid_shape, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var("data", shape=data_shape, dtype=dtype)
    grid = relay.var("grid", shape=grid_shape, dtype=dtype)
    y = relay.image.grid_sample(data, grid)
    mod["main"] = relay.Function([data, grid], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    grid_np = np.random.uniform(size=grid_shape).astype(dtype)
    check_result([data_np, grid_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_grid_sample():
    verify_any_grid_sample(
        data_shape=(relay.Any(), 4, 16, 32),
        grid_shape=(4, 2, 8, 8),
        static_data_shape=(4, 4, 16, 32),
        ref_out_shape=(4, 4, 8, 8),
    )
    verify_any_grid_sample(
        data_shape=(relay.Any(), 4, 16, 32),
        grid_shape=(4, 2, 32, 32),
        static_data_shape=(4, 4, 16, 32),
        ref_out_shape=(4, 4, 32, 32),
    )


def verify_any_affine_grid(num_batch, static_num_batch, target_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data_shape = (num_batch, 2, 3)
    static_data_shape = (static_num_batch, 2, 3)
    data = relay.var("data", shape=data_shape, dtype=dtype)
    y = relay.image.affine_grid(data, target_shape)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    check_result([data_np], mod, ref_out_shape, assert_shape=True)


@tvm.testing.uses_gpu
def test_any_affine_grid():
    verify_any_affine_grid(
        num_batch=relay.Any(),
        static_num_batch=1,
        target_shape=(16, 32),
        ref_out_shape=(1, 2, 16, 32),
    )
    verify_any_affine_grid(
        num_batch=relay.Any(),
        static_num_batch=8,
        target_shape=(32, 32),
        ref_out_shape=(8, 2, 32, 32),
    )


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
    index1 = relay.var("index1", shape=(relay.Any(), 1), dtype="int64")
    out = relay.adv_index([data, index0, index1])
    mod = tvm.IRModule()
    mod["main"] = relay.Function([data, index0, index1], out)
    np_data_shape = (5, 5, 10)
    np_index0_shape = (1, 4)
    np_index1_shape = (4, 1)
    np_data = np.random.uniform(size=np_data_shape).astype("float32")
    np_index0 = np.random.uniform(0, np_data_shape[0], size=np_index0_shape).astype("int64")
    np_index1 = np.random.uniform(0, np_data_shape[0], size=np_index1_shape).astype("int64")
    ref_res = np_data[tuple([np_index0, np_index1])]
    print(ref_res.shape)
    check_result([np_data, np_index0, np_index1], mod, ref_res)


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


def verify_any_where(
    cond_shape, x_shape, y_shape, cond_np_shape, x_np_shape, y_np_shape, y_np_shape_invalid=None
):
    dtype = "float32"
    cond = relay.var("cond", shape=cond_shape, dtype="bool")
    x = relay.var("x", shape=x_shape, dtype=dtype)
    y = relay.var("y", shape=y_shape, dtype=dtype)
    z = relay.where(cond, x, y)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([cond, x, y], z)

    cond_np = np.random.randn(*cond_np_shape) > 0
    x_np = np.random.randn(*x_np_shape).astype(dtype)
    y_np = np.random.randn(*y_np_shape).astype(dtype)
    expected = np.where(cond_np, x_np, y_np)

    check_result([cond_np, x_np, y_np], mod, expected)

    # verify invalid broadcasting check
    if y_np_shape_invalid:
        y_np_bad = np.random.randn(*y_np_shape_invalid).astype(dtype)
        try:
            check_result([cond_np, x_np, y_np_bad], mod, expected)
        except tvm.error.TVMError as e:
            error_msg = str(e).split("\n")[-1]
            assert "Invalid broadcast shapes" in error_msg


@tvm.testing.uses_gpu
def test_any_where():
    verify_any_where(any_dims(1), (5,), (5,), (5,), (5,), (5,))
    verify_any_where(any_dims(1), any_dims(1), (5,), (5,), (5,), (5,))
    verify_any_where(any_dims(1), any_dims(1), any_dims(1), (5,), (5,), (5,))
    verify_any_where((5,), any_dims(1), any_dims(1), (5,), (5,), (5,))

    # where with broadcast
    verify_any_where(any_dims(1), any_dims(1), any_dims(1), (5,), (1,), (5,))
    verify_any_where(any_dims(1), any_dims(2), any_dims(2), (5,), (5, 5), (5, 5))
    verify_any_where(any_dims(1), any_dims(1), any_dims(2), (5,), (5,), (5, 5))
    verify_any_where(
        any_dims(2), any_dims(2), any_dims(2), (3, 4), (3, 1), (1, 4), y_np_shape_invalid=(2, 4)
    )

    # Test scalar where in a dynamically shaped graph
    x = relay.var("x", shape=any_dims(1), dtype="int64")
    y = relay.var("y", shape=any_dims(2), dtype="float32")

    left = relay.take(x, relay.const(1, dtype="int32")) + relay.const(4, "int64")
    right = relay.const(4, "int64")
    where = relay.where(relay.const(False, "bool"), left, right)
    z = relay.take(y, where, axis=1)

    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], z)

    x_np = np.random.randn(2).astype("int64")
    y_np = np.random.randn(2, 6).astype("float32")
    expected = y_np[:, 4]

    check_result([x_np, y_np], mod, expected)


@tvm.testing.uses_gpu
def test_non_max_suppression():
    x0 = relay.var("x0", relay.ty.TensorType((1, relay.Any(), 6), "float32"))
    x1 = relay.var("x1", relay.ty.TensorType((1,), "int32"))
    x2 = relay.var("x2", relay.ty.TensorType((1, relay.Any()), "int32"))
    x3 = relay.var("x3", relay.ty.TensorType((), "int32"))
    z = relay.vision.non_max_suppression(
        x0,
        x1,
        x2,
        x3,
        iou_threshold=0.5,
        force_suppress=True,
        top_k=2,
        return_indices=True,
        invalid_to_bottom=False,
    )
    z = z.astuple()
    func = relay.Function([x0, x1, x2, x3], z)
    mod = tvm.IRModule()
    mod["main"] = func

    np_data = np.array(
        [
            [
                [0, 0.8, 1, 20, 25, 45],
                [1, 0.7, 30, 60, 50, 80],
                [0, 0.4, 4, 21, 19, 40],
                [2, 0.9, 35, 61, 52, 79],
                [1, 0.5, 100, 60, 70, 110],
            ]
        ]
    ).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 3, 4, -1]]).astype("int32")
    np_max_output_size = -1
    np_indices_result = np.array([[4, 0, -1, -1, -1]])
    np_valid_box_count = np.array([[2]]).astype("int32")

    check_result(
        [np_data, np_valid_count, np_indices, np_max_output_size],
        mod,
        [np_indices_result, np_valid_box_count],
        only_vm=False,
    )

    np_data = np.zeros((1, 0, 6)).astype("float32")
    np_valid_count = np.array([0]).astype("int32")
    np_indices = np.zeros((1, 0)).astype("int32")
    np_max_output_size = -1
    np_indices_result = np.zeros((1, 0))
    np_valid_box_count = np.array([[0]]).astype("int32")

    check_result(
        [np_data, np_valid_count, np_indices, np_max_output_size],
        mod,
        [np_indices_result, np_valid_box_count],
        only_vm=False,
    )


@tvm.testing.uses_gpu
def test_all_class_non_max_suppression():
    def verify_all_class_non_max_suppression(
        boxes_np,
        scores_np,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        expected,
        output_format="onnx",
    ):
        batch_size = boxes_np.shape[0]
        num_classes = scores_np.shape[1]
        num_boxes = relay.Any()
        boxes = relay.var("boxes", relay.ty.TensorType((batch_size, num_boxes, 4), "float32"))
        scores = relay.var(
            "scores", relay.ty.TensorType((batch_size, num_classes, num_boxes), "float32")
        )

        nms_out = relay.vision.all_class_non_max_suppression(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_format
        )

        if output_format == "onnx":
            three = relay.const(np.array([3]), dtype="int64")
            begin = relay.const(np.array([0, 0]), dtype="int64")
            end = relay.op.concatenate([nms_out[1], three], axis=0)
            strides = relay.const(np.array([1, 1]), dtype="int64")
            out = relay.op.strided_slice(nms_out[0], begin, end, strides)
            mod = tvm.IRModule()
            mod["main"] = relay.Function([boxes, scores], out)
            check_result([boxes_np, scores_np], mod, [expected])
        else:
            out = nms_out.tuple_value
            mod = tvm.IRModule()
            mod["main"] = relay.Function([boxes, scores], out)
            check_result([boxes_np, scores_np], mod, expected)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.5, 0.5, 0.4, 0.4],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.9, 0.9],
                [0.5, 0.5, 1.0, 1.0],
            ],
        ]
    ).astype("float32")

    scores = np.array(
        [
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.8, 0.2, 0.6, 0.3, 0.9]],
        ]
    ).astype("float32")

    max_output_boxes_per_class = 2
    iou_threshold = 0.8
    score_threshold = 0.4

    expected = np.array([[0, 0, 4], [0, 0, 2], [0, 1, 4], [0, 1, 0]])

    verify_all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, expected
    )

    expected = [
        np.array(
            [[[0, 4], [0, 2], [1, 4], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]]
        ),
        np.array(
            [
                [
                    0.9,
                    0.6,
                    0.9,
                    0.8,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ]
        ),
        np.array([4]),
    ]

    verify_all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        expected,
        output_format="tensorflow",
    )

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 0.9, 1.2],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.2, 0.3], [0.3, 0.2]]]).astype(np.float32)
    iou_threshold = 0.3
    score_threshold = 0.15

    expected = np.array([[0, 0, 1], [0, 1, 0]])

    verify_all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, expected
    )

    # zero box detection case
    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.2]]]).astype(np.float32)
    score_threshold = 0.4

    expected = np.zeros((0, 3))

    verify_all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, expected
    )


@tvm.testing.uses_gpu
def test_gather_nd():
    def verify_gather_nd(data_shape, indices_shape, data_shape_np, indices_shape_np, batch_dims=0):
        x = relay.var("x", relay.TensorType(data_shape, "float32"))
        y = relay.var("y", relay.TensorType(indices_shape, "int32"))
        z = relay.gather_nd(x, y, batch_dims=batch_dims, index_rank=indices_shape[0])

        mod = tvm.IRModule()
        mod["main"] = relay.Function([x, y], z)

        data_np = np.random.uniform(size=data_shape_np).astype("float32")
        indices_np = np.random.randint(low=0, high=2, size=indices_shape_np, dtype="int32")

        ref_res = ref_funcs.gather_nd(data_np, indices_np, batch_dims)
        check_result([data_np, indices_np], mod, [ref_res])

    verify_gather_nd((2, 2), (2, relay.Any()), (2, 2), (2, 3))
    verify_gather_nd((relay.Any(), 2), (2, relay.Any()), (2, 2), (2, 3))
    verify_gather_nd((relay.Any(), 2), (1, relay.Any()), (10, 2), (1, 10), 1)
    verify_gather_nd(
        (relay.Any(), 2, 2, 3, 4), (3, relay.Any(), relay.Any()), (3, 2, 2, 3, 4), (3, 3, 2), 2
    )


@tvm.testing.uses_gpu
def test_scatter_nd():
    def verify_scatter_nd(data_np, indices_np, updates_np, ref_res):
        indices_shape = (2, relay.Any())
        updates_shape = (relay.Any(),)
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices = relay.var("indices", relay.TensorType(indices_shape, str(indices_np.dtype)))
        updates = relay.var("updates", relay.TensorType(updates_shape, str(updates_np.dtype)))

        out = relay.op.scatter_nd(data, indices, updates, "add")

        mod = tvm.IRModule()
        mod["main"] = relay.Function([data, indices, updates], out)

        check_result([data_np, indices_np, updates_np], mod, [ref_res])

    data = np.zeros((2, 2)).astype("int64")
    indices = np.array([[1, 1, 0], [0, 1, 0]])
    updates = np.array([2, 3, 0])
    out = np.array([[0, 0], [2, 3]])
    verify_scatter_nd(data, indices, updates, out)


@tvm.testing.uses_gpu
def test_scatter_nd_any_updates():
    def verify_scatter_nd_any_updates(data_np, indices_np, updates_np, ref_res):
        indices_shape = (2, relay.Any())
        updates_shape = (2, relay.Any())
        data = relay.var("data", shape=data_np.shape, dtype=str(data_np.dtype))
        indices = relay.var("indices", relay.TensorType(indices_shape, str(indices_np.dtype)))
        updates = relay.var("updates", relay.TensorType(updates_shape, str(updates_np.dtype)))

        out = relay.op.scatter_nd(data, indices, updates, "add")

        mod = tvm.IRModule()
        mod["main"] = relay.Function([data, indices, updates], out)

        check_result([data_np, indices_np, updates_np], mod, [ref_res], only_vm=True)

    data = np.zeros((3, 3)).astype("int64")
    indices = np.array([[1, 1], [0, 1]])
    updates = np.array([[2, 2], [1, 1]])
    out = np.array([[0, 0, 0], [0, 0, 0], [2, 2, 1]])
    verify_scatter_nd_any_updates(data, indices, updates, out)


@tvm.testing.uses_gpu
def test_gather():
    def verify_gather(data_shape, indices_shape, data_shape_np, indices_shape_np, axis):
        x = relay.var("x", relay.TensorType(data_shape, "float32"))
        y = relay.var("y", relay.TensorType(indices_shape, "int32"))
        z = relay.gather(x, axis, y)

        mod = tvm.IRModule()
        mod["main"] = relay.Function([x, y], z)

        data_np = np.random.uniform(size=data_shape_np).astype("float32")
        indices_np = np.random.randint(low=0, high=2, size=indices_shape_np, dtype="int32")

        ref_res = tvm.topi.testing.gather_python(data_np, axis, indices_np)
        check_result([data_np, indices_np], mod, [ref_res])

    verify_gather((relay.Any(),), (relay.Any(),), (10,), (10,), 0)
    verify_gather((2, 2), (2, relay.Any()), (2, 2), (2, 3), 1)
    verify_gather((relay.Any(), 2), (2, relay.Any()), (2, 2), (2, 3), 1)
    verify_gather((relay.Any(), relay.Any()), (relay.Any(), relay.Any()), (2, 3), (1, 3), 0)


@tvm.testing.uses_gpu
def test_searchsorted():
    def verify_searchsorted(
        sorted_sequence_shape, values_shape, sorted_sequence_shape_np, values_shape_np
    ):
        x = relay.var("x", relay.TensorType(sorted_sequence_shape, "float32"))
        y = relay.var("y", relay.TensorType(values_shape, "float32"))
        z = relay.searchsorted(x, y)

        mod = tvm.IRModule()
        mod["main"] = relay.Function([x, y], z)

        x_np = np.sort(np.random.uniform(size=sorted_sequence_shape_np).astype("float32"), axis=-1)
        y_np = np.random.uniform(size=values_shape_np).astype("float32")

        ref_res = searchsorted_ref(x_np, y_np, False, "int32")
        check_result([x_np, y_np], mod, [ref_res])

    for shape_np, values_shape_np in zip([(8, 9, 10), (10,), (11,)], [(8, 9, 20), (5,), (8, 9, 7)]):
        sorted_sequence_shape = (relay.Any(),) * len(shape_np)
        values_shape = (relay.Any(),) * len(values_shape_np)

        verify_searchsorted(
            sorted_sequence_shape,
            values_shape,
            shape_np,
            values_shape_np,
        )


if __name__ == "__main__":
    tvm.testing.main()
