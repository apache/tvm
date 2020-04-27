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

def int32(val):
    return relay.const(val, 'int32')

def any_dims(ndim):
    shape = []
    for _ in range(ndim):
        shape.append(relay.Any())
    return tuple(shape)

# TODO(@wweic): because vm doesn't support heterogeneous exec, we can only test
# shape function on CPU.

def verify_any_broadcast(x_shape, y_shape, x_np_shape, y_np_shape, op, np_op):
    dtype = 'float32'
    x = relay.var('x', shape=x_shape, dtype=dtype)
    y = relay.var('y', shape=y_shape, dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], op(x, y))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    y_np = np.random.uniform(size=y_np_shape).astype(dtype)
    res_np = np_op(x_np, y_np)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(x_np, y_np)
        tvm.testing.assert_allclose(result.asnumpy(), res_np)

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
    dtype = 'float32'
    x = relay.var('x', shape=x_shape, dtype=dtype)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], op(x))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    res_np = np_op(x_np)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(x_np)
        tvm.testing.assert_allclose(result.asnumpy(), res_np)

def test_any_elemwise():
    verify_any_elemwise((relay.Any(),), (3,), relay.sqrt, np.sqrt)
    verify_any_elemwise((relay.Any(), 2), (5, 2), relay.negative, np.negative)
    verify_any_elemwise((relay.Any(), relay.Any()), (5, 4), relay.exp, np.exp)

def test_any_broadcast_fail():
    # Test broadcast with incompatible values at runtime
    def check_fail(x_shape, y_shape, x_np_shape, y_np_shape, op, np_op):
        try:
            verify_any_broadcast(
                x_shape, y_shape, x_np_shape, y_np_shape, op, np_op)
        except tvm._ffi.base.TVMError:
            pass
        else:
            assert False

    check_fail((relay.Any(),), (3, 2), (1,), (4, 2), relay.add, np.add)
    check_fail((relay.Any(), 2), (3, 2), (4, 2), (4, 2), relay.add, np.add)
    check_fail((relay.Any(), 2), (3, relay.Any()), (1, 2), (4, 1), relay.add, np.add)
    check_fail((relay.Any(), 2), (3, 3), (1, 3), (3, 3), relay.add, np.add)
    check_fail((relay.Any(),), (3, 2), (2), (4, 2), relay.add, np.add)


def verify_any_full(x_shape, x_np_shape, relay_op, np_op, dtype='float32'):
    x = relay.var('x', shape=x_shape, dtype=dtype)
    mod = tvm.IRModule()
    mod['main'] = relay.Function([x], relay.zeros_like(x))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    res_np = np.zeros_like(x_np)
    for kind in ['debug', 'vm']:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target='llvm')
        result = ex.evaluate()(x_np).asnumpy()
        tvm.testing.assert_allclose(result, res_np)

def test_any_full():
    # zeros, zeros_like, ones, ones_like
    verify_any_full(any_dims(3), (2, 3, 5), relay.zeros, np.zeros, "float32")
    verify_any_full(any_dims(3), (225, 115, 15), relay.zeros, np.zeros, "float32")
    verify_any_full(any_dims(5), (10, 11, 12, 13, 14), relay.zeros, np.zeros, "int32")
    verify_any_full(any_dims(3), (2, 3, 5), relay.zeros_like, np.zeros_like, "float32")
    verify_any_full(any_dims(3), (225, 115, 15), relay.zeros_like, np.zeros_like, "float32")
    verify_any_full(any_dims(5), (10, 11, 12, 13, 14), relay.zeros_like, np.zeros_like, "int32")
    verify_any_full(any_dims(3), (2, 3, 5), relay.ones, np.ones, "float32")
    verify_any_full(any_dims(3), (225, 115, 15), relay.ones, np.ones, "float32")
    verify_any_full(any_dims(5), (10, 11, 12, 13, 14), relay.ones, np.ones, "int32")
    verify_any_full(any_dims(3), (2, 3, 5), relay.ones_like, np.ones_like, "float32")
    verify_any_full(any_dims(3), (225, 115, 15), relay.ones_like, np.ones_like, "float32")
    verify_any_full(any_dims(5), (10, 11, 12, 13, 14), relay.ones_like, np.ones_like, "int32")

def test_any_concat():
    x = relay.var('x', shape=(relay.Any(), 2), dtype="float32")
    y = relay.var('y', shape=(1, 2), dtype="float32")
    xx = x - relay.expr.const(3.0)
    yy = y * relay.expr.const(5.0)
    z = relay.op.concatenate([xx, yy], axis=0)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x, y], z)
    x_np = np.random.uniform(size=(3, 2)).astype('float32')
    y_np = np.random.uniform(size=(1, 2)).astype('float32')
    ref = np.concatenate([x_np - 3.0, y_np * 5.0], axis=0)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(x_np, y_np)
        tvm.testing.assert_allclose(result.asnumpy(), ref)

def verify_any_reshape(x_shape, newshape, x_np_shape, out_shape):
    x = relay.var('x', shape=x_shape, dtype="float32")
    y = relay.reshape(x, newshape=newshape)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    data = np.random.uniform(size=x_np_shape).astype('float32')
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data).asnumpy()
        assert result.shape == out_shape
        tvm.testing.assert_allclose(result.flatten(), data.flatten())

def test_any_reshape():
    verify_any_reshape(any_dims(3), (1, -1), (2, 3, 4), (1, 24))
    verify_any_reshape(any_dims(3), (0, -1), (2, 3, 4), (2, 12))
    verify_any_reshape(any_dims(3), (0, -2), (2, 3, 4), (2, 3, 4))
    verify_any_reshape(any_dims(3), (-4, 2, -1, -2), (6, 3, 4), (2, 3, 3, 4))
    verify_any_reshape(any_dims(3), (-4, -1, 2, -3), (6, 3, 4), (3, 2, 12))

def verify_any_argwhere(x_shape, x_np_shape, dtype="bool"):
    x = relay.var('x', shape=x_shape, dtype=dtype)
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
    data = relay.var('data', shape=data_shape, dtype='float32')
    indices = relay.var('indices', shape=indices_shape, dtype='int32')
    y = relay.take(data, indices, axis=axis)
    mod["main"] = relay.Function([data, indices], y)
    data_np = np.random.uniform(size=data_np_shape).astype('float32')
    if axis is None:
        max_index = data_np.size
    else:
        max_index = data_np.shape[axis]
    indices_np = np.random.randint(max_index, size=indices_np_shape).astype('int32')
    ref = np.take(data_np, indices_np, axis=axis)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np, indices_np)
        tvm.testing.assert_allclose(result.asnumpy(), ref)

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

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        res = ex.evaluate()(x_data)
        tvm.testing.assert_allclose(res.asnumpy(), ref_res, rtol=1e-5)

def test_any_tile():
    verify_any_tile(any_dims(3), (3, 2, 1), (2, 3, 4), (3, 2, 1))
    verify_any_tile(any_dims(3), (1, 2), (2, 3, 4), (1, 2))
    verify_any_tile(any_dims(2), (3, 2, 1), (2, 3), (3, 2, 1))
    verify_any_tile(any_dims(3), (1,), (2, 3, 4), (1,))

def test_any_shape_of():
    x = relay.var('x', shape=any_dims(2), dtype='float32')
    y = relay.shape_of(x)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    data = np.random.uniform(size=(3, 4)).astype('float32')
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), np.array([3,4]).astype("int64"))

    x = relay.var('x', shape=any_dims(3), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(1, 'int32'))
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y1)
    data = np.random.uniform(size=(2, 3, 4)).astype('float32')
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), np.array(3).astype("int64"))

def verify_any_reduce(reduce_op, data_shape, axis, exclude, keepdims,
                      static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "bool" if reduce_op == relay.all else "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = reduce_op(data, axis, keepdims, exclude)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_reduce():
    verify_any_reduce(relay.argmax, any_dims(3), None, False, False, (3, 4, 5), ())
    verify_any_reduce(relay.argmin, any_dims(4), 1, False, True, (3, 4, 5, 6), (3, 1, 5, 6))
    verify_any_reduce(relay.all, any_dims(3), (1, 2), True, False, (3, 4, 5), (4, 5))
    verify_any_reduce(relay.max, any_dims(4), -1, True, True, (3, 4, 5, 6), (1, 1, 1, 6))
    verify_any_reduce(relay.min, any_dims(3), (0, 1), False, False, (4, 5, 6), (6,))
    verify_any_reduce(relay.prod, any_dims(4), 2, True, True, (3, 4, 5, 6), (1, 1, 5, 1))
    verify_any_reduce(relay.mean, any_dims(2), 0, False, False, (1, 2), (2,))
    verify_any_reduce(relay.variance, any_dims(5), (2, 4), False, False, (3, 4, 5, 6, 7), (3, 4, 6))

def verify_any_layout_transform(data_shape, src_layout, dst_layout, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.layout_transform(data, src_layout, dst_layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_layout_transform():
    verify_any_layout_transform(any_dims(4), "NCHW", "NHWC", (3, 4, 5, 6), (3, 5, 6, 4))
    verify_any_layout_transform(any_dims(5), "NCHW16c", "NCHW2c", (1, 2, 8, 8, 16), (1, 16, 8, 8, 2))
    verify_any_layout_transform(any_dims(5), "NCHW6n", "NHWC", (3, 4, 5, 6, 6), (18, 5, 6, 4))
    verify_any_layout_transform(any_dims(4), "NCHW", "NCHW4c", (3, 4, 5, 6), (3, 1, 5, 6, 4))
    verify_any_layout_transform((16, 1), "CH", "C4cH", (16, 1), (4, 4, 1))

def verify_any_expand_dims(data_shape, axis, num_newaxis, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.expand_dims(data, axis=axis, num_newaxis=num_newaxis)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_expand_dims():
    verify_any_expand_dims(any_dims(3), 1, 2, (1, 2, 3), (1, 1, 1, 2, 3))
    verify_any_expand_dims(any_dims(3), -1, 2, (1, 2, 3), (1, 2, 3, 1, 1))

def verify_any_transpose(data_shape, axes, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.transpose(data, axes=axes)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.transpose(data_np, axes)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        tvm.testing.assert_allclose(result.asnumpy(), ref_out)

def test_any_transpose():
    verify_any_transpose(any_dims(3), (1, 0, 2), (10, 3, 2))
    verify_any_transpose(any_dims(3), None, (2, 3, 4))
    verify_any_transpose(any_dims(6), (0, 1, 3, 2, 5, 4), (11, 12, 2, 1, 9, 17))

def verify_any_squeeze(data_shape, axis, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.squeeze(data, axis=axis)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.squeeze(data_np, axis)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        tvm.testing.assert_allclose(result.asnumpy(), ref_out)

def test_any_squeeze():
    verify_any_squeeze((1, relay.Any(), relay.Any()), (0,), (1, 9, 8))
    verify_any_squeeze((1, relay.Any(), relay.Any(), 1, relay.Any(), relay.Any()), (0, 3), (1, 12, 2, 1, 9, 17))

def test_any_reshape_like():
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=(relay.Any(), 3, 10), dtype=dtype)
    shape_like = relay.var('data', shape=(relay.Any(), 5, 6), dtype=dtype)
    y = relay.reshape_like(data, shape_like)
    mod["main"] = relay.Function([data, shape_like], y)
    data_np = np.random.uniform(size=(3, 3, 10)).astype(dtype)
    shape_like_np = np.random.uniform(size=(3, 5, 6)).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np, shape_like_np)
        assert result.asnumpy().shape == shape_like_np.shape, \
            "Shape mismatch: expect %s but got %s." % (str(shape_like_np.shape), str(result.asnumpy().shape))

def verify_any_conv2d_NCHWc(data_shape, kernel_shape, strides, padding, dilation,
                            data_layout, kernel_layout, out_layout,
                            static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    kernel = relay.var('kernel', shape=kernel_shape, dtype=dtype)
    y = relay.nn.contrib_conv2d_nchwc(data, kernel, strides, padding, dilation,
                                      kernel_size=kernel_shape[2:4],
                                      channels=kernel_shape[0]*kernel_shape[-1],
                                      data_layout=data_layout, kernel_layout=kernel_layout,
                                      out_layout=out_layout)
    mod["main"] = relay.Function([data, kernel], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    kernel_np = np.random.uniform(size=kernel_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np, kernel_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

# TODO(@kevinthesun): Need to fix the compute in conv2d_NCHWc to support any
@pytest.mark.skip
def test_any_conv2d_NCHWc():
    verify_any_conv2d_NCHWc((relay.Any(), 8, relay.Any(), relay.Any(), 8), (8, 8, 3, 3, 8, 8), (1, 1), (1, 1), (1, 1),
                            "NCHW8c", "OIHW8i8o", "NCHW8c", (1, 8, 224, 224, 8), (1, 8, 224, 224, 8))
    verify_any_conv2d_NCHWc((relay.Any(), 8, relay.Any(), relay.Any(), 8), (8, 8, 3, 3, 8, 8), (1, 1), (1, 1), (2, 2),
                            "NCHW8c", "OIHW8i8o", "NCHW8c", (1, 8, 224, 224, 8), (1, 8, 222, 222, 8))

def verify_any_pool2d(pool_type, data_shape, pool_size, strides, padding,
                      layout, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    pool_func = relay.nn.max_pool2d if pool_type == "max" else relay.nn.avg_pool2d
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = pool_func(data, pool_size, strides, padding, layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_pool2d():
    verify_any_pool2d("max", (relay.Any(), 3, relay.Any(), relay.Any()),
                      (3, 3), (1, 1), (1, 1), "NCHW", (2, 3, 220, 220), (2, 3, 220, 220))
    verify_any_pool2d("avg", (relay.Any(), relay.Any(), relay.Any(), 4),
                      (1, 1), (2, 2), (0, 0), "NHWC", (3, 220, 220, 4), (3, 110, 110, 4))
    verify_any_pool2d("max", (relay.Any(), 3, relay.Any(), relay.Any(), 4),
                      (3, 3), (2, 2), (1, 1), "NCHW4c", (2, 3, 220, 220, 4), (2, 3, 110, 110, 4))

def verify_any_global_pool2d(pool_type, data_shape, layout, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    pool_func = relay.nn.global_max_pool2d if pool_type == "max" else relay.nn.global_avg_pool2d
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = pool_func(data, layout)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_global_pool2d():
    verify_any_global_pool2d("max", (relay.Any(), 3, relay.Any(), relay.Any()),
                      "NCHW", (2, 3, 220, 220), (2, 3, 1, 1))
    verify_any_global_pool2d("avg", (relay.Any(), relay.Any(), relay.Any(), 4),
                      "NHWC", (3, 220, 220, 4), (3, 1, 1, 4))
    verify_any_global_pool2d("max", (relay.Any(), 3, relay.Any(), relay.Any(), 4),
                      "NCHW4c", (2, 3, 220, 220, 4), (2, 3, 1, 1, 4))

def verify_any_split(data_shape, indices_or_sections, axis, static_data_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.split(data, indices_or_sections, axis)
    mod["main"] = relay.Function([data], y.astuple())
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        for ret, ref_ret in zip(result, ref_out_shape):
            assert ret.asnumpy().shape == ref_ret, \
                "Shape mismatch: expect %s but got %s." % (str(ref_ret), str(ret.asnumpy().shape))

def test_any_split():
    verify_any_split((relay.Any(), 4), 2, 1, (9, 4), [(9, 2), (9, 2)])
    verify_any_split((relay.Any(), 12), (1, 4, 8), 1, (7, 12), [(7, 1), (7, 3), (7, 4)])

def test_any_batch_flatten():
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=any_dims(3), dtype=dtype)
    y = relay.nn.batch_flatten(data)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=(3, 3, 10)).astype(dtype)
    ref_out_shape = (3, 30)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def verify_any_dense(data_shape, weight_shape, units, static_data_shape,
                     static_weight_shape, ref_out_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    weight = relay.var('weight', shape=weight_shape, dtype=dtype)
    y = relay.nn.dense(data, weight, units)
    mod["main"] = relay.Function([data, weight], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    weight_np = np.random.uniform(size=static_weight_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np, weight_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_dense():
    verify_any_dense(any_dims(2), any_dims(2), None, (4, 16), (8, 16), (4, 8))
    verify_any_dense(any_dims(2), (50, relay.Any()), 50, (4, 40), (50, 40), (4, 50))

def verify_any_pad(data_shape, pad_width, static_data_shape):
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.nn.pad(data, pad_width)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_out = np.pad(data_np, pad_width)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        tvm.testing.assert_allclose(result.asnumpy(), ref_out)

def test_any_pad():
    verify_any_pad(any_dims(3), ((0, 0), (1, 1), (2, 2)), (1, 2, 3))
    verify_any_pad(any_dims(4), ((1, 0), (1, 3), (0, 2), (9, 0)), (13, 11, 3, 1))

def verify_any_dilate(data_shape, strides, static_data_shape):
    assert len(data_shape) == len(strides)
    mod = tvm.IRModule()
    dtype = "float32"
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.nn.dilate(data, strides)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    ref_shape = tuple((static_data_shape[i] - 1) * strides[i] + 1
                      for i in range(len(static_data_shape)))
    ref_out = np.zeros(shape=ref_shape, dtype=dtype)
    ref_out[tuple(slice(None, None, strides[i]) for i in range(len(data_shape)))] = data_np

    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        tvm.testing.assert_allclose(result.asnumpy(), ref_out)

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
    data = relay.var('data', shape=data_shape, dtype=dtype)
    y = relay.nn.softmax(data, axis)
    mod["main"] = relay.Function([data], y)
    data_np = np.random.uniform(size=static_data_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data_np)
        assert result.asnumpy().shape == ref_out_shape, \
            "Shape mismatch: expect %s but got %s." % (str(ref_out_shape), str(result.asnumpy().shape))

def test_any_softmax():
    verify_any_softmax(any_dims(3), -1, (1, 2, 3), (1, 2, 3))
    verify_any_softmax(any_dims(4), 2, (13, 11, 3, 1), (13, 11, 3, 1))

def test_fused_ops():
    x = relay.var('x', shape=(relay.Any(), relay.Any()), dtype='float32')
    y0 = x + relay.const(1.0, 'float32')
    y1 = y0 * relay.const(2.0, 'float32')
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y1)
    data = np.random.uniform(size=(5, 4)).astype('float32')
    for kind in ["vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), (data + 1) * 2)

def test_arange_with_dynamic_shape():
    # m, n, k = relay.ShapeVar('m'), relay.ShapeVar('n'), relay.ShapeVar('k')
    m, n, k = relay.Any(), relay.Any(), relay.Any()
    x = relay.var('x', shape=(m, n, k), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(0, 'int32'))
    y2 = relay.op.arange(y1, dtype="int32")
    y3 = y2 + relay.const(1, dtype="int32")
    data = np.random.rand(10, 5, 3).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y3)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), np.array(range(10)).astype("int32")+1)

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
    i = relay.var('i', shape=(), dtype='int32')
    st = relay.var('st', shape=(relay.Any(), 1), dtype='int32')

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1,1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = while_loop(_cond, [i, st], _body)
    start = relay.var('start', shape=(), dtype='int32')
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    mod = tvm.IRModule()
    mod["main"] = func
    data = np.array(0.0, dtype='int32')
    ref = np.array([0] + list(range(10))).reshape((11, 1)).astype("int32")
    # TODO(@jroesch): After LambdaLift pass, TypeInfer pass will fail
    # so currently we cannot run this test case on VM
    for kind in ["debug"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        np.testing.assert_allclose(result.asnumpy(), ref)

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
    i = relay.var('i', shape=(), dtype='int32')
    st = relay.var('st', shape=(1, 1), dtype='int32')

    def _cond(i, st):
        return relay.op.min(relay.op.less(i, int32(10)))

    def _body(i, st):
        i_vec = relay.op.reshape(i, (1,1))
        ret = relay.op.concatenate([st, i_vec], axis=0)
        return i + int32(1), ret

    loop = while_loop(_cond, [i, st], _body)
    start = relay.var('start', shape=(), dtype='int32')
    body = loop(start, relay.op.reshape(relay.const(0), newshape=(1, 1)))
    func = relay.Function([start], relay.TupleGetItem(body, 1))
    try:
        func = infer_type(func)
        assert False
    except Exception as e:
        assert "in particular dimension 0 conflicts 2 does not match 1" in str(e)

if __name__ == "__main__":
    test_any_full()
    test_any_broadcast()
    test_any_elemwise()
    test_any_broadcast_fail()
    test_any_concat()
    test_any_reshape()
    test_any_take()
    test_any_tile()
    test_any_split()
    test_any_shape_of()
    test_any_reduce()
    test_any_layout_transform()
    test_any_expand_dims()
    test_any_transpose()
    test_any_squeeze()
    test_any_reshape_like()
    test_any_conv2d_NCHWc()
    test_any_pool2d()
    test_any_global_pool2d()
    test_any_batch_flatten()
    test_any_dense()
    test_any_pad()
    test_any_softmax()
    test_fused_ops()
    test_arange_with_dynamic_shape()
    test_recursive_concat()
    test_recursive_concat_with_wrong_annotation()
