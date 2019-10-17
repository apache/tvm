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
    mod = relay.module.Module()
    mod["main"] = relay.Function([x, y], op(x, y))
    x_np = np.random.uniform(size=x_np_shape).astype(dtype)
    y_np = np.random.uniform(size=y_np_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(x_np, y_np)
        tvm.testing.assert_allclose(result.asnumpy(), np_op(x_np, y_np))

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


def test_any_concat():
    x = relay.var('x', shape=(relay.Any(), 2), dtype="float32")
    y = relay.var('y', shape=(1, 2), dtype="float32")
    z = relay.op.concatenate([x, y], axis=0)
    mod = relay.module.Module()
    mod["main"] = relay.Function([x, y], z)
    x_np = np.random.uniform(size=(3, 2)).astype('float32')
    y_np = np.random.uniform(size=(1, 2)).astype('float32')
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(x_np, y_np)
        ref = np.concatenate([x_np, y_np], axis=0)
        tvm.testing.assert_allclose(result.asnumpy(), ref)

def verify_any_reshape(x_shape, newshape, x_np_shape, out_shape):
    x = relay.var('x', shape=x_shape, dtype="float32")
    y = relay.reshape(x, newshape=newshape)
    mod = relay.module.Module()
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
    mod = relay.module.Module()
    mod["main"] = relay.Function([x], y)
    data = np.random.choice([0, 1, 2, 3], size=x_np_shape).astype(dtype)
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data).asnumpy()
        expected = np.argwhere(data)
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
    mod = relay.Module()
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

def test_any_shape_of():
    x = relay.var('x', shape=any_dims(2), dtype='float32')
    y = relay.shape_of(x)
    mod = relay.module.Module()
    mod["main"] = relay.Function([x], y)
    data = np.random.uniform(size=(3, 4)).astype('float32')
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), np.array([3,4]).astype("int64"))

    x = relay.var('x', shape=any_dims(3), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(1, 'int32'))
    mod = relay.module.Module()
    mod["main"] = relay.Function([x], y1)
    data = np.random.uniform(size=(2, 3, 4)).astype('float32')
    for kind in ["debug", "vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), np.array(3).astype("int64"))

def test_fused_ops():
    x = relay.var('x', shape=(relay.Any(), relay.Any()), dtype='float32')
    y0 = x + relay.const(1.0, 'float32')
    y1 = y0 * relay.const(2.0, 'float32')
    mod = relay.module.Module()
    mod["main"] = relay.Function([x], y1)
    data = np.random.uniform(size=(5, 4)).astype('float32')
    for kind in ["vm"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        tvm.testing.assert_allclose(result.asnumpy(), (data + 1) * 2)

def test_arange_with_dynamic_shape():
    m, n, k = relay.ShapeVar('m'), relay.ShapeVar('n'), relay.ShapeVar('k')
    x = relay.var('x', shape=(m.var, n.var, k.var), dtype='float32')
    y0 = relay.shape_of(x)
    y1 = relay.take(y0, relay.const(0, 'int32'))
    y2 = relay.op.arange(y1, dtype="int32")
    y3 = y2 + relay.const(1, dtype="int32")
    data = np.random.rand(10, 5, 3).astype('float32')
    mod = relay.module.Module()
    mod["main"] = relay.Function([x], y3, type_params=[m, n, k])
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
    mod = relay.module.Module()
    mod["main"] = func
    data = np.array(0.0, dtype='int32')
    # TODO(@jroesch): After LambdaLift pass, TypeInfer pass will fail
    # so currently we cannot run this test case on VM
    for kind in ["debug"]:
        ex = relay.create_executor(kind, mod=mod, ctx=tvm.cpu(), target="llvm")
        result = ex.evaluate()(data)
        ref = np.array([0] + list(range(10))).reshape((11, 1)).astype("int32")
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
    test_any_broadcast()
    test_any_broadcast_fail()
    test_any_concat()
    test_any_reshape()
    test_any_take()
    test_any_shape_of()
    test_fused_ops()
    test_arange_with_dynamic_shape()
    test_recursive_concat()
    test_recursive_concat_with_wrong_annotation()
