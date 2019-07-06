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
import tvm
import numpy as np
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import ctx_list
import topi.testing

def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def test_binary_op():
    def check_binary_op(opfunc, ref):
        n = tvm.var("n")
        t1 = relay.TensorType((5, n, 5))
        t2 = relay.TensorType((n, 1))
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        # test printer
        assert ("{}(%x, %y)".format(z.op.name)) in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == t1

        if ref is not None:
            t1 = relay.TensorType((5, 10, 5))
            t2 = relay.TensorType((5, 10, 5))
            x = relay.var("x", t1)
            y = relay.var("y", t2)
            z = opfunc(x, y)
            x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
            y_data = np.random.rand(5, 10, 5).astype(t2.dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            for target, ctx in ctx_list():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    for opfunc, ref in [(relay.power, np.power)]:
        check_binary_op(opfunc, ref)


def test_cmp_type():
    for op, ref in ((relay.greater, np.greater),
               (relay.greater_equal, np.greater_equal),
               (relay.less, np.less),
               (relay.less_equal, np.less_equal),
               (relay.equal, np.equal),
               (relay.not_equal, np.not_equal)):
        x = relay.var("x", relay.TensorType((10, 4), "float32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "float32"))
        z = op(x, y)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "bool")

        if ref is not None:
            x_shape = (10, 4)
            y_shape = (5, 10, 1)
            t1 = relay.TensorType(x_shape)
            t2 = relay.TensorType(y_shape)
            x = relay.var("x", t1)
            y = relay.var("y", t2)
            z = op(x, y)
            x_data = np.random.rand(*x_shape).astype(t1.dtype)
            y_data = np.random.rand(*y_shape).astype(t2.dtype)
            ref_res = ref(x_data, y_data)
            func = relay.Function([x, y], z)

            for target, ctx in ctx_list():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


def test_binary_int_broadcast():
    for op, ref in [(relay.right_shift, np.right_shift),
               (relay.left_shift, np.left_shift),
                (relay.mod, np.mod),
               (relay.maximum, np.maximum),
               (relay.minimum, np.minimum)]:
        x = relay.var("x", relay.TensorType((10, 4), "int32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "int32"))
        z = op(x, y)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "int32")

    if ref is not None:
        x_shape = (10, 4)
        y_shape = (5, 10, 1)
        t1 = relay.TensorType(x_shape, 'int32')
        t2 = relay.TensorType(y_shape, 'int32')
        x_data = np.random.rand(*x_shape).astype(t1.dtype)
        y_data = np.random.rand(*y_shape).astype(t2.dtype)
        func = relay.Function([x, y], z)
        ref_res = ref(x_data, y_data)

        for target, ctx in ctx_list():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data, y_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


def test_where():
    shape = (3, 4)
    dtype = "float32"
    cond = relay.var("cond", relay.TensorType(shape, dtype))
    x = relay.var("x", relay.TensorType(shape, dtype))
    y = relay.var("y", relay.TensorType(shape, dtype))
    z = relay.where(cond, x, y)
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType(shape, dtype)

    func = relay.Function([cond, x, y], z)
    condition = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    x = np.random.uniform(size=shape).astype(dtype)
    y = np.random.uniform(size=shape).astype(dtype)
    ref_res = np.where(condition, x, y)
    for target, ctx in ctx_list():
        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(condition, x, y)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)


def verify_reduce(funcs, data, axis, keepdims, exclude, output, dtype="float32"):
    test_func = funcs[0]
    ref_func = funcs[1]
    dtype = "bool" if ref_func in [np.all] else dtype

    x = relay.var("x", relay.TensorType(data, dtype))
    z = test_func(x, axis, keepdims, exclude)
    zz = run_infer_type(z)
    if axis:
        assert "axis=" in z.astext()
    if keepdims:
        assert "keepdims=" in z.astext()
    if exclude:
        assert "exclude=" in z.astext()
    out_type = "int32" if test_func in [relay.argmin, relay.argmax] else dtype
    assert zz.checked_type == relay.ty.TensorType(output, out_type)

    if all(isinstance(v, tvm.expr.Var) == 1 for v in data):
        return

    func = relay.Function([x], z)
    x_data = np.random.choice([True, False], size=data) if ref_func in [np.all] \
        else np.random.uniform(size=data).astype(dtype)

    if ref_func in [np.sum]:
        ref_res = ref_func(x_data + 0, axis=axis, dtype=dtype, keepdims=keepdims)
    elif ref_func in [np.max, np.min, np.mean, np.prod]:
        ref_res = ref_func(x_data + 0, axis=axis, keepdims=keepdims)
    else: #argmin/argmax
        if axis and not isinstance(axis, int) and len(axis) > 1 :
            return
        ref_res = ref_func(x_data + 0, axis=axis, keepdims=keepdims)

    for target, ctx in ctx_list():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

def test_reduce_functions():
    def _with_keepdims(func):
        def _wrapper(data, axis=None, keepdims=False):
            if not keepdims:
                return func(data, axis=axis)
            else:
                if axis is not None:
                    axis = axis if isinstance(axis, int) else axis[0]
                    out_shape = list(data.shape)
                    out_shape[axis] = 1
                else:
                    out_shape = [1 for _ in range(len(data.shape))]
                return func(data, axis=axis).reshape(out_shape)
        return _wrapper

    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    for func in [[relay.sum, np.sum],
                 [relay.max, np.max],
                 [relay.min, np.min],
                 [relay.mean, np.mean],
                 [relay.prod, np.prod],
                 [relay.all, np.all],
                 [relay.argmin, _with_keepdims(np.argmin)],
                 [relay.argmax, _with_keepdims(np.argmax)]]:
        verify_reduce(func, (d1, d2, d3, d4), None, False, False, ())
        verify_reduce(func, (d1, d2, d3, d4), 2, True, False, (d1, d2, 1, d4))
        verify_reduce(func, (d1, d2, d3, d4), 0, True, False, (1, d2, d3, d4))
        verify_reduce(func, (d1, d2, d3), 1, True, False, (d1, 1, d3))
        verify_reduce(func, (d1, d2, d3), 0, True, False, (1, d2, d3))
        verify_reduce(func, (d1, d2, d3), None, True, False, (1, 1, 1))
        verify_reduce(func, (d1, d2, d3), (0, 1), True, False, (1, 1, d3))
        verify_reduce(func, (2, 3, 4), 1, True, False, (2, 1, 4))
        verify_reduce(func, (2, 3, 4), (1,), True, False, (2, 1, 4))
        verify_reduce(func, (2, 3, 4), -1, True, False, (2, 3, 1))
        verify_reduce(func, (2, 3, 4), (0, 1, 2), False, False, ())
        verify_reduce(func, (4, 4, 3), None, False, False, ())
        verify_reduce(func, (4, 4, 3), (0, 2), False, False, (4,))
        verify_reduce(func, (128, 24, 128), (0, 1), False, False, (128,))
        verify_reduce(func, (128, 24, 128), (0, 2), False, False, (24,))
        verify_reduce(func, (128, 24, 128), (0, 1), True, False, (1, 1, 128))
        verify_reduce(func, (128, 24, 128), (0, 2), True, False, (1, 24, 1))


def test_strided_slice():
    def verify(dshape, begin, end, strides, output, test_ref=True):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.strided_slice(x, begin=begin, end=end, strides=strides)
        func = relay.Function([x], z)
        func = run_infer_type(func)
        text = func.astext()
        assert "begin=" in text
        assert "end=" in text
        if output:
            assert func.body.checked_type == relay.ty.TensorType(output, "float32")
        if not test_ref:
            return
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = topi.testing.strided_slice_python(
            x_data, begin, end, strides)
        for target, ctx in ctx_list():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    d1, d2, d3, d4 = tvm.var("d1"), tvm.var("d2"), tvm.var("d3"), tvm.var("d4")
    verify((d1, d2, 3), [None, None, 1], [None, None, 2], None, (d1, d2, 1), False)
    verify((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], (1, 3, 3))
    verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    verify((3, 4, 3), [1, 0, 0], [2, 2, 3], [1, 1, 2], (1, 2, 2))
    verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1], [4, 4, 3], None, (2, 3, 3))


if __name__ == "__main__":
    test_strided_slice()
    test_binary_op()
    test_cmp_type()
    test_binary_int_broadcast()
    test_where()
    test_reduce_functions()
