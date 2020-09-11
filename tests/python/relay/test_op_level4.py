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
from tvm import te
import numpy as np
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
import tvm.topi.testing
import tvm.testing


@tvm.testing.uses_gpu
def test_binary_op():
    def check_binary_op(opfunc, ref):
        n = te.size_var("n")
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

            for target, ctx in tvm.testing.enabled_targets():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    for opfunc, ref in [(relay.power, np.power)]:
        check_binary_op(opfunc, ref)


@tvm.testing.uses_gpu
def test_cmp_type():
    for op, ref in (
        (relay.greater, np.greater),
        (relay.greater_equal, np.greater_equal),
        (relay.less, np.less),
        (relay.less_equal, np.less_equal),
        (relay.equal, np.equal),
        (relay.not_equal, np.not_equal),
    ):
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

            for target, ctx in tvm.testing.enabled_targets():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


@tvm.testing.uses_gpu
def test_binary_int_broadcast_1():
    for op, ref in [(relay.right_shift, np.right_shift), (relay.left_shift, np.left_shift)]:
        x = relay.var("x", relay.TensorType((10, 4), "int32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "int32"))
        z = op(x, y)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "int32")

        if ref is not None:
            x_shape = (10, 4)
            y_shape = (5, 10, 1)
            t1 = relay.TensorType(x_shape, "int32")
            t2 = relay.TensorType(y_shape, "int32")
            x_data = np.random.randint(1, 10000, size=(x_shape)).astype(t1.dtype)
            y_data = np.random.randint(1, 31, size=(y_shape)).astype(t2.dtype)
            func = relay.Function([x, y], z)
            ref_res = ref(x_data, y_data)

            for target, ctx in tvm.testing.enabled_targets():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


@tvm.testing.uses_gpu
def test_binary_int_broadcast_2():
    for op, ref in [(relay.maximum, np.maximum), (relay.minimum, np.minimum), (relay.mod, np.mod)]:
        x = relay.var("x", relay.TensorType((10, 4), "int32"))
        y = relay.var("y", relay.TensorType((5, 10, 1), "int32"))
        z = op(x, y)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType((5, 10, 4), "int32")

        if ref is not None:
            x_shape = (10, 4)
            y_shape = (5, 10, 1)
            t1 = relay.TensorType(x_shape, "int32")
            t2 = relay.TensorType(y_shape, "int32")
            x_data = np.random.randint(1, 10000, size=(x_shape)).astype(t1.dtype)
            y_data = np.random.randint(1, 10000, size=(y_shape)).astype(t2.dtype)
            func = relay.Function([x, y], z)
            ref_res = ref(x_data, y_data)

            for target, ctx in tvm.testing.enabled_targets():
                intrp = relay.create_executor("graph", ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data, y_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)


@tvm.testing.uses_gpu
def test_where():
    def run(func, inputs, ref_res):
        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(*inputs)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

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

    run(func, [condition, x, y], ref_res)

    x = relay.const(1)
    y = relay.const(-1)
    shape = (3,)
    dtype = "float32"
    cond = relay.var("cond", relay.TensorType(shape, "bool"))
    z = relay.where(cond, x, y)

    func = relay.Function([cond], z)
    condition = np.array([1, 0, 1], dtype=np.bool)
    ref_res = np.where(condition, 1, -1)

    run(func, [condition], ref_res)


def verify_reduce(funcs, data, axis, keepdims, exclude, output, dtype="float32"):
    test_func = funcs[0]
    ref_func = funcs[1]
    dtype = "bool" if ref_func in [np.all, np.any] else dtype

    x = relay.var("x", relay.TensorType(data, dtype))
    if test_func == relay.logsumexp:
        z = test_func(x, axis, keepdims)
    else:
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

    if all(isinstance(v, tvm.tir.Var) == 1 for v in data):
        return

    func = relay.Function([x], z)
    x_data = (
        np.random.choice([True, False], size=data)
        if ref_func in [np.all]
        else np.random.uniform(size=data).astype(dtype)
    )

    if ref_func in [np.sum]:
        ref_res = ref_func(x_data + 0, axis=axis, dtype=dtype, keepdims=keepdims)
    elif ref_func in [np.max, np.min, np.mean, np.prod]:
        ref_res = ref_func(x_data + 0, axis=axis, keepdims=keepdims)
    else:  # argmin/argmax
        if axis and not isinstance(axis, int) and len(axis) > 1:
            return
        ref_res = ref_func(x_data + 0, axis=axis, keepdims=keepdims)

    for target, ctx in tvm.testing.enabled_targets():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
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

    def _np_log_sum_exp(x, axis, keepdims=False):
        max_x = np.max(x, axis=axis, keepdims=True)
        x = np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))
        x = x + max_x
        if not keepdims:
            x = np.squeeze(x, axis=axis)
        return x

    def _unbiased_relay_wrapper(f):
        def _unbiased_func(x, axis=None, keepdims=False, exclude=False):
            return f(x, axis=axis, keepdims=keepdims, exclude=exclude, unbiased=True)

        return _unbiased_func

    def _unbiased_np_wrapper(f):
        def _unbiased_func(a, axis=None, dtype=None, keepdims=None):
            return f(a, axis=axis, dtype=dtype, ddof=1, keepdims=keepdims)

        return _unbiased_func

    d1, d2, d3, d4 = te.var("d1"), te.var("d2"), te.var("d3"), te.var("d4")
    for func in [
        [relay.sum, np.sum],
        [relay.max, np.max],
        [relay.min, np.min],
        [relay.mean, np.mean],
        [relay.variance, np.var],
        [_unbiased_relay_wrapper(relay.variance), _unbiased_np_wrapper(np.var)],
        [relay.std, np.std],
        [_unbiased_relay_wrapper(relay.std), _unbiased_np_wrapper(np.std)],
        [relay.prod, np.prod],
        [relay.all, np.all],
        [relay.any, np.any],
        [relay.logsumexp, _np_log_sum_exp],
        [relay.argmin, _with_keepdims(np.argmin)],
        [relay.argmax, _with_keepdims(np.argmax)],
    ]:
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


def verify_mean_var_std(funcs, shape, axis, keepdims):
    test_func = funcs[0]
    ref_func = funcs[1]
    dtype = "float32"

    x = relay.var("x", relay.TensorType(shape, dtype))
    z = test_func(x, axis, keepdims)
    func = relay.Function([x], z.astuple())
    x_data = np.random.uniform(size=shape).astype(dtype)
    ref_mean = np.mean(x_data, axis=axis, dtype=dtype, keepdims=keepdims)
    ref_res = ref_func(x_data, axis=axis, dtype=dtype, keepdims=keepdims)

    for target, ctx in tvm.testing.enabled_targets():
        intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
        intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
        op_res1 = intrp1.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res1[0].asnumpy(), ref_mean, rtol=1e-5)
        tvm.testing.assert_allclose(op_res1[1].asnumpy(), ref_res, rtol=1e-5)
        op_res2 = intrp2.evaluate(func)(x_data)
        tvm.testing.assert_allclose(op_res2[0].asnumpy(), ref_mean, rtol=1e-5)
        tvm.testing.assert_allclose(op_res2[1].asnumpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_mean_var_std():
    for func in [[relay.mean_variance, np.var], [relay.mean_std, np.std]]:
        verify_mean_var_std(func, (2, 3, 4), 1, True)
        verify_mean_var_std(func, (2, 3, 4), (1,), True)
        verify_mean_var_std(func, (2, 3, 4), -1, True)
        verify_mean_var_std(func, (2, 3, 4), (0, 1, 2), False)
        verify_mean_var_std(func, (4, 4, 3), None, False)
        verify_mean_var_std(func, (4, 4, 3), (0, 2), False)
        verify_mean_var_std(func, (128, 24, 128), (0, 1), False)
        verify_mean_var_std(func, (128, 24, 128), (0, 2), False)
        verify_mean_var_std(func, (128, 24, 128), (0, 1), True)
        verify_mean_var_std(func, (128, 24, 128), (0, 2), True)


@tvm.testing.uses_gpu
def test_strided_slice():
    def verify(dshape, begin, end, strides, output, slice_mode="end", test_ref=True, dtype="int32"):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(x_data, begin, end, strides, slice_mode)

        if strides:
            z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
        else:
            z = relay.strided_slice(x, begin=begin, end=end, slice_mode=slice_mode)
        func = relay.Function([x], z)

        func = run_infer_type(func)
        text = func.astext()
        assert "begin=" in text
        assert "end=" in text

        if output:
            assert func.body.checked_type == relay.ty.TensorType(output, "float32")

        if not test_ref:
            return
        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    verify((1, 3, 10, 10), [0, 0, 0, 0], [-1, 3, 10, 10], [1], (0, 3, 10, 10), dtype="int64")
    verify(
        (1, 224, 224, 3),
        [0, 20, 20, 0],
        [1, 140, 140, 3],
        [1, 1, 1, 1],
        (1, 120, 120, 3),
        dtype="int64",
    )
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], (1, 3, 3), dtype="int16")
    verify((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify(
        (3, 4, 3), [1, 0, 0], [3, -1, 3], [1, 1, 1], (2, 4, 3), slice_mode="size", test_ref=False
    )
    verify((3, 4, 3), [1, 0, 0], [-1, 2, 3], [1, 1, 1], (2, 2, 3), slice_mode="size", test_ref=True)


# TODO(mbrookhart): enable once vm supports heterogenous execution
# @tvm.testing.uses_gpu
def test_dyn_strided_slice():
    def verify(dshape, begin, end, strides, output, slice_mode="end", test_ref=True, dtype="int32"):
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(x_data, begin, end, strides, slice_mode)

        x = relay.var("x", relay.TensorType((relay.Any(),) * ndim, "float32"))
        if strides:
            z = relay.strided_slice(x, begin=begin, end=end, strides=strides, slice_mode=slice_mode)
        else:
            z = relay.strided_slice(x, begin=begin, end=end, slice_mode=slice_mode)
        func = relay.Function([x], z)

        func = run_infer_type(func)
        text = func.astext()
        assert "begin=" in text
        assert "end=" in text

        if not test_ref:
            return
        for target, ctx in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            intrp = relay.create_executor("vm", mod=mod, ctx=ctx, target=target)
            op_res = intrp.evaluate()(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    verify((1, 3, 10, 10), [0, 0, 0, 0], [-1, 3, 10, 10], [1], (0, 3, 10, 10), dtype="int64")
    verify(
        (1, 224, 224, 3),
        [0, 20, 20, 0],
        [1, 140, 140, 3],
        [1, 1, 1, 1],
        (1, 120, 120, 3),
        dtype="int64",
    )
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], (1, 3, 3), dtype="int16")
    verify((3, 4, 3), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 1000, 3], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 4], None, (2, 3, 3))
    verify((3, 4, 3), [1, 1, 0], [4, 4, 3], None, (2, 3, 3))
    # TODO(mbrookhart): fix static strided_slice with dynamic input and negative begin
    # verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    # verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify(
        (3, 4, 3), [1, 0, 0], [3, -1, 3], [1, 1, 1], (2, 4, 3), slice_mode="size", test_ref=False
    )
    verify((3, 4, 3), [1, 0, 0], [-1, 2, 3], [1, 1, 1], (2, 2, 3), slice_mode="size", test_ref=True)


@tvm.testing.uses_gpu
def test_strided_set():
    def verify(dshape, begin, end, strides, vshape, test_ref=True):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        v = relay.var("v", relay.TensorType(vshape, "float32"))
        begin_c = relay.const(begin, dtype="int32")
        end_c = relay.const(end, dtype="int32")
        if strides:
            strides_c = relay.const(strides, dtype="int32")
            z = relay.strided_set(x, v, begin=begin_c, end=end_c, strides=strides_c)
        else:
            z = relay.strided_set(x, v, begin=begin_c, end=end_c)
        func = relay.Function([x, v], z)
        func = run_infer_type(func)
        text = func.astext()
        assert "strided_set" in text
        print(text)
        assert func.body.checked_type == relay.ty.TensorType(dshape, "float32")
        if not test_ref:
            return
        x_data = np.random.uniform(size=dshape).astype("float32")
        v_data = np.random.uniform(size=vshape).astype("float32")
        ref_res = tvm.topi.testing.strided_set_python(x_data, v_data, begin, end, strides)
        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data, v_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res)

    verify((3, 4, 16), [0, 0, 0], [4, -5, 4], [1, -1, 2], (3, 1, 2))
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
    test_strided_set()
    test_binary_op()
    test_cmp_type()
    test_binary_int_broadcast_1()
    test_binary_int_broadcast_2()
    test_where()
    test_reduce_functions()
    test_mean_var_std()
