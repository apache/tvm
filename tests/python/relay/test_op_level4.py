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
import sys

import numpy as np
import numpy.random
import pytest
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay, te
from tvm.relay import transform
from tvm.relay.testing import run_infer_type

executor_kind = tvm.testing.parameter("graph", "vm")


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

            for target, dev in tvm.testing.enabled_targets():
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data, y_data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res)

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

            for target, dev in tvm.testing.enabled_targets():
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data, y_data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res)


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

            for target, dev in tvm.testing.enabled_targets():
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data, y_data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res)


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

            for target, dev in tvm.testing.enabled_targets():
                op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                    x_data, y_data
                )
                tvm.testing.assert_allclose(op_res.numpy(), ref_res)


@tvm.testing.uses_gpu
def test_where(executor_kind):
    def run(func, inputs, ref_res):
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                *inputs
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    def verify(x_np, y_np, cond_np):
        ref_res = np.where(cond_np, x_np, y_np)

        args = []
        args_np = []
        vs = []

        cond = relay.var("cond", relay.TensorType(cond_np.shape, "bool"))

        args.append(cond)
        args_np.append(cond_np)

        for v_name, v_np in [("x", x_np), ("y", y_np)]:
            if len(v_np.shape) == 0:
                v = relay.const(v_np.item())
            else:
                v = relay.var(v_name, relay.TensorType(v_np.shape, dtype))
                args.append(v)
                args_np.append(v_np)
            vs.append(v)

        z = relay.where(cond, vs[0], vs[1])

        func = relay.Function(args, z)

        run(func, args_np, ref_res)

    dtype = "float32"

    x_np = np.random.uniform(size=(3, 4)).astype(dtype)
    y_np = np.random.uniform(size=(3, 4)).astype(dtype)
    cond_np = np.random.uniform(low=-1, high=1, size=(3, 4)) > 0

    verify(x_np, y_np, cond_np)

    x_np = np.array(1.0, dtype)
    y_np = np.array(-1.0, dtype)
    cond_np = np.array([1, 0, 1], dtype=bool)

    verify(x_np, y_np, cond_np)

    x_np = np.arange(10).astype(dtype)
    y_np = 10 * x_np
    cond_np = x_np < 5

    verify(x_np, y_np, cond_np)

    x_np = np.array([[1, 2], [3, 4]], dtype)
    y_np = np.array([[5, 6], [7, 8]], dtype)
    cond_np = np.array([[1], [0]], dtype=bool)

    verify(x_np, y_np, cond_np)
    verify(x_np, y_np, cond_np.T)

    x_np = np.random.randn(1, 12, 8, 8).astype(dtype)
    y_np = np.array(-1.0, dtype)
    cond_np = np.random.randn(1, 1, 8, 8) > 0

    verify(x_np, y_np, cond_np)

    x_np, y_np = np.ogrid[:3, :4]
    cond_np = np.where(x_np < y_np, x_np, 10 + y_np).astype(bool)

    verify(x_np.astype(dtype), y_np.astype(dtype), cond_np)


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


class TestReduceFunctions:
    funcs = {
        "sum": (relay.sum, np.sum),
        "max": (relay.max, np.max),
        "min": (relay.min, np.min),
        "mean": (relay.mean, np.mean),
        "var": (relay.variance, np.var),
        "unbiased_var": (_unbiased_relay_wrapper(relay.variance), _unbiased_np_wrapper(np.var)),
        "std": (relay.std, np.std),
        "unbiased_std": (_unbiased_relay_wrapper(relay.std), _unbiased_np_wrapper(np.std)),
        "prod": (relay.prod, np.prod),
        "all": (relay.all, np.all),
        "any": (relay.any, np.any),
        "logsumexp": (relay.logsumexp, _np_log_sum_exp),
        "argmin": (relay.argmin, _with_keepdims(np.argmin)),
        "argmax": (relay.argmax, _with_keepdims(np.argmax)),
    }
    relay_func, ref_func = tvm.testing.parameters(
        *funcs.values(),
        ids=list(funcs),
    )

    d1, d2, d3, d4 = te.var("d1"), te.var("d2"), te.var("d3"), te.var("d4")

    data, axis, keepdims, exclude, output = tvm.testing.parameters(
        ((d1, d2, d3, d4), None, False, False, ()),
        ((d1, d2, d3, d4), 2, True, False, (d1, d2, 1, d4)),
        ((d1, d2, d3, d4), 0, True, False, (1, d2, d3, d4)),
        ((d1, d2, d3), 1, True, False, (d1, 1, d3)),
        ((d1, d2, d3), 0, True, False, (1, d2, d3)),
        ((d1, d2, d3), None, True, False, (1, 1, 1)),
        ((d1, d2, d3), (0, 1), True, False, (1, 1, d3)),
        ((2, 3, 4), 1, True, False, (2, 1, 4)),
        ((2, 3, 4), (1,), True, False, (2, 1, 4)),
        ((2, 3, 4), -1, True, False, (2, 3, 1)),
        ((2, 3, 4), (0, 1, 2), False, False, ()),
        ((4, 4, 3), None, False, False, ()),
        ((4, 4, 3), (0, 2), False, False, (4,)),
        ((128, 24, 128), (0, 1), False, False, (128,)),
        ((128, 24, 128), (0, 2), False, False, (24,)),
        ((128, 24, 128), (0, 1), True, False, (1, 1, 128)),
        ((128, 24, 128), (0, 2), True, False, (1, 24, 1)),
    )

    def test_reduce(
        self,
        target,
        dev,
        relay_func,
        ref_func,
        executor_kind,
        data,
        axis,
        keepdims,
        exclude,
        output,
    ):
        dtype = "bool" if ref_func in [np.all, np.any] else "float32"
        out_type = "int32" if relay_func in [relay.argmin, relay.argmax] else dtype

        target = tvm.target.Target(target)
        if target.kind.name == "vulkan" and dtype == "bool":
            pytest.xfail("Known failing test on vulkan runtime")

        x = relay.var("x", relay.TensorType(data, dtype))
        if relay_func == relay.logsumexp:
            z = relay_func(x, axis, keepdims)
        else:
            z = relay_func(x, axis, keepdims, exclude)
        zz = run_infer_type(z)
        if axis:
            assert "axis=" in z.astext()
        if keepdims:
            assert "keepdims=" in z.astext()
        if exclude:
            assert "exclude=" in z.astext()
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

        op_res1 = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5)


@tvm.testing.uses_gpu
def test_argmin_argmax_get_last_elements():
    def get_test_case(shape, gt_func, test_argmin=False):
        total_ele = np.product(shape)
        arr = np.zeros(total_ele)
        target_value = -1 if test_argmin else 1
        arr[: total_ele // 3] = target_value
        np.random.shuffle(arr)
        arr = arr.reshape(shape)
        ans = gt_func(np.flip(arr))
        return arr, len(arr) - ans - 1

    funcs_and_gt_funcs = [(relay.argmax, np.argmax), (relay.argmin, np.argmin)]
    lengths = [5, 10, 15]
    for func, gt_func in funcs_and_gt_funcs:
        for shape in lengths:
            x_in = relay.var("x_in", shape=[shape])
            output = func(x_in, select_last_index=True)
            arr, ans = get_test_case(shape, gt_func, test_argmin=func == relay.argmin)

            mod = tvm.IRModule.from_expr(output)
            for target, dev in tvm.testing.enabled_targets():
                op_res = relay.create_executor(
                    "graph", mod=mod, device=dev, target=target
                ).evaluate()(arr)
                assert op_res.numpy().item() == ans


def verify_mean_var_std(executor_kind, funcs, shape, axis, keepdims, dtype="float32"):
    test_func = funcs[0]
    ref_func = funcs[1]

    x = relay.var("x", relay.TensorType(shape, dtype))
    z = test_func(x, axis, keepdims)
    func = relay.Function([x], z.astuple())
    x_data = np.random.uniform(size=shape).astype("float32")
    ref_mean = np.mean(x_data, axis=axis, dtype="float32", keepdims=keepdims).astype(dtype)
    ref_res = ref_func(x_data, axis=axis, dtype="float32", keepdims=keepdims).astype(dtype)

    for target, dev in tvm.testing.enabled_targets():
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data.astype(dtype)
        )
        # FP16 is always a little less accurate.
        if dtype == "float16":
            rtol, atol = (1e-2, 1e-2)
        else:
            rtol, atol = (1e-5, 1e-5)
        tvm.testing.assert_allclose(op_res[0].numpy(), ref_mean, rtol=rtol, atol=atol)
        tvm.testing.assert_allclose(op_res[1].numpy(), ref_res, rtol=rtol, atol=atol)


@tvm.testing.uses_gpu
def test_mean_var_std(executor_kind):
    for func in [[relay.mean_variance, np.var], [relay.mean_std, np.std]]:
        verify_mean_var_std(executor_kind, func, (2, 3, 4), 1, True)
        verify_mean_var_std(executor_kind, func, (2, 3, 4), (1,), True)
        verify_mean_var_std(executor_kind, func, (2, 3, 4), -1, True)
        verify_mean_var_std(executor_kind, func, (2, 3, 4), (0, 1, 2), False)
        verify_mean_var_std(executor_kind, func, (4, 4, 3), None, False)
        verify_mean_var_std(executor_kind, func, (4, 4, 3), (0, 2), False)
        verify_mean_var_std(executor_kind, func, (128, 24, 128), (0, 1), False)
        verify_mean_var_std(executor_kind, func, (128, 24, 128), (0, 2), False)
        verify_mean_var_std(executor_kind, func, (128, 24, 128), (0, 1), True)
        verify_mean_var_std(executor_kind, func, (128, 24, 128), (0, 2), True)
        # Test FP16 reduction with large indices.
        verify_mean_var_std(executor_kind, func, (128, 24, 128), (0, 2), True, "float16")
        verify_mean_var_std(executor_kind, func, (128, 24, 128), None, False, "float16")


@tvm.testing.uses_gpu
def test_strided_slice():
    def verify(
        dshape,
        begin,
        end,
        strides,
        output,
        axes=None,
        slice_mode="end",
        test_ref=True,
        dtype="int32",
        unknown_dim_value=10,
    ):
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)

        # Resolve unknown dimensions to create test case:
        dshape = list(dshape)
        for i, d in enumerate(dshape):
            if not isinstance(d, int):
                dshape[i] = unknown_dim_value
        x_data = np.random.uniform(size=dshape).astype("float32")

        ref_res = tvm.topi.testing.strided_slice_python(
            x_data,
            begin,
            end,
            strides,
            slice_mode,
            axes=axes,
        )

        if strides:
            z = relay.strided_slice(
                x, begin=begin, end=end, strides=strides, axes=axes, slice_mode=slice_mode
            )
        else:
            z = relay.strided_slice(x, begin=begin, end=end, axes=axes, slice_mode=slice_mode)
        func = relay.Function([x], z)

        func = run_infer_type(func)
        text = func.astext()
        assert "begin=" in text
        assert "end=" in text

        if output:
            assert func.body.checked_type == relay.ty.TensorType(output, "float32")

        if not test_ref:
            return
        for target, dev in tvm.testing.enabled_targets():
            # Need VM to run tests with non-static dimensions
            op_res = relay.create_executor("vm", device=dev, target=target).evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.numpy(), ref_res)

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

    # Test backwards slicing.
    verify((3, 4, 3), [-1, -1, -1], [-5, -5, -5], [-1, -1, -1], (3, 4, 3))
    # Test slicing with overlarge indices.
    verify((3, 4, 3), [0, 0, 0], [np.iinfo(np.int32).max] * 3, [1, 1, 1], (3, 4, 3))
    # Test slice mode.
    verify(
        (3, 4, 3), [1, 0, 0], [3, -1, 3], [1, 1, 1], (2, 4, 3), slice_mode="size", test_ref=False
    )

    verify((3, 4, 3), [1, 0, 0], [-1, 2, 3], [1, 1, 1], (2, 2, 3), slice_mode="size", test_ref=True)
    verify((3, 4, 3), [1], [4], None, None, axes=[1])

    # Test Any dims for simple cases
    verify((3, relay.Any()), [0], [1], [1], None, axes=[1], unknown_dim_value=10)
    verify((relay.Any(), 3), [0], [1], [1], None, axes=[1], unknown_dim_value=10)
    verify(
        (relay.Any(), relay.Any(), relay.Any()),
        [0, 1, 2],
        [5, 5, 5],
        [1, 2, 1],
        None,
        unknown_dim_value=10,
    )


@tvm.testing.uses_gpu
def test_dyn_strided_slice():
    def verify(
        dshape,
        begin,
        end,
        strides,
        output,
        axes=None,
        ishape=None,
        slice_mode="end",
        test_ref=True,
        dtype="int32",
    ):
        ndim = len(dshape)
        begin = begin if begin else [0] * ndim
        end = end if end else list(dshape)

        # target numpy result
        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.strided_slice_python(
            x_data, begin, end, strides, slice_mode, axes=axes
        )

        if ishape is None:
            ishape = (relay.Any(),) * ndim

        x = relay.var("x", relay.TensorType(ishape, "float32"))
        if strides:
            z = relay.strided_slice(
                x, begin=begin, end=end, strides=strides, axes=axes, slice_mode=slice_mode
            )
        else:
            z = relay.strided_slice(x, begin=begin, end=end, axes=axes, slice_mode=slice_mode)
        func = relay.Function([x], z)

        func = run_infer_type(func)
        text = func.astext()
        assert "begin=" in text
        assert "end=" in text

        if not test_ref:
            return
        for target, dev in tvm.testing.enabled_targets():
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res)

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
    verify((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], (1, 4, 3))
    verify((3, 4, 3), [1, -1, 0], [2, -3, 3], [1, -1, 1], (1, 2, 3))
    verify(
        (3, 4, 3), [1, 0, 0], [3, -1, 3], [1, 1, 1], (2, 4, 3), slice_mode="size", test_ref=False
    )
    verify((3, 4, 3), [1, 0, 0], [-1, 2, 3], [1, 1, 1], (2, 2, 3), slice_mode="size", test_ref=True)
    verify(
        (3, 4, 3, 2),
        [1, 0],
        [3, 1],
        [1, 1],
        None,
        axes=[1, 3],
        ishape=(relay.Any(), 4, relay.Any(), 2),
    )


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
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
                x_data, v_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res)

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
    tvm.testing.main()
