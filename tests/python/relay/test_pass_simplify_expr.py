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
from math import sqrt
import pytest
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass, run_infer_type

import numpy as np


def test_simplify_reshape():
    def before():
        x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, -1))
        y = relay.reshape(y, newshape=(4, 8, -1, 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, -1))
        return relay.Function([x, w], y)

    def expected():
        x = relay.var("x", shape=(1, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(32, 16, 16))
        return relay.Function([x, w], y)

    def symbolic():
        b = tvm.te.size_var("b")
        x = relay.var("x", shape=(b, 16, 16, 16), dtype="float32")
        w = relay.var("w", shape=(32, 16, 3, 3), dtype="float32")
        y = relay.nn.conv2d(x, w, padding=(1, 1))
        y = relay.reshape(y, newshape=(1, 16, -1))
        y = relay.reshape(y, newshape=(4, 8, -1, 16))
        y = relay.reverse_reshape(y, newshape=(32, 0, -1))
        return relay.Function([x, w], y)

    z = before()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)

    z = symbolic()
    zz = run_opt_pass(z, transform.SimplifyExpr())
    after = run_opt_pass(symbolic(), transform.InferType())
    assert tvm.ir.structural_equal(zz, after)


def test_simplify_transpose():
    # Test a series of transpose and layout_transform ops
    def before1():
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
        y = relay.transpose(x, axes=[0, 2, 3, 1])  # To NHWC
        y = relay.layout_transform(y, "NHWC", "HWCN")  # To HWCN
        y = relay.transpose(y, axes=[3, 0, 1, 2])  # To NHWC
        return relay.Function([x], y)

    def expected1():
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
        y = relay.transpose(x, axes=[0, 2, 3, 1])  # To NHWC
        return relay.Function([x], y)

    # Test that all transpose ops can be cancelled
    def before2():
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
        y = relay.nn.relu(x)
        y = relay.transpose(y, axes=[0, 2, 3, 1])  # To NHWC
        y = relay.transpose(y, axes=[1, 2, 3, 0])  # To HWCN
        y = relay.transpose(y, axes=[3, 2, 0, 1])  # To NCHW
        return relay.Function([x], y)

    def expected2():
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
        y = relay.nn.relu(x)
        return relay.Function([x], y)

    # Test default axis (reverse) and negative axis
    def before3():
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
        y = relay.nn.relu(x)
        y = relay.transpose(y)  # Reverse
        y = relay.transpose(y)  # Reverse
        y = relay.transpose(y, axes=[0, 2, -1, 1])
        y = relay.transpose(y)  # Reverse
        y = relay.transpose(y)  # Reverse
        return relay.Function([x], y)

    def expected3():
        x = relay.var("x", shape=(1, 3, 224, 224), dtype="float32")  # NCHW
        y = relay.nn.relu(x)
        y = relay.transpose(y, axes=[0, 2, 3, 1])
        return relay.Function([x], y)

    # Test a series of transpose and rank changing layout_transform
    def before4():
        """
        Simplify transpose->layout_transform and its inverse.

        Input:
        NHWC -> NCHW -> NCHW4c -> op -> NCHW4c -> NCHW -> NHWC

        Simplified:
        NHWC -> NCHW4c -> op -> NCHW4c -> NHWC
        """
        x = relay.var("x", shape=(1, 56, 56, 128), dtype="float32")
        y = relay.transpose(x, axes=[0, 3, 1, 2])
        y = relay.layout_transform(y, "NCHW", "NCHW4c")
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW4c", "NCHW")
        y = relay.transpose(y, axes=[0, 2, 3, 1])
        return relay.Function([x], y)

    def expected4():
        x = relay.var("x", shape=(1, 56, 56, 128), dtype="float32")  # NHWC
        y = relay.layout_transform(x, "NHWC", "NCHW4c")  # To NCHW4c
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW4c", "NHWC")  # To NHWC
        return relay.Function([x], y)

    def before5():
        """
        Simplify layout_transform->layout_transform and its inverse.

        Input:
        NHWC -> NCHW -> NCHW4c -> op -> NCHW4c -> NCHW -> NHWC

        Simplified:
        NHWC -> NCHW4c -> op -> NCHW4c -> NHWC
        """
        x = relay.var("x", shape=(1, 56, 56, 128), dtype="float32")  # NHWC
        y = relay.layout_transform(x, "NHWC", "NCHW")  # To NCHW
        y = relay.layout_transform(y, "NCHW", "NCHW4c")  # To NCHW4c
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW4c", "NCHW")  # To NCHW
        y = relay.layout_transform(y, "NCHW", "NHWC")  # To NHWC
        return relay.Function([x], y)

    def expected5():
        x = relay.var("x", shape=(1, 56, 56, 128), dtype="float32")  # NHWC
        y = relay.layout_transform(x, "NHWC", "NCHW4c")  # To NCHW4c
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW4c", "NHWC")  # To NHWC
        return relay.Function([x], y)

    def before6():
        """
        Remove trivial layout_transform->layout_transform.

        Input:
        NCHW -> NHWC -> NCHW -> op

        Simplified:
        NHWC -> op
        """

        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.layout_transform(x, "NCHW", "NHWC")
        y = relay.layout_transform(y, "NHWC", "NCHW")
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def expected6():
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.nn.relu(x)
        return relay.Function([x], y)

    def before7():
        """
        Remove trivial layout_transform->layout_transform.

        Input:
        NCHW4c -> NCHW8c -> NCHW4c -> op

        Simplified:
        NCHW4c -> op
        """
        x = relay.var("x", shape=(1, 32, 56, 56, 4), dtype="float32")
        y = relay.layout_transform(x, "NCHW4c", "NCHW8c")
        y = relay.layout_transform(y, "NCHW8c", "NCHW4c")
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def expected7():
        x = relay.var("x", shape=(1, 32, 56, 56, 4), dtype="float32")
        y = relay.nn.relu(x)
        return relay.Function([x], y)

    def before8():
        """
        Simplify layout_transform->layout_transform with rank contraction and expansion

        Input:
        NCHW4c -> NCHW -> NCHW8c -> op

        Simplified:
        NCHW4c -> NCHW8c -> op
        """
        x = relay.var("x", shape=(1, 32, 56, 56, 4), dtype="float32")
        y = relay.layout_transform(x, "NCHW4c", "NCHW")
        y = relay.layout_transform(y, "NCHW", "NCHW8c")
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def expected8():
        x = relay.var("x", shape=(1, 32, 56, 56, 4), dtype="float32")
        y = relay.layout_transform(x, "NCHW4c", "NCHW8c")
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def before9():
        """
        Remove trivial layout_transform->layout_transform.

        Input:
        NCHW -> NCHW4c -> NCHW -> op

        Simplified:
        NCHW -> op
        """
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.layout_transform(x, "NCHW", "NCHW4c")
        y = relay.layout_transform(y, "NCHW4c", "NCHW")
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def expected9():
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.nn.relu(x)
        return relay.Function([x], y)

    def before10():
        """
        Simplify layout_transform->layout_transform without rank change to transpose.

        Input:
        NCHW -> NHWC -> CHWN -> op

        Simplified:
        NCHW -> CHWN -> op
        """
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.layout_transform(x, "NCHW", "NHWC")
        y = relay.layout_transform(y, "NHWC", "CHWN")
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def expected10():
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.transpose(x, axes=[1, 2, 3, 0])
        y = relay.nn.relu(y)
        return relay.Function([x], y)

    def before11():
        """
        Remove trivial no op transpose ops

        Input:
        op1 -> relay.transpose(x, axes=[0, 1, 2, 3]) -> op2

        Simplified:
        op1 -> op2
        """
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.transpose(x, axes=[0, 1, 2, 3])
        y = relay.nn.relu(y)
        y = relay.layout_transform(y, "NCHW", "NCHW")
        return relay.Function([x], y)

    def expected11():
        x = relay.var("x", shape=(1, 128, 56, 56), dtype="float32")
        y = relay.nn.relu(x)
        return relay.Function([x], y)

    for before, expected in [
        [before1(), expected1()],
        [before2(), expected2()],
        [before3(), expected3()],
        [before4(), expected4()],
        [before5(), expected5()],
        [before6(), expected6()],
        [before7(), expected7()],
        [before8(), expected8()],
        [before9(), expected9()],
        [before10(), expected10()],
        [before11(), expected11()],
    ]:
        after = run_opt_pass(before, transform.SimplifyExpr())
        expected = run_opt_pass(expected, transform.InferType())
        assert tvm.ir.structural_equal(after, expected), "\nafter: {} \nexpected: {}".format(
            after, expected
        )


def test_simplify_full_elementwise():
    def validate(shape, value, dtype):
        def before_left(x, elem_op, full):
            return elem_op(full, x)

        def after_left(x, elem_op, value):
            if elem_op == relay.add and value == 0:
                return x
            elif elem_op == relay.multiply and (value == 1 or (value > 1 and dtype == "bool")):
                return x
            return elem_op(relay.const(value, dtype), x)

        def before_right(x, elem_op, full):
            return elem_op(x, full)

        def after_right(x, elem_op, value):
            if elem_op in [relay.add, relay.subtract] and value == 0:
                return x
            elif elem_op in [relay.multiply, relay.divide] and (
                value == 1 or (value > 1 and dtype == "bool")
            ):
                return x
            return elem_op(x, relay.const(value, dtype))

        x = relay.var("x", shape=shape, dtype=dtype)
        elem_ops = [relay.add, relay.multiply, relay.subtract, relay.divide]
        full_ops = []
        if value == 0:
            full_ops.append(relay.zeros(shape, dtype))
            full_ops.append(relay.zeros_like(x))
        if value == 1:
            full_ops.append(relay.ones(shape, dtype))
            full_ops.append(relay.ones_like(x))
        else:
            full_ops.append(relay.full(relay.const(value, dtype), shape))
            full_ops.append(relay.full_like(x, relay.const(value, dtype)))
        for op in elem_ops:
            for full in full_ops:
                z = before_left(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(after_left(x, op, value), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

                z = before_right(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(after_right(x, op, value), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

        # Test the case in which x is broadcast to full's shape
        full_ops = []
        if value == 0:
            full_ops.append(relay.zeros(shape * 2, dtype))
        if value == 1:
            full_ops.append(relay.ones(shape * 2, dtype))
        else:
            full_ops.append(relay.full(relay.const(value, dtype), shape * 2))
        for op in elem_ops:
            for full in full_ops:
                z = before_left(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(before_left(x, op, full), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

                z = before_right(x, op, full)
                zz = run_opt_pass(z, transform.SimplifyExpr())
                after = run_opt_pass(before_right(x, op, full), transform.InferType())
                assert tvm.ir.structural_equal(zz, after)

    for shape in [[10], [10, 10], [10, 10, 10]]:
        for dtype in ["float32", "int32", "bool"]:
            for value in [0, 1, 2]:
                validate(shape, value, dtype)


def test_eliminate_identity():
    def check(x, y=None, do_nothing=False):
        expected = run_infer_type(x)
        if do_nothing:
            actual = run_opt_pass(x, transform.SimplifyExpr())
            assert tvm.ir.structural_equal(actual, expected)
        else:
            assert y is not None
            actual = run_opt_pass(y, transform.SimplifyExpr())
            assert tvm.ir.structural_equal(actual, expected)

    shape = [2, 3, 4]
    dtype = "float32"
    x = relay.var("x", shape=shape, dtype=dtype)
    x = run_opt_pass(x, transform.InferType())

    for op, op_like, id_op, const in [
        (relay.zeros, relay.zeros_like, relay.add, relay.const(0, dtype)),
        (relay.ones, relay.ones_like, relay.multiply, relay.const(1, dtype)),
    ]:
        check(x, id_op(op_like(x), x))
        check(x, id_op(op(shape, dtype), x))
        check(x, id_op(const, x))
        check(x, id_op(op(shape[1:], dtype), x))
        check(x, id_op(x, op_like(x)))
        check(x, id_op(x, op(shape, dtype)))
        check(x, id_op(x, const))
        check(x, id_op(x, op(shape[1:], dtype)))
        check(id_op(x, op([2] + shape, dtype)), do_nothing=True)
        check(id_op(op([2] + shape, dtype), x), do_nothing=True)

    for op, op_like, id_op, const in [
        (relay.zeros, relay.zeros_like, relay.subtract, relay.const(0, dtype)),
        (relay.ones, relay.ones_like, relay.divide, relay.const(1, dtype)),
    ]:
        check(x, id_op(x, op_like(x)))
        check(x, id_op(x, const))
        check(x, id_op(x, op(shape, dtype)))
        check(x, id_op(x, op(shape[1:], dtype)))
        check(id_op(x, op([2] + shape, dtype)), do_nothing=True)
        check(id_op(const, x), id_op(op(shape, dtype), x))
        check(id_op(const, x), id_op(op_like(x), x))


def test_simplify_same_cast():
    dtype = "int32"
    data = relay.var("data", shape=(3, 4, 5), dtype=dtype)
    expr1 = relay.cast(data, dtype)
    dtype_like = relay.var("dtype_like", shape=(2, 2, 2), dtype=dtype)
    expr2 = relay.cast_like(data, dtype_like)

    expected = run_infer_type(data)
    actual1 = run_opt_pass(expr1, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual1, expected)
    actual2 = run_opt_pass(expr2, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual2, expected)


def test_simplify_consecutive_cast():
    x = relay.var("x", shape=(3, 4, 5), dtype="int8")
    y = relay.var("y", shape=(3, 4), dtype="int64")
    z = relay.var("z", shape=(3,), dtype="float32")

    expr1 = relay.cast(x, "int16")
    expr2 = relay.cast(expr1, "int32")
    expr3 = relay.cast_like(expr2, y)
    expr4 = relay.cast_like(expr3, z)

    actual1 = run_opt_pass(expr2, relay.transform.SimplifyExpr())
    expected = run_infer_type(relay.cast(x, "int32"))
    assert tvm.ir.structural_equal(actual1, expected)
    actual2 = run_opt_pass(expr3, relay.transform.SimplifyExpr())
    expected = run_infer_type(relay.cast(x, "int64"))
    assert tvm.ir.structural_equal(actual2, expected)
    actual3 = run_opt_pass(expr4, relay.transform.SimplifyExpr())
    expected = run_infer_type(relay.cast(x, "float32"))
    assert tvm.ir.structural_equal(actual3, expected)

    # cannot simplify the narrow cast
    x = relay.var("x", shape=(3, 4, 5), dtype="float32")
    y = relay.var("y", shape=(3, 4), dtype="float32")
    expr1 = relay.cast(x, "int32")
    expr2 = relay.cast_like(expr1, y)
    actual = run_opt_pass(expr2, relay.transform.SimplifyExpr())
    expected = run_infer_type(relay.cast(expr1, "float32"))
    assert tvm.ir.structural_equal(actual, expected)

    x = relay.var("x", shape=(3, 4), dtype="int64")
    expr1 = relay.cast(x, "bool")
    expr2 = relay.cast(expr1, "int32")
    actual = run_opt_pass(expr2, relay.transform.SimplifyExpr())
    expected = run_infer_type(expr2)
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_reshape_like():
    data = relay.var("data", shape=(2, 3, 4), dtype="float32")
    shape_like = relay.var("shape_like", shape=(6, 2, 2), dtype="float32")
    expr = relay.reshape_like(data, shape_like)

    expected = run_infer_type(relay.reshape(data, (6, 2, 2)))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_reshape_like_attrs():
    data = relay.var("data", shape=(2, 3, 4), dtype="float32")
    shape_like = relay.var("shape_like", shape=(6, 2, 2), dtype="float32")
    expr = relay.reshape_like(data, shape_like, lhs_begin=2, rhs_begin=1)

    expected = run_infer_type(relay.reshape(data, (2, 3, 2, 2)))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_zeros_like():
    dtype = "int32"
    shape_like = relay.var("shape_like", shape=(3, 4, 5), dtype=dtype)
    expr = relay.zeros_like(shape_like)

    expected = run_infer_type(relay.zeros((3, 4, 5), dtype))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_ones_like():
    dtype = "int32"
    shape_like = relay.var("shape_like", shape=(3, 4, 5), dtype=dtype)
    expr = relay.ones_like(shape_like)

    expected = run_infer_type(relay.ones((3, 4, 5), dtype))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_full_like():
    dtype = "int32"
    shape_like = relay.var("shape_like", shape=(3, 4, 5), dtype=dtype)
    fill_value = relay.var("fill", relay.TensorType((), "float32"))
    expr = relay.full_like(shape_like, fill_value)

    expected = run_infer_type(relay.full(fill_value, (3, 4, 5), dtype))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_collapse_sum_like():
    data = relay.var("data", shape=(3, 3, 3), dtype="float32")
    shape_like = relay.var("shape_like", shape=(3,), dtype="float32")
    expr = relay.collapse_sum_like(data, shape_like)

    expected = run_infer_type(relay.collapse_sum_to(data, (3,)))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_broadcast_to_like():
    data = relay.var("data", shape=(3,), dtype="float32")
    shape_like = relay.var("shape_like", shape=(3, 3, 3), dtype="float32")
    expr = relay.broadcast_to_like(data, shape_like)

    expected = run_infer_type(relay.broadcast_to(data, (3, 3, 3)))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_cast_like():
    dim_any = tvm.tir.Any()
    data = relay.var("data", shape=(3, dim_any, 5), dtype="float32")
    dtype_like = relay.var("dtype_like", shape=(dim_any, 3, 3), dtype="int32")
    expr = relay.cast_like(data, dtype_like)

    expected = run_infer_type(relay.cast(data, "int32"))
    actual = run_opt_pass(expr, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_concretize_multiple():
    x = relay.var("x", shape=(2, 3), dtype="float32")
    y = relay.var("y", shape=(3,), dtype="float32")
    l = x + y

    dl = relay.ones_like(l)
    dx = relay.zeros_like(x)
    dy = relay.zeros_like(y)
    dx = dx + relay.collapse_sum_like(dl, dx)
    dy = dy + relay.collapse_sum_like(dl, dy)
    ret = relay.Tuple([dx, dy])

    dl_c = relay.ones((2, 3), "float32")
    # NOTE: these are removed by EliminateIdentity
    # dx_c = relay.zeros((2, 3), "float32")
    # dy_c = relay.zeros((3,), "float32")
    dx_c = relay.collapse_sum_to(dl_c, (2, 3))
    dy_c = relay.collapse_sum_to(dl_c, (3,))
    ret_c = relay.Tuple([dx_c, dy_c])

    expected = run_infer_type(ret_c)
    actual = run_opt_pass(ret, relay.transform.SimplifyExpr())
    assert tvm.ir.structural_equal(actual, expected)


def test_simplify_mul_add():
    def check_simple_fold(origin_exprs, expect_expr):
        for origin_expr in origin_exprs:
            simple_expr = run_opt_pass(origin_expr, transform.SimplifyExpr())
            assert tvm.ir.structural_equal(simple_expr, expect_expr)

    n = 32
    c1_val = np.random.uniform(size=n).astype("float32")
    c2_val = np.random.uniform(size=n).astype("float32")
    c3_val = np.random.uniform(size=n).astype("float32")

    x = relay.var("x", shape=(n,), dtype="float32")
    c1 = relay.const(c1_val)
    c2 = relay.const(c2_val)
    c3 = relay.const(c3_val)

    # add-add -> add
    origin_exprs = [
        x + c1 + c2,
        c1 + x + c2,
    ]
    expect_expr = x + relay.const(c1_val + c2_val)
    check_simple_fold(origin_exprs, expect_expr)

    # mul-mul -> mul
    origin_exprs = [
        x * c1 * c2,
        c1 * x * c2,
    ]
    expect_expr = x * relay.const(c1_val * c2_val)
    check_simple_fold(origin_exprs, expect_expr)

    # add-mul -> mul-add
    origin_exprs = [
        (x + c1) * c2,
        (c1 + x) * c2,
        c2 * (x + c1),
        c2 * (c1 + x),
    ]
    expect_expr = x * c2 + relay.const(c1_val * c2_val)
    check_simple_fold(origin_exprs, expect_expr)

    # add-mul-add -> mul-add
    origin_exprs = [
        (x + c1) * c2 + c3,
        (c1 + x) * c2 + c3,
        c2 * (x + c1) + c3,
        c2 * (c1 + x) + c3,
        c3 + (x + c1) * c2,
        c3 + (c1 + x) * c2,
        c3 + c2 * (x + c1),
        c3 + c2 * (c1 + x),
    ]
    expect_expr = x * c2 + relay.const(c1_val * c2_val + c3_val)
    check_simple_fold(origin_exprs, expect_expr)

    # mul-add-mul -> mul-add
    origin_exprs = [
        (x * c1 + c2) * c3,
        (c1 * x + c2) * c3,
        (c2 + x * c1) * c3,
        (c2 + c1 * x) * c3,
        c3 * (x * c1 + c2),
        c3 * (c1 * x + c2),
        c3 * (c2 + x * c1),
        c3 * (c2 + c1 * x),
    ]
    expect_expr = x * relay.const(c1_val * c3_val) + relay.const(c2_val * c3_val)
    check_simple_fold(origin_exprs, expect_expr)


def test_simplify_rsqrt():
    shape = (32, 1, 1)
    x = relay.var("x", shape=shape, dtype="float32")

    def before(c):
        return relay.const(c) / relay.sqrt(x)

    def expected(c):
        if c == 1:
            return relay.rsqrt(x)
        else:
            return relay.const(c) * relay.rsqrt(x)

    for c in [1.0, 2.0, 2.5]:
        opt = run_opt_pass(before(c), transform.SimplifyExpr())
        after = run_opt_pass(expected(c), transform.InferType())
        assert tvm.ir.structural_equal(opt, after)


def test_simplify_dq_argmax():
    shape = (4, 32, 1, 1)
    x = relay.var("x", shape=shape, dtype="int8")

    def before():
        y = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(0))
        return relay.op.argmax(y, axis=1)

    def expected():
        return relay.op.argmax(x, axis=1)

    opt = run_opt_pass(before(), transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(opt, after)


def test_simplify_dq_argmin():
    shape = (4, 32, 1, 1)
    x = relay.var("x", shape=shape, dtype="int8")

    def before():
        y = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(0))
        return relay.op.argmin(y, axis=1)

    def expected():
        return relay.op.argmin(x, axis=1)

    opt = run_opt_pass(before(), transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(opt, after)


def test_simplify_dq_argsort():
    shape = (4, 32, 1, 1)
    x = relay.var("x", shape=shape, dtype="int8")

    def before():
        y = relay.qnn.op.dequantize(x, relay.const(2.0), relay.const(0))
        return relay.op.argsort(y, axis=1)

    def expected():
        return relay.op.argsort(x, axis=1)

    opt = run_opt_pass(before(), transform.SimplifyExpr())
    after = run_opt_pass(expected(), transform.InferType())
    assert tvm.ir.structural_equal(opt, after)


def test_simplify_clip_cast():
    def before1():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "uint8")
        cast = relay.cast(cast, "int32")
        return relay.Function([x], cast)

    def expected1():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        return relay.Function([x], clip)

    def before2():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "uint8")
        cast = relay.cast(cast, "int32")
        return relay.Function([x], cast)

    def expected2():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        return relay.Function([x], clip)

    def before3():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "uint8")
        cast = relay.cast(cast, "int16")
        cast = relay.cast(cast, "int32")
        return relay.Function([x], cast)

    def expected3():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        return relay.Function([x], clip)

    def before4():
        x = relay.var("x", shape=(4, 8), dtype="float32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "uint8")
        cast = relay.cast(cast, "int16")
        cast = relay.cast(cast, "int32")
        return relay.Function([x], cast)

    def expected4():
        x = relay.var("x", shape=(4, 8), dtype="float32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "int32")
        return relay.Function([x], cast)

    def before5():
        x = relay.var("x", shape=(4, 8), dtype="float32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "int8")
        cast = relay.cast(cast, "int16")
        cast = relay.cast(cast, "int32")
        return relay.Function([x], cast)

    def expected5():
        x = relay.var("x", shape=(4, 8), dtype="float32")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        cast = relay.cast(clip, "int8")
        cast = relay.cast(cast, "int32")
        return relay.Function([x], cast)

    for before, expected in [
        [before1(), expected1()],
        [before2(), expected2()],
        [before3(), expected3()],
        [before4(), expected4()],
        [before5(), expected5()],
    ]:
        after = run_opt_pass(before, transform.SimplifyExpr())
        expected = run_opt_pass(expected, transform.InferType())
        assert tvm.ir.structural_equal(after, expected), "\nafter: {} \nexpected: {}".format(
            after, expected
        )


def test_simplify_cast_clip():
    def before1():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        cast = relay.cast(x, "uint8")
        clip = relay.clip(cast, a_min=0.0, a_max=255.0)
        return relay.Function([x], clip)

    def expected1():
        x = relay.var("x", shape=(4, 8), dtype="int32")
        cast = relay.cast(x, "uint8")
        return relay.Function([x], cast)

    def before2():
        x = relay.var("x", shape=(4, 8), dtype="uint8")
        clip = relay.clip(x, a_min=0.0, a_max=255.0)
        return relay.Function([x], clip)

    def expected2():
        x = relay.var("x", shape=(4, 8), dtype="uint8")
        return relay.Function([x], x)

    def before3():
        x = relay.var("x", shape=(4, 8), dtype="float32")
        cast = relay.cast(x, "bfloat16")
        clip = relay.clip(cast, a_min=-0.2, a_max=0.4)
        return relay.Function([x], clip)

    def expected3():
        x = relay.var("x", shape=(4, 8), dtype="float32")
        cast = relay.cast(x, "bfloat16")
        clip = relay.clip(cast, a_min=-0.2, a_max=0.4)
        return relay.Function([x], clip)

    for before, expected in [
        [before1(), expected1()],
        [before2(), expected2()],
        [before3(), expected3()],
    ]:
        after = run_opt_pass(before, transform.SimplifyExpr())
        expected = run_opt_pass(expected, transform.InferType())
        assert tvm.ir.structural_equal(after, expected), "\nafter: {} \nexpected: {}".format(
            after, expected
        )


def test_simplify_add():
    x = relay.var("x", shape=(1, 3, 100, 100), dtype="float32")

    def before():
        return relay.add(x, x)

    def expected():
        s = relay.const(2.0)
        return relay.multiply(x, s)

    opt = run_opt_pass(before(), transform.SimplifyExpr())
    ref = run_infer_type(expected())
    assert tvm.ir.structural_equal(opt, ref)


def test_binomials():
    def check_simple_fold(origin_exprs, expect_exprs):
        for origin_expr in origin_exprs:
            simple_expr = run_opt_pass(origin_expr, transform.SimplifyExpr())
            match = False
            for expected in expect_exprs:
                e = run_opt_pass(expected, transform.EliminateCommonSubexpr())
                match = match or tvm.ir.structural_equal(simple_expr, e)
                if match:
                    break
            assert match

    def gen_expected_expressions(x, y, a, b, c, dtype):
        if c == 1 and a > 1:
            swap = a
            a = c
            c = swap
            swap = x
            x = y
            y = swap

        det = b * b - 4 * a * c
        if det < 0:
            return gen_expressions(x, y, a, b, c)

        p_val = (b + sqrt(det)) / (2 * a)
        q_val = (b - sqrt(det)) / (2 * a)
        p = relay.const(p_val, dtype=dtype)
        q = relay.const(q_val, dtype=dtype)
        first_exp = [x + y, y + x] if p_val == 1 else [x + p * y, p * y + x, x + y * p, y * p + x]
        second_exp = [x + y, y + x] if q_val == 1 else [x + q * y, q * y + x, x + y * q, y * q + x]
        final_exp = []
        for f in first_exp:
            for s in second_exp:
                final_exp.append(f * s)
                if not p_val == q_val:
                    final_exp.append(s * f)
        return final_exp

    def gen_expressions(x, y, a, b, c):
        first_exp = [x * x] if a == 1 else [a * x * x, x * a * x, x * x * a]
        second_exp = (
            [x * y, y * x]
            if b == 1
            else [b * x * y, x * b * y, x * y * b, b * y * x, y * b * x, y * x * b]
        )
        third_exp = [y * y] if c == 1 else [c * y * y, y * c * y, y * y * c]
        final_exp = []
        for f in first_exp:
            for s in second_exp:
                for t in third_exp:
                    final_exp.append(f + s + t)
                    final_exp.append(f + t + s)
                    final_exp.append(s + f + t)
                    final_exp.append(s + t + f)
                    final_exp.append(t + f + s)
                    final_exp.append(t + s + f)
        return final_exp

    n = 5
    dtypes = ["int32", "float32", "float64"]
    for dtype in dtypes:
        x = relay.var("x", shape=(n,), dtype=dtype)
        y = relay.var("y", shape=(n,), dtype=dtype)

        a = relay.const(1, dtype=dtype)
        b = relay.const(2, dtype=dtype)
        c = relay.const(1, dtype=dtype)
        origin_exprs = gen_expressions(x, y, a, b, c)
        expect_expr = gen_expected_expressions(x, y, 1, 2, 1, dtype)
        check_simple_fold(origin_exprs, expect_expr)

        a = relay.const(6, dtype=dtype)
        b = relay.const(5, dtype=dtype)
        c = relay.const(1, dtype=dtype)
        origin_exprs = gen_expressions(x, y, a, b, c)
        expect_expr = gen_expected_expressions(x, y, 6, 5, 1, dtype)
        check_simple_fold(origin_exprs, expect_expr)

        a = relay.const(1, dtype=dtype)
        b = relay.const(1, dtype=dtype)
        c = relay.const(1, dtype=dtype)
        origin_exprs = gen_expressions(x, y, a, b, c)
        expect_expr = gen_expected_expressions(x, y, 1, 1, 1, dtype)
        check_simple_fold(origin_exprs, expect_expr)

        a = relay.const(1, dtype=dtype)
        b = relay.const(4, dtype=dtype)
        c = relay.const(4, dtype=dtype)
        origin_exprs = gen_expressions(x, y, a, b, c)
        expect_expr = gen_expected_expressions(x, y, 1, 4, 4, dtype)
        check_simple_fold(origin_exprs, expect_expr)


if __name__ == "__main__":
    tvm.testing.main()
