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

    for (op, op_like, id_op, const) in [
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

    for (op, op_like, id_op, const) in [
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


if __name__ == "__main__":
    pytest.main([__file__])
