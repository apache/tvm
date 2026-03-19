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
# ruff: noqa: RUF005
import numpy as np

import tvm
import tvm.testing


def lower_intrin(params, stmt):
    """wrapper to call transformation in stmt"""
    lower_expr = isinstance(stmt, tvm.tirx.PrimExpr)
    stmt = tvm.tirx.Evaluate(stmt) if lower_expr else stmt
    mod = tvm.IRModule.from_expr(
        tvm.tirx.PrimFunc(params, stmt).with_attr("target", tvm.target.Target("llvm"))
    )
    mod = tvm.transform.Sequential(
        [tvm.tirx.transform.Simplify(), tvm.tirx.transform.LowerIntrin()]
    )(mod)
    func = mod["main"]
    stmt = func.body
    return stmt.value if lower_expr else stmt.body


def check_value(expr, variables, data, fref):
    """
    Check that expr evaluates to fref(*row) for each row in data.
    variables: list of TIR vars [x] or [x, y] bound to the columns of data.
    data: list of tuples, each tuple has len(variables) elements.
    """
    n = len(data)
    num_vars = len(variables)
    assert num_vars >= 1 and all(len(row) == num_vars for row in data)

    # Build input and output buffers
    input_bufs = [
        tvm.tirx.decl_buffer((n,), dtype=variables[i].dtype, name=f"v{i}") for i in range(num_vars)
    ]
    out_buf = tvm.tirx.decl_buffer((n,), dtype=expr.dtype, name="C")

    # Build loop body: for each i, bind variables[j] = input_bufs[j][i], then store expr to out
    loop_var = tvm.tirx.Var("i", "int32")

    def make_store(i_var):
        # Build the expression with each variable bound to the corresponding buffer load
        result = expr
        for j in range(num_vars - 1, -1, -1):
            result = tvm.tirx.Let(variables[j], tvm.tirx.BufferLoad(input_bufs[j], [i_var]), result)
        return tvm.tirx.BufferStore(out_buf, result, [i_var])

    loop = tvm.tirx.For(
        loop_var,
        tvm.tirx.const(0, "int32"),
        tvm.tirx.const(n, "int32"),
        tvm.tirx.ForKind.SERIAL,
        make_store(loop_var),
    )

    prim_func = tvm.tirx.PrimFunc(input_bufs + [out_buf], loop)
    prim_func = prim_func.with_attr({"tirx.noalias": True, "global_symbol": "main"})
    f = tvm.compile(prim_func, "llvm")

    arrays = [
        tvm.runtime.tensor(np.array([row[j] for row in data], dtype=variables[j].dtype))
        for j in range(num_vars)
    ]
    c = tvm.runtime.tensor(np.zeros(n, dtype=expr.dtype))
    f(*arrays, c)
    cref = np.array([fref(*row) for row in data])
    np.testing.assert_equal(c.numpy(), cref)


def get_ref_data():
    """Get reference data for every pairs"""
    import itertools

    x = range(-10, 10)
    y = list(range(-10, 10))
    y.remove(0)
    return list(itertools.product(x, y))


@tvm.testing.requires_llvm
def test_lower_floordiv():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = tvm.tirx.Var("x", dtype)
        y = tvm.tirx.Var("y", dtype)
        zero = tvm.tirx.const(0, dtype)
        # no constraints
        res = lower_intrin([x, y], tvm.tirx.floordiv(x, y))
        check_value(res, [x, y], data, lambda a, b: a // b)
        # rhs >= 0
        res = lower_intrin([x, y], tvm.tirx.Select(y >= 0, tvm.tirx.floordiv(x, y), zero))
        check_value(res, [x, y], data, lambda a, b: a // b if b > 0 else 0)
        # involves max
        res = lower_intrin(
            [x, y], tvm.tirx.Select(y >= 0, tvm.tirx.max(tvm.tirx.floordiv(x, y), zero), zero)
        )
        check_value(res, [x, y], data, lambda a, b: max(a // b, 0) if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(
            [x, y], tvm.tirx.Select(tvm.tirx.all(y >= 0, x >= 0), tvm.tirx.floordiv(x, y), zero)
        )
        check_value(res, [x, y], data, lambda a, b: a // b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin([x, y], tvm.tirx.floordiv(x, tvm.tirx.const(8, dtype=dtype)))
        check_value(res, [x, y], [(a, b) for a, b in data if b == 8], lambda a, b: a // b)
        # floordiv(x + m, k), m and k are positive constant. 2 <= m <= k-1.
        res = lower_intrin(
            [x, y],
            tvm.tirx.floordiv(x + tvm.tirx.const(4, dtype=dtype), tvm.tirx.const(5, dtype=dtype)),
        )
        check_value(res, [x, y], [(a, b) for a, b in data if b == 5], lambda a, b: (a + 4) // b)


@tvm.testing.requires_llvm
def test_lower_floormod():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = tvm.tirx.Var("x", dtype)
        y = tvm.tirx.Var("y", dtype)
        zero = tvm.tirx.const(0, dtype)
        # no constraints
        res = lower_intrin([x, y], tvm.tirx.floormod(x, y))
        check_value(res, [x, y], data, lambda a, b: a % b)
        # rhs >= 0
        res = lower_intrin([x, y], tvm.tirx.Select(y >= 0, tvm.tirx.floormod(x, y), zero))
        check_value(res, [x, y], data, lambda a, b: a % b if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(
            [x, y], tvm.tirx.Select(tvm.tirx.all(y >= 0, x >= 0), tvm.tirx.floormod(x, y), zero)
        )
        check_value(res, [x, y], data, lambda a, b: a % b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin([x, y], tvm.tirx.floormod(x, tvm.tirx.const(8, dtype=dtype)))
        check_value(res, [x, y], [(a, b) for a, b in data if b == 8], lambda a, b: a % b)
        # floormod(x + m, k), m and k are positive constant. 2 <= m <= k-1.
        res = lower_intrin(
            [x, y],
            tvm.tirx.floormod(x + tvm.tirx.const(4, dtype=dtype), tvm.tirx.const(5, dtype=dtype)),
        )
        check_value(res, [x, y], [(a, b) for a, b in data if b == 5], lambda a, b: (a + 4) % b)


@tvm.testing.requires_llvm
def test_lower_floordiv_overflow_checks():
    """
    Regression tests for overflow checks in TryFindShiftCoefficientForPositiveRange.
    Divisor is constant 3 (not 1 to avoid CSE, not power-of-two so we don't take the shift path).
    Reuses lower_intrin and check_value; overflow tests use one var [x].
    """
    # Check 3: (b-1) - a_min must not overflow (numerator and C++ int64).
    # x (int64) full range -> min_value = -2^63. With b = 3: numerator = 2 - (-2^63) > LLONG_MAX.
    x = tvm.tirx.Var("x", "int64")
    res = lower_intrin([x], tvm.tirx.floordiv(x, tvm.tirx.const(3, "int64")))
    data_check3 = [(-(2**63),), (0,), (100,)]
    check_value(res, [x], data_check3, lambda a: a // 3)

    # Check 4: c_value * b_value must not overflow dtype.
    # x (int16) full range -> min_value = -32768, c = ceil(32770/3) = 10923; 10923*3 > 32767.
    x = tvm.tirx.Var("x", "int16")
    res = lower_intrin([x], tvm.tirx.floordiv(x, tvm.tirx.const(3, "int16")))
    data_check4 = [(-32768,), (0,), (100,)]
    check_value(res, [x], data_check4, lambda a: a // 3)

    # Check 5: a_max + b*c must not overflow (offset numerator).
    # tirx.min(tirx.max(x, -10), 32758) can give bounds [-10, 32758]; b=3, c=4; a_max + 12 > 32767.
    # In practice this path may not be triggered. This test still validates correct lowering.
    x = tvm.tirx.Var("x", "int16")
    clamped = tvm.tirx.min(
        tvm.tirx.max(x, tvm.tirx.const(-10, "int16")), tvm.tirx.const(32758, "int16")
    )
    res = lower_intrin([x], tvm.tirx.floordiv(clamped, tvm.tirx.const(3, "int16")))
    data_check5 = [(-10,), (0,), (32758,), (32757,)]
    check_value(res, [x], data_check5, lambda a: (min(max(a, -10), 32758)) // 3)


if __name__ == "__main__":
    test_lower_floordiv()
    test_lower_floormod()
    test_lower_floordiv_overflow_checks()
