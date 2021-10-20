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
import tvm.testing
from tvm import te
import numpy as np


def lower_intrin(params, stmt):
    """wrapper to call transformation in stmt"""
    lower_expr = isinstance(stmt, tvm.tir.PrimExpr)
    stmt = tvm.tir.Evaluate(stmt) if lower_expr else stmt
    mod = tvm.IRModule.from_expr(
        tvm.tir.PrimFunc(params, stmt).with_attr("target", tvm.target.Target("llvm"))
    )
    mod = tvm.transform.Sequential([tvm.tir.transform.Simplify(), tvm.tir.transform.LowerIntrin()])(
        mod
    )
    func = mod["main"]
    stmt = func.body
    return stmt.value if lower_expr else stmt.body


def check_value(expr, vx, vy, data, fref):
    n = len(data)
    A = te.placeholder((n,), name="A", dtype=expr.dtype)
    B = te.placeholder((n,), name="B", dtype=expr.dtype)

    def make_binds(i):
        x = expr
        x = tvm.tir.Let(vx, A[i], x)
        x = tvm.tir.Let(vy, B[i], x)
        return x

    C = te.compute((n,), make_binds)
    s = te.create_schedule([C.op])

    f = tvm.build(s, [A, B, C], "llvm")
    a = tvm.nd.array(np.array([x for x, y in data], dtype=expr.dtype))
    b = tvm.nd.array(np.array([y for x, y in data], dtype=expr.dtype))
    c = tvm.nd.array(np.zeros(len(data), dtype=expr.dtype))
    f(a, b, c)
    cref = np.array([fref(x, y) for x, y in data])
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
        x = te.var("x", dtype=dtype)
        y = te.var("y", dtype=dtype)
        zero = tvm.tir.const(0, dtype)
        # no constraints
        res = lower_intrin([x, y], tvm.te.floordiv(x, y))
        check_value(res, x, y, data, lambda a, b: a // b)
        # rhs >= 0
        res = lower_intrin([x, y], tvm.tir.Select(y >= 0, tvm.te.floordiv(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a // b if b > 0 else 0)
        # involves max
        res = lower_intrin(
            [x, y], tvm.tir.Select(y >= 0, tvm.te.max(tvm.te.floordiv(x, y), zero), zero)
        )
        check_value(res, x, y, data, lambda a, b: max(a // b, 0) if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(
            [x, y], tvm.tir.Select(tvm.tir.all(y >= 0, x >= 0), tvm.te.floordiv(x, y), zero)
        )
        check_value(res, x, y, data, lambda a, b: a // b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin([x, y], tvm.te.floordiv(x, tvm.tir.const(8, dtype=dtype)))
        check_value(res, x, y, [(a, b) for a, b in data if b == 8], lambda a, b: a // b)


@tvm.testing.requires_llvm
def test_lower_floormod():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = te.var("x", dtype=dtype)
        y = te.var("y", dtype=dtype)
        zero = tvm.tir.const(0, dtype)
        # no constraints
        res = lower_intrin([x, y], tvm.te.floormod(x, y))
        check_value(res, x, y, data, lambda a, b: a % b)
        # rhs >= 0
        res = lower_intrin([x, y], tvm.tir.Select(y >= 0, tvm.te.floormod(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a % b if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(
            [x, y], tvm.tir.Select(tvm.tir.all(y >= 0, x >= 0), tvm.te.floormod(x, y), zero)
        )
        check_value(res, x, y, data, lambda a, b: a % b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin([x, y], tvm.te.floormod(x, tvm.tir.const(8, dtype=dtype)))
        check_value(res, x, y, [(a, b) for a, b in data if b == 8], lambda a, b: a % b)


if __name__ == "__main__":
    test_lower_floordiv()
    test_lower_floormod()
