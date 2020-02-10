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

def lower_intrin(stmt):
    """wrapper to call transformation in stmt"""
    lower_expr = isinstance(stmt, tvm.expr.PrimExpr)
    stmt = tvm.stmt.Evaluate(stmt) if lower_expr else stmt
    stmt = tvm.ir_pass.CanonicalSimplify(stmt)
    stmt  = tvm.ir_pass._LowerIntrinStmt(stmt, "llvm")
    return stmt.value if lower_expr else stmt.body


def check_value(expr, vx, vy, data, fref):
    n = len(data)
    A = tvm.placeholder((n,), name="A", dtype=expr.dtype)
    B = tvm.placeholder((n,), name="B", dtype=expr.dtype)

    def make_binds(i):
        x = expr
        x = tvm.expr.Let(vx, A[i], x)
        x = tvm.expr.Let(vy, B[i], x)
        return x

    C = tvm.compute((n,), make_binds)
    s = tvm.create_schedule([C.op])

    if not tvm.module.enabled("llvm"):
        return

    f = tvm.build(s, [A, B, C], "llvm")
    a = tvm.nd.array(np.array([x for x, y in data], dtype=expr.dtype))
    b = tvm.nd.array(np.array([y for x, y in data], dtype=expr.dtype))
    c = tvm.nd.array(np.zeros(len(data), dtype=expr.dtype))
    f(a, b, c)
    cref = np.array([fref(x, y) for x, y in data])
    np.testing.assert_equal(c.asnumpy(), cref)



def get_ref_data():
    """Get reference data for every pairs"""
    import itertools
    x = range(-10, 10)
    y = list(range(-10, 10))
    y.remove(0)
    return list(itertools.product(x, y))


def test_lower_floordiv():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = tvm.var("x", dtype=dtype)
        y = tvm.var("y", dtype=dtype)
        zero = tvm.const(0, dtype)
        # no constraints
        res = lower_intrin(tvm.floordiv(x, y))
        check_value(res, x, y, data, lambda a, b: a // b)
        # rhs >= 0
        res = lower_intrin(tvm.expr.Select(y >= 0, tvm.floordiv(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a // b if b > 0 else 0)
        # involves max
        res = lower_intrin(tvm.expr.Select(y >= 0, tvm.max(tvm.floordiv(x, y), zero), zero))
        check_value(res, x, y, data, lambda a, b: max(a // b, 0) if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(tvm.expr.Select(tvm.all(y >= 0, x >= 0), tvm.floordiv(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a // b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin(tvm.floordiv(x, tvm.const(8, dtype=dtype)))
        check_value(res, x, y, [(a, b) for a, b in data if b == 8], lambda a, b: a // b)


def test_lower_floormod():
    data = get_ref_data()
    for dtype in ["int32", "int64", "int16"]:
        x = tvm.var("x", dtype=dtype)
        y = tvm.var("y", dtype=dtype)
        zero = tvm.const(0, dtype)
        # no constraints
        res = lower_intrin(tvm.floormod(x, y))
        check_value(res, x, y, data, lambda a, b: a % b)
        # rhs >= 0
        res = lower_intrin(tvm.expr.Select(y >= 0, tvm.floormod(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a % b if b > 0 else 0)
        # lhs >= 0
        res = lower_intrin(tvm.expr.Select(tvm.all(y >= 0, x >= 0), tvm.floormod(x, y), zero))
        check_value(res, x, y, data, lambda a, b: a % b if b > 0 and a >= 0 else 0)
        # const power of two
        res = lower_intrin(tvm.floormod(x, tvm.const(8, dtype=dtype)))
        check_value(res, x, y, [(a, b) for a, b in data if b == 8], lambda a, b: a % b)



if __name__ == "__main__":
    test_lower_floordiv()
    test_lower_floormod()
