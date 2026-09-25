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
import operator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import tvm_ffi

import tvm
from tvm import tirx
from tvm.ir import prim
from tvm.sym.int_set import neg_inf, pos_inf


@pytest.mark.parametrize(
    "operation,node_type",
    [
        (operator.add, prim.Add),
        (operator.sub, prim.Sub),
        (operator.mul, prim.Mul),
        (tirx.div, prim.Div),
        (tirx.truncmod, prim.Mod),
        (operator.floordiv, prim.FloorDiv),
        (operator.mod, prim.FloorMod),
        (operator.lt, prim.LT),
        (operator.le, prim.LE),
        (operator.eq, prim.EQ),
        (operator.ne, prim.NE),
        (operator.gt, prim.GT),
        (operator.ge, prim.GE),
        (operator.lshift, prim.LShift),
        (operator.rshift, prim.RShift),
        (operator.and_, prim.BitwiseAnd),
        (operator.or_, prim.BitwiseOr),
        (operator.xor, prim.BitwiseXor),
        (tirx.min, prim.Min),
        (tirx.max, prim.Max),
    ],
)
def test_eager_construction(operation, node_type):
    a, b = prim.IntImm("int32", 6), prim.IntImm("int32", 2)
    with prim.OpConstFoldScope(False):
        expr = operation(a, b)
        assert isinstance(expr, node_type)
        assert expr.a.same_as(a)
        assert expr.b.same_as(b)
        simplified = tvm.sym.Analyzer().simplify(expr)
        assert not prim.op_const_fold_enabled()
    tvm.ir.assert_structural_equal(simplified, tvm.sym.Analyzer().simplify(expr))
    # Raw node constructors preserve nodes regardless of construction policy.
    assert isinstance(node_type(a, b), node_type)


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
def test_default_and_identity_folding(dtype):
    a, b = prim.const(1, dtype), prim.const(2, dtype)
    x = tirx.Var("x", dtype)
    assert (a + b).value == 3
    assert (x + 0).same_as(x)
    with prim.OpConstFoldScope(False):
        assert isinstance(a + b, prim.Add)
        assert isinstance(x + 0, prim.Add)
        assert isinstance(x * 1, prim.Mul)
        assert isinstance(-a, prim.Mul)
    assert (a + b).value == 3


def test_nested_exception_and_reentry():
    scope = prim.OpConstFoldScope(False)
    query = tvm_ffi.get_global_func("prim.op_const_fold_enabled")
    assert query() and prim.op_const_fold_enabled()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unwind"):
            with scope:
                assert not query()
                with prim.OpConstFoldScope(True):
                    assert query()
                    with scope:
                        assert not query()
                    assert query()
                assert not query()
                raise RuntimeError("unwind")
        assert query() and prim.op_const_fold_enabled()


def test_thread_isolation():
    # Reusing one context object must also keep each thread's saved state separate.
    shared = prim.OpConstFoldScope(False)
    barrier = Barrier(2)

    def worker():
        assert prim.op_const_fold_enabled()
        with shared:
            barrier.wait()
            assert not prim.op_const_fold_enabled()
            barrier.wait()
        assert prim.op_const_fold_enabled()

    with ThreadPoolExecutor(1) as executor:
        with prim.OpConstFoldScope(False):
            with shared:
                future = executor.submit(worker)
                barrier.wait()
                assert not prim.op_const_fold_enabled()
                barrier.wait()
            assert not prim.op_const_fold_enabled()
        future.result()
    assert prim.op_const_fold_enabled()


def test_non_kernel_folding_and_normalization():
    one, two = prim.const(1), prim.const(2)
    with prim.OpConstFoldScope(False):
        assert isinstance(one.astype("int64"), prim.Cast)
        assert one.astype("int32").same_as(one)
        promoted = one + prim.const(2, "int64")
        assert promoted.ty.dtype == "int64"
        assert promoted.a.ty.dtype == promoted.b.ty.dtype == "int64"
        vector = prim.Broadcast(one, 4) + two
        assert vector.ty.dtype == "int32x4"
        assert isinstance(vector.b, prim.Broadcast)
        assert isinstance(tirx.if_then_else(prim.const(True), one, two), tvm.ir.Call)
        assert isinstance(tirx.likely(prim.const(True)), tvm.ir.Call)
        assert isinstance(tirx.abs(prim.const(-3)), prim.Select)
        assert isinstance(tirx.ceildiv(one, two), prim.FloorDiv)
        for op in [tirx.abs, tirx.ceil, tirx.floor, tirx.round, tirx.trunc, tirx.nearbyint]:
            assert isinstance(op(prim.const(-1.5, "float32")), tvm.ir.Call)
        with pytest.raises((TypeError, tvm.error.InternalError)):
            tirx.bitwise_and(prim.const(1.0), one)
        with pytest.raises(tvm.error.InternalError, match="Shift amount"):
            one << 32
        with pytest.raises(tvm.error.InternalError, match="Cannot match type"):
            prim.Broadcast(one, 4) + prim.Broadcast(two, 8)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "operation", [tirx.div, tirx.truncmod, tirx.floordiv, tirx.floormod, tirx.ceildiv]
)
def test_invalid_divisor(enabled, operation):
    with prim.OpConstFoldScope(enabled):
        with pytest.raises(tvm.error.InternalError, match="Divide by zero"):
            operation(prim.const(1), prim.const(0))


@pytest.mark.parametrize("enabled", [False, True])
def test_unsigned_subtraction_validation(enabled):
    with prim.OpConstFoldScope(enabled):
        with pytest.raises(tvm.error.InternalError, match="negative uint"):
            prim.const(0, "uint32") - prim.const(1, "uint32")


def test_type_directed_normalization():
    # Intrinsics with an integer-only answer retain their type semantics.
    x = tirx.Var("x", "int32")
    u = tirx.Var("u", "uint32")
    with prim.OpConstFoldScope(False):
        assert tirx.abs(u).same_as(u)
        for op in [tirx.ceil, tirx.floor, tirx.round, tirx.trunc, tirx.nearbyint]:
            assert op(x).same_as(x)
        assert tirx.isnan(x).value == 0
        assert tirx.isinf(x).value == 0
        shape = tvm.relax.ShapeExpr([prim.const(4)])
        assert isinstance(shape.values[0], prim.IntImm)
        assert shape.values[0].ty.dtype == "int64"
        assert not prim.op_const_fold_enabled()


def test_explicit_symbolic_analysis():
    x, y = tirx.Var("x", "int32"), tirx.Var("y", "int32")
    everything = tvm.sym.IntervalSet(neg_inf(), pos_inf())
    expression = prim.Min(x, y)
    indices = [x // 2 * 2 + x % 2]
    ranges = {x: tvm.ir.Range(0, 8), y: tvm.ir.Range(0, 4)}
    with prim.OpConstFoldScope(False):
        analyzer = tvm.sym.Analyzer()
        result = analyzer.int_set(expression, {x: everything, y: everything})
        assert result.is_everything()
        bounded = analyzer.int_set(prim.Min(x, prim.const(4)), {x: everything})
        assert bounded.min_value.same_as(neg_inf())
        assert bounded.max_value.value == 4
        with analyzer.constraint_scope((x + 0) > 0):
            assert analyzer.simplify(x > 0).value == 1
        assert not prim.op_const_fold_enabled()
        coefficients = tvm.sym.detect_linear_equation(4 * x + 2 * y + 3, [x, y])
        assert [coefficient.value for coefficient in coefficients] == [4, 2, 3]
        simplified = tvm.sym.iter_map_simplify(indices, ranges)
        tvm.ir.assert_structural_equal(simplified[0], x)
        assert not prim.op_const_fold_enabled()
