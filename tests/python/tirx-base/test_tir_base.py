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
# ruff: noqa: E711, F841
import itertools

import numpy as np
import pytest
import tvm_ffi

import tvm
from tvm import tirx
from tvm.ir.transform import PassContext
from tvm.script import tirx as T


def build_tir_func(func):
    func = func.with_attr("global_symbol", "main")
    pass_ctx = PassContext.current()
    if pass_ctx.config.get("tirx.noalias", True):
        func = func.with_attr("tirx.noalias", True)
    mod = tvm.IRModule({"main": func})
    func = tvm.compile(mod)
    return func


def test_scalar_add():
    # All these types should be interchangeable with each other
    # E.g. float16 + float32 upconverts the float16 --> float32
    # Meanwhile if an int or float or together the int will be
    # cast to the float type.
    lhs_types = ["float32", "float16", "int32", "int64"]
    rhs_types = ["float32", "float16"]
    for lhs_type, rhs_type in itertools.product(lhs_types, rhs_types):
        # Input vars should be float32, we will cast to test for upcasting between them
        lhs_input = tirx.Var("lhs", "float32")
        rhs_input = tirx.Var("rhs", "float32")
        lhs = tirx.Cast(lhs_type, lhs_input)
        rhs = tirx.Cast(rhs_type, rhs_input)
        output = lhs + rhs
        output = tirx.Return(output)
        func = tirx.PrimFunc([lhs_input, rhs_input], output)
        func = build_tir_func(func)
        out = func(1.0, 2.0)
        assert out == 3.0


def assignment_helper(store_dtype, value_dtype):
    store = tirx.Var("store", ty=store_dtype)
    value = tirx.Var("value", ty=value_dtype)
    tirx.Let(store, value, body=store)


def test_fail_implicit_downcasts_same_type():
    # These lists should be sorted
    bits = [8, 16, 32, 64]
    for type in ["float", "int", "uint"]:
        for i in range(len(bits) - 1):
            with pytest.raises(RuntimeError):
                assignment_helper(
                    store_dtype=f"{type}{bits[i]}", value_dtype=f"{type}{bits[i + 1]}"
                )


def test_cast_between_types():
    # We should only be able to assign values with the same types
    bits = [16, 32]
    types = ["float", "int", "uint"]
    for store_type, store_bits, value_type, value_bits in itertools.product(
        types, bits, types, bits
    ):
        store_dtype = f"{store_type}{store_bits}"
        value_dtype = f"{value_type}{value_bits}"
        if store_dtype == value_dtype:
            assignment_helper(store_dtype, value_dtype)
        else:
            # TODO: we might want to allow casts between uint and int types
            with pytest.raises(RuntimeError):
                assignment_helper(store_dtype, value_dtype)


def test_return_const():
    a = tirx.const(0)
    b = tirx.Return(a)
    func = tirx.PrimFunc([], b)
    func = build_tir_func(func)
    out = func()
    assert out == 0


def test_return_accepts_expr_and_roundtrips():
    value = tvm.relax.ShapeExpr([2, 3])
    span = tvm.ir.Span(tvm.ir.SourceName("return_test"), 1, 1, 1, 9)
    stmt = tirx.Return(value, span)

    assert stmt.value.same_as(value)
    assert stmt.span.same_as(span)
    assert not tvm.ir.is_prim_expr(stmt.value)

    restored = tvm.ir.load_json(tvm.ir.save_json(stmt))
    tvm.ir.assert_structural_equal(restored, stmt)
    assert tvm_ffi.structural_hash(restored) == tvm_ffi.structural_hash(stmt)

    with pytest.raises(tvm.error.InternalError):
        tirx.Return(None)


def test_stmt_span_not_structural():
    span_a = tvm.ir.Span(tvm.ir.SourceName("a.py"), 1, 1, 1, 2)
    span_b = tvm.ir.Span(tvm.ir.SourceName("b.py"), 10, 10, 3, 4)
    stmt_a = tirx.Evaluate(tirx.IntImm("int32", 0), span_a)
    stmt_b = tirx.Evaluate(tirx.IntImm("int32", 0), span_b)

    assert tvm_ffi.structural_equal(stmt_a, stmt_b)
    assert tvm_ffi.structural_hash(stmt_a) == tvm_ffi.structural_hash(stmt_b)


def test_return_stmt_functor_traversal_and_mutation():
    x = tirx.Var("x", "int32")
    span = tvm.ir.Span(tvm.ir.SourceName("return_test"), 1, 1, 1, 9)
    stmt = tirx.Return(x + 1, span)
    visited = []

    tirx.stmt_functor.post_order_visit(stmt, visited.append)
    assert any(node.same_as(x) for node in visited)
    assert any(isinstance(node, tirx.Return) for node in visited)

    rewritten = tirx.stmt_functor.substitute(stmt, {x: tirx.IntImm("int32", 4)})
    expected = tirx.Return(tirx.Add(tirx.IntImm("int32", 4), tirx.IntImm("int32", 1)), span)
    tvm.ir.assert_structural_equal(rewritten, expected)
    assert rewritten.span.same_as(span)


def test_control_flow_jump():
    @T.prim_func(s_tir=True)
    def func(a: T.float32, b: T.float32):
        if True:
            return a
        return b

    func = build_tir_func(func)
    out = func(1.0, 2.0)
    assert out == 1.0


def test_break_loop():
    @T.prim_func(s_tir=True)
    def func(In: T.Buffer((2,), "int32"), Out: T.Buffer((2,), "int32")):
        Out[0] = 0
        Out[1] = 1
        for i in range(10):
            for j in range(10):
                if i * 10 + j == In[0]:
                    Out[0] = i + j
                    break
            if Out[0] > 0:
                break
        while Out[1] > 0:
            Out[1] = Out[1] + 1
            if Out[1] > In[1]:
                break

    func = build_tir_func(func)
    a = np.asarray([49, 8], "int32")
    b = np.zeros([2], "int32")
    if not hasattr(b, "__dlpack__"):
        return
    func(a, b)
    assert b[0] == 13
    assert b[1] == 9


def test_continue_loop():
    @T.prim_func(s_tir=True)
    def func(Out: T.Buffer((2,), "int32")):
        T.func_attr({"global_symbol": "main"})
        Out[0] = 0
        Out[1] = 0
        for i in range(10):
            for j in range(10):
                if (i * 10 + j) % 3 != 0:
                    continue
                Out[0] = Out[0] + 1
        k = T.decl_buffer([], "int32")
        k[()] = 0
        while k[()] < Out[0]:
            k[()] = k[()] + 1
            if k[()] % 6 == 0:
                Out[1] = Out[1] + 1
                continue

    func = build_tir_func(func)
    b = np.zeros([2], "int32")
    if not hasattr(b, "__dlpack__"):
        return
    func(b)
    assert b[0] == 34
    assert b[1] == 5


def test_exception():
    with pytest.raises(TypeError):
        x = tirx.Var(name=1, ty="int")


def test_eq_ops():
    # NOTE: the `== None` / `!= None` below are intentional and must NOT be
    # rewritten as `is None` / `is not None`. This test exercises the overloaded
    # `__eq__` / `__ne__` operators on `IntImm` / `StringImm`; the `is` operators
    # bypass those overloads and would defeat the test.
    a = tirx.IntImm("int8", 1)
    with pytest.raises(ValueError):
        assert a != None
    with pytest.raises(ValueError):
        assert not a == None
    b = tirx.StringImm("abc")
    assert b != None
    assert not b == None


if __name__ == "__main__":
    test_scalar_add()
    test_return_const()
    test_control_flow_jump()
    test_exception()
    test_eq_ops()
