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
# pylint: disable=missing-docstring


import pytest

import tvm.testing
from tvm import ir, tirx
from tvm.ir import Range
from tvm.script import ir as I
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import ir_builder as T


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


def test_prim_func_symbolic_buffer_param_roundtrip():
    n = tirx.Var("n", "int32")
    A = tirx.decl_buffer(shape=[n + 1, n], dtype="float32", name="A", layout=None)
    func = (
        tirx.PrimFunc(params=[A], body=tirx.Evaluate(n))
        .with_attr("global_symbol", "main")
        .with_attr("s_tir", True)
    )

    source = func.script(extra_config={"script.use_pep695": False})
    assert "T.Buffer((n + 1, n)" in source
    assert source.index('n = I.dynamic("n", dtype="int32")') < source.index("T.evaluate(n)")
    tvm.ir.assert_structural_equal(tvm.script.from_source(source), func)


def test_prim_func_compound_buffer_shape_first_use_roundtrip():
    n = tirx.Var("n", "int32")
    A = tirx.decl_buffer(shape=[tirx.max(n, 1)], dtype="float32", name="A", layout=None)
    func = (
        tirx.PrimFunc(params=[A], body=tirx.Evaluate(n))
        .with_attr("global_symbol", "main")
        .with_attr("s_tir", True)
    )

    source = func.script(extra_config={"script.use_pep695": False})
    assert "T.Buffer((T.max(n, 1),)" in source
    assert source.index('n = I.dynamic("n", dtype="int32")') < source.index("T.evaluate(n)")
    tvm.ir.assert_structural_equal(tvm.script.from_source(source), func)


def test_prim_func_symbolic_alloc_buffer_roundtrip():
    size = tirx.Var("size", "int32")
    buf = tirx.decl_buffer(shape=[size], dtype="float32", name="buf", layout=None)
    func = tirx.PrimFunc(
        params=[],
        body=tirx.SeqStmt([tirx.AllocBuffer(buf), tirx.Evaluate(tirx.BufferLoad(buf, [0]))]),
    ).with_attr("s_tir", True)

    source = func.script()
    assert "T.alloc_buffer((size,))" in source
    tvm.ir.assert_structural_equal(tvm.script.from_source(source, check_well_formed=False), func)


def test_buffer():
    a = tirx.decl_buffer((128, 128), "float16", name="A")
    _assert_print(
        a,
        """A = T.Buffer((128, 128), "float16")
A""",
    )


def test_buffer_region():
    src = tirx.decl_buffer((128, 128), "float32", name="src")
    obj = tirx.BufferRegion(
        src,
        [
            Range(64, 128),
            Range(64, 128),
        ],
    )
    _assert_print(
        obj,
        """
src = T.Buffer((128, 128))
src[64:128, 64:128]
""",
    )


def test_buffer_load():
    a = tirx.decl_buffer((128, 128), "float16", name="A")
    obj = tirx.BufferLoad(a, [128, 128])
    _assert_print(
        obj,
        """
A = T.Buffer((128, 128), "float16")
A[128, 128]
""",
    )


def test_buffer_store():
    a = tirx.decl_buffer((128, 128), "float16", name="A")
    with IRBuilder() as ib:
        T.buffer_store(a, a[128, 128] + 1, [128, 128])
    obj = ib.get()
    _assert_print(
        obj,
        """
A = T.Buffer((128, 128), "float16")
A[128, 128] = A[128, 128] + T.float16(1.0)
""",
    )


def test_for():
    with IRBuilder() as ib:
        with T.grid(128, 128, 128) as (i, j, k):
            ib.name_many(["i", "j", "k"], [i, j, k])
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
for i, j, k in T.grid(128, 128, 128):
    T.evaluate(0)
""",
    )


def test_attr_stmt():
    with IRBuilder() as ib:
        with T.attr("pragma", "unroll", 1):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.attr("pragma", "unroll", 1):
    T.evaluate(0)
""",
    )


def test_assert_stmt():
    with IRBuilder() as ib:
        with T.prim_func():
            T.assert_(True, "assertion")
            T.evaluate(T.call_extern("int32", "after_assert"))
    obj = ib.get().body
    _assert_print(
        obj,
        """
assert T.bool(True), ("RuntimeError", ["assertion"])
T.call_extern("int32", "after_assert")
""",
    )


def test_while():
    with IRBuilder() as ib:
        x = I.dynamic("v", "int32")
        with T.while_(x < 10):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
v = I.dynamic("v", dtype="int32")
while v < 10:
    T.evaluate(0)
""",
    )


def test_allocate():
    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("test")
            buf = T.alloc_buffer([128, 128], "float32")
            T.evaluate(1)
    obj = ib.get()
    _assert_print(
        obj.body,
        """
buffer = T.alloc_buffer((128, 128))
T.evaluate(1)
""",
    )


def test_allocate_with_decl_buffer_sugar():
    # AllocBuffer and DeclBuffer are flat siblings
    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("test")
            buf = T.alloc_buffer([128, 128], "float32")
            buf2 = T.decl_buffer([128, 128], "float32", data=buf.data)
            T.evaluate(1)
    obj = ib.get()
    _assert_print(
        obj.body,
        """
buffer = T.alloc_buffer((128, 128))
buffer_1 = T.decl_buffer((128, 128), data=buffer.data)
T.evaluate(1)
""",
    )


def test_allocate_with_decl_buffer_sugar_multi_usage():
    # AllocBuffer and DeclBuffer are flat siblings
    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("test")
            buf = T.alloc_buffer([128, 128], "float32")
            buf2 = T.decl_buffer([128, 128], "float32", data=buf.data)
            T.evaluate(buf.data)
    obj = ib.get()
    _assert_print(
        obj.body,
        """
buffer = T.alloc_buffer((128, 128))
buffer_1 = T.decl_buffer((128, 128), data=buffer.data)
T.evaluate(buffer.data)
""",
    )


def test_allocate_with_decl_buffer_no_sugar_mismatch():
    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("test")
            buf = T.alloc_buffer([128, 128], "float32")
            buf2 = T.decl_buffer([256, 256], "float32", data=buf.data)
            T.evaluate(buf.data)
    obj = ib.get()
    _assert_print(
        obj.body,
        """
buffer = T.alloc_buffer((128, 128))
buffer_1 = buffer.view(256, 256)
T.evaluate(buffer.data)
""",
    )


def test_decl_buffer():
    # DeclBuffer is flat: we need a frame to hold multiple stmts
    with IRBuilder() as ib:
        with T.prim_func():
            T.func_name("test")
            buf = T.decl_buffer((10, 10), data=T.ptr("float32"))
            T.evaluate(1)
    obj = ib.get()
    # Print only the body (skip PrimFunc wrapper)
    _assert_print(
        obj.body,
        """
v = T.handle("float32", "global")
buffer = T.decl_buffer((10, 10), data=v)
T.evaluate(1)
""",
    )


def test_seq_stmt():
    with IRBuilder() as ib:
        with T.serial(10):
            T.evaluate(1)
            T.evaluate(2)
    obj = ib.get().body
    _assert_print(
        obj,
        """
T.evaluate(1)
T.evaluate(2)
""",
    )


def test_if_then_else():
    with IRBuilder() as ib:
        with T.if_(I.dynamic("v", "int32") == 1):
            with T.then_():
                T.evaluate(0)

    obj = ib.get()
    _assert_print(
        obj,
        """
v = I.dynamic("v", dtype="int32")
if v == 1:
    T.evaluate(0)
""",
    )


def test_evaluate():
    with IRBuilder() as ib:
        T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
T.evaluate(0)
""",
    )


def test_var():
    a = tirx.Var("a", "float32")
    _assert_print(
        a,
        """
a = I.dynamic("a", dtype="float32")
a""",
    )

    a = tirx.Var("a", "handle")
    _assert_print(
        a,
        """
a = T.handle()
a""",
    )

    a = tirx.Var("a", ir.PointerType(ir.PrimType("void"), "shared"))
    _assert_print(
        a,
        """
a = T.handle(storage_scope="shared")
a""",
    )


def test_iter_var():
    a = tirx.IterVar((0, 8), "a", iter_type=tirx.IterVar.DataPar)
    _assert_print(
        a,
        """
a = I.dynamic("a", dtype="int32")
T.iter_var(a, T.Range(0, 8), "DataPar", "")
""",
    )


def test_string_imm():
    s = tvm.ir.StringImm("str")
    _assert_print(s, '"str"')


def test_cast():
    obj = tirx.Cast("float64", tirx.Var("a", "float32"))
    _assert_print(
        obj,
        """
a = I.dynamic("a", dtype="float32")
T.Cast("float64", a)
""",
    )


def test_llvm_intrin_imm():
    a = tirx.call_llvm_intrin("int32x4", "llvm.donothing")
    _assert_print(a, 'T.call_llvm_intrin("int32x4", "llvm.donothing")')
    a = tirx.call_llvm_pure_intrin("int32x4", "llvm.donothing")
    _assert_print(a, 'T.call_llvm_pure_intrin("int32x4", "llvm.donothing")')


def test_binary_arith():
    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    for op, sign in [
        (tirx.Add, "+"),
        (tirx.Sub, "-"),
        (tirx.Mul, "*"),
        (tirx.Mod, "truncmod"),
        (tirx.FloorDiv, "//"),
        (tirx.FloorMod, "%"),
        (tirx.LT, "<"),
        (tirx.LE, "<="),
        (tirx.EQ, "=="),
        (tirx.NE, "!="),
        (tirx.GT, ">"),
        (tirx.GE, ">="),
    ]:
        obj = op(a, b)
        if sign.isalpha():
            expected = f"""
a = I.dynamic("a", dtype="int32")
b = I.dynamic("b", dtype="int32")
T.{sign}(a, b)"""
        else:
            expected = f"""
a = I.dynamic("a", dtype="int32")
b = I.dynamic("b", dtype="int32")
a {sign} b"""
        _assert_print(obj, expected)


def test_binary_arith_const():
    a = tirx.IntImm("int64", 3)
    b = tirx.IntImm("int64", 4)
    for op, name in [
        (tirx.Add, "Add"),
        (tirx.Sub, "Sub"),
        (tirx.Mul, "Mul"),
        (tirx.Div, "Div"),
        (tirx.Mod, "truncmod"),
        (tirx.FloorDiv, "FloorDiv"),
        (tirx.FloorMod, "FloorMod"),
        (tirx.LT, "LT"),
        (tirx.LE, "LE"),
        (tirx.EQ, "EQ"),
        (tirx.NE, "NE"),
        (tirx.GT, "GT"),
        (tirx.GE, "GE"),
    ]:
        obj = op(a, b)
        expected = f"""
T.{name}({a!s}, {b!s})"""
        _assert_print(obj, expected)


def test_int_div():
    a = tirx.Var("a", "int32")
    b = tirx.Var("b", "int32")
    _assert_print(
        tirx.Div(a, b),
        """
a = I.dynamic("a", dtype="int32")
b = I.dynamic("b", dtype="int32")
T.Div(a, b)
""",
    )


def test_logical():
    a = tirx.Var("a", "bool")
    b = tirx.Var("b", "bool")
    _assert_print(
        tirx.And(a, b),
        """
a = I.dynamic("a", dtype="bool")
b = I.dynamic("b", dtype="bool")
a and b
""",
    )
    _assert_print(
        tirx.Or(a, b),
        """
a = I.dynamic("a", dtype="bool")
b = I.dynamic("b", dtype="bool")
a or b
""",
    )
    _assert_print(
        tirx.Not(a),
        """
a = I.dynamic("a", dtype="bool")
not a
""",
    )


def test_select():
    obj = tirx.Select(True, 0, 2)
    _assert_print(
        obj,
        """T.Select(T.bool(True), 0, 2)
""",
    )


@pytest.mark.parametrize(
    "lanes, scripted_lanes", [(32, "32"), (tvm.tirx.vscale() * 8, "T.vscale() * 8")]
)
def test_ramp(lanes, scripted_lanes):
    a = tirx.Var("a", "int32")
    obj = tirx.Ramp(a, 1, lanes)
    _assert_print(
        obj,
        f"""
a = I.dynamic("a", dtype="int32")
T.Ramp(a, 1, {scripted_lanes})
""",
    )


@pytest.mark.parametrize(
    "lanes, scripted_lanes", [(4, "4"), (tvm.tirx.vscale() * 4, "T.vscale() * 4")]
)
def test_broadcast(lanes, scripted_lanes):
    obj = tirx.Broadcast(0, lanes)
    _assert_print(
        obj,
        f"""
T.Broadcast(0, {scripted_lanes})
""",
    )


def test_let_expr():
    x = tirx.Var("x", "int32")
    obj = tirx.Let(x, 1, x + 1)
    _assert_print(
        obj,
        """
x = I.dynamic("x", dtype="int32")
T.Let(x + 1, where={x: 1})
""",
    )


def test_call():
    obj = tirx.atan(T.float32(1.0))
    _assert_print(
        obj,
        """
T.atan(T.float32(1.0))
""",
    )


def test_comm_reducer():
    obj = T.comm_reducer(lambda x, y: x + y, identity=[T.float32(0)])
    _assert_print(
        obj,
        """
T.comm_reducer(lambda x, y: x + y, [T.float32(0.0)])
""",
    )


def test_int_imm():
    obj = T.int16(1)
    _assert_print(
        obj,
        """
T.int16(1)
""",
    )


def test_float_imm():
    obj = T.float16(1)
    _assert_print(
        obj,
        """
T.float16(1.0)
""",
    )


def test_range():
    obj = Range(0, 10)
    _assert_print(
        obj,
        """
I.Range(0, 10)
""",
    )


def test_prim_type():
    obj = ir.PrimType("float32")
    _assert_print(obj, "T.float32")


def test_pointer_type():
    obj = ir.PointerType(ir.PrimType("int32"), "global")
    _assert_print(obj, 'T.handle("int32", "global")')

    obj = ir.PointerType(ir.PrimType("void"))
    _assert_print(obj, "T.handle")

    obj = ir.PointerType(ir.PrimType("void"), "shared")
    _assert_print(obj, 'T.handle(storage_scope="shared")')


def test_tuple_type():
    obj = ir.TupleType([ir.PrimType("float32"), ir.PrimType("int32")])
    _assert_print(obj, "T.Tuple(T.float32, T.int32)")


if __name__ == "__main__":
    tvm.testing.main()
