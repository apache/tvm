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
from contextlib import contextmanager

from tvm import ir, tir
from tvm.ir import Range
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T
from tvm.script.printer import default


@contextmanager
def verbose_expr():
    try:
        default.verbose_expr(True)
        yield
    finally:
        default.verbose_expr(False)


def _assert_print(obj, expected):
    with verbose_expr():
        assert repr(obj).strip() == expected.strip()


def test_prim_func():
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    func = tir.PrimFunc(
        params=[a, b],
        ret_type=None,
        buffer_map={
            a: tir.decl_buffer(shape=[128, 128], dtype="float32", name="A"),
            b: tir.decl_buffer(shape=[256, 256], dtype="float32", name="B"),
        },
        body=tir.Evaluate(0),
    )
    _assert_print(
        func,
        expected="""
@T.prim_func
def main(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (256, 256))
    T.evaluate(0)""",
    )


def test_block_realize():
    i = tir.Var("i", "int32")
    j = tir.Var("j", "int32")
    k = tir.Var("k", "int32")
    with IRBuilder() as ib:
        with T.block(name="block", no_realize=False):
            vi = ib.name("vi", T.axis.spatial(128, i))
            vj = ib.name("vj", T.axis.spatial(64, j))
            vk = ib.name("vk", T.axis.reduce(32, k))
            T.reads()
            T.writes()
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
i = T.var("int32")
j = T.var("int32")
k = T.var("int32")
with T.block("block"):
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(64, j)
    vk = T.axis.reduce(32, k)
    T.reads()
    T.writes()
    T.evaluate(0)""",
    )


def test_block():
    i = tir.Var("i", "int32")
    j = tir.Var("j", "int32")
    k = tir.Var("k", "int32")
    with IRBuilder() as ib:
        with T.block(name="block", no_realize=False):
            vi = ib.name("vi", T.axis.spatial(128, i))
            vj = ib.name("vj", T.axis.spatial(64, j))
            vk = ib.name("vk", T.axis.reduce(32, k))
            T.reads()
            T.writes()
            T.evaluate(0)
    obj = ib.get().block
    _assert_print(
        obj,
        """
with T.block("block", no_realize=True):
    vi = T.axis.spatial(128)
    vj = T.axis.spatial(64)
    vk = T.axis.reduce(32)
    T.reads()
    T.writes()
    T.evaluate(0)""",
    )


def test_match_buffer_region():
    src = tir.decl_buffer((128, 128), "float32", name="src")
    tgt = tir.decl_buffer((64, 64), "float32", name="tgt")
    obj = tir.MatchBufferRegion(
        tgt,
        tir.BufferRegion(
            src,
            [
                Range(64, 128),
                Range(64, 128),
            ],
        ),
    )
    _assert_print(
        obj,
        """
src = T.buffer_decl((128, 128))
tgt = T.match_buffer(src[64:128, 64:128], (64, 64))
""",
    )


def test_buffer():
    a = tir.decl_buffer((128, 128), "float16", name="A")
    _assert_print(
        a,
        """A = T.buffer_decl((128, 128), "float16")
A""",
    )


def test_buffer_region():
    src = tir.decl_buffer((128, 128), "float32", name="src")
    obj = tir.BufferRegion(
        src,
        [
            Range(64, 128),
            Range(64, 128),
        ],
    )
    _assert_print(
        obj,
        """
src = T.buffer_decl((128, 128))
src[64:128, 64:128]
""",
    )


def test_buffer_load():
    a = tir.decl_buffer((128, 128), "float16", name="A")
    obj = tir.BufferLoad(a, [128, 128])
    _assert_print(
        obj,
        """
A = T.buffer_decl((128, 128), "float16")
A[128, 128]
""",
    )


def test_buffer_store():
    a = tir.decl_buffer((128, 128), "float16", name="A")
    with IRBuilder() as ib:
        T.buffer_store(a, a[128, 128] + 1, [128, 128])
    obj = ib.get()
    _assert_print(
        obj,
        """
A = T.buffer_decl((128, 128), "float16")
A[128, 128] = A[128, 128] + T.float16(1)
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


def test_let_stmt():
    with IRBuilder() as ib:
        with T.let(T.var("float32"), T.float32(10)):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.let(v, T.float32(10)):
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
        with T.Assert(1, "assertion"):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.Assert(1, "assertion"):
    T.evaluate(0)
""",
    )


def test_while():
    with IRBuilder() as ib:
        x = T.var("int32")
        with T.While(x < 10):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
v = T.var("int32")
while v < 10:
    T.evaluate(0)
""",
    )


def test_allocate():
    with IRBuilder() as ib:
        with T.allocate([128, 128], "float32"):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.allocate([128, 128], "float32", "global") as v:
    T.evaluate(0)
""",
    )


def test_decl_buffer():
    with IRBuilder() as ib:
        with T.decl_buffer((10, 10), data=T.ptr("float32")):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.decl_buffer((10, 10)) as buffer:
    T.evaluate(0)
""",
    )


def test_prefetch():
    a = tir.decl_buffer((128, 128), "float16", name="A")
    with IRBuilder() as ib:
        T.prefetch(a, [Range(0, 64), Range(0, 64)])
    obj = ib.get()
    _assert_print(
        obj,
        """
A = T.buffer_decl((128, 128), "float16")
T.prefetch(A, [T.Range(0, 64), T.Range(0, 64)])
""",
    )


def test_seq_stmt():
    with IRBuilder() as ib:
        with T.serial(10):
            T.evaluate(0)
            T.evaluate(1)
    obj = ib.get().body
    _assert_print(
        obj,
        """
T.evaluate(0)
T.evaluate(1)
""",
    )


def test_if_then_else():
    with IRBuilder() as ib:
        with T.If(T.var("int32") == 1):
            with T.Then():
                T.evaluate(0)

    obj = ib.get()
    _assert_print(
        obj,
        """
v = T.var("int32")
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


def test_buffer_realize():
    with IRBuilder() as ib:
        a = tir.decl_buffer((128, 128), "float32", name="A")
        with T.realize(a[0:128, 0:128], "test_storage_scope", True):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
A = T.buffer_decl((128, 128))
with T.realize(A[0:128, 0:128], "test_storage_scope"):
    T.evaluate(0)
""",
    )


def test_var():
    a = tir.Var("a", "float32")
    _assert_print(
        a,
        """
a = T.var("float32")
a""",
    )


def test_size_var():
    a = tir.SizeVar("a", "float32")
    _assert_print(
        a,
        """
a = T.var("float32")
a""",
    )


def test_iter_var():
    a = tir.IterVar((0, 8), "a", iter_type=tir.IterVar.DataPar)
    _assert_print(
        a,
        """
a = T.var("int32")
T.iter_var(a, T.Range(0, 8), "DataPar", "")
""",
    )


def test_string_imm():
    s = tir.StringImm("str")
    _assert_print(s, '"str"')


def test_cast():
    obj = tir.Cast("float64", tir.Var("a", "float32"))
    _assert_print(
        obj,
        """
a = T.var("float32")
T.Cast("float64", a)
""",
    )


def test_binary_arith():
    a = tir.Var("a", "float32")
    b = tir.Var("b", "float32")
    for op, sign in [
        (tir.Add, "+"),
        (tir.Sub, "-"),
        (tir.Mul, "*"),
        (tir.Div, "/"),
        (tir.Mod, "truncmod"),
        (tir.FloorDiv, "//"),
        (tir.FloorMod, "%"),
        (tir.LT, "<"),
        (tir.LE, "<="),
        (tir.EQ, "=="),
        (tir.NE, "!="),
        (tir.GT, ">"),
        (tir.GE, ">="),
    ]:
        obj = op(a, b)
        if sign.isalpha():
            expected = """
a = T.var("float32")
b = T.var("float32")
T.{}(a, b)""".format(
                sign
            )
        else:
            expected = """
a = T.var("float32")
b = T.var("float32")
a {} b""".format(
                sign
            )
        _assert_print(obj, expected)


def test_logical():
    a = T.var("bool", "a")
    b = T.var("bool", "b")
    _assert_print(
        tir.And(a, b),
        """
a = T.var("bool")
b = T.var("bool")
a and b
""",
    )
    _assert_print(
        tir.Or(a, b),
        """
a = T.var("bool")
b = T.var("bool")
a or b
""",
    )
    _assert_print(
        tir.Not(a),
        """
a = T.var("bool")
not a
""",
    )


def test_select():
    obj = tir.Select(True, 0, 2)
    _assert_print(
        obj,
        """T.Select(True, 0, 2)
""",
    )


def test_ramp():
    a = tir.Var("a", "int32")
    obj = tir.Ramp(a, 1, 32)
    _assert_print(
        obj,
        """
a = T.var("int32")
T.Ramp(a, 1, 32)
""",
    )


def test_broadcast():
    obj = tir.Broadcast(0, 4)
    _assert_print(
        obj,
        """
T.Broadcast(0, 4)
""",
    )


def test_let_expr():
    x = tir.Var("x", "int32")
    obj = tir.Let(x, 1, x + 1)
    _assert_print(
        obj,
        """
x = T.var("int32")
T.let(x, 1, x + 1)
""",
    )


def test_call():
    obj = tir.atan(T.float32(1.0))
    _assert_print(
        obj,
        """
T.atan(T.float32(1))
""",
    )


def test_comm_reducer():
    obj = T.comm_reducer(lambda x, y: x + y, identity=[T.float32(0)])
    _assert_print(
        obj,
        """
T.comm_reducer(lambda x, y: x + y, [T.float32(0)])
""",
    )


def test_any():
    obj = tir.Any()
    _assert_print(
        obj,
        """
T.Any()
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
T.float16(1)
""",
    )


def test_range():
    obj = Range(0, 10)
    _assert_print(
        obj,
        """
T.Range(0, 10)
""",
    )


def test_prim_type():
    obj = ir.PrimType("float32")
    _assert_print(obj, "T.float32")


def test_pointer_type():
    obj = ir.PointerType(ir.PrimType("int32"), "global")
    _assert_print(obj, 'T.Ptr("int32", "global")')


def test_tuple_type():
    obj = ir.TupleType([ir.PrimType("float32"), ir.PrimType("int32")])
    _assert_print(obj, "T.Tuple(T.float32, T.int32)")


if __name__ == "__main__":
    test_prim_func()
    test_block_realize()
    test_block()
    test_buffer()
    test_buffer_region()
    test_buffer_load()
    test_buffer_store()
    test_match_buffer_region()
    test_for()
    test_let_stmt()
    test_attr_stmt()
    test_assert_stmt()
    test_while()
    test_allocate()
    test_decl_buffer()
    test_prefetch()
    test_seq_stmt()
    test_if_then_else()
    test_evaluate()
    test_buffer_realize()
    test_var()
    test_size_var()
    test_iter_var()
    test_string_imm()
    test_cast()
    test_binary_arith()
    test_logical()
    test_select()
    test_ramp()
    test_broadcast()
    test_let_expr()
    test_call()
    test_comm_reducer()
    test_any()
    test_int_imm()
    test_float_imm()
    test_range()
    test_prim_type()
    test_pointer_type()
    test_tuple_type()
