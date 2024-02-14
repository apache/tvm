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

import re

import tvm.testing
from tvm import ir, tir
from tvm.ir import Range
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


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
    ).with_attr("global_symbol", "main")
    _assert_print(
        func,
        expected="""
# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(0)""",
    )


def test_prim_func_no_sugar_inlined_buffer():
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    func = tir.PrimFunc(
        params=[a, b],
        ret_type=None,
        buffer_map={
            a: tir.decl_buffer(shape=[128, 128], dtype="float32", name="A"),
            b: tir.decl_buffer(shape=[256, 256], dtype="float32", name="B"),
        },
        body=tir.Evaluate(a),
    ).with_attr("global_symbol", "main")
    _assert_print(
        func,
        expected="""
# from tvm.script import tir as T

@T.prim_func
def main(a: T.handle, B: T.Buffer((256, 256), "float32")):
    A = T.match_buffer(a, (128, 128))
    T.evaluate(a)
""",
    )


def test_prim_func_no_sugar_shared_buffer_data():
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    buffer_data = tir.decl_buffer(shape=[128, 128], dtype="float32", name="A").data
    func = tir.PrimFunc(
        params=[a, b],
        ret_type=None,
        buffer_map={
            a: tir.decl_buffer(shape=[128, 128], dtype="float32", name="A", data=buffer_data),
            b: tir.decl_buffer(shape=[256, 256], dtype="float32", name="B", data=buffer_data),
        },
        body=tir.Evaluate(0),
    ).with_attr("global_symbol", "main")
    _assert_print(
        func,
        expected="""
# from tvm.script import tir as T

@T.prim_func
def main(a: T.handle, b: T.handle):
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (256, 256), data=A.data)
    T.evaluate(0)
""",
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
i = T.int32()
j = T.int32()
k = T.int32()
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
src = T.Buffer((128, 128))
tgt = T.match_buffer(src[64:128, 64:128], (64, 64))
""",
    )


def test_buffer():
    a = tir.decl_buffer((128, 128), "float16", name="A")
    _assert_print(
        a,
        """A = T.Buffer((128, 128), "float16")
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
src = T.Buffer((128, 128))
src[64:128, 64:128]
""",
    )


def test_buffer_load():
    a = tir.decl_buffer((128, 128), "float16", name="A")
    obj = tir.BufferLoad(a, [128, 128])
    _assert_print(
        obj,
        """
A = T.Buffer((128, 128), "float16")
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
A = T.Buffer((128, 128), "float16")
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
        with T.LetStmt(T.float32(10)) as v:
            ib.name("v", v)
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.LetStmt(T.float32(10)) as v:
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
        with T.Assert(True, "assertion"):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.Assert(T.bool(True), "assertion"):
    T.evaluate(0)
""",
    )


def test_while():
    with IRBuilder() as ib:
        x = T.int32()
        with T.While(x < 10):
            T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
v = T.int32()
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


def test_allocate_with_decl_buffer_sugar():
    with IRBuilder() as ib:
        with T.allocate([128, 128], "float32") as buffer_data:
            with T.decl_buffer([128, 128], "float32", data=buffer_data) as buffer:
                T.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.decl_buffer((128, 128)) as buffer:
    T.evaluate(0)
""",
    )


def test_allocate_with_decl_buffer_sugar_multi_usage():
    with IRBuilder() as ib:
        with T.allocate([128, 128], "float32") as buffer_data:
            with T.decl_buffer([128, 128], "float32", data=buffer_data) as buffer:
                T.evaluate(buffer_data)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.decl_buffer((128, 128)) as buffer:
    T.evaluate(buffer.data)
""",
    )


def test_allocate_with_decl_buffer_no_sugar_mismatch():
    with IRBuilder() as ib:
        with T.allocate([128, 128], "float32") as buffer_data:
            with T.decl_buffer([256, 256], "float32", data=buffer_data) as buffer:
                T.evaluate(buffer_data)
    obj = ib.get()
    _assert_print(
        obj,
        """
with T.allocate([128, 128], "float32", "global") as v:
    buffer = T.decl_buffer((256, 256), data=v)
    T.evaluate(v)
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
v = T.handle("float32", "global")
with T.decl_buffer((10, 10), data=v) as buffer:
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
A = T.Buffer((128, 128), "float16")
T.prefetch(A, [T.Range(0, 64), T.Range(0, 64)])
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
        with T.If(T.int32() == 1):
            with T.Then():
                T.evaluate(0)

    obj = ib.get()
    _assert_print(
        obj,
        """
v = T.int32()
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
A = T.Buffer((128, 128))
with T.realize(A[0:128, 0:128], "test_storage_scope"):
    T.evaluate(0)
""",
    )


def test_var():
    a = tir.Var("a", "float32")
    _assert_print(
        a,
        """
a = T.float32()
a""",
    )


def test_size_var():
    a = tir.SizeVar("a", "float32")
    _assert_print(
        a,
        """
a = T.float32(is_size_var=True)
a""",
    )


def test_iter_var():
    a = tir.IterVar((0, 8), "a", iter_type=tir.IterVar.DataPar)
    _assert_print(
        a,
        """
a = T.int32()
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
a = T.float32()
T.Cast("float64", a)
""",
    )


def test_llvm_intrin_imm():
    a = tir.call_llvm_intrin("int32x4", "llvm.donothing", T.uint32(0))
    _assert_print(a, 'T.call_llvm_intrin("int32x4", "llvm.donothing", T.uint32(0))')
    a = tir.call_llvm_pure_intrin("int32x4", "llvm.donothing", T.uint32(0))
    _assert_print(a, 'T.call_llvm_pure_intrin("int32x4", "llvm.donothing", T.uint32(0))')


def test_binary_arith():
    a = tir.Var("a", "int32")
    b = tir.Var("b", "int32")
    for op, sign in [
        (tir.Add, "+"),
        (tir.Sub, "-"),
        (tir.Mul, "*"),
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
a = T.int32()
b = T.int32()
T.{}(a, b)""".format(
                sign
            )
        else:
            expected = """
a = T.int32()
b = T.int32()
a {} b""".format(
                sign
            )
        _assert_print(obj, expected)


def test_binary_arith_const():
    a = tir.IntImm("int64", 3)
    b = tir.IntImm("int64", 4)
    for op, name in [
        (tir.Add, "Add"),
        (tir.Sub, "Sub"),
        (tir.Mul, "Mul"),
        (tir.Div, "Div"),
        (tir.Mod, "truncmod"),
        (tir.FloorDiv, "FloorDiv"),
        (tir.FloorMod, "FloorMod"),
        (tir.LT, "LT"),
        (tir.LE, "LE"),
        (tir.EQ, "EQ"),
        (tir.NE, "NE"),
        (tir.GT, "GT"),
        (tir.GE, "GE"),
    ]:
        obj = op(a, b)
        expected = """
T.{}({}, {})""".format(
            name, str(a), str(b)
        )
        _assert_print(obj, expected)


def test_int_div():
    a = tir.Var("a", "int32")
    b = tir.Var("b", "int32")
    _assert_print(
        tir.Div(a, b),
        """
a = T.int32()
b = T.int32()
T.Div(a, b)
""",
    )


def test_logical():
    a = tir.Var("a", "bool")
    b = tir.Var("b", "bool")
    _assert_print(
        tir.And(a, b),
        """
a = T.bool()
b = T.bool()
a and b
""",
    )
    _assert_print(
        tir.Or(a, b),
        """
a = T.bool()
b = T.bool()
a or b
""",
    )
    _assert_print(
        tir.Not(a),
        """
a = T.bool()
not a
""",
    )


def test_select():
    obj = tir.Select(True, 0, 2)
    _assert_print(
        obj,
        """T.Select(T.bool(True), 0, 2)
""",
    )


def test_ramp():
    a = tir.Var("a", "int32")
    obj = tir.Ramp(a, 1, 32)
    _assert_print(
        obj,
        """
a = T.int32()
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
x = T.int32()
T.Let(x + 1, where={x: 1})
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
I.Range(0, 10)
""",
    )


def test_prim_type():
    obj = ir.PrimType("float32")
    _assert_print(obj, "T.float32")


def test_pointer_type():
    obj = ir.PointerType(ir.PrimType("int32"), "global")
    _assert_print(obj, 'T.handle("int32", "global")')


def test_tuple_type():
    obj = ir.TupleType([ir.PrimType("float32"), ir.PrimType("int32")])
    _assert_print(obj, "T.Tuple(T.float32, T.int32)")


def test_remap():
    from tvm.script import tir as T

    @T.prim_func
    def block_with_remap_implicitly():
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
            with T.block("update"):
                v0 = T.axis.spatial(128, i0 + 1)
                v1 = T.axis.spatial(128, i1)
                v2 = T.axis.reduce(128, i2)
                v3 = T.axis.spatial(128, i3 - 1)
                v4 = T.axis.reduce(128, i4)
                v5 = T.axis.spatial(128, i5)

    @T.prim_func
    def block_with_remap_explicitly():
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
            with T.block("update"):
                v0 = T.axis.spatial(128, i0 + 1)
                v1, v2 = T.axis.remap("SR", [i1, i2])
                v3 = T.axis.spatial(128, i3 - 1)
                v4, v5 = T.axis.remap("RS", [i4, i5])

    expected_output = """
# from tvm.script import tir as T

@T.prim_func
def main():
    # with T.block("root"):
    for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
        with T.block("update"):
            v0 = T.axis.spatial(128, i0 + 1)
            v1, v2 = T.axis.remap("SR", [i1, i2])
            v3 = T.axis.spatial(128, i3 - 1)
            v4, v5 = T.axis.remap("RS", [i4, i5])
            T.reads()
            T.writes()
            T.evaluate(0)"""
    _assert_print(block_with_remap_explicitly.with_attr("global_symbol", "main"), expected_output)
    _assert_print(block_with_remap_implicitly.with_attr("global_symbol", "main"), expected_output)


def test_root_block():
    from tvm.script import tir as T

    @T.prim_func
    def root_block_implicitly():
        a = T.alloc_buffer([128, 128])
        for i, j in T.grid(128, 128):
            with T.block():
                T.evaluate(0)

    @T.prim_func
    def root_block_explicitly():
        with T.block("root"):
            a = T.alloc_buffer([128, 128])
            for i, j in T.grid(128, 128):
                with T.block():
                    T.evaluate(0)

    expected_output = """
# from tvm.script import tir as T

@T.prim_func
def main():
    # with T.block("root"):
    a = T.alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with T.block(""):
            T.reads()
            T.writes()
            T.evaluate(0)
    """
    _assert_print(root_block_implicitly.with_attr("global_symbol", "main"), expected_output)
    _assert_print(root_block_explicitly.with_attr("global_symbol", "main"), expected_output)


def test_private_primfunc():
    from tvm.script import tir as T

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
# from tvm.script import tir as T

@T.prim_func(private=True)
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(0)""",
    )


def test_prim_func_different_symbol():
    from tvm.script import tir as T

    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
        T.func_attr({"global_symbol": "func"})
        T.evaluate(0)

    expected_output = """
# from tvm.script import tir as T

@T.prim_func
def func(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(0)
    """
    _assert_print(main, expected_output)


def test_variable_with_cpp_address():
    """The show_object_address option displays the C++ addressess

    Because the C++ address may vary with each execution, the output
    produced with this option cannot be compared to a fixed string.
    Instead, this test uses the normal script output to generate a
    regular expression against with the test output must match.  The
    regular expression validates that all names have been appended
    with "_0x" followed by a hexadecimal number, and that the address
    is the same for each variable.
    """
    from tvm.script import tir as T

    # The test function has all named objects suffixed with "_name",
    # to avoid spurious replacement when generating the expected
    # regex.
    @T.prim_func
    def func(a_name: T.handle):
        N_name = T.int64()
        A_name = T.match_buffer(a_name, N_name, "float32")
        for i_name in range(N_name):
            A_name[i_name] = A_name[i_name] + 1.0

    without_address = func.script(show_object_address=False)
    script = func.script(show_object_address=True)

    expected_regex = re.escape(without_address)
    for name in ["a_name", "A_name", "N_name", "i_name"]:
        # Replace all occurrences with a backref to an earlier match
        expected_regex = expected_regex.replace(name, rf"(?P={name})")
        # Then replace the first such backref with a capturing group.
        expected_regex = expected_regex.replace(
            rf"(?P={name})", rf"(?P<{name}>{name}_0x[A-Fa-f0-9]+)", 1
        )

    assert re.match(expected_regex, script)


if __name__ == "__main__":
    tvm.testing.main()
