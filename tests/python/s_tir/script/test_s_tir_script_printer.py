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
# ruff: noqa: E501, F841
"""S-TIR script printer."""

import io
import re
import tokenize

import pytest
from tvm_ffi.access_path import AccessPath

import tvm
import tvm.testing
from tvm import IRModule, s_tir, tirx
from tvm.ir import Range
from tvm.s_tir.script.ir_builder import prim_func as build_prim_func
from tvm.script import ir as I
from tvm.script import ir_builder as IB
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import ir_builder as TB


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
    tvm.ir.assert_structural_equal(
        tvm.script.from_source(
            source, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir}
        ),
        func,
    )


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
    tvm.ir.assert_structural_equal(
        tvm.script.from_source(
            source, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir}
        ),
        func,
    )


def test_prim_func_symbolic_alloc_buffer_roundtrip():
    size = tirx.Var("size", "int32")
    buf = tirx.decl_buffer(shape=[size], dtype="float32", name="buf", layout=None)
    func = tirx.PrimFunc(
        params=[],
        body=tirx.SeqStmt([tirx.AllocBuffer(buf), tirx.Evaluate(tirx.BufferLoad(buf, [0]))]),
    ).with_attr("s_tir", True)

    source = func.script()
    assert "T.alloc_buffer((size,))" in source
    tvm.ir.assert_structural_equal(
        tvm.script.from_source(
            source,
            check_well_formed=False,
            extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
        ),
        func,
    )


def test_prim_func():
    A = tirx.decl_buffer(shape=[128, 128], dtype="float32", name="A")
    B = tirx.decl_buffer(shape=[256, 256], dtype="float32", name="B")
    func = (
        tirx.PrimFunc(
            params=[A, B],
            ret_type=None,
            body=tirx.Evaluate(0),
        )
        .with_attr("global_symbol", "main")
        .with_attr("s_tir", True)
    )
    _assert_print(
        func,
        expected="""
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(0)""",
    )


def test_prim_func_buffer_data_use():
    A = tirx.decl_buffer(shape=[128, 128], dtype="float32", name="A")
    B = tirx.decl_buffer(shape=[256, 256], dtype="float32", name="B")
    func = (
        tirx.PrimFunc(
            params=[A, B],
            ret_type=None,
            body=tirx.Evaluate(A.data),
        )
        .with_attr("global_symbol", "main")
        .with_attr("s_tir", True)
    )
    _assert_print(
        func,
        expected="""
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(A.data)
""",
    )


def test_prim_func_buffer_data_argument_is_scope_hint():
    buffer_data = tirx.decl_buffer(shape=[128, 128], dtype="float32", name="A").data
    A = tirx.decl_buffer(shape=[128, 128], dtype="float32", name="A", data=buffer_data)
    B = tirx.decl_buffer(shape=[256, 256], dtype="float32", name="B", data=buffer_data)
    func = (
        tirx.PrimFunc(
            params=[A, B],
            ret_type=None,
            body=tirx.Evaluate(0),
        )
        .with_attr("global_symbol", "main")
        .with_attr("s_tir", True)
    )
    _assert_print(
        func,
        expected="""
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(0)
""",
    )


def test_block_realize():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    k = tirx.Var("k", "int32")
    with IRBuilder() as ib:
        with Ts.sblock(name="block", no_realize=False):
            vi = ib.name("vi", Ts.axis.spatial(128, i))
            vj = ib.name("vj", Ts.axis.spatial(64, j))
            vk = ib.name("vk", Ts.axis.reduce(32, k))
            Ts.reads()
            Ts.writes()
            TB.evaluate(0)
    obj = ib.get()
    _assert_print(
        obj,
        """
i = I.dynamic("i", dtype="int32")
j = I.dynamic("j", dtype="int32")
k = I.dynamic("k", dtype="int32")
with Ts.sblock("block"):
    vi = Ts.axis.spatial(128, i)
    vj = Ts.axis.spatial(64, j)
    vk = Ts.axis.reduce(32, k)
    Ts.reads()
    Ts.writes()
    T.evaluate(0)""",
    )


def test_block():
    i = tirx.Var("i", "int32")
    j = tirx.Var("j", "int32")
    k = tirx.Var("k", "int32")
    with IRBuilder() as ib:
        with Ts.sblock(name="block", no_realize=False):
            vi = ib.name("vi", Ts.axis.spatial(128, i))
            vj = ib.name("vj", Ts.axis.spatial(64, j))
            vk = ib.name("vk", Ts.axis.reduce(32, k))
            Ts.reads()
            Ts.writes()
            TB.evaluate(0)
    obj = ib.get().block
    _assert_print(
        obj,
        """
with Ts.sblock("block", no_realize=True):
    vi = Ts.axis.spatial(128)
    vj = Ts.axis.spatial(64)
    vk = Ts.axis.reduce(32)
    Ts.reads()
    Ts.writes()
    T.evaluate(0)""",
    )


def test_match_buffer_region():
    src = tirx.decl_buffer((128, 128), "float32", name="src")
    tgt = tirx.decl_buffer((64, 64), "float32", name="tgt")
    obj = s_tir.MatchBufferRegion(
        tgt,
        tirx.BufferRegion(
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
tgt = Ts.match_buffer(src[64:128, 64:128], (64, 64))
""",
    )


def test_bind():
    with IRBuilder() as ib:
        with build_prim_func():
            v = TB.bind(TB.float32(10))
            ib.name("v", v)
            TB.evaluate(1)
    obj = ib.get()
    _assert_print(
        obj,
        """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func(private=True)
def main():
    v: T.let[T.float32] = T.float32(10.0)
    T.evaluate(1)
""",
    )


def test_remap():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def block_with_remap_implicitly():
        for i0, i1, i2, i3, i4, i5 in TB.grid(128, 128, 128, 128, 128, 128):
            with Ts.sblock("update"):
                v0 = Ts.axis.spatial(128, i0 + 1)
                v1 = Ts.axis.spatial(128, i1)
                v2 = Ts.axis.reduce(128, i2)
                v3 = Ts.axis.spatial(128, i3 - 1)
                v4 = Ts.axis.reduce(128, i4)
                v5 = Ts.axis.spatial(128, i5)

    @Ts.prim_func
    def block_with_remap_explicitly():
        for i0, i1, i2, i3, i4, i5 in TB.grid(128, 128, 128, 128, 128, 128):
            with Ts.sblock("update"):
                v0 = Ts.axis.spatial(128, i0 + 1)
                v1, v2 = Ts.axis.remap("SR", [i1, i2])
                v3 = Ts.axis.spatial(128, i3 - 1)
                v4, v5 = Ts.axis.remap("RS", [i4, i5])

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main():
    # with Ts.sblock("root"):
    for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
        with Ts.sblock("update"):
            v = Ts.axis.spatial(128, i0 + 1)
            v_1, v_2 = Ts.axis.remap("SR", [i1, i2])
            v_3 = Ts.axis.spatial(128, i3 - 1)
            v_4, v_5 = Ts.axis.remap("RS", [i4, i5])
            Ts.reads()
            Ts.writes()
            T.evaluate(0)"""
    _assert_print(block_with_remap_explicitly.with_attr("global_symbol", "main"), expected_output)
    _assert_print(block_with_remap_implicitly.with_attr("global_symbol", "main"), expected_output)


def test_root_block():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def root_block_implicitly():
        a = Ts.sblock_alloc_buffer([128, 128])
        for i, j in TB.grid(128, 128):
            with Ts.sblock():
                TB.evaluate(0)

    @Ts.prim_func
    def root_block_explicitly():
        with Ts.sblock("root"):
            a = Ts.sblock_alloc_buffer([128, 128])
            for i, j in TB.grid(128, 128):
                with Ts.sblock():
                    TB.evaluate(0)

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main():
    # with Ts.sblock("root"):
    buffer = Ts.sblock_alloc_buffer((128, 128))
    for i, j in T.grid(128, 128):
        with Ts.sblock(""):
            Ts.reads()
            Ts.writes()
            T.evaluate(0)
    """
    _assert_print(root_block_implicitly.with_attr("global_symbol", "main"), expected_output)
    _assert_print(root_block_explicitly.with_attr("global_symbol", "main"), expected_output)


def test_private_primfunc():
    A = tirx.decl_buffer(shape=[128, 128], dtype="float32", name="A")
    B = tirx.decl_buffer(shape=[256, 256], dtype="float32", name="B")
    func = tirx.PrimFunc(
        params=[A, B],
        ret_type=None,
        body=tirx.Evaluate(0),
    ).with_attr("s_tir", True)
    _assert_print(
        func,
        expected="""
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func(private=True)
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.evaluate(0)""",
    )


def test_prim_func_different_symbol():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main(A: TB.Buffer((128, 128), "float32"), B: TB.Buffer((256, 256), "float32")):
        TB.func_attr({"global_symbol": "func"})
        TB.evaluate(0)

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
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
    from tvm.script import tirx as TB

    # The test function has all named objects suffixed with "_name",
    # to avoid spurious replacement when generating the expected
    # regex.
    N_name = I.dynamic("N_name")

    @Ts.prim_func
    def func(A_name: TB.Buffer(N_name, "float32")):
        for i_name in range(N_name):
            A_name[i_name] = A_name[i_name] + 1.0

    without_address = func.script(show_object_address=False)
    script = func.script(show_object_address=True)

    # Address suffixes belong to identifiers, not display-name string literals
    # passed to constructors such as I.dynamic("N_name").
    names = {"a_name", "A_name", "N_name", "i_name"}
    line_offsets = [0]
    for line in without_address.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(line))
    parts = []
    seen = set()
    cursor = 0
    for token in tokenize.generate_tokens(io.StringIO(without_address).readline):
        name = token.string
        if token.type != tokenize.NAME or name not in names:
            continue
        start = line_offsets[token.start[0] - 1] + token.start[1]
        end = line_offsets[token.end[0] - 1] + token.end[1]
        parts.append(re.escape(without_address[cursor:start]))
        parts.append(rf"(?P={name})" if name in seen else rf"(?P<{name}>{name}_0x[A-Fa-f0-9]+)")
        seen.add(name)
        cursor = end
    parts.append(re.escape(without_address[cursor:]))
    assert re.fullmatch("".join(parts), script)


def test_return_statement():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def func():
        return TB.int32(5)

    expected_output = """
# from tvm.script import s_tir as Ts

@Ts.prim_func
def func():
    return 5
    """
    _assert_print(func, expected_output)
    assert func.script(verbose_expr=True, syntax_sugar=False).strip() == expected_output.strip()


CUSTOM_FLOAT_DTYPES = [
    # Float8 variants
    "float8_e3m4",
    "float8_e4m3",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
    # Float6 variants
    "float6_e2m3fn",
    "float6_e3m2fn",
    # Float4 variant
    "float4_e2m1fn",
]


@pytest.mark.parametrize("dtype", CUSTOM_FLOAT_DTYPES)
def test_custom_float_types(dtype):
    from tvm.script import tirx as TB

    @Ts.prim_func
    def func():
        TB.evaluate(getattr(TB, dtype)(0.0))

    expected_output = f"""
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def func():
    T.evaluate(T.{dtype}(0.0))
"""
    _assert_print(func, expected_output)


def test_predicated_load_store():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main(A: TB.Buffer((128, 128), "float32"), B: TB.Buffer((256, 256), "float32")):
        TB.func_attr({"global_symbol": "func"})
        a_load = TB.meta_var(
            TB.call_intrin(
                "float32x4",
                "tirx.masked_load",
                A,
                0,
                TB.Ramp(0, 4, 4),
                TB.Broadcast(TB.bool(False), 4),
            )
        )
        TB.evaluate(
            TB.call_intrin(
                "void",
                "tirx.masked_store",
                A,
                a_load,
                0,
                TB.Ramp(0, 2, 4),
                TB.Broadcast(TB.bool(False), 4),
            )
        )

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def func(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    a_load: T.let[T.float32x4] = T.masked_load("float32x4", A, 0, T.Ramp(0, 4, 4), T.Broadcast(T.bool(False), 4))
    T.masked_store(A, a_load, 0, T.Ramp(0, 2, 4), T.Broadcast(T.bool(False), 4))
    """
    _assert_print(main, expected_output)


def test_predicated_buffer_load_store():
    a = tirx.Var("a", "handle")
    b = tirx.Var("b", "handle")
    buffers = {
        a: tirx.decl_buffer(shape=[128, 128], dtype="float32", name="A"),
        b: tirx.decl_buffer(shape=[256, 256], dtype="float32", name="B"),
    }
    buffer_load = tirx.call_intrin(
        "float32x4",
        "tirx.masked_load",
        buffers[b],
        0,
        tirx.Ramp(0, 4, 4),
        tirx.Broadcast(tirx.IntImm("bool", 0), 4),
    )
    body = tirx.Evaluate(
        tirx.call_intrin(
            "void",
            "tirx.masked_store",
            buffers[a],
            buffer_load,
            0,
            tirx.Ramp(0, 2, 4),
            tirx.Broadcast(tirx.IntImm("bool", 0), 4),
        )
    )
    func = tirx.PrimFunc(
        params=[buffers[a], buffers[b]],
        ret_type=None,
        body=body,
    ).with_attr("s_tir", True)

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func(private=True)
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    T.masked_store(A, T.masked_load("float32x4", B, 0, T.Ramp(0, 4, 4), T.Broadcast(T.bool(False), 4)), 0, T.Ramp(0, 2, 4), T.Broadcast(T.bool(False), 4))
    """
    _assert_print(func, expected_output)


def test_predicated_scalable_load_store():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main(A: TB.Buffer((128, 128), "float32"), B: TB.Buffer((256, 256), "float32")):
        TB.func_attr({"global_symbol": "func"})
        mask = TB.meta_var(TB.get_active_lane_mask("uint1xvscalex4", 0, 13))
        a_load = TB.meta_var(
            TB.call_intrin(
                "float32xvscalex4", "tirx.masked_load", A, 0, TB.Ramp(0, 4, TB.vscale() * 4), mask
            )
        )
        TB.evaluate(
            TB.call_intrin(
                "void", "tirx.masked_store", A, a_load, 0, TB.Ramp(0, 2, TB.vscale() * 4), mask
            )
        )

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def func(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
    mask: T.let[T.uint1xvscalex4] = T.get_active_lane_mask("uint1xvscalex4", 0, 13)
    a_load: T.let[T.float32xvscalex4] = T.masked_load("float32xvscalex4", A, 0, T.Ramp(0, 4, T.vscale() * 4), mask)
    T.masked_store(A, a_load, 0, T.Ramp(0, 2, T.vscale() * 4), mask)
    """
    _assert_print(main, expected_output)


def test_masked_load_prevents_scalar_allocation_init_fusion():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main():
        A = TB.alloc_buffer((1,), "float32x4")
        A[0] = TB.masked_load("float32x4", A, 0, TB.Broadcast(TB.bool(True), 4))

    source = main.script()
    assert "A = T.alloc_buffer" in source
    assert "A[0] = T.masked_load" in source
    tvm.ir.assert_structural_equal(
        tvm.script.from_source(
            source, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir}
        ),
        main,
    )


def test_vload_with_explicit_scalable_data_type():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main(A: TB.Buffer((128,), "float32"), B: TB.Buffer((128,), "float32")):
        B[0 : TB.vscale() * 4] = A.vload([TB.Ramp(0, 1, TB.vscale() * 4)], dtype="float32xvscalex4")

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
    B[0:T.vscale() * 4] = A[T.Ramp(0, 1, T.vscale() * 4)]
    """
    _assert_print(main, expected_output)


def test_vectorize_llvm_pure_intrin():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main(A: TB.Buffer((4,), "float32"), B: TB.Buffer((4,), "float32")):
        A[TB.Ramp(0, 1, 4)] = TB.call_llvm_pure_intrin(
            "float32x4", "llvm.sqrt", B[TB.Ramp(0, 1, 4)]
        )

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    A[0:4] = T.call_llvm_pure_intrin("float32x4", "llvm.sqrt", B[T.Ramp(0, 1, 4)])
    """
    _assert_print(main, expected_output)


def test_func_with_loop_jumps():
    from tvm.script import tirx as TB

    @Ts.prim_func
    def main(A: TB.Buffer((4,), "float32"), B: TB.Buffer((4,), "float32")):
        for i in range(1000):
            if i % 13 == 0:
                A[1] = A[1] + 1
                continue
            if A[0] >= B[0]:
                break

    expected_output = """
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
    for i in range(1000):
        if i % 13 == 0:
            A[1] = A[1] + T.float32(1.0)
            continue
        if A[0] >= B[0]:
            break
    """
    _assert_print(main, expected_output)


def opt_gemm_lower():
    """Representative vector GEMM lowering with base and offset MAC accesses."""

    @tvm.script.ir_module
    class Module:
        @Ts.prim_func
        def mmult(
            A_1: T.Buffer([16384], elem_offset=0, align=64, offset_factor=1),
            B_1: T.Buffer([1024, 1024], elem_offset=0, align=64, offset_factor=1),
            C_1: T.Buffer([16384], elem_offset=0, align=64, offset_factor=1),
        ) -> None:
            # function attr dict
            T.func_attr({"tirx.noalias": True})
            # body
            packedB = T.alloc_buffer((32768,))
            for x in T.parallel(0, 32):
                for y in T.serial(0, 1024):
                    packedB[T.ramp(((x * 32768) + (y * 32)), 1, 32)] = B_1[y, T.ramp(x * 32, 1, 32)]
            for x_outer in T.parallel(0, 32):
                C_global = T.alloc_buffer((1024,))
                for y_outer in T.serial(0, 32):
                    for x_c_init in T.serial(0, 32):
                        C_global[T.ramp((x_c_init * 32), 1, 32)] = T.broadcast(T.float32(0), 32)
                    for k_outer in T.serial(0, 256):
                        for x_c in T.serial(0, 32):
                            C_global[T.ramp((x_c * 32), 1, 32)] = C_global[
                                T.ramp((x_c * 32), 1, 32)
                            ] + (
                                T.broadcast(
                                    A_1[(((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4))],
                                    32,
                                )
                                * packedB[T.ramp(((y_outer * 32768) + (k_outer * 128)), 1, 32)]
                            )
                            C_global[T.ramp((x_c * 32), 1, 32)] = C_global[
                                T.ramp((x_c * 32), 1, 32)
                            ] + (
                                T.broadcast(
                                    A_1[
                                        ((((x_outer * 32768) + (x_c * 1024)) + (k_outer * 4)) + 3),
                                    ],
                                    32,
                                )
                                * packedB[
                                    T.ramp((((y_outer * 32768) + (k_outer * 128)) + 96), 1, 32)
                                ]
                            )
                    for x_inner in T.serial(0, 32):
                        for y_inner in T.serial(0, 32):
                            C_1[
                                (
                                    (((x_outer * 32768) + (x_inner * 1024)) + (y_outer * 32))
                                    + y_inner
                                )
                            ] = C_global[((x_inner * 32) + y_inner)]

    return Module


def opt_conv_tensorcore_lower():
    """Representative WMMA lowering with scalar/vector copies and zero/nonzero fragments."""

    @Ts.prim_func
    def func(
        A: T.Buffer((16, 14, 14, 16, 16, 16), "float16"),
        W: T.Buffer((3, 3, 16, 32, 16, 16), "float16"),
        Conv: T.Buffer((16, 14, 14, 32, 16, 16), "float32"),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "default_function", "tirx.noalias": True})
        # body
        A_1 = T.decl_buffer([12845056], dtype="float16", data=A.data)
        W_1 = T.decl_buffer([1179648], dtype="float16", data=W.data)
        Conv_1 = T.decl_buffer([25690112], data=Conv.data)
        bx = T.env_thread("blockIdx.x")
        by = T.env_thread("blockIdx.y")
        bz = T.env_thread("blockIdx.z")
        tx = T.env_thread("threadIdx.x")
        ty = T.env_thread("threadIdx.y")
        tz = T.env_thread("threadIdx.z")
        T.launch_thread(bz, 196)
        Conv_wmma_accumulator = T.alloc_buffer((2048,), scope="wmma.accumulator")
        Apad_shared = T.alloc_buffer((12288,), "float16", scope="shared")
        W_shared = T.alloc_buffer((12288,), "float16", scope="shared")
        Apad_shared_wmma_matrix_a = T.alloc_buffer((512,), "float16", scope="wmma.matrix_a")
        W_shared_wmma_matrix_b = T.alloc_buffer((1024,), "float16", scope="wmma.matrix_b")
        T.launch_thread(bx, 2)
        T.launch_thread(by, 4)
        T.launch_thread(ty, 4)
        T.launch_thread(tz, 2)
        T.evaluate(
            T.tvm_fill_fragment(
                Conv_wmma_accumulator.data, 16, 16, 16, 0, T.float32(0), dtype="handle"
            )
        )
        T.evaluate(
            T.tvm_fill_fragment(
                Conv_wmma_accumulator.data, 16, 16, 16, 7, T.float32(0), dtype="handle"
            )
        )
        for ic_outer in T.serial(0, 8):
            for kh in T.serial(0, 3):
                for ax2 in T.serial(0, 3):
                    with T.launch_thread(tx, 32):
                        Apad_shared[((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx)] = (
                            T.if_then_else(
                                (
                                    (
                                        (
                                            (1 <= (T.floordiv(bz, 14) + kh))
                                            and ((T.floordiv(bz, 14) + kh) < 15)
                                        )
                                        and (1 <= (ax2 + T.floormod(bz, 14)))
                                    )
                                    and ((ax2 + T.floormod(bz, 14)) < 15)
                                ),
                                A_1[
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                ((bx * 6422528) + (ty * 1605632))
                                                                + (tz * 802816)
                                                            )
                                                            + (kh * 57344)
                                                        )
                                                        + (bz * 4096)
                                                    )
                                                    + (ax2 * 4096)
                                                )
                                                + (ic_outer * 512)
                                            )
                                            + tx
                                        )
                                        - 61440
                                    ),
                                ],
                                T.float16(0),
                                dtype="float16",
                            )
                        )
                    T.launch_thread(tx, 32)
                    Apad_shared[(((((ty * 3072) + (tz * 1536)) + (ax2 * 512)) + tx) + 480)] = (
                        T.if_then_else(
                            (
                                (
                                    (
                                        (1 <= (T.floordiv(bz, 14) + kh))
                                        and ((T.floordiv(bz, 14) + kh) < 15)
                                    )
                                    and (1 <= (ax2 + T.floormod(bz, 14)))
                                )
                                and ((ax2 + T.floormod(bz, 14)) < 15)
                            ),
                            A_1[
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            ((bx * 6422528) + (ty * 1605632))
                                                            + (tz * 802816)
                                                        )
                                                        + (kh * 57344)
                                                    )
                                                    + (bz * 4096)
                                                )
                                                + (ax2 * 4096)
                                            )
                                            + (ic_outer * 512)
                                        )
                                        + tx
                                    )
                                    - 60960
                                ),
                            ],
                            T.float16(0),
                            dtype="float16",
                        )
                    )
                with T.launch_thread(tx, 32):
                    W_shared[T.ramp((((ty * 512) + (tz * 256)) + (tx * 8)), 1, 8)] = W_1[
                        T.ramp(
                            (
                                (
                                    (
                                        (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                        + (ty * 512)
                                    )
                                    + (tz * 256)
                                )
                                + (tx * 8)
                            ),
                            1,
                            8,
                        )
                    ]
                with T.launch_thread(tx, 32):
                    W_shared[T.ramp(((((ty * 512) + (tz * 256)) + (tx * 8)) + 10240), 1, 8)] = W_1[
                        T.ramp(
                            (
                                (
                                    (
                                        (
                                            (((kh * 393216) + (ic_outer * 16384)) + (by * 2048))
                                            + (ty * 512)
                                        )
                                        + (tz * 256)
                                    )
                                    + (tx * 8)
                                )
                                + 270336
                            ),
                            1,
                            8,
                        )
                    ]
                for ic_inner in T.serial(0, 2):
                    for kw in T.serial(0, 3):
                        T.evaluate(
                            T.tvm_load_matrix_sync(
                                Apad_shared_wmma_matrix_a.data,
                                16,
                                16,
                                16,
                                0,
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float16"),
                                    Apad_shared.data,
                                    (((ty * 3072) + (kw * 512)) + (ic_inner * 256)),
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                        T.evaluate(
                            T.tvm_load_matrix_sync(
                                Apad_shared_wmma_matrix_a.data,
                                16,
                                16,
                                16,
                                1,
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float16"),
                                    Apad_shared.data,
                                    ((((ty * 3072) + (kw * 512)) + (ic_inner * 256)) + 1536),
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                        T.evaluate(
                            T.tvm_load_matrix_sync(
                                W_shared_wmma_matrix_b.data,
                                16,
                                16,
                                16,
                                0,
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float16"),
                                    W_shared.data,
                                    (((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)),
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                        T.evaluate(
                            T.tvm_load_matrix_sync(
                                W_shared_wmma_matrix_b.data,
                                16,
                                16,
                                16,
                                3,
                                T.tvm_access_ptr(
                                    T.type_annotation(dtype="float16"),
                                    W_shared.data,
                                    ((((kw * 4096) + (ic_inner * 2048)) + (tz * 1024)) + 768),
                                    256,
                                    1,
                                    dtype="handle",
                                ),
                                16,
                                "row_major",
                                dtype="handle",
                            )
                        )
                        T.evaluate(
                            T.tvm_mma_sync(
                                Conv_wmma_accumulator.data,
                                0,
                                Apad_shared_wmma_matrix_a.data,
                                0,
                                W_shared_wmma_matrix_b.data,
                                0,
                                Conv_wmma_accumulator.data,
                                0,
                                dtype="handle",
                            )
                        )
                        T.evaluate(
                            T.tvm_mma_sync(
                                Conv_wmma_accumulator.data,
                                7,
                                Apad_shared_wmma_matrix_a.data,
                                1,
                                W_shared_wmma_matrix_b.data,
                                3,
                                Conv_wmma_accumulator.data,
                                7,
                                dtype="handle",
                            )
                        )
        T.evaluate(
            T.tvm_store_matrix_sync(
                Conv_wmma_accumulator.data,
                16,
                16,
                16,
                0,
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"),
                    Conv_1.data,
                    (
                        ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                        + (tz * 1024)
                    ),
                    256,
                    2,
                    dtype="handle",
                ),
                16,
                "row_major",
                dtype="handle",
            )
        )
        T.evaluate(
            T.tvm_store_matrix_sync(
                Conv_wmma_accumulator.data,
                16,
                16,
                16,
                7,
                T.tvm_access_ptr(
                    T.type_annotation(dtype="float32"),
                    Conv_1.data,
                    (
                        (
                            ((((bx * 12845056) + (ty * 3211264)) + (bz * 8192)) + (by * 2048))
                            + (tz * 1024)
                        )
                        + 1606400
                    ),
                    256,
                    2,
                    dtype="handle",
                ),
                16,
                "row_major",
                dtype="handle",
            )
        )

    return func


def opt_conv_tensorcore_mod_host():
    """Representative packed host ABI checks and device/kernel calls."""

    @Ts.prim_func
    def opt_conv_tensorcore_mod_host(
        args: T.handle,
        arg_type_ids: T.Buffer((3,), "int32"),
        num_args: T.int32,
        out_ret_value: T.handle,
        out_ret_tcode: T.handle,
        resource_handle: T.handle,
    ) -> T.int32:
        # function attr dict
        T.func_attr(
            {
                "tirx.noalias": True,
                "global_symbol": "default_function",
                "tirx.is_entry_func": True,
                "calling_conv": 1,
            }
        )
        # body
        stack_tcode_data: T.let[T.handle("int32")] = T.tvm_stack_alloca(
            "arg_tcode", 10, dtype="handle"
        )
        stack_tcode = T.decl_buffer([9], "int32", data=stack_tcode_data)
        stack_value: T.let[T.handle] = T.tvm_stack_alloca("arg_value", 10, dtype="handle")
        assert num_args == 3, "default_function: num_args should be 3"
        arg0: T.let[T.handle] = T.tvm_struct_get(args, 0, 12, dtype="handle")
        arg0_code: T.let[T.int32] = arg_type_ids[0]
        arg1: T.let[T.handle] = T.tvm_struct_get(args, 1, 12, dtype="handle")
        arg2: T.let[T.handle] = T.tvm_struct_get(args, 2, 12, dtype="handle")

        A: T.let[T.handle] = T.tvm_struct_get(arg0, 0, 1, dtype="handle")
        T.attr(A, "storage_alignment", 128)
        arg0_shape_data: T.let[T.handle("int64")] = T.tvm_struct_get(
            arg0, 0, 2, dtype=T.handle("int64").ty
        )
        arg0_shape = T.decl_buffer([6], "int64", data=arg0_shape_data)
        arg0_strides_data: T.let[T.handle("int64")] = T.tvm_struct_get(
            arg0, 0, 3, dtype=T.handle("int64").ty
        )
        arg0_strides = T.decl_buffer([6], "int64", data=arg0_strides_data)

        dev_id: T.let[T.int32] = T.tvm_struct_get(arg0, 0, 9, dtype="int32")

        W: T.let[T.handle] = T.tvm_struct_get(arg1, 0, 1, dtype="handle")
        T.attr(W, "storage_alignment", 128)

        Conv: T.let[T.handle] = T.tvm_struct_get(arg2, 0, 1, dtype="handle")
        T.attr(Conv, "storage_alignment", 128)

        assert (((arg0_code == 3) or (arg0_code == 13)) or (arg0_code == 7)) or (arg0_code == 4), (
            "default_function: Expect arg[0] to be pointer"
        )
        assert 6 == T.tvm_struct_get(arg0, 0, 4, dtype="int32"), "arg0.ndim is expected to equal 6"
        assert (
            (T.tvm_struct_get(arg0, 0, 5, dtype="uint8") == T.uint8(2))
            and (T.tvm_struct_get(arg0, 0, 6, dtype="uint8") == T.uint8(16))
        ) and (T.tvm_struct_get(arg0, 0, 7, dtype="uint16") == T.uint16(1)), (
            "arg0.dtype is expected to be float16"
        )
        assert 16 == T.cast(arg0_shape[0], "int32"), (
            "Argument arg0.shape[0] has an unsatisfied constraint"
        )
        assert 14 == T.cast(arg0_shape[1], "int32"), (
            "Argument arg0.shape[1] has an unsatisfied constraint"
        )
        if not (T.isnullptr(arg0_strides.data, dtype="bool")):
            assert (
                (
                    (
                        (
                            (1 == T.cast(arg0_strides[5], "int32"))
                            and (16 == T.cast(arg0_strides[4], "int32"))
                        )
                        and (256 == T.cast(arg0_strides[3], "int32"))
                    )
                    and (4096 == T.cast(arg0_strides[2], "int32"))
                )
                and (57344 == T.cast(arg0_strides[1], "int32"))
            ) and (802816 == T.cast(arg0_strides[0], "int32")), (
                "arg0.strides: expected to be compact array"
            )
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(arg0, 0, 8, dtype="uint64"), (
            "Argument arg0.byte_offset has an unsatisfied constraint"
        )
        assert 2 == T.tvm_struct_get(arg0, 0, 10, dtype="int32"), (
            "Argument arg0.device_type has an unsatisfied constraint"
        )
        assert (
            (T.tvm_struct_get(arg2, 0, 5, dtype="uint8") == T.uint8(2))
            and (T.tvm_struct_get(arg2, 0, 6, dtype="uint8") == T.uint8(32))
        ) and (T.tvm_struct_get(arg2, 0, 7, dtype="uint16") == T.uint16(1)), (
            "arg2.dtype is expected to be float32"
        )
        assert dev_id == T.tvm_struct_get(arg2, 0, 9, dtype="int32"), (
            "Argument arg2.device_id has an unsatisfied constraint"
        )
        T.evaluate(T.tvm_struct_set(stack_value, 0, 12, T.cast(2, "int64"), dtype="int32"))
        stack_tcode[0] = 0
        T.evaluate(T.tvm_struct_set(stack_value, 1, 12, T.cast(dev_id, "int64"), dtype="int32"))
        stack_tcode[1] = 0
        T.evaluate(T.tvm_call_packed_lowered("__tvm_set_device", stack_value, 0, 2, dtype="int32"))
        T.attr(0, "compute_scope", "default_function_compute_")
        T.evaluate(T.tvm_struct_set(stack_value, 0, 12, A, dtype="int32"))
        stack_tcode[0] = 3
        T.evaluate(T.tvm_struct_set(stack_value, 1, 12, W, dtype="int32"))
        stack_tcode[1] = 3
        T.evaluate(T.tvm_struct_set(stack_value, 2, 12, Conv, dtype="int32"))
        stack_tcode[2] = 3
        T.evaluate(T.tvm_struct_set(stack_value, 3, 12, T.cast(196, "int64"), dtype="int32"))
        stack_tcode[3] = 0
        T.evaluate(T.tvm_struct_set(stack_value, 4, 12, T.cast(2, "int64"), dtype="int32"))
        stack_tcode[4] = 0
        T.evaluate(T.tvm_struct_set(stack_value, 5, 12, T.cast(4, "int64"), dtype="int32"))
        stack_tcode[5] = 0
        T.evaluate(T.tvm_struct_set(stack_value, 6, 12, T.cast(4, "int64"), dtype="int32"))
        stack_tcode[6] = 0
        T.evaluate(T.tvm_struct_set(stack_value, 7, 12, T.cast(2, "int64"), dtype="int32"))
        stack_tcode[7] = 0
        T.evaluate(T.tvm_struct_set(stack_value, 8, 12, T.cast(32, "int64"), dtype="int32"))
        stack_tcode[8] = 0
        T.evaluate(
            T.tvm_call_packed_lowered("default_function_kernel0", stack_value, 0, 9, dtype="int32")
        )

    return opt_conv_tensorcore_mod_host


def select():
    @Ts.prim_func
    def select(A: T.Buffer((), "float32")) -> None:
        A[()] = T.Select(True, 1, 2)

    return select


def minmax():
    @Ts.prim_func
    def minmax(A: T.Buffer((), "float32")) -> None:
        A[()] = T.min(1, 2)
        A[()] = T.max(1, 2)

    return minmax


def abs():
    @Ts.prim_func
    def abs(A: T.Buffer((128, 128), "float32")) -> None:
        for i, j in T.grid(128, 128):
            with Ts.sblock("A"):
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = T.abs(A[vi, vj])

    return abs


def constant_folding():
    @Ts.prim_func
    def constant_folding(A: T.Buffer((), "float32")) -> None:
        A[()] = T.min(2.2, 5.2)
        A[()] = T.max(T.float32(2.2), T.float32(T.float32(5.2)))
        A[()] = T.min(2.2, 5.0)

    return constant_folding


def simplify_bracket():
    # uninitialized variables
    a = T.dynamic("a", "int32")
    b = T.dynamic("b", "int32")
    c = T.dynamic("c", "int32")
    d = T.dynamic("d", "int32")

    @Ts.prim_func(check_well_formed=False)
    def simplify_bracket() -> None:
        T.evaluate(a + b * (c + d))

    return simplify_bracket


def var_with_same_name():
    @Ts.prim_func
    def var_with_same_name(A: T.Buffer((16, 16), "float32")) -> None:
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = 0
        for i, j in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                A[vi, vj] = 0

    return var_with_same_name


def test_same_name_var():
    func = var_with_same_name()
    out_str = func.script()
    rt_func = tvm.script.from_source(
        out_str,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(func, rt_func)
    assert out_str.count("for i, j in T.grid(16, 16)") == 2
    assert out_str.find("i_") == -1
    assert out_str.find("i_") == -1


def primfunc_with_allocate_annotations():
    @Ts.prim_func
    def primfunc_with_allocate_annotations(
        placeholder_29: T.Buffer([802816], dtype="uint8", elem_offset=0, align=64, offset_factor=1),
        T_cast_7: T.Buffer([200704], dtype="int16", elem_offset=0, align=64, offset_factor=1),
    ) -> None:
        # function attr dict
        T.func_attr(
            {"global_symbol": "tvmgen_default_fused_nn_max_pool2d_cast", "tirx.noalias": True}
        )

        # body
        tensor_2 = T.alloc_buffer((200704,), "uint8", annotations={"attr1_key": "attr1_value"})
        for ax0_ax1_fused_4 in T.serial(0, 56):
            for ax2_4 in T.serial(0, 56):
                for ax3_init in T.serial(0, 64):
                    tensor_2[(((ax0_ax1_fused_4 * 3584) + (ax2_4 * 64)) + ax3_init)] = T.uint8(0)
                for rv0_rv1_fused_1, ax3_2 in T.grid(9, 64):
                    tensor_2[(((ax0_ax1_fused_4 * 3584) + (ax2_4 * 64)) + ax3_2)] = T.max(
                        tensor_2[(((ax0_ax1_fused_4 * 3584) + (ax2_4 * 64)) + ax3_2)],
                        T.if_then_else(
                            (
                                (((ax0_ax1_fused_4 * 2) + T.floordiv(rv0_rv1_fused_1, 3)) < 112)
                                and (((ax2_4 * 2) + T.floormod(rv0_rv1_fused_1, 3)) < 112)
                            ),
                            placeholder_29[
                                (
                                    (
                                        (
                                            (
                                                (ax0_ax1_fused_4 * 14336)
                                                + (T.floordiv(rv0_rv1_fused_1, 3) * 7168)
                                            )
                                            + (ax2_4 * 128)
                                        )
                                        + (T.floormod(rv0_rv1_fused_1, 3) * 64)
                                    )
                                    + ax3_2
                                )
                            ],
                            T.uint8(0),
                            dtype="uint8",
                        ),
                    )
        for ax0_ax1_fused_5 in T.serial(0, 56):
            for ax2_5, ax3_3 in T.grid(56, 64):
                T_cast_7[(((ax0_ax1_fused_5 * 3584) + (ax2_5 * 64)) + ax3_3)] = T.cast(
                    tensor_2[(((ax0_ax1_fused_5 * 3584) + (ax2_5 * 64)) + ax3_3)], "int16"
                )

    return primfunc_with_allocate_annotations


def comm_reducer_single_reduce_group():
    @Ts.prim_func
    def comm_reducer_single_reduce_group(
        A: T.Buffer([16384], dtype="float32"), b: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        threadIdx_x = T.env_thread("threadIdx.x")

        for i in T.serial(0, 128):
            T.launch_thread(threadIdx_x, 128)
            reduce_temp0 = T.alloc_buffer((1,), scope="local")
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.float32(0)]), "reduce_scope", T.int32(0)
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        A[i * 128 + threadIdx_x],
                        True,
                        reduce_temp0.data,
                        threadIdx_x,
                        dtype="handle",
                    )
                )

    return comm_reducer_single_reduce_group


def comm_reducer_multiple_reduce_groups():
    @Ts.prim_func
    def comm_reducer_multiple_reduce_groups(
        A: T.Buffer([16384], dtype="float32"), b: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        threadIdx_x = T.env_thread("threadIdx.x")

        for i in T.serial(0, 128):
            T.launch_thread(threadIdx_x, 128)
            reduce_temp0 = T.alloc_buffer((1,), scope="local")
            with T.attr(
                T.comm_reducer(
                    lambda x0, x1, y0, y1: (
                        T.Select((x1 >= y1), x0, y0),
                        T.Select((x1 >= y1), x1, y1),
                    ),
                    [T.int32(-1), T.min_value("float32")],
                ),
                "reduce_scope",
                T.int32(0),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        A[i * 128 + threadIdx_x],
                        True,
                        reduce_temp0.data,
                        threadIdx_x,
                        dtype="handle",
                    )
                )

    return comm_reducer_multiple_reduce_groups


def multiple_commreducer():
    # normal_reduce_temp0 is treated as uninitialized value
    @Ts.prim_func(check_well_formed=False)
    def multiple_commreducer() -> None:
        normal_reduce_temp0 = T.Buffer([1], dtype="float32", strides=[1], scope="local")
        normal_reduce_temp1 = T.Buffer([1], dtype="float32", strides=[1], scope="local")
        reduce_temp0 = T.Buffer([1], dtype="float32", strides=[1], scope="local")
        reduce_temp1 = T.Buffer([1], dtype="float32", strides=[1], scope="local")
        for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
            with Ts.sblock("T_softmax_maxelem_cross_thread_reduction"):
                T.attr(
                    T.comm_reducer(lambda x, y: T.max(x, y), [T.min_value("float32")]),
                    "reduce_scope",
                    T.int32(0),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0.data,
                        ax0_1,
                        dtype="handle",
                    )
                )
        for ax0_1 in T.thread_binding(0, 32, thread="threadIdx.x"):
            with Ts.sblock("T_softmax_expsum_cross_thread_reduction"):
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]), "reduce_scope", T.int32(0)
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp1[0],
                        True,
                        reduce_temp1.data,
                        ax0_1,
                        dtype="handle",
                    )
                )

    return multiple_commreducer


def func_div_mod():
    # not well-formed: free variables
    a = T.dynamic("a", "int32")
    b = T.dynamic("b", "int32")

    @Ts.prim_func(check_well_formed=False)
    def func_div_mod():
        T.evaluate(a // b)
        T.evaluate(a % b)
        T.evaluate(T.truncmod(a, b))

    return func_div_mod


def test_div_mod():
    func = func_div_mod()
    rt_func = tvm.script.from_source(
        func.script(),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(func, rt_func, True)

    assert isinstance(func.body[0].value, tvm.tirx.FloorDiv)
    assert isinstance(func.body[1].value, tvm.tirx.FloorMod)
    assert isinstance(func.body[2].value, tvm.tirx.Mod)


def func_with_target_spec_by_config():
    @Ts.prim_func
    def func_with_target_spec_by_config() -> None:
        T.func_attr(
            {
                "kTarget": T.target(
                    {
                        "max_num_threads": 1024,
                        "arch": "sm_70",
                        "thread_warp_size": 32,
                        "kind": "cuda",
                        "tag": "",
                        "keys": ["cuda", "gpu"],
                        "host": T.target({"kind": "llvm", "tag": "", "keys": ["cpu"]}),
                    }
                )
            }
        )
        T.evaluate(0)

    return func_with_target_spec_by_config


def func_with_target_spec_by_str():
    @Ts.prim_func
    def func_with_target_spec_by_str() -> None:
        T.func_attr({"kTarget": T.target("nvidia/nvidia-a100")})
        T.evaluate(0)

    return func_with_target_spec_by_str


def func_with_target_and_host_spec_by_str():
    @Ts.prim_func
    def func():
        T.func_attr({"target": T.target("nvidia/nvidia-a100", host="llvm")})
        T.evaluate(0)

    return func


def func_T_ptr_let_statement():
    @Ts.prim_func
    def func_T_ptr_let_statement(
        args: T.handle, arg_type_ids_handle: T.handle("int32"), num_args: T.int32
    ) -> None:
        # The T.Ptr declaration in the parameter list should parse
        # correctly, and should be usable as the data pointer in a buffer.
        arg_type_ids = T.decl_buffer([2], dtype="int32", data=arg_type_ids_handle)

        arg0: T.let[T.handle] = T.tvm_struct_get(args, 0, 12, dtype="handle")
        arg1: T.let[T.handle] = T.tvm_struct_get(args, 1, 12, dtype="handle")

        # The ABI field is an opaque pointer.  Retag it explicitly before
        # binding it to the buffer's exact element pointer type.
        A_data: T.let[T.handle("float32")] = T.reinterpret(
            T.handle("float32").ty,
            T.tvm_struct_get(arg0, 0, 1, dtype="handle"),
        )

        # The buffer declaration has a data pointer defined earlier in
        # this function.  It should only be defined after the data pointer
        # has been defined, and should not be hoisted into the header of
        # the function as other buffer_decl statements can be.
        A = T.decl_buffer([1024], dtype="float32", data=A_data)
        B_data: T.let[T.handle("float32")] = T.reinterpret(
            T.handle("float32").ty,
            T.tvm_struct_get(arg1, 0, 1, dtype="handle"),
        )
        B = T.decl_buffer([1024], dtype="float32", data=B_data)

        B[0] = A[0]

    return func_T_ptr_let_statement


def func_T_ptr_allocate():
    @Ts.prim_func
    def func_T_ptr_allocate() -> None:
        A = T.alloc_buffer((1024,))
        A[0] = 0.0

    return func_T_ptr_allocate


def llvm_intrin_call():
    @Ts.prim_func
    def ctpop(A: T.Buffer((16,), "uint8"), B: T.Buffer((16,), "uint8")) -> None:
        for i in range(0, 16):
            with Ts.sblock("A"):
                vi = Ts.axis.remap(
                    "S",
                    [
                        i,
                    ],
                )
                B[vi] = T.call_llvm_pure_intrin(
                    T.llvm_lookup_intrinsic_id("llvm.ctpop.i8"),
                    A[vi],
                    dtype="uint8",
                )

    return ctpop


def string_annotation_escaping():
    @Ts.prim_func
    def string_annotation_of_special_chars():
        T.func_attr(
            {
                "key1": '"\'hello\t\r"',
                "key2": """
            %1 = add i32 %0, %0
            %2 = add i32 %0, %1
            %3 = add i32 %1, %2
            """,
            }
        )
        T.evaluate(0)

    return string_annotation_of_special_chars


def pointer_type():
    @Ts.prim_func
    def func_with_ptr_type_annotations(x: T.handle("int32"), y: T.handle("int32", "shared")):
        xx = T.alloc_buffer((16,), "int32")
        yy = T.alloc_buffer((16,), "int32", scope="shared")
        a: T.let[T.handle("int32")] = T.address_of(xx[0], dtype="handle")
        b: T.let[T.handle("int32", "shared")] = T.address_of(yy[0], dtype="handle")
        T.evaluate(T.call_extern("copy", a, b, dtype=""))

    return func_with_ptr_type_annotations


def buffer_ramp_access_as_slice_index():
    @Ts.prim_func
    def buffer_ramp_access(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((128,), "float32"),
    ) -> None:
        for i in range(128):
            A[i : i + 1 : 1] = i
        for i in range(4):
            B[i * 32 : i * 32 + 32] = A[T.Ramp(i * 32, 1, 32)] + T.broadcast(1.0, 32)
        for i in range(4):
            C[i : i + 128 : 4] = B[T.Ramp(i, 4, 32)] + T.broadcast(1.0, 32)

    return buffer_ramp_access


def ramp_int64():
    @Ts.prim_func
    def func() -> None:
        T.evaluate(T.Ramp(T.int64(0), 1, 3))

    return func


def scalable_vectors():
    @Ts.prim_func
    def func(A: T.Buffer((200,), "float32")):
        A[T.Ramp(11, 2, 4 * tirx.vscale())] = T.Broadcast(125, 4 * tirx.vscale())

    return func


def predicated_buffer_load_store():
    @Ts.prim_func
    def func(A: T.Buffer((4,), "float32"), B: T.Buffer((8,), "float32")):
        for i_0 in range(4):
            load_a = T.meta_var(
                T.call_intrin(
                    "float32x4",
                    "tirx.masked_load",
                    A,
                    T.Ramp(i_0, 1, 4),
                    T.Broadcast(T.bool(True), 4),
                )
            )
            T.evaluate(
                T.call_intrin(
                    "void",
                    "tirx.masked_store",
                    B,
                    load_a,
                    T.Ramp(0, 2, 4),
                    T.Broadcast(T.bool(True), 4),
                )
            )

    return func


def let_expression():
    x = T.dynamic("x", "int32")

    @Ts.prim_func
    def func():
        T.evaluate(T.Let(x + 1, where={x: 1}))

    return func


def test_void_ptr_vs_handle():
    """An untyped handle is the canonical void-pointer type."""

    # Generates PointerType(PrimType::Void())
    @Ts.prim_func
    def void_ptr(out_ret_value: T.handle("void")):
        T.evaluate(out_ret_value)

    # Generates PointerType::VoidPointerTy()
    @Ts.prim_func
    def handle(out_ret_value: T.handle):
        T.evaluate(out_ret_value)

    tvm.ir.assert_structural_equal(void_ptr.params[0].ty, handle.params[0].ty)
    script = void_ptr.script()
    assert "out_ret_value: T.handle" in script
    assert 'T.handle("void")' not in script
    tvm.ir.assert_structural_equal(
        void_ptr,
        tvm.script.from_source(
            script,
            extra_vars={
                "I": tvm.script.ir,
                "T": tvm.script.tirx,
                "Ts": tvm.script.s_tir,
                "R": tvm.script.relax,
            },
        ),
    )

    @Ts.prim_func
    def scoped_void_ptr(out_ret_value: T.handle("void", "shared")):
        T.evaluate(out_ret_value)

    scoped_script = scoped_void_ptr.script()
    assert 'out_ret_value: T.handle(storage_scope="shared")' in scoped_script
    assert 'T.handle("void"' not in scoped_script
    tvm.ir.assert_structural_equal(
        scoped_void_ptr,
        tvm.script.from_source(
            scoped_script,
            extra_vars={
                "I": tvm.script.ir,
                "T": tvm.script.tirx,
                "Ts": tvm.script.s_tir,
                "R": tvm.script.relax,
            },
        ),
    )


def void_ptr():
    @Ts.prim_func
    def func(out_ret_value: T.handle("void")):
        T.evaluate(out_ret_value)

    return func


def decl_buffer():
    @Ts.prim_func
    def func(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")) -> None:
        A_flattened = T.decl_buffer(data=A.data, shape=(256,), dtype="float32")
        B_flattened = T.decl_buffer(data=B.data, shape=(256,), dtype="float32")
        C_alias = T.decl_buffer(data=A_flattened.data, shape=(256,), dtype="float32")
        for i in range(256):
            B_flattened[i] = A_flattened[i] + C_alias[i] + T.float32(1.0)

    return func


def allocate_and_decl_buffer():
    @Ts.prim_func
    def func(A: T.Buffer((16,), "float32"), B: T.Buffer((16,), "float32")) -> None:
        D = T.alloc_buffer((16,))
        for i in range(4):
            C = T.alloc_buffer((4,))
            for j in range(4):
                C[j] = A[i * 4 + j] + T.float32(1.0)
            for j in range(4):
                D[j] = C[j]
            for j in range(4):
                B[i * 4 + j] = D[j]

    return func


def alloc_buffer_example():
    @Ts.prim_func
    def func(A: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32")):
        B = T.alloc_buffer((128,), "float32")
        for i in range(128):
            B[i] = A[i] * T.float32(2)
        for i in range(128):
            C[i] = B[i] + T.float32(1)

    return func


def float_infinity():
    @Ts.prim_func
    def func(
        placeholder: T.Buffer((1, 512, 768), "float32"), T_isinf: T.Buffer((1, 512, 768), "bool")
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tirx.noalias": True})
        # body
        # with Ts.sblock("root")
        for i0, i1, i2 in T.grid(1, 512, 768):
            with Ts.sblock("T_isinf"):
                ax0, ax1, ax2 = Ts.axis.remap("SSS", [i0, i1, i2])
                Ts.reads(placeholder[ax0, ax1, ax2])
                Ts.writes(T_isinf[ax0, ax1, ax2])
                T_isinf[ax0, ax1, ax2] = T.fabs(
                    placeholder[ax0, ax1, ax2], dtype="float32"
                ) == T.float32("inf") and not (T.isnan(placeholder[ax0, ax1, ax2], dtype="bool"))

    return func


def minimal_i32_literal():
    @Ts.prim_func
    def func() -> None:
        T.evaluate(T.int32(-2147483648))
        T.evaluate(-T.int64(2147483648))

    return func


def bool_primitive():
    @Ts.prim_func
    def func() -> None:
        T.evaluate(T.bool(True))

    return func


def bool_cast():
    # uninitialized var
    a = T.dynamic("a", "bool")

    @Ts.prim_func(check_well_formed=False)
    def func() -> None:
        T.evaluate(T.bool(T.int32(0)))
        T.evaluate(a == T.bool(False))

    return func


def nested_boolean_expressions():
    expressions = {
        "and_lhs_and": lambda i, j, k: tirx.all(tirx.all(i, j), k),
        "and_rhs_and": lambda i, j, k: tirx.all(i, tirx.all(j, k)),
        "and_lhs_or": lambda i, j, k: tirx.all(tirx.any(i, j), k),
        "and_rhs_or": lambda i, j, k: tirx.all(i, tirx.any(j, k)),
        "or_lhs_and": lambda i, j, k: tirx.any(tirx.all(i, j), k),
        "or_rhs_and": lambda i, j, k: tirx.any(i, tirx.all(j, k)),
        "or_lhs_or": lambda i, j, k: tirx.any(tirx.any(i, j), k),
        "or_rhs_or": lambda i, j, k: tirx.any(i, tirx.any(j, k)),
        "and_of_ors": lambda i, j, k: tirx.all(
            tirx.any(i, j), tirx.any(j, k), tirx.any(i, k), i, j, k
        ),
        "or_of_ands": lambda i, j, k: tirx.any(
            tirx.all(i, j), tirx.all(j, k), tirx.all(i, k), i, j, k
        ),
    }

    def make_ir_generator(name, expression):
        def inner():
            @Ts.prim_func
            def func(A: T.Buffer(1, "bool"), i: T.bool, j: T.bool, k: T.bool):
                A[0] = expression(i, j, k)

            return func

        inner.__name__ = f"nested_boolean_expr_{name}"
        return inner

    for name, expression in expressions.items():
        generator = make_ir_generator(name, expression)

        yield generator


def multi_env_threads():
    @Ts.prim_func
    def func(A: T.Buffer(128, "float32"), C: T.Buffer(128, "float32")):
        B = Ts.sblock_alloc_buffer([128], dtype="float32")
        for i in T.thread_binding(128, thread="threadIdx.x"):
            B[i] = A[i] + 1.0
        for i in T.thread_binding(128, thread="threadIdx.x"):
            C[i] = B[i] + 2.0

    mod = tvm.s_tir.transform.LowerOpaqueBlock()(
        tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    )
    return mod["main"]


def intrinsic_pow():
    @Ts.prim_func
    def func():
        T.pow(T.float32(1), T.float32(1))

    return func


def tvm_shfl_builtins():
    @Ts.prim_func
    def func(
        A: T.handle("float32"),
        B: T.handle("float32"),
        C: T.handle("float32"),
    ):
        blockIdx_x = T.launch_thread("blockIdx.x", 1)
        threadIdx_x = T.launch_thread("threadIdx.x", 32)
        A_warp = T.alloc_buffer((1,), scope="local")
        B_warp = T.alloc_buffer((1,), scope="local")
        red_buf0 = T.alloc_buffer((1,), scope="local")
        A_warp_1 = T.decl_buffer((32,), data=A_warp.data, scope="local")
        A_1 = T.decl_buffer((32,), data=A)  # A is a handle param
        A_warp_1[0] = A_1[threadIdx_x]
        B_warp_1 = T.decl_buffer((32,), data=B_warp.data, scope="local")
        T.tvm_storage_sync("warp")
        B_warp_1[0] = T.tvm_warp_shuffle(
            T.tvm_warp_activemask(), A_warp_1[0], threadIdx_x % 4 * 8 + threadIdx_x // 4, 32, 32
        ) + T.float32(1)
        red_buf0_1 = T.decl_buffer((1,), data=red_buf0.data, scope="local")
        with T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.int32(0),
        ):
            mask = T.alloc_buffer((1,), "uint32", scope="local")
            t0 = T.alloc_buffer((1,), scope="local")
            red_buf0_1[0] = A_warp_1[0]
            mask_1 = T.decl_buffer((1,), "uint32", data=mask.data, scope="local")
            mask_1[0] = T.tvm_warp_activemask()
            t0_1 = T.decl_buffer((1,), data=t0.data, scope="local")
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 16, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 8, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 4, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 2, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            t0_1[0] = T.tvm_warp_shuffle_down(mask_1[0], red_buf0_1[0], 1, 32, 32)
            red_buf0_1[0] = red_buf0_1[0] + t0_1[0]
            red_buf0_1[0] = T.tvm_warp_shuffle(mask_1[0], red_buf0_1[0], 0, 32, 32)
            # NOTE(Zihao): test tvm_warp_shuffle_up
            red_buf0_1[0] = T.tvm_warp_shuffle_up(mask_1[0], red_buf0_1[0], 0, 32, 32)
        if threadIdx_x == 0:
            C_1 = T.decl_buffer((1,), data=C)
            C_1[0] = red_buf0_1[0]
        B_1 = T.decl_buffer((32,), data=B)
        B_1[threadIdx_x] = B_warp_1[0]

    return func


def make_packed_api_result():
    @Ts.prim_func
    def func(A: T.Buffer(64, "float32")):
        T.func_attr({"global_symbol": "main", "target": T.target("cuda")})
        bx = T.launch_thread("blockIdx.x", 64)
        T.evaluate(A[bx])

    mod = tvm.IRModule.from_expr(func)
    return tvm.tirx.transform.MakePackedAPI()(mod)


def tvm_struct_set_generated_in_cpp():
    """Ensure same dtype for tvm_struct_set in Python/C++

    The TVMStructSet method in C++, used internally by
    LowerTVMBuiltin, and the Python method `T.tvm_struct_set`, used
    when parsing TVMScript should use the same dtype "int32".
    """

    @I.ir_module
    class Module:
        @Ts.prim_func
        def tir_packed_call(A: T.Buffer(16)):
            T.attr(0, "device_id", 0)
            T.attr(0, "device_type", 0)
            T.evaluate(
                T.tvm_call_cpacked(
                    "tvm_test_cpacked",
                    T.tvm_stack_make_array(
                        A.data,
                        T.tvm_stack_make_shape(16, dtype="handle"),
                        T.reinterpret(T.uint64(0), dtype="handle"),
                        T.uint32(1),
                        T.Cast("float32", 0),
                        0,
                        dtype="handle",
                    ),
                    dtype="int32",
                )
            )

    return tvm.tirx.transform.LowerTVMBuiltin()(Module)


def undefined_data_ptr_in_decl_buffer():
    """The T.decl_buffer syntax should not introduce an Allocate

    While T.decl_buffer can be used to represent an
    Allocate/DeclBuffer pair, performing a round-trip through
    TVMScript should not introduce an Allocate node.
    """

    # uninitialized var
    @Ts.prim_func(check_well_formed=False)
    def func():
        data_ptr = T.handle("float32")
        buf = T.decl_buffer(shape=[1], dtype="float32", data=data_ptr)
        T.evaluate(buf[0])

    return func


def op_of_literal():
    op_list = [
        (T.exp, 0),
        (T.exp2, 0),
        (T.exp10, 0),
        (T.erf, 0.0),
        (T.tanh, 0.0),
        (T.sigmoid, 0.0),
        (T.log, 0.0),
        (T.log2, 0.0),
        (T.log1p, 0.0),
        (T.tan, 0.0),
        (T.cos, 0.0),
        (T.acos, 0.0),
        (T.acosh, 0.0),
        (T.sin, 0.0),
        (T.sinh, 0.0),
        (T.asin, 0.0),
        (T.asinh, 0.0),
        (T.atan, 0.0),
        (T.atanh, 0.0),
        (T.atan2, (1.0, 0.0)),
        (T.sqrt, 0.0),
        (T.rsqrt, 1.0),
        (T.nextafter, (0.0, 1.0)),
        (T.hypot, (1.0, 1.0)),
        (T.copysign, (1.0, 1.0)),
        (T.popcount, 0),
        (T.fmod, (1.0, 1.0)),
    ]

    def make_ir_generator(op, arg):
        def inner():
            call_expr = op(*arg) if isinstance(arg, tuple) else op(arg)

            @Ts.prim_func
            def func():
                T.evaluate(call_expr)

            return func

        inner.__name__ = f"{op.__name__}_of_literal"
        return inner

    for op, arg in op_list:
        yield make_ir_generator(op, arg)


def test_address_of_buffer():
    @Ts.prim_func
    def func(A: T.Buffer((128, 128), "float32")):
        T.evaluate(T.address_of(A))

    assert "T.address_of(A[0, 0])" in func.script()


@Ts.prim_func
def _func():
    T.evaluate(-1)
    T.evaluate(1)
    T.evaluate(2)
    T.evaluate(3)
    T.evaluate(4)
    T.evaluate(5)
    T.evaluate(6)
    T.evaluate(7)


def test_annotation_multi_access_paths():
    result = _func.with_attr("global_symbol", "main").script(
        path_to_annotate={
            AccessPath.root().attr("body").attr("seq").array_item(1): "annotation 1",
            AccessPath.root().attr("body").attr("seq").array_item(3): "annotation 3",
            AccessPath.root().attr("body").attr("seq").array_item(5): "annotation 5",
            AccessPath.root().attr("body").attr("seq").array_item(7): "annotation 7",
        }
    )
    assert (
        result
        == """# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main():
    T.evaluate(-1)
    T.evaluate(1)  # annotation 1
    T.evaluate(2)
    T.evaluate(3)  # annotation 3
    T.evaluate(4)
    T.evaluate(5)  # annotation 5
    T.evaluate(6)
    T.evaluate(7)  # annotation 7"""
    )


def test_annotate_from_multi_obj():
    result = _func.with_attr("global_symbol", "main").script(
        obj_to_annotate={
            _func.body.seq[1]: "annotation 1",
            _func.body.seq[3]: "annotation 3",
            _func.body.seq[5]: "annotation 5",
            _func.body.seq[7]: "annotation 7",
        }
    )
    assert (
        result
        == """# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main():
    T.evaluate(-1)
    T.evaluate(1)  # annotation 1
    T.evaluate(2)
    T.evaluate(3)  # annotation 3
    T.evaluate(4)
    T.evaluate(5)  # annotation 5
    T.evaluate(6)
    T.evaluate(7)  # annotation 7"""
    )


def test_disable_concise_scoping_when_scope_annotated():
    @Ts.prim_func
    def _func():
        x: T.int32 = 1
        y: T.int32 = x + 1
        T.evaluate(y - 1)

    # Explicit scalar declarations lower to AllocBuffer + BufferStore (local_scalar).
    # The printer fuses each pair into one line; annotate the allocation for y.
    result = _func.with_attr("global_symbol", "main").script(
        obj_to_annotate={
            _func.body.seq[2]: "annotation 1",
        }
    )
    assert (
        result
        == """# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@Ts.prim_func
def main():
    x: T.int32 = 1
    y: T.int32 = x + 1  # annotation 1
    T.evaluate(y - 1)"""
    )


def _assert_module_representations(obj, expected):
    assert str(obj).strip() == expected.strip()
    assert repr(obj).strip() == expected.strip()
    if isinstance(obj, IRModule):
        assert obj.script().strip() == expected.strip()


def test_ir_module():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with IB.ir_module():
            with build_prim_func():
                TB.func_name_("foo")
    mod = ib.get()
    _assert_module_representations(
        mod,
        """
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import s_tir as Ts

@I.ir_module
class Module:
    @Ts.prim_func
    def foo():
        T.evaluate(0)""",
    )


def test_str_metadata():
    # This test is to check we reuse the existing metadata element for the same tvm.ir.StringImm
    # So metadata["ir.StringImm"][0] will occur in the printed script for three times
    str_imm = tvm.ir.StringImm("aaa\nbbb\n")

    @I.ir_module
    class Module:
        @Ts.prim_func
        def foo() -> None:
            A = str_imm
            B = str_imm

        @Ts.prim_func
        def foo1() -> None:
            A = str_imm

    printed_str = Module.script(verbose_expr=True)
    assert (
        printed_str.count('metadata["ir.StringImm"][0]') == 3
        and printed_str.count('metadata["ir.StringImm"][1]') == 0
    )


@pytest.mark.parametrize(
    "ir_generator",
    [
        opt_gemm_lower,
        opt_conv_tensorcore_lower,
        opt_conv_tensorcore_mod_host,
        comm_reducer_single_reduce_group,
        comm_reducer_multiple_reduce_groups,
        multiple_commreducer,
        llvm_intrin_call,
        multi_env_threads,
        tvm_shfl_builtins,
        make_packed_api_result,
        tvm_struct_set_generated_in_cpp,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_lowered(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


@pytest.mark.parametrize(
    "ir_generator",
    [
        select,
        minmax,
        abs,
        constant_folding,
        simplify_bracket,
        ramp_int64,
        let_expression,
        float_infinity,
        minimal_i32_literal,
        bool_primitive,
        bool_cast,
        *nested_boolean_expressions(),
        intrinsic_pow,
        *op_of_literal(),
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_expressions(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


@pytest.mark.parametrize(
    "ir_generator",
    [
        primfunc_with_allocate_annotations,
        func_T_ptr_let_statement,
        func_T_ptr_allocate,
        pointer_type,
        buffer_ramp_access_as_slice_index,
        scalable_vectors,
        predicated_buffer_load_store,
        void_ptr,
        decl_buffer,
        allocate_and_decl_buffer,
        alloc_buffer_example,
        undefined_data_ptr_in_decl_buffer,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_buffers(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


@pytest.mark.parametrize(
    "ir_generator",
    [
        func_with_target_spec_by_config,
        func_with_target_spec_by_str,
        func_with_target_and_host_spec_by_str,
        string_annotation_escaping,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_metadata(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm.script.ir,
            "T": tvm.script.tirx,
            "Ts": tvm.script.s_tir,
            "R": tvm.script.relax,
        },
    )
    tvm.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)


# Import-time construction also checks the annotated S-TIR API.
@Ts.prim_func
def lowered_loop_split(
    A: T.Buffer([128, 128], dtype="float32"), B: T.Buffer([128], dtype="float32")
) -> None:
    reduce_temp0 = Ts.sblock_alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    normal_reduce_temp0 = Ts.sblock_alloc_buffer([1], dtype="float32", strides=[1], scope="local")
    for i in T.serial(0, 128):
        for ki in T.thread_binding(0, 32, thread="threadIdx.x"):
            normal_reduce_temp0[0] = T.float32(0)
            for ko in T.serial(0, 4):
                with Ts.sblock("B_normal_reduction"):
                    vi = Ts.axis.S(128, i)
                    vk = Ts.axis.R(128, ko * 32 + ki)
                    Ts.reads([A[vi, vk], normal_reduce_temp0[0]])
                    Ts.writes([normal_reduce_temp0[0]])
                    normal_reduce_temp0[0] = normal_reduce_temp0[0] + A[vi, vk]
            with Ts.sblock("B_cross_thread_reduction"):
                Ts.reads([normal_reduce_temp0[0]])
                Ts.writes([reduce_temp0[0]])
                T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.float32(0)]),
                    "reduce_scope",
                    T.int32(0),
                )
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        normal_reduce_temp0[0],
                        True,
                        reduce_temp0.data,
                        ki,
                        dtype="handle",
                    )
                )
            with Ts.sblock("B_write_back"):
                vi = Ts.axis.S(128, i)
                Ts.reads([reduce_temp0[0]])
                Ts.writes([B[vi]])
                B[vi] = reduce_temp0[0]


if __name__ == "__main__":
    tvm.testing.main()
