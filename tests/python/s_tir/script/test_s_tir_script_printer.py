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
# ruff: noqa: E501, F401, F841

import io
import re
import tokenize

import pytest

import tvm.testing
from tvm import s_tir, tirx
from tvm.ir import Range
from tvm.s_tir.script.ir_builder import prim_func as build_prim_func
from tvm.script import ir as I
from tvm.script import s_tir as Ts
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
            T.evaluate(0)
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
            T.evaluate(0)
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
            v = T.bind(T.float32(10))
            ib.name("v", v)
            T.evaluate(1)
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
    from tvm.script import tirx as T

    @Ts.prim_func
    def block_with_remap_implicitly():
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
            with Ts.sblock("update"):
                v0 = Ts.axis.spatial(128, i0 + 1)
                v1 = Ts.axis.spatial(128, i1)
                v2 = Ts.axis.reduce(128, i2)
                v3 = Ts.axis.spatial(128, i3 - 1)
                v4 = Ts.axis.reduce(128, i4)
                v5 = Ts.axis.spatial(128, i5)

    @Ts.prim_func
    def block_with_remap_explicitly():
        for i0, i1, i2, i3, i4, i5 in T.grid(128, 128, 128, 128, 128, 128):
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
    from tvm.script import tirx as T

    @Ts.prim_func
    def root_block_implicitly():
        a = Ts.sblock_alloc_buffer([128, 128])
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                T.evaluate(0)

    @Ts.prim_func
    def root_block_explicitly():
        with Ts.sblock("root"):
            a = Ts.sblock_alloc_buffer([128, 128])
            for i, j in T.grid(128, 128):
                with Ts.sblock():
                    T.evaluate(0)

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
    from tvm.script import tirx as T

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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
        T.func_attr({"global_symbol": "func"})
        T.evaluate(0)

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
    from tvm.script import tirx as T

    # The test function has all named objects suffixed with "_name",
    # to avoid spurious replacement when generating the expected
    # regex.
    N_name = I.dynamic("N_name")

    @Ts.prim_func
    def func(A_name: T.Buffer(N_name, "float32")):
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
    from tvm.script import tirx as T

    @Ts.prim_func
    def func():
        return T.int32(5)

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
    from tvm.script import tirx as T

    @Ts.prim_func
    def func():
        T.evaluate(getattr(T, dtype)(0.0))

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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
        T.func_attr({"global_symbol": "func"})
        a_load = T.meta_var(
            T.call_intrin(
                "float32x4",
                "tirx.masked_load",
                A,
                0,
                T.Ramp(0, 4, 4),
                T.Broadcast(T.bool(False), 4),
            )
        )
        T.evaluate(
            T.call_intrin(
                "void",
                "tirx.masked_store",
                A,
                a_load,
                0,
                T.Ramp(0, 2, 4),
                T.Broadcast(T.bool(False), 4),
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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((256, 256), "float32")):
        T.func_attr({"global_symbol": "func"})
        mask = T.meta_var(T.get_active_lane_mask("uint1xvscalex4", 0, 13))
        a_load = T.meta_var(
            T.call_intrin(
                "float32xvscalex4", "tirx.masked_load", A, 0, T.Ramp(0, 4, T.vscale() * 4), mask
            )
        )
        T.evaluate(
            T.call_intrin(
                "void", "tirx.masked_store", A, a_load, 0, T.Ramp(0, 2, T.vscale() * 4), mask
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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main():
        A = T.alloc_buffer((1,), "float32x4")
        A[0] = T.masked_load("float32x4", A, 0, T.Broadcast(T.bool(True), 4))

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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        B[0 : T.vscale() * 4] = A.vload([T.Ramp(0, 1, T.vscale() * 4)], dtype="float32xvscalex4")

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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        A[T.Ramp(0, 1, 4)] = T.call_llvm_pure_intrin("float32x4", "llvm.sqrt", B[T.Ramp(0, 1, 4)])

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
    from tvm.script import tirx as T

    @Ts.prim_func
    def main(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
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
            T.continue_loop()
        if A[0] >= B[0]:
            T.break_loop()
    """
    _assert_print(main, expected_output)


if __name__ == "__main__":
    tvm.testing.main()
