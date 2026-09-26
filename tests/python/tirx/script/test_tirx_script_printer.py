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

"""TIRx script printer."""

import pytest
from tvm_ffi import get_global_func

import tvm
import tvm.script
import tvm.testing
from tvm import ir, tirx
from tvm import tirx as tir
from tvm.ir import Range, assert_structural_equal
from tvm.runtime.script_printer import _script
from tvm.script import ir as I
from tvm.script import ir_builder as IB
from tvm.script import tirx as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda import op as cuda_op
from tvm.tirx.script import ir_builder as TB
from tvm.tirx.trn import op as trn_op


def test_failed_invalid_prefix():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with IB.ir_module():
            with TB.prim_func():
                TB.func_name_("foo")
    mod = ib.get()

    with pytest.raises(RuntimeError):
        mod.script(ir_prefix="2I")


def test_config_extension_passthrough():
    make_config = get_global_func("node.PrinterConfig")
    cfg = make_config(
        {
            "extension.option": 7,
            "custom_key": "value",
            "syntax_sugar": False,
            "render_invisible_path_info": True,
            "tirx.prefix": "invalid-prefix",
            "extra_config": {
                "extension.option": 9,
                "render_invisible_path_info": False,
                "tirx.prefix": "Custom",
            },
        }
    )
    assert cfg.extra_config["extension.option"] == 9
    assert cfg.extra_config["custom_key"] == "value"
    assert "syntax_sugar" not in cfg.extra_config
    assert "extra_config" not in cfg.extra_config
    assert not cfg.syntax_sugar
    assert not cfg.render_invisible_path_info
    assert make_config({}).syntax_sugar
    assert _script(tirx.Var("Custom", "int32"), cfg) == "Custom_1"


@pytest.mark.parametrize("key", ["tirx.prefix", "relax.prefix", "s_tir.prefix"])
@pytest.mark.parametrize("value", ["2prefix", 17])
@pytest.mark.parametrize("nested", [False, True])
def test_config_validates_dialect_prefixes(key, value, nested):
    config = {key: value}
    if nested:
        config = {"extra_config": config}
    with pytest.raises((RuntimeError, TypeError)):
        get_global_func("node.PrinterConfig")(config)


@pytest.mark.parametrize(
    "prefixes",
    [{}, {"tirx.prefix": "CustomT", "relax.prefix": "CustomR", "s_tir.prefix": "CustomTs"}],
)
def test_config_reserves_dialect_prefixes_before_variable_definition(prefixes):
    tir_prefix = prefixes.get("tirx.prefix", "T")
    relax_prefix = prefixes.get("relax.prefix", "R")
    for name in [tir_prefix, relax_prefix, prefixes.get("s_tir.prefix", "Ts")]:
        var = tirx.Var(name, "int32")
        assert var.script(verbose_expr=True, extra_config=prefixes).strip() == (
            f'{name}_1 = I.dynamic("{name}", dtype="int32")\n{name}_1'
        )


def test_buffer():
    a = tirx.decl_buffer((128, 128), "float16", name="A")
    _assert_print(
        a,
        """A = T.Buffer((128, 128), "float16")
A""",
    )


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


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
        TB.buffer_store(a, a[128, 128] + 1, [128, 128])
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
        with TB.grid(128, 128, 128) as (i, j, k):
            ib.name_many(["i", "j", "k"], [i, j, k])
            TB.evaluate(0)
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
        with TB.attr("pragma", "unroll", 1):
            TB.evaluate(0)
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
        with TB.prim_func():
            TB.assert_(True, "assertion")
            TB.evaluate(TB.call_extern("int32", "after_assert"))
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
        with TB.while_(x < 10):
            TB.evaluate(0)
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
        with TB.prim_func():
            TB.func_name_("test")
            buf = TB.alloc_buffer([128, 128], "float32")
            TB.evaluate(1)
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
        with TB.prim_func():
            TB.func_name_("test")
            buf = TB.alloc_buffer([128, 128], "float32")
            buf2 = TB.decl_buffer([128, 128], "float32", data=buf.data)
            TB.evaluate(1)
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
        with TB.prim_func():
            TB.func_name_("test")
            buf = TB.alloc_buffer([128, 128], "float32")
            buf2 = TB.decl_buffer([128, 128], "float32", data=buf.data)
            TB.evaluate(buf.data)
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
        with TB.prim_func():
            TB.func_name_("test")
            buf = TB.alloc_buffer([128, 128], "float32")
            buf2 = TB.decl_buffer([256, 256], "float32", data=buf.data)
            TB.evaluate(buf.data)
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
        with TB.prim_func():
            TB.func_name_("test")
            buf = TB.decl_buffer((10, 10), data=TB.ptr("float32"))
            TB.evaluate(1)
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
        with TB.serial(10):
            TB.evaluate(1)
            TB.evaluate(2)
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
        with TB.if_(I.dynamic("v", "int32") == 1):
            with TB.then_():
                TB.evaluate(0)

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
        TB.evaluate(0)
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
    obj = tirx.atan(TB.float32(1.0))
    _assert_print(
        obj,
        """
T.atan(T.float32(1.0))
""",
    )


def test_comm_reducer():
    obj = TB.comm_reducer(lambda x, y: x + y, identity=[TB.float32(0)])
    _assert_print(
        obj,
        """
T.comm_reducer(lambda x, y: x + y, [T.float32(0.0)])
""",
    )


def test_int_imm():
    _assert_print(tirx.IntImm("int64", 1), "T.int64(1)")
    obj = TB.int16(1)
    _assert_print(
        obj,
        """
T.int16(1)
""",
    )


def test_float_imm():
    obj = TB.float16(1)
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


def test_nested_seqstmt_roundtrip():
    original = nested_seqstmt()
    parsed = tvm.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx},
    )
    tvm.ir.assert_structural_equal(original, parsed, True)


def nested_seqstmt():
    """Nested SeqStmt should be normalized to flat SeqStmt

    Nested SeqStmt are representable in the TIR structures, but are
    flattened when converted to TVMScript.  Previously, this could
    cause failures to round-trip through TVMScript, including
    erroneous use of TVMScript's concise-scoping rules.  This was
    resolved by normalizing nested SeqStmt in TIR, such that the use
    of `tirx.SeqStmt` below results in a single flat `tirx.SeqStmt`
    containing the three `tirx.Evaluate` calls.
    """
    func = tvm.tirx.PrimFunc(
        params=[],
        body=tvm.tirx.SeqStmt(
            [
                tvm.tirx.SeqStmt([tvm.tirx.Evaluate(0), tvm.tirx.Evaluate(1)]),
                tvm.tirx.Evaluate(2),
            ]
        ),
    )

    return func


def test_print_kwargs_schedule_op_full_code():
    # fmt: off
    @T.prim_func
    def test():
        A = T.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], T.float32(1.25), dispatch="v10", bar=7, foo=42)
    # fmt: on

    expected = (
        "# from tvm.script import tirx as T\n"
        "# from tvm.tirx.layout import Axis\n\n"
        "@T.prim_func\n"
        "def test():\n"
        "    A = T.alloc_buffer((16,))\n"
        '    T.tile.memset(A[0:16], T.float32(1.25), dispatch="v10", bar=7, foo=42)'
    )
    code = test.script()
    assert code == expected
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_default_script_prefix_tirx_irmodule_non_main():
    """IRModule with non-main TIRx PrimFunc should default to T prefix."""
    mod = tvm.IRModule({"foo": _make_minimal_tirx_prim_func()})
    code = mod.script()
    assert "# from tvm.script import tirx as T" in code
    assert "# from tvm.script import tir as T" not in code
    assert "@T.prim_func" in code
    assert "def foo(" in code
    parsed = from_source(code)
    assert parsed.script() == code
    assert_structural_equal(mod, parsed)


def _make_minimal_tirx_prim_func():
    source = (
        "# from tvm.script import tirx as T\n\n"
        "@T.prim_func()\n"
        'def f(A: T.Buffer((1,), "float32")):\n'
        "    A[0] = T.float32(1)"
    )
    return from_source(source)


def test_printer_cuda_namespace_printf():
    node = tir.Evaluate(cuda_op.cuda_printf("x=%d", tir.IntImm("int32", 1)))
    _assert_namespace_print(node, 'T.cuda.printf("x=%d", 1)')


def _assert_namespace_print(obj, expected):
    # Standalone TIR nodes use the canonical tirx script prefix.
    out = obj.script(verbose_expr=True, extra_config={"tirx.prefix": "T"}).strip()
    assert out == expected.strip()


def test_printer_cuda_cluster_sync():
    node = tir.Evaluate(cuda_op.cuda_cluster_sync())
    _assert_namespace_print(node, "T.cuda.cluster_sync()")


def test_printer_cuda_namespace_mbarrier_wait():
    node = tir.Evaluate(cuda_op.cuda_mbarrier_wait(tir.IntImm("int32", 0), tir.IntImm("int32", 0)))
    _assert_namespace_print(node, "T.cuda.mbarrier_wait(0, 0)")


def test_printer_nvshmem_namespace():
    node = tir.Evaluate(cuda_op.nvshmem_fence())
    _assert_namespace_print(node, "T.nvshmem.fence()")


def test_printer_ptx_more():
    r = tir.Var("r", "handle")
    s = tir.Var("s", "handle")
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    b = tir.Var("b", "handle")
    _assert_namespace_print(
        cuda_op.cuda_tcgen05_encode_matrix_descriptor(d, a, 1, 2, 0),
        "d = T.handle()\na = T.handle()\nT.cuda.tcgen05.encode_matrix_descriptor(d, a, 1, 2, 0)",
    )
    _assert_namespace_print(
        cuda_op.cuda_tcgen05_encode_instr_descriptor(
            d,
            d_dtype="f16",
            a_dtype="f16",
            b_dtype="f16",
            M=16,
            N=16,
            K=16,
            trans_a=True,
            trans_b=False,
            n_cta_groups=1,
            neg_a=False,
            neg_b=False,
            sat_d=False,
            is_sparse=False,
        ),
        'd = T.handle()\nT.cuda.tcgen05.encode_instr_descriptor(d, "f16", "f16", "f16", 16, 16, 16, T.bool(True), T.bool(False), 1, T.bool(False), T.bool(False), T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_namespace_print(
        cuda_op.cuda_tcgen05_encode_instr_descriptor_block_scaled(
            d,
            d_dtype="f16",
            a_dtype="f16",
            b_dtype="f16",
            sfa_dtype="f16",
            sfb_dtype="f16",
            sfa_tmem_addr=a,
            sfb_tmem_addr=b,
            M=16,
            N=16,
            K=16,
            trans_a=True,
            trans_b=False,
            is_sparse=True,
            n_cta_groups=1,
            neg_a=False,
            neg_b=False,
        ),
        "d = T.handle()\n"
        "a = T.handle()\n"
        "b = T.handle()\n"
        'T.cuda.tcgen05.encode_instr_descriptor_block_scaled(d, "f16", "f16", "f16", "f16", "f16", a, b, 16, 16, 16, T.bool(True), T.bool(False), 1, T.bool(False), T.bool(False), T.bool(True))',  # noqa: E501
    )


def test_printer_cuda_mbarrier_wait_var():
    bar = tir.Var("bar", "handle")
    _assert_namespace_print(
        cuda_op.cuda_mbarrier_wait(bar, 1), "bar = T.handle()\nT.cuda.mbarrier_wait(bar, 1)"
    )
    _assert_namespace_print(cuda_op.cuda_cluster_sync(), "T.cuda.cluster_sync()")


def test_printer_cuda_more():
    p = tir.Var("p", "handle")
    _assert_namespace_print(cuda_op.cuda_thread_fence(), "T.cuda.thread_fence()")
    _assert_namespace_print(cuda_op.cuda_warp_sync(), "T.cuda.warp_sync()")
    _assert_namespace_print(cuda_op.cuda_cta_sync(), "T.cuda.cta_sync()")
    _assert_namespace_print(cuda_op.cuda_grid_sync(), "T.cuda.grid_sync()")
    _assert_namespace_print(cuda_op.cuda_cluster_sync(), "T.cuda.cluster_sync()")
    _assert_namespace_print(cuda_op.cuda_syncthreads_and(1), "T.cuda.syncthreads_and(1)")
    _assert_namespace_print(cuda_op.cuda_syncthreads_or(1), "T.cuda.syncthreads_or(1)")
    _assert_namespace_print(cuda_op.cuda_nano_sleep(100), "T.cuda.nano_sleep(100)")
    _assert_namespace_print(
        cuda_op.cuda_atomic_add(p, tir.IntImm("int32", 1)),
        "p = T.handle()\nT.cuda.atomic_add(p, 1)",
    )
    _assert_namespace_print(
        cuda_op.cuda_atomic_cas(p, 1, 2), "p = T.handle()\nT.cuda.atomic_cas(p, 1, 2)"
    )
    _assert_namespace_print(
        cuda_op.cuda_ldg(p, "float32"), 'p = T.handle()\nT.cuda.ldg(p, "float32")'
    )
    _assert_namespace_print(
        cuda_op.cuda_func_call("f", 1, source_code=""), 'T.cuda.func_call("f", 1, source_code="")'
    )


def test_printer_cuda_low_level_warp_intrinsics_roundtrip():
    @T.prim_func
    def kernel(x: T.int32):
        mask = T.cuda.__activemask()
        T.evaluate(T.cuda.__shfl_sync(mask, x, 0, 32))
        T.evaluate(T.cuda.__shfl_up_sync(mask, x, 1, 32))
        T.evaluate(T.cuda.__shfl_down_sync(mask, x, 1, 32))
        T.evaluate(T.cuda.__shfl_xor_sync(mask, x, 1, 32))

    code = kernel.script()
    assert "T.cuda.__activemask()" in code
    assert "T.cuda.__shfl_sync(" in code
    assert "T.cuda.__shfl_up_sync(" in code
    assert "T.cuda.__shfl_down_sync(" in code
    assert "T.cuda.__shfl_xor_sync(" in code
    assert "T.tirx." not in code
    assert (
        tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}).script()
        == code
    )


def test_printer_webgpu_namespace_roundtrip():
    @T.prim_func
    def kernel(x: T.int32):
        T.evaluate(T.webgpu.subgroup_shuffle(x, 0))
        T.evaluate(T.webgpu.subgroup_shuffle_up(x, 1))
        T.evaluate(T.webgpu.subgroup_shuffle_down(x, 1))

    code = kernel.script()
    assert "T.webgpu.subgroup_shuffle(" in code
    assert "T.webgpu.subgroup_shuffle_up(" in code
    assert "T.webgpu.subgroup_shuffle_down(" in code
    assert "T.tirx." not in code
    assert (
        tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}).script()
        == code
    )


def test_printer_nvshmem_more():
    p = tir.Var("p", "handle")
    _assert_namespace_print(cuda_op.nvshmem_my_pe(), "T.nvshmem.my_pe()")
    _assert_namespace_print(cuda_op.nvshmem_n_pes(), "T.nvshmem.n_pes()")
    _assert_namespace_print(
        cuda_op.nvshmem_signal_op(p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.signal_op(p, 1, "set", 0)',
    )
    _assert_namespace_print(
        cuda_op.nvshmem_wait_until(p, "eq", 0),
        'p = T.handle()\nT.nvshmem.wait_until(p, "eq", 0, "uint64_t")',
    )
    _assert_namespace_print(cuda_op.nvshmem_quiet(), "T.nvshmem.quiet()")
    _assert_namespace_print(cuda_op.nvshmem_barrier_all(), "T.nvshmem.barrier_all()")
    _assert_namespace_print(
        cuda_op.nvshmem_getmem_nbi(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.getmem_nbi(p, p, 16, 0)",
    )
    _assert_namespace_print(
        cuda_op.nvshmem_getmem_nbi_warp(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.getmem_nbi.warp(p, p, 16, 0)",
    )
    _assert_namespace_print(
        cuda_op.nvshmem_putmem_nbi_block(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.putmem_nbi.block(p, p, 16, 0)",
    )
    _assert_namespace_print(
        cuda_op.nvshmem_putmem_nbi(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.putmem_nbi(p, p, 16, 0)",
    )
    _assert_namespace_print(
        cuda_op.nvshmem_putmem_nbi_warp(p, p, 16, 0),
        "p = T.handle()\nT.nvshmem.putmem_nbi.warp(p, p, 16, 0)",
    )
    _assert_namespace_print(
        cuda_op.nvshmem_putmem_signal_nbi(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi(p, p, 16, p, 1, "set", 0)',
    )
    _assert_namespace_print(
        cuda_op.nvshmem_putmem_signal_nbi_warp(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi.warp(p, p, 16, p, 1, "set", 0)',
    )
    _assert_namespace_print(
        cuda_op.nvshmem_putmem_signal_nbi_block(p, p, 16, p, 1, "set", 0),
        'p = T.handle()\nT.nvshmem.putmem_signal_nbi.block(p, p, 16, p, 1, "set", 0)',
    )


def test_printer_nki_namespace():
    A = tir.decl_buffer([1], dtype="float16", name="A")
    B = tir.decl_buffer([1], dtype="float16", name="B")
    a0 = A[0]
    b0 = B[0]
    _assert_namespace_print(
        trn_op.nki_load(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.load(A, B)',
    )
    _assert_namespace_print(
        trn_op.nki_store(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.store(A, B)',
    )
    _assert_namespace_print(
        trn_op.nki_tensor_copy(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.tensor_copy(A, B)',
    )
    _assert_namespace_print(
        trn_op.nki_matmul(a0, a0, b0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        "T.nki.matmul(A, A, B, T.bool(True))",
    )
    _assert_namespace_print(
        trn_op.nki_activation(a0, b0, "relu", 0.0, 1.0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.activation(A, B, "relu", T.float32(0.0), T.float32(1.0))',
    )
    _assert_namespace_print(
        trn_op.nki_memset(a0, 0),
        'A = T.Buffer((1,), "float16")\nT.nki.memset(A, 0)',
    )
    _assert_namespace_print(
        trn_op.nki_identity(a0, 1),
        'A = T.Buffer((1,), "float16")\nT.nki.identity(A, 1)',
    )
    _assert_namespace_print(
        trn_op.nki_reciprocal(a0, b0),
        'A = T.Buffer((1,), "float16")\nB = T.Buffer((1,), "float16")\nT.nki.reciprocal(A, B)',
    )
    _assert_namespace_print(
        trn_op.nki_tensorreduce(a0, b0, "sum", False, 0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.tensorreduce(A, B, "sum", T.bool(False), 0)',
    )
    _assert_namespace_print(
        trn_op.nki_tensortensor(a0, a0, b0, "add"),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.tensortensor(A, A, B, "add")',
    )
    _assert_namespace_print(
        trn_op.nki_tensorscalar(a0, a0, 1.0, "mul", False),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.tensorscalar(A, A, T.float32(1.0), "mul", T.bool(False))',
    )
    _assert_namespace_print(
        trn_op.nki_tensorscalar_reduce(a0, a0, 1.0, "mul", "sum", False),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.tensorscalar_reduce(A, A, T.float32(1.0), "mul", "sum", T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_namespace_print(
        trn_op.nki_scalar_tensor_tensor(a0, a0, 1.0, a0, "add", "add"),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.scalar_tensor_tensor(A, A, T.float32(1.0), A, "add", "add", T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_namespace_print(
        trn_op.nki_scalar_tensor_scalar(a0, a0, 1.0, 1.0, "add", "add"),
        'A = T.Buffer((1,), "float16")\n'
        'T.nki.scalar_tensor_scalar(A, A, T.float32(1.0), T.float32(1.0), "add", "add", T.bool(False), T.bool(False))',  # noqa: E501
    )
    _assert_namespace_print(
        trn_op.nki_activation_reduce(a0, a0, b0, "relu", "sum", 0.0, 1.0),
        'A = T.Buffer((1,), "float16")\n'
        'B = T.Buffer((1,), "float16")\n'
        'T.nki.activation_reduce(A, A, B, "relu", "sum", T.float32(0.0), T.float32(1.0))',
    )
    _assert_namespace_print(
        trn_op.nki_affine_select(a0, a0, a0, 1.0),
        'A = T.Buffer((1,), "float16")\nT.nki.affine_select(A, A, A, T.float32(1.0))',
    )


def test_printer_ptx_mma_and_wgmma():
    r = tir.Var("r", "handle")
    d = tir.Var("d", "handle")
    a = tir.Var("a", "handle")
    tir.Var("b", "handle")
    _assert_namespace_print(
        cuda_op.cuda_wgmma_encode_matrix_descriptor(d, a, 1, 1, 0),
        "d = T.handle()\na = T.handle()\nT.cuda.wgmma.encode_matrix_descriptor(d, a, 1, 1, 0)",
    )
    _assert_namespace_print(cuda_op.cuda_wgmma_noop_barrier(0), "T.cuda.wgmma.noop_barrier(0)")
