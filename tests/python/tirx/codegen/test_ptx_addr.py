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
"""Tests for ``T.ptx.addr(base, byte_offset)``."""

import pytest

import tvm
from tvm.ir import Call, Op
from tvm.runtime import const
from tvm.script import tirx as T
from tvm.tirx.expr import Broadcast, CallEffectKind

TARGET = tvm.target.Target("cuda")


def _cuda_source(func) -> str:
    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": func}), target=TARGET, tir_pipeline="tirx")
    return mod.mod.imports[0].inspect_source("cuda")


def _calls(func, op_name):
    calls = []

    def visit(node):
        if isinstance(node, Call) and getattr(node.op, "name", None) == op_name:
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(func.body, visit)
    return calls


def test_ptx_addr_registration_and_table_capabilities():
    from tvm.backend.cuda.codegen.registry import CODEGEN_REGISTRY
    from tvm.backend.cuda.ptx.table import TABLE

    op = Op.get("tirx.ptx.addr")
    assert int(op.get_attr("TCallEffectKind")) == CallEffectKind.Pure.value
    assert op.get_attr("TScriptPrinterName") == "ptx.addr"
    assert "tirx.ptx.addr" in CODEGEN_REGISTRY

    addresses = [slot for entry in TABLE.values() for slot in entry.operands if slot.kind == "addr"]
    assert len(addresses) == 145
    assert sum(slot.allow_imm_offset for slot in addresses) == 115
    assert sum(slot.bracket is not None and not slot.allow_imm_offset for slot in addresses) == 5
    assert sum(slot.space == "tmem" and not slot.allow_imm_offset for slot in addresses) == 25


def test_ptx_addr_table_validation_rejects_wrong_operand_classes():
    from tvm.backend.cuda.ptx.table import (
        InstructionEntry,
        OperandSlot,
        _validate_imm_offset_slots,
    )

    bad_entries = (
        InstructionEntry("reg", (OperandSlot("x", allow_imm_offset=True),)),
        InstructionEntry("ptr", (OperandSlot("x", kind="ptr", allow_imm_offset=True),)),
        InstructionEntry(
            "composite",
            (OperandSlot("x", kind="addr", bracket="pair", allow_imm_offset=True),),
        ),
        InstructionEntry(
            "tmem",
            (OperandSlot("x", kind="addr", space="tmem", allow_imm_offset=True),),
        ),
    )
    with pytest.raises(ValueError, match="reg.x.*kind='reg'"):
        _validate_imm_offset_slots(bad_entries)


def test_ptx_addr_coercion_ir_order_and_shared_codegen():
    @T.prim_func
    def kernel(global_buf: T.Buffer((8,), "uint64"), raw_shared: T.uint32, raw_global: T.uint64):
        T.device_entry()
        tx = T.thread_id([32])
        shared_buf = T.alloc_buffer((8,), "uint64", scope="shared")
        value = T.local_scalar("uint64")
        if tx == 0:
            T.ptx.ld.shared.b64(value, T.ptx.addr(shared_buf.data, 4))
            T.ptx.ld.shared.b64(value, T.ptx.addr(raw_shared, 8))
            T.ptx.ld.global_.b64(value, T.ptx.addr(global_buf.data, 12))
            T.ptx.ld.global_.b64(value, T.ptx.addr(raw_global, 16))

    calls = _calls(kernel, "tirx.ptx.addr")
    assert len(calls) == 4
    assert getattr(calls[0].args[0].op, "name", None) == "tirx.cuda.cvta_generic_to_shared"
    assert not isinstance(calls[1].args[0], Call)
    assert getattr(calls[2].args[0].op, "name", None) == "tirx.buffer_data"
    assert getattr(calls[3].args[0].op, "name", None) == "tirx.reinterpret"
    assert [int(call.args[1]) for call in calls] == [4, 8, 12, 16]
    assert all(call.ty == call.args[0].ty for call in calls)

    source = _cuda_source(kernel)
    assert "ld.shared.b64 %0, [%1+4];" in source
    assert "ld.shared.b64 %0, [%1+8];" in source
    # The shared pointer needs cvta; the raw uint32 shared-window address does not.
    assert source.count("__cvta_generic_to_shared") == 1


def test_ptx_addr_scalar_vector_cache_predicate_and_multi_address_codegen():
    @T.prim_func
    def kernel(
        src: T.Buffer((64,), "uint32"),
        dst: T.Buffer((64,), "uint32"),
        policy: T.Buffer((1,), "uint64"),
    ):
        T.device_entry()
        tx = T.thread_id([32])
        shared_buf = T.alloc_buffer((64,), "uint32", scope="shared")
        barrier = T.alloc_buffer((1,), "uint64", scope="shared")
        values = T.alloc_local((2,), "uint32")
        T.ptx.ld.global_.b32(values[0], T.ptx.addr(src.data, 16))
        T.ptx.ld.global_.L2__cache_hint.b32(values[1], T.ptx.addr(src.data, -16), policy[0])
        T.ptx.ld.shared.v2.b32(values[0], values[1], T.ptx.addr(shared_buf.data, 8))
        T.ptx.st.global_.b32(T.ptx.addr(dst.data, 0), values[0])
        T.ptx.st.global_.v2.b32(T.ptx.addr(dst.data, 32), values[0], values[1], pred=tx == 0)
        T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"](
            T.ptx.addr(shared_buf.data, 16),
            T.ptx.addr(src.data, -16),
            T.uint32(16),
            T.ptx.addr(barrier.data, 8),
        )

    source = _cuda_source(kernel)
    assert "ld.global.b32 %0, [%1+16];" in source
    assert "ld.global.L2::cache_hint.b32 %0, [%1+-16], %2;" in source
    assert "ld.shared.v2.b32 {%0, %1}, [%2+8];" in source
    assert "st.global.b32 [%0], %1;" in source
    assert "@p st.global.v2.b32 [%0+32], {%1, %2};" in source
    assert (
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0+16], [%1+-16], %2, [%3+8];"
    ) in source


def test_ptx_addr_zero_sign_boundaries_and_helper_names():
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, tokens_for

    entry = TABLE["ld"]
    tokens = tokens_for(entry, space="global", type="b32")
    bare = render_variant(entry, tokens)
    zero = render_variant(entry, tokens, addr_offsets=((0, 0),))
    positive = render_variant(entry, tokens, addr_offsets=((0, 16),))
    negative = render_variant(entry, tokens, addr_offsets=((0, -16),))
    low = render_variant(entry, tokens, addr_offsets=((0, -(1 << 31)),))
    high = render_variant(entry, tokens, addr_offsets=((0, (1 << 31) - 1),))

    assert zero == bare
    assert bare[1] == "tvm_builtin_ptx_ld_global_b32"
    assert positive[1].endswith("_addr0_p16")
    assert negative[1].endswith("_addr0_m16")
    assert "[%1+16]" in positive[2]
    assert "[%1+-16]" in negative[2]
    assert "[%1+-2147483648]" in low[2]
    assert "[%1+2147483647]" in high[2]

    from tvm.backend.cuda.ptx.table import renderings

    cp_entry = TABLE["cp_async_ca"]
    cp_tokens, cp_dtypes, cp_predicated, cp_imms, cp_sinks = next(iter(renderings(cp_entry)))
    _, helper, source = render_variant(
        cp_entry,
        cp_tokens,
        cp_predicated,
        cp_dtypes,
        cp_imms,
        cp_sinks,
        addr_offsets=((0, 16), (1, -16)),
    )
    assert helper.endswith("_addr0_p16_addr1_m16")
    assert "[%0+16], [%1+-16]" in source

    for value in (-(1 << 31) - 1, 1 << 31):
        with pytest.raises(ValueError, match="outside int32 range"):
            render_variant(entry, tokens, addr_offsets=((0, value),))


def test_ptx_addr_unrolled_expression_and_dynamic_rejection():
    @T.prim_func
    def unrolled(src: T.Buffer((16,), "uint32")):
        T.device_entry()
        tx = T.thread_id([32])
        value = T.local_scalar("uint32")
        if tx == 0:
            for i in T.unroll(3):
                T.ptx.ld.global_.b32(value, T.ptx.addr(src.data, i * 16))

    source = _cuda_source(unrolled)
    assert "ld.global.b32 %0, [%1];" in source
    assert "ld.global.b32 %0, [%1+16];" in source
    assert "ld.global.b32 %0, [%1+32];" in source

    @T.prim_func
    def thread_dynamic(src: T.Buffer((16,), "uint32")):
        T.device_entry()
        tx = T.thread_id([32])
        value = T.local_scalar("uint32")
        T.ptx.ld.global_.b32(value, T.ptx.addr(src.data, tx * 4))

    @T.prim_func
    def loop_dynamic(src: T.Buffer((16,), "uint32")):
        T.device_entry()
        tx = T.thread_id([32])
        value = T.local_scalar("uint32")
        if tx == 0:
            for i in T.serial(2):
                T.ptx.ld.global_.b32(value, T.ptx.addr(src.data, i * 4))

    for func in (thread_dynamic, loop_dynamic):
        with pytest.raises(
            (ValueError, tvm.error.InternalError), match="must become a compile-time"
        ):
            _cuda_source(func)


def test_ptx_addr_offset_type_and_range_rejections():
    for value in (True, 1.5, Broadcast(const(1, "int32"), 4)):
        with pytest.raises(ValueError, match="byte_offset"):
            T.ptx.addr(None, value)
    for value in (-(1 << 31) - 1, 1 << 31):
        with pytest.raises(ValueError, match="outside signed int32 range"):
            T.ptx.addr(None, value)
    with pytest.raises(ValueError, match="cannot be nested"):
        T.ptx.addr(T.ptx.addr(None, 0), 4)


def test_ptx_addr_pointer_and_raw_address_validation():
    with pytest.raises(
        (ValueError, tvm.error.DiagnosticError), match="uint32 address requires shared"
    ):

        @T.prim_func
        def global_u32(raw: T.uint32):
            T.device_entry()
            value = T.local_scalar("uint32")
            T.ptx.ld.global_.b32(value, T.ptx.addr(raw, 4))

    with pytest.raises(
        (ValueError, tvm.error.DiagnosticError), match="does not support T.ptx.addr"
    ):

        @T.prim_func
        def ptr_operand(src: T.Buffer((8,), "uint32")):
            T.device_entry()
            result = T.local_scalar("uint32")
            T.ptx.isspacep.global_(result, T.ptx.addr(src.data, 4))


def test_ptx_addr_tma_tmem_and_independent_immediate_rejections():
    with pytest.raises(
        (ValueError, tvm.error.DiagnosticError), match="does not support T.ptx.addr"
    ):

        @T.prim_func
        def tma(tmap: T.Buffer((8,), "uint64")):
            T.device_entry()
            shared_buf = T.alloc_buffer((16,), "uint32", scope="shared")
            barrier = T.alloc_buffer((1,), "uint64", scope="shared")
            T.ptx["cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes"](
                shared_buf.data, T.ptx.addr(tmap.data, 16), T.int32(0), barrier.data
            )

    with pytest.raises(
        (ValueError, tvm.error.DiagnosticError), match="does not support T.ptx.addr"
    ):

        @T.prim_func
        def tmem(raw: T.uint32):
            T.device_entry()
            value = T.local_scalar("uint32")
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x1.b32"](value, T.ptx.addr(raw, 16))

    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, variants

    entry = TABLE["tcgen05_ld_split"]
    tokens = variants(entry)[0]
    _, helper, source = render_variant(entry, tokens, imms=("16",))
    assert helper.endswith("_16")
    assert "[%" in source and "], 16;" in source
    with pytest.raises(ValueError, match="does not support an immediate offset"):
        render_variant(entry, tokens, imms=("16",), addr_offsets=((0, 16),))


def test_ptx_addr_printer_script_and_json_roundtrip():
    @T.prim_func
    def kernel(src: T.Buffer((8,), "uint32"), dst: T.Buffer((8,), "uint32")):
        T.device_entry()
        value = T.local_scalar("uint32")
        T.ptx.ld.global_.b32(value, T.ptx.addr(src.data, -16))
        T.ptx.st.global_.b32(T.ptx.addr(dst.data, 16), value)

    script = kernel.script()
    assert script.count("T.ptx.addr(") == 2
    tvm.ir.assert_structural_equal(kernel, tvm.script.from_source(script))
    tvm.ir.assert_structural_equal(kernel, tvm.ir.load_json(tvm.ir.save_json(kernel)))


def test_ptx_addr_legacy_positional_offsets_rejected():
    with pytest.raises((ValueError, tvm.error.DiagnosticError)):

        @T.prim_func
        def scalar_load(src: T.Buffer((8,), "uint32")):
            T.device_entry()
            value = T.local_scalar("uint32")
            T.ptx.ld.global_.b32(value, src.data, 16)

    with pytest.raises((ValueError, tvm.error.DiagnosticError)):

        @T.prim_func
        def vector_load(src: T.Buffer((8,), "uint32")):
            T.device_entry()
            values = T.alloc_local((2,), "uint32")
            T.ptx.ld.global_.v2.b32(values[0], values[1], src.data, 16)

    with pytest.raises((ValueError, tvm.error.DiagnosticError)):

        @T.prim_func
        def scalar_store(dst: T.Buffer((8,), "uint32")):
            T.device_entry()
            T.ptx.st.global_.b32(dst.data, 16, T.uint32(0))

    with pytest.raises((ValueError, tvm.error.DiagnosticError)):

        @T.prim_func
        def vector_store(dst: T.Buffer((8,), "uint32")):
            T.device_entry()
            T.ptx.st.global_.v2.b32(dst.data, 16, T.uint32(0), T.uint32(0))


def test_ptx_addr_unconsumed_codegen_diagnostic():
    from tvm.backend.cuda.codegen.registry import CODEGEN_REGISTRY

    with pytest.raises(ValueError, match="must be consumed by a PTX address operand"):
        CODEGEN_REGISTRY["tirx.ptx.addr"]([const(0, "uint64"), const(16, "int32")])
