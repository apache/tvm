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
"""Tests for the table-driven PTX dialect (``T.ptx``)."""

import os
import re
import shutil

import numpy as np
import pytest

import tvm
from tvm.ir import Op
from tvm.script import tirx as T
from tvm.testing import env

TARGET = tvm.target.Target("cuda")

requires_nvcc = pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available")


def _cuda_source(func) -> str:
    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": func}), target=TARGET, tir_pipeline="tirx")
    return mod.mod.imports[0].inspect_source("cuda")


# Architecture the certification assembles at. Families whose ISA floor is
# higher (tcgen05, clusterlaunchcontrol, several cp.async.bulk forms) MUST be
# certified at their own floor: assembling them below it makes ptxas report
# legal variants as illegal, which would then get baked into a check().
PTX_ARCH = os.environ.get("PTX_ARCH", "sm_90")


def _assert_ptxas_ok(src: str, rdc: bool = False, arch: str = PTX_ARCH) -> None:
    """Assemble through ptxas (cubin) — `-ptx` alone never validates inline asm."""
    from tvm.support import nvcc

    options = ["-rdc=true"] if rdc else None
    nvcc.compile_cuda(src, target_format="cubin", arch=arch, options=options, compiler="nvcc")


def test_ptx_registration():
    from tvm.backend.cuda.intrinsics.registry import CODEGEN_REGISTRY
    from tvm.backend.cuda.ptx.table import TABLE

    assert hasattr(T, "ptx")
    for entry in TABLE.values():
        op = Op.get(entry.op_name)  # raises if unregistered
        assert op.get_attr("TCallEffectKind") is not None, entry.name
        family = entry.family  # several entries may share a mnemonic
        assert op.get_attr("TScriptPrinterName") == f"ptx.{family}", entry.name
        assert entry.op_name in CODEGEN_REGISTRY, entry.name


def test_ptx_prefetch_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        T.ptx.prefetch.global_.L2(A.ptr_to([0]))
        A[tx] = T.float32(0)  # keep-alive store so the buffer is not elided

    src = _cuda_source(kernel)
    assert "prefetch.global.L2 [%0];" in src
    assert "tvm_builtin_ptx_prefetch_global_L2" in src


def test_ptx_ld_st_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            # Declare the register, then name it as an operand — the PTX model.
            val = T.local_scalar("uint32")
            T.ptx.ld.global_.acquire.gpu.b32(val, A.ptr_to([0]))
            T.ptx.st.release.gpu.global_.b32(B.ptr_to([0]), val)
        B[tx] = B[tx]

    src = _cuda_source(kernel)
    # Modifier tokens render in table slot order (sem, scope, space, type)
    # regardless of the order they were written in the chain.
    assert "ld.acquire.gpu.global.b32 %0, [%1];" in src
    assert "st.release.gpu.global.b32 [%0], %1;" in src
    assert "tvm_builtin_ptx_ld_acquire_gpu_global_b32" in src
    assert "tvm_builtin_ptx_st_release_gpu_global_b32" in src


def test_ptx_st_shared_coercion():
    @T.prim_func
    def kernel(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        if tx == 0:
            # Shared-space slot fed a shared-scope pointer: engine must
            # auto-wrap with cvta_generic_to_shared.
            T.ptx.st.shared__cta.b32(smem.ptr_to([0]), T.uint32(7))
        T.cuda.cta_sync()
        out[0] = smem[0]

    src = _cuda_source(kernel)
    assert "st.shared::cta.b32 [%0], %1;" in src
    assert "cvta_generic_to_shared" in src


def test_ptx_explicit_cvta():
    @T.prim_func
    def kernel(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "uint64")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        smem[tx % 4] = T.uint32(0)
        if tx == 0:
            T.ptx.cvta.to.shared.u64(out[0], smem.data)

    src = _cuda_source(kernel)
    assert "cvta.to.shared.u64 %0, %1;" in src
    assert "tvm_builtin_ptx_cvta_to_shared_u64" in src


def test_ptx_red_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (1,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.ptx.red.relaxed.gpu.global_.add.u32(A.ptr_to([0]), T.uint32(1))
        A[0] = A[0]

    src = _cuda_source(kernel)
    assert "red.relaxed.gpu.global.add.u32 [%0], %1;" in src


def test_ptx_predication_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        flag: T.uint32 = T.uint32(0)
        if tx == 0:
            flag = T.uint32(1)
        T.ptx.red.relaxed.gpu.global_.add.u32(A.ptr_to([0]), T.uint32(1), pred=flag)
        A[tx] = A[tx]

    src = _cuda_source(kernel)
    assert "tvm_builtin_ptx_red_relaxed_gpu_global_add_u32_pred" in src
    assert "setp.ne.b32 p, %2, 0; @p red.relaxed.gpu.global.add.u32 [%0], %1;" in src


def test_ptx_string_form_matches_chain():
    chain_call = T.ptx.ld.global_.acquire.gpu.b32
    string_call = T.ptx["ld.acquire.gpu.global.b32"]

    def make(fn):
        @T.prim_func
        def kernel(a_ptr: T.handle, b_ptr: T.handle):
            A = T.match_buffer(a_ptr, (32,), "uint32")
            B = T.match_buffer(b_ptr, (32,), "uint32")
            T.device_entry()
            T.cta_id([1])
            tx = T.thread_id([32])
            if tx == 0:
                fn(B[0], A.ptr_to([0]))
            B[tx] = B[tx]

        return kernel

    tvm.ir.assert_structural_equal(make(chain_call), make(string_call))


def test_ptx_trace_time_errors():
    # Global ld fed a raw uint32 address.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="shared state space"):

        @T.prim_func
        def bad_global_addr(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.ld.global_.b32(out[0], T.uint32(0))

    # Bogus modifier token.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def bad_modifier(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.ld.global_.bogus.b32(out[0], T.uint32(0))

    # Value dtype of the wrong *width*. A bit type accepts any dtype of its own
    # width (see test_ptx_bit_width_axis), so the rejection is about size.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must have dtype"):

        @T.prim_func
        def bad_value_dtype(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.st.global_.b32(A.ptr_to([0]), T.float64(1.0))

    # Missing required modifier (no type token).
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="missing required modifier"):

        @T.prim_func
        def missing_type(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.st.global_(A.ptr_to([0]), T.uint32(0))

    # Per-slot legal tokens whose combination is illegal PTX: acquire
    # requires a scope — rejected by the entry's check function.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="requires a scope"):

        @T.prim_func
        def acquire_without_scope(out: T.Buffer((1,), "uint32"), a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.ld.global_.acquire.b32(out[0], A.ptr_to([0]))

    # A float is not an address in any state space.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="pointer or uint64 handle"):

        @T.prim_func
        def bad_addr_dtype(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.ld.global_.b32(out[0], T.float32(0))


def test_ptx_destination_errors():
    """A destination is a register the caller declared: it must be a writable lvalue."""
    # Destination of the wrong width (a .b32 load into a 64-bit slot).
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must have dtype"):

        @T.prim_func
        def wrong_dst_dtype(out: T.Buffer((1,), "float64"), a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.ld.global_.b32(out[0], A.ptr_to([0]))

    # A T.let binding is immutable, so it cannot be written into. This is the
    # gate that keeps the analyzer from re-expanding one call into N.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="writable scalar"):

        @T.prim_func
        def let_destination(out: T.Buffer((1,), "uint32"), a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            bound: T.let = out[0] + T.uint32(1)
            T.ptx.ld.global_.b32(bound, A.ptr_to([0]))

    # An rvalue is not a destination either.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="writable scalar"):

        @T.prim_func
        def rvalue_destination(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.ld.global_.b32(T.uint32(0), A.ptr_to([0]))

    # @p is rejected on any instruction that writes a destination.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="without a destination"):

        @T.prim_func
        def predicated_destination(out: T.Buffer((1,), "uint32"), a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.ld.global_.b32(out[0], A.ptr_to([0]), pred=T.uint32(1))


def test_ptx_register_group_codegen():
    """A `.lanes > 1` operand renders as braces in the asm, flat params in C.

    `{%1, %2}` is ONE PTX operand occupying two registers (ISA 9.7.9.4), which
    is why the lane count appears nowhere in the instruction text -- `mov.b64`
    names the aggregate width, and the operand shape carries the rest.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        packed = T.local_scalar("uint64")
        lo = T.local_scalar("float32")
        hi = T.local_scalar("float32")
        T.ptx.mov.b64(packed, A[0], A[1])  # pack: one dst, a 2-register source
        T.ptx.mov.b64(lo, hi, packed)  # unpack: a 2-register dst, one source
        A[tx % 4] = lo + hi

    src = _cuda_source(kernel)
    assert "mov.b64 %0, {%1, %2};" in src
    assert "mov.b64 {%0, %1}, %2;" in src
    # Both shapes share the mnemonic; the operand shape picks the entry, which
    # is the same information ptxas resolves them by.
    assert "tvm_builtin_ptx_mov_pack_b32x2_b64_u64_f32(" in src
    assert "tvm_builtin_ptx_mov_unpack_b32x2_b64_f32_u64(" in src


def test_ptx_register_group_errors():
    """A register group is one operand: its arity is fixed and its lanes agree."""
    # No `mov` shape takes two operands, so nothing in the family matches.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match=r"expects \d+ operand"):

        @T.prim_func
        def wrong_arity(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (4,), "uint32")
            T.device_entry()
            packed = T.local_scalar("uint64")
            T.ptx.mov.b64(packed, A[0])
            A[0] = T.uint32(0)

    # Each lane of a destination group is its own register the caller declared,
    # so each has to be an lvalue -- not just the first.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="writable scalar"):

        @T.prim_func
        def non_lvalue_lane(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (4,), "uint32")
            T.device_entry()
            packed = T.local_scalar("uint64")
            lo = T.local_scalar("uint32")
            T.ptx.mov.b64(lo, A[0] + T.uint32(1), packed)
            A[0] = T.uint32(0)

    # Lanes disagreeing on dtype: legal for each lane alone (both are 32-bit),
    # but the group is one operand with one C parameter type, so binding the
    # odd lane to it would be a numeric conversion.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must have one dtype"):

        @T.prim_func
        def mixed_lane_dtypes(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (4,), "uint32")
            T.device_entry()
            packed = T.local_scalar("uint64")
            f = T.local_scalar("float32")
            u = T.local_scalar("uint32")
            T.ptx.mov.b64(packed, f, u)
            A[0] = T.uint32(0)

    # A bare float literal names no dtype: on a .b32 lane it could be the
    # float's bits or the number, and it used to silently become T.uint32(1).
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="is ambiguous"):

        @T.prim_func
        def bare_float_literal(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (4,), "uint32")
            T.device_entry()
            packed = T.local_scalar("uint64")
            T.ptx.mov.b64(packed, 1.5, 2.5)
            A[0] = T.uint32(0)

    # An explicit constant is accepted and picks the float32 helper.
    @T.prim_func
    def typed_literal(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        packed = T.local_scalar("uint64")
        T.ptx.mov.b64(packed, T.float32(1.5), T.float32(2.5))
        A[tx % 4] = A[tx % 4]

    assert "mov_pack_b32x2_b64_u64_f32" in _cuda_source(typed_literal)


def test_ptx_optional_operand_arity_dispatch():
    """A no-count and a counted syntax line share a mnemonic, split by arity.

    This only works because `pred` is keyword-only: the old positional-pred
    fallback let every entry also accept arity+1 calls, so both lines matched a
    two-operand call and dispatch was ambiguous. The pred marker in the Call
    layout keeps the printed form exact (a predicated no-count arrive must not
    re-parse as a counted one).
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        bar = T.alloc_buffer((2,), "uint64", scope="shared")
        T.ptx.bar.sync(T.uint32(0))
        T.ptx.bar.sync(T.uint32(0), T.uint32(64))
        T.ptx.mbarrier.arrive.shared.b64(bar.ptr_to([0]))
        T.ptx.mbarrier.arrive.shared.b64(bar.ptr_to([0]), T.uint32(2))
        T.ptx.mbarrier.arrive.shared.b64(bar.ptr_to([1]), pred=T.uint32(1))
        A[tx % 4] = A[tx % 4]

    src = _cuda_source(kernel)
    assert "bar.sync %0;" in src
    assert "bar.sync %0, %1;" in src
    assert "mbarrier.arrive.shared.b64 _, [%0];" in src
    assert "mbarrier.arrive.shared.b64 _, [%0], %1;" in src
    assert "@p mbarrier.arrive.shared.b64 _, [%0];" in src

    # The predicated no-count arrive survives a print/parse round trip as
    # itself -- the pred marker is what stops the count entry from absorbing
    # the predicate as a count.
    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)


def test_ptx_bit_width_axis():
    """A `.bN` operand takes any dtype of that width, each with its own helper.

    PTX ISA 5.2: "The bit-size type is compatible with any fundamental type
    having the same size." The helper follows the dtype the caller actually
    holds, so the value binds its own register class and no conversion is
    emitted -- handing a float to a uint32_t parameter would instead be a
    numeric conversion and emit `cvt.rzi.u32.f32`.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle, o_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "float32")
        Out = T.match_buffer(o_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        fv = T.local_scalar("float32")
        sv = T.local_scalar("int32")
        T.ptx.ld.global_.b32(fv, A.ptr_to([0]))  # .b32 destination, float32
        T.ptx.ld.global_.b32(sv, A.ptr_to([1]))  # .b32 destination, int32
        T.ptx.st.global_.b32(Out.ptr_to([0]), fv)  # .b32 source, float32
        Out[tx % 4] = A[tx % 4]

    src = _cuda_source(kernel)
    # One instruction text, three signatures, named by the dtypes they carry.
    assert "tvm_builtin_ptx_ld_global_b32_f32(float& __d" in src
    assert "tvm_builtin_ptx_ld_global_b32_s32(int32_t& __d" in src
    assert "tvm_builtin_ptx_st_global_b32_f32(const void* __addr, float __value" in src
    assert '"=f"(__d)' in src and '"=r"(__d)' in src
    assert src.count("ld.global.b32 %0, [%1];") >= 1
    # The float binds "f" directly. Routing it through the canonical uint32_t
    # parameter would have been a numeric conversion, not a bit pun.
    assert "cvt." not in src
    # The canonical dtype keeps the unsuffixed name it had before the axis.
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, tokens_for

    ld = TABLE["ld"]
    tokens = tokens_for(ld, space="global", type="b32")
    assert render_variant(ld, tokens)[1] == "tvm_builtin_ptx_ld_global_b32"


def test_ptx_u64_address_handle_is_reinterpreted():
    """A 64-bit address handle is accepted via an explicit, visible cast.

    Some PTX address operands reach us as u64 handles rather than typed
    pointers (``T.address_of(tensormap)`` is the motivating case). PTX binds
    both to the same "l" register, so the conversion must exist in the IR
    rather than being punned inside the helper.
    """
    handle = tvm.tirx.Var("h", "uint64")
    call = T.ptx.prefetch.tensormap(handle)
    addr = call.args[0]
    assert addr.op.name == "tirx.reinterpret", addr
    assert addr.args[0].same_as(handle)


def test_ptx_parser_roundtrip():
    """script() output re-parses to a structurally equal PrimFunc."""

    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        if tx == 0:
            val = T.local_scalar("uint32")
            smem_addr = T.local_scalar("uint64")
            lo = T.local_scalar("float32")
            hi = T.local_scalar("float32")
            packed = T.local_scalar("uint64")
            # Several `mov` entries share one mnemonic; the printed form names
            # the family and reparsing re-dispatches on the operand shape.
            T.ptx.mov.b64(packed, lo, hi)
            T.ptx.mov.b64(lo, hi, packed)
            T.ptx.ld.global_.acquire.gpu.b32(val, A.ptr_to([0]))
            T.ptx.st.shared__cta.b32(smem.ptr_to([0]), val)
            T.ptx.red.relaxed.gpu.global_.add.u32(B.ptr_to([0]), T.uint32(1), pred=val)
            T.ptx.prefetch.global_.L2(A.ptr_to([16]))
            T.ptx.cvta.to.shared.u64(smem_addr, smem.data)
        T.cuda.cta_sync()
        B[tx] = smem[tx % 4]

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)


def test_ptx_pred_operand_roundtrip():
    """A `.pred` operand survives print/parse, tag and all.

    The printed text carries neither the `T.ptx.pred(...)` wrapper nor the
    register class -- a predicate and an integer share the same uint32 carrier
    -- so the marker has to name the position. Without that, reparsing would
    re-dispatch on a bare uint32 and could pick a *different* entry of the same
    family (which is exactly what the src-size / ignore-src pair does).
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            tmem = T.local_scalar("uint32")
            desc = T.local_scalar("uint64")
            idesc = T.local_scalar("uint32")
            flag = T.local_scalar("uint32")
            T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                tmem, desc, desc, idesc, 0, 0, 0, 0, T.ptx.pred(flag)
            )
        A[tx] = A[tx]

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)


def test_ptx_pred_operand_rejects_untagged_integer():
    """An untagged integer at a `.pred` position is refused, by name.

    The carrier is shared, so accepting it would erase the only thing that
    tells the two `cp.async` optional-operand syntax lines apart. A bool needs
    no tag -- it already says what it is.
    """
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match=r"T\.ptx\.pred"):

        @T.prim_func
        def untagged_integer():
            T.device_entry()
            T.cta_id([1])
            tmem = T.local_scalar("uint32")
            desc = T.local_scalar("uint64")
            idesc = T.local_scalar("uint32")
            flag = T.local_scalar("uint32")
            T.ptx["tcgen05.mma.cta_group::1.kind::f16"](tmem, desc, desc, idesc, 0, 0, 0, 0, flag)

    # A bool expression carries the class in its own dtype, so it needs no tag.
    @T.prim_func
    def bool_needs_no_tag(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            tmem = T.local_scalar("uint32")
            desc = T.local_scalar("uint64")
            idesc = T.local_scalar("uint32")
            T.ptx["tcgen05.mma.cta_group::1.kind::f16"](
                tmem, desc, desc, idesc, 0, 0, 0, 0, tx == 0
            )
        A[tx] = A[tx]

    assert bool_needs_no_tag is not None


def test_ptx_printer_form():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.ptx.ld.global_.acquire.gpu.b32(B[0], A.ptr_to([0]))
        B[tx] = B[tx]

    script = kernel.script()
    assert "T.ptx.ld(" in script
    assert '"acquire"' in script


# ---------------------------------------------------------------------------
# Registered-instruction unit tests: pin down the engine's generated helpers
# and its trace-time coercion so the behavior is readable here, not implicit.
# ---------------------------------------------------------------------------


def test_ptx_helper_source_golden():
    """Exact generated helper source, one per family (executable documentation)."""
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, tokens_for

    def render(name, predicated=False, dtypes=None, **by_name):
        entry = TABLE[name]
        return render_variant(entry, tokens_for(entry, **by_name), predicated, dtypes)[2]

    # tokens_for is what lets the goldens below name their modifiers. Positional
    # tuples silently shift when a slot is inserted, which is the one edit this
    # table invites; naming makes that a loud error instead.
    assert tokens_for(TABLE["ld"], space="global", type="b32") == (
        ("", "", "", "global", "", "", "", "", "b32")
    )
    with pytest.raises(ValueError, match="no modifier slot named"):
        tokens_for(TABLE["ld"], storage="global", type="b32")
    with pytest.raises(ValueError, match="not in"):
        tokens_for(TABLE["ld"], space="tmem", type="b32")
    with pytest.raises(ValueError, match="is required"):
        tokens_for(TABLE["ld"], space="global")

    assert render("prefetch", space="global", level="L2") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_prefetch_global_L2"
        "(const void* __addr) {\n"
        '  asm volatile("prefetch.global.L2 [%0];" :  : "l"(__addr) : "memory");\n'
        "}\n"
    )
    # A destination is an ordinary operand taken by reference, so the C
    # parameter list is the PTX operand list in order and the helper is void.
    assert render("ld", sem="acquire", scope="gpu", space="global", type="b32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_ld_acquire_gpu_global_b32"
        "(uint32_t& __d, const void* __addr) {\n"
        '  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(__d) : "l"(__addr)'
        ' : "memory");\n'
        "}\n"
    )
    # 8-bit destination: no asm constraint of its own, so it rides a 16-bit
    # carrier register and is narrowed into the reference afterwards. The asm
    # block still holds exactly one instruction.
    assert render("ld", space="global", type="b8") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_ld_global_b8"
        "(uint8_t& __d, const void* __addr) {\n"
        "  uint16_t __d_reg;\n"
        '  asm volatile("ld.global.b8 %0, [%1];" : "=h"(__d_reg) : "l"(__addr) : "memory");\n'
        "  __d = (uint8_t)__d_reg;\n"
        "}\n"
    )
    # Shared-space addr slot: helper takes uint32_t (post-coercion form), not void*.
    assert render("st", space="shared::cta", type="b32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_st_shared__cta_b32"
        "(uint32_t __addr, uint32_t __value) {\n"
        '  asm volatile("st.shared::cta.b32 [%0], %1;" :  : "r"(__addr), "r"(__value)'
        ' : "memory");\n'
        "}\n"
    )
    # Register-only op: plain asm (asm_volatile=False), no memory clobber.
    assert render("cvta", dir="to", space="shared", type="u64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_cvta_to_shared_u64"
        "(uint64_t& __d, const void* __ptr) {\n"
        '  asm("cvta.to.shared.u64 %0, %1;" : "=l"(__d) : "l"(__ptr));\n'
        "}\n"
    )
    # Predicated twin (framework-level @p): extra uint32 pred param,
    # setp + @p guard wrapping the same instruction.
    assert render(
        "red", predicated=True, sem="relaxed", scope="gpu", space="global", op="add", type="u32"
    ) == (
        "__forceinline__ __device__ void tvm_builtin_ptx_red_relaxed_gpu_global_add_u32_pred"
        "(const void* __addr, uint32_t __value, uint32_t __pred) {\n"
        '  asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; '
        '@p red.relaxed.gpu.global.add.u32 [%0], %1; }"'
        ' :  : "l"(__addr), "r"(__value), "r"(__pred) : "memory");\n'
        "}\n"
    )
    # Mixed-space operands: per-operand space/dtype pick each carrier —
    # shared addrs are uint32, the global addr is a pointer.
    assert render(
        "cp_async_bulk_g2s_cta",
        api="async",
        kind="bulk",
        dst="shared::cta",
        src="global",
        completion="mbarrier::complete_tx::bytes",
    ) == (
        "__forceinline__ __device__ void tvm_builtin_ptx_cp_async_bulk_g2s_cta_async_bulk"
        "_shared__cta_global_mbarrier__complete_tx__bytes"
        "(uint32_t __dst_mem, const void* __src_mem, uint32_t __size, uint32_t __mbar) {\n"
        '  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes '
        '[%0], [%1], %2, [%3];" :  : "r"(__dst_mem), "l"(__src_mem), "r"(__size), "r"(__mbar)'
        ' : "memory");\n'
        "}\n"
    )

    # Register group: `{%1, %2}` is ONE PTX operand spanning two registers, so
    # the group is braces in the asm text and flat C parameters around it. An
    # unpack turns that into two "=" outputs.
    assert render("mov_pack_b32x2", type="b64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mov_pack_b32x2_b64"
        "(uint64_t& __d, uint32_t __a0, uint32_t __a1) {\n"
        '  asm("mov.b64 %0, {%1, %2};" : "=l"(__d) : "r"(__a0), "r"(__a1));\n'
        "}\n"
    )
    assert render("mov_unpack_b32x2", type="b64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mov_unpack_b32x2_b64"
        "(uint32_t& __d0, uint32_t& __d1, uint64_t __a) {\n"
        '  asm("mov.b64 {%0, %1}, %2;" : "=r"(__d0), "=r"(__d1) : "l"(__a));\n'
        "}\n"
    )
    # 128-bit destination on the "q" constraint, and asm_volatile=False: a
    # register shuffle nvcc is free to common up.
    assert render("mov_pack_b64x2", type="b128") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mov_pack_b64x2_b128"
        "(__uint128_t& __d, uint64_t __a0, uint64_t __a1) {\n"
        '  asm("mov.b128 %0, {%1, %2};" : "=q"(__d) : "l"(__a0), "l"(__a1));\n'
        "}\n"
    )
    # Same instruction text, f32 lanes: the dtype axis renames the helper and
    # moves the lanes to "f", and nothing else moves. This is the shape the
    # packed-f32x2 call sites use.
    assert render("mov_pack_b32x2", type="b64", dtypes=("uint64", "float32")) == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mov_pack_b32x2_b64_u64_f32"
        "(uint64_t& __d, float __a0, float __a1) {\n"
        '  asm("mov.b64 %0, {%1, %2};" : "=l"(__d) : "f"(__a0), "f"(__a1));\n'
        "}\n"
    )


def test_ptx_coercion_ir_forms():
    """The trace-time addr coercion, written down as IR-level assertions."""
    from tvm.ir.type import PointerType, PrimType

    shared_ptr = tvm.tirx.Var("p", PointerType(PrimType("uint32"), "shared"))
    global_ptr = tvm.tirx.Var("g", PointerType(PrimType("uint32"), "global"))
    raw_u32 = tvm.tirx.Var("a", "uint32")
    val = tvm.tirx.Var("v", "uint32")

    # Shared slot + shared pointer: engine wraps with an explicit cvta call node.
    call = T.ptx.st.shared__cta.b32(shared_ptr, val)
    addr = call.args[0]
    assert addr.op.name == "tirx.cuda.cvta_generic_to_shared"
    assert addr.args[0].same_as(shared_ptr)

    # Shared slot + raw uint32: passthrough, no conversion inserted.
    call = T.ptx.st.shared__cta.b32(raw_u32, val)
    assert call.args[0].same_as(raw_u32)

    # Global slot + global pointer: passthrough.
    call = T.ptx.st.release.gpu.global_.b32(global_ptr, val)
    assert call.args[0].same_as(global_ptr)

    # Modifiers ride as trailing positional string args in slot order, then
    # the pred marker ("pred" or "") that makes the printed form re-parse
    # exactly instead of guessing from the argument count.
    assert [str(a).strip('"') for a in call.args[2:]] == [
        "",
        "release",
        "gpu",
        "global",
        "",
        "",
        "b32",
        "",
    ]

    # A shared-space slot converts whatever pointer it is given: TIRx pointer
    # scopes are not a reliable discriminator (a shared buffer's ptr_to()
    # reports 'global'), and the legacy helpers converted unconditionally too.
    call = T.ptx.st.shared__cta.b32(global_ptr, val)
    assert call.args[0].op.name == "tirx.cuda.cvta_generic_to_shared"

    # Predication: pred rides after the operands (codegen derives the
    # predicated form from the arg count).
    flag = tvm.tirx.Var("f", "uint32")
    call = T.ptx.st.release.gpu.global_.b32(global_ptr, val, pred=flag)
    assert call.args[2].same_as(flag)
    assert len(call.args) == 2 + 1 + 7 + 1  # operands + pred + slot tokens + marker
    # @p on an instruction with a destination is rejected: a false predicate
    # leaves it unwritten while "=" tells nvcc its prior value is dead.
    dst = tvm.tirx.Var("d", "uint32")
    with pytest.raises(ValueError, match="without a destination"):
        T.ptx.ld.global_.b32(dst, global_ptr, pred=flag)


# fp16/bf16 dtypes bring in __half / __nv_bfloat16 and their bit-cast helpers.
_CERT_PRELUDE = "#include <cstdint>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>"

_ASM_RE = re.compile(r'asm(?: volatile)?\("(.*?)"\s*:', re.S)
_BLOCK_RE = re.compile(r"^\{ (?P<body>.*) \}$")
# The sanctioned in-block boundary conversions, and nothing else: predicate
# register declarations, setp conversions in (@p's own guard and pred_src
# operands), selp materializations out (pred_dst operands).
# The asm block's sanctioned non-instructions: `render.BRIDGE`'s register
# declarations and the conversions that move a value between the class the ISA
# names and the carrier inline asm can bind. Never semantics -- see BRIDGE.
_BOUNDARY_PREFIXES = (
    ".reg .pred ",
    ".reg .b8 ",
    "setp.ne.b32 ",
    "selp.b32 ",
    "cvt.u8.u16 ",
    "cvt.u16.u8 ",
)


def _as_render_args(rendering):
    """`renderings` yields (tokens, dtypes, predicated, imms); render_variant
    takes (tokens, predicated, dtypes, imms)."""
    tokens, dtypes, predicated, imms = rendering
    return tokens, predicated, dtypes, imms


def _sole_instruction(asm_text):
    """The single PTX statement in ``asm_text``, or None if it is not exactly one.

    The sanctioned boundary conversions are peeled first: ``@p``'s guard,
    pred_src's setp, pred_dst's selp. They convert values at the block
    boundary; they never add a second instruction.
    """
    m = _BLOCK_RE.match(asm_text)
    if m:
        stmts = [f"{part.strip()};" for part in m.group("body").split(";") if part.strip()]
        core = [st for st in stmts if not st.startswith(_BOUNDARY_PREFIXES)]
        if len(core) != 1:
            return None
        body = core[0].removeprefix("@p ")
    else:
        body = asm_text
    if body.count(";") != 1 or not body.endswith(";"):
        return None
    return body


def test_ptx_single_instruction_invariant():
    """Every ptx variant emits exactly ONE native PTX instruction.

    This is the dialect's defining constraint, enforced mechanically so it
    cannot erode as the table grows: the only multi-statement form allowed is
    the framework-level ``@p`` wrapper, which guards a single instruction
    rather than adding one. cvta coercion is a separate IR node and must never
    appear inside a helper body.

    ``RAW_ENTRIES`` below is the one exemption, and it is a list of names, not
    a predicate: the entries whose helper body the table cannot derive because
    one operand is typed ``.b8`` and inline asm has no 8-bit constraint, so the
    value has to be staged through a block-local ``.reg .b8``. Naming them here
    and asserting the set equals the table's own ``raw_render`` entries means a
    new one can never be added without editing this test. Every other assertion
    still applies to them.
    """
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, renderings

    # Empty, and it should stay that way: the one shape that used to need a
    # hand-written body (`.e2m1x2`'s .b8 operand) turned out to be a dtype with
    # a bridge, not an irregular family.
    RAW_ENTRIES = set()
    assert RAW_ENTRIES == {name for name, e in TABLE.items() if e.raw_render}, (
        "the set of hand-written (raw_render) entries changed; each one is a "
        "permanent exemption from the single-instruction invariant, so it has "
        "to be added here deliberately"
    )

    asm_re, sole_instruction = _ASM_RE, _sole_instruction
    checked = 0
    for entry in TABLE.values():
        raw = entry.raw_render is not None
        for tokens, dtypes, predicated, imms in renderings(entry):
            opcode, helper, source = render_variant(entry, tokens, predicated, dtypes, imms)
            asm_blocks = asm_re.findall(source)
            assert len(asm_blocks) == 1, f"{opcode}: {len(asm_blocks)} asm blocks, expected 1"
            if raw:
                # The `.reg .b8` prologue is several statements, which is why
                # this entry is exempt. The instruction must still be in there
                # as its own statement. The helper name needs no assertion:
                # `render_variant` passes the name it derived into
                # `raw_render`, so a raw body cannot declare a different one.
                assert f"; {opcode} " in asm_blocks[0], (
                    f"{opcode}: raw body does not emit the opcode: {asm_blocks[0]!r}"
                )
            else:
                instr = sole_instruction(asm_blocks[0])
                assert instr is not None, f"{opcode}: not a single statement: {asm_blocks[0]!r}"
                assert instr.startswith(opcode + " ") or instr == opcode + ";", (
                    f"{opcode}: emitted instruction does not match the opcode: {instr!r}"
                )
            assert "cvta" not in source or entry.name == "cvta", (
                f"{opcode}: cvta must be a separate IR node, never inside a helper"
            )
            # Every helper is void: a PTX destination is an operand, never a C
            # return value.
            assert source.startswith("__forceinline__ __device__ void "), (
                f"{opcode}: helper must be void, got {source.splitlines()[0]!r}"
            )
            assert "return" not in source, f"{opcode}: helper must not return a value"
            checked += 1
    assert checked > 0


def test_ptx_single_instruction_invariant_detects_violations():
    """Falsify the probe: it must reject every shape the invariant forbids.

    A guard that has never been shown to fail is worth nothing, so exercise
    the real helper the invariant test uses.
    """
    forbidden = {
        # two chained instructions
        "bundle": 'asm volatile("mov.u32 %0, 1; add.u32 %0, %0, 2;" : "=r"(x));',
        # spin loop with a label and a branch (the mbarrier.try_wait shape)
        "spin": 'asm volatile("{ LAB: mbarrier.try_wait.b64 p, [%0]; @!p bra LAB; }" :: "r"(a));',
        # a prologue that computes rather than converts: `shl` is arithmetic,
        # so it is a second instruction no matter that it feeds the first.
        # (Contrast the b8 staging below, which is a sanctioned conversion.)
        "prologue": 'asm volatile("{ .reg .b32 t; shl.b32 t, %1, 4; cvt.f32 %0, t; }" : "=r"(d));',
        # a bridge conversion on a register class that has no bridge: the
        # prefixes are a closed set, not "anything that looks like a cvt".
        "unsanctioned": 'asm volatile("{ cvt.u32.u16 %0, %1; st.b32 [%2], %0; }" : "=r"(d));',
    }
    for shape, src in forbidden.items():
        assert _sole_instruction(_ASM_RE.findall(src)[0]) is None, f"{shape} slipped through"

    # The sanctioned wrappers are NOT violations -- each carries exactly one
    # instruction, everything else being a `render.BRIDGE` boundary conversion.
    guarded = "{ .reg .pred p; setp.ne.b32 p, %2, 0; @p red.relaxed.gpu.global.add.u32 [%0], %1; }"
    assert _sole_instruction(guarded) == "red.relaxed.gpu.global.add.u32 [%0], %1;"
    staged = "{ .reg .b8 raw_a; cvt.u8.u16 raw_a, %1; cvt.rn.f16x2.e2m1x2 %0, raw_a; }"
    assert _sole_instruction(staged) == "cvt.rn.f16x2.e2m1x2 %0, raw_a;"


def test_ptx_all_variants_render_unique():
    """Every legal variant renders; helper names (incl. @p twins) are unique."""
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, renderings, variants

    names = set()
    total = 0
    for entry in TABLE.values():
        assert variants(entry), f"{entry.name}: check() filtered out every combination"
        for tokens, dtypes, predicated, imms in renderings(entry):
            opcode, helper, source = render_variant(entry, tokens, predicated, dtypes, imms)
            assert helper not in names, f"helper name collision: {helper}"
            names.add(helper)
            if predicated:  # framework-level @p twin, guarded inside the block
                assert f"@p {opcode} " in source or f"@p {opcode};" in source
            else:
                # The instruction may open the asm text or sit inside a
                # boundary-conversion block after a setp.
                assert (
                    f'"{opcode} ' in source
                    or f'"{opcode};"' in source
                    or f"; {opcode} " in source
                    or f"; {opcode};" in source
                )
            total += not predicated  # a @p twin is not a separate variant
    assert total == 110894  # update when the table grows


def test_ptx_no_instruction_registered_twice():
    """One PTX instruction, one entry.

    The table's law is the ISA: an entry models one syntax group, and no two
    entries may model the same one. Strip the helper name and the parameter
    names off a rendering and what is left is the instruction itself plus its
    operand constraints — if two entries ever produce the same one, the ISA
    line has been registered twice and calls to it are unresolvable.
    """
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, renderings

    owners = {}
    for entry in TABLE.values():
        for tokens, dtypes, predicated, imms in renderings(entry):
            _, _, source = render_variant(entry, tokens, predicated, dtypes, imms)
            instruction = re.sub(r"__[A-Za-z_][A-Za-z0-9_]*", "ARG", source)
            instruction = re.sub(r"tvm_builtin_\w+", "FN", instruction)
            first = owners.setdefault(instruction, entry.name)
            assert first == entry.name, f"{first} and {entry.name} both register\n{instruction}"


def test_ptx_dispatch_unambiguous():
    """No two entries may accept the same call.

    Models what the engine resolves by, and only that: the written tokens (slot
    names and order are invisible to `_fill`), the operand count, and each
    position's acceptance class. Declared spaces that `_coerce_address` treats
    alike collapse into one class, which is what makes this stricter than the
    rendering check above — two entries can emit different assembly and still
    leave a call with nothing to choose between them.
    """
    from tvm.backend.cuda.ptx.table import (
        TABLE,
        mods,
        operand_dtypes,
        operand_layout,
        operand_space,
        operand_type,
        variants,
    )

    def accepts(slot, mod_map, pred_is_distinct=True):
        if slot.kind == "addr":
            space = operand_space(slot, mod_map)
            if space == "tmem":
                return ("addr", "tmem")
            return ("addr", "shared*" if space.startswith("shared") else "generic")
        if slot.kind != "reg":
            return (slot.kind,)
        if pred_is_distinct and operand_type(slot, mod_map) == "pred":
            # A `.pred` operand shares its uint32 carrier with every integer
            # one, so it is only a class of its own because `T.ptx.pred(...)`
            # evidences it at the call. `pred_is_distinct=False` models the
            # engine as it would be WITHOUT that tag -- see the falsification
            # twin below, which is what proves this key can see the collision
            # it exists to prevent.
            return (slot.rw, "pred")
        return (slot.rw, tuple(sorted(operand_dtypes(slot, mod_map))))

    owners = {}
    for entry in TABLE.values():
        for tokens in variants(entry):
            mod_map = mods(entry, tokens)
            layout = operand_layout(entry, mod_map)
            shape = tuple(accepts(s, mod_map) for s, _, n in layout for _ in range(n))
            key = (entry.family, frozenset(t for t in tokens if t), shape)
            first = owners.setdefault(key, entry.name)
            assert first == entry.name, (
                f"{first} and {entry.name} both accept "
                f"T.ptx.{entry.family} with {sorted(key[1])} and {len(shape)} operand(s)"
            )


def test_ptx_dispatch_model_detects_a_collapsed_class():
    """Falsify the discriminator: erasing a class must make it go red.

    `.pred` is only an acceptance class of its own because `T.ptx.pred(...)`
    evidences it -- the carrier it rides is the same uint32 every integer
    operand uses. Collapse that distinction and the two `cp.async` lines whose
    only difference is `{, src-size}` vs `{, ignore-src}` become
    indistinguishable, which is the defect this guard exists to catch. A guard
    that has only ever been seen to pass proves nothing.
    """
    from tvm.backend.cuda.ptx.table import (
        TABLE,
        mods,
        operand_dtypes,
        operand_layout,
        operand_space,
        variants,
    )

    def accepts(slot, mod_map):
        # The same key as test_ptx_dispatch_unambiguous, minus the pred class.
        if slot.kind == "addr":
            space = operand_space(slot, mod_map)
            if space == "tmem":
                return ("addr", "tmem")
            return ("addr", "shared*" if space.startswith("shared") else "generic")
        if slot.kind != "reg":
            return (slot.kind,)
        return (slot.rw, tuple(sorted(operand_dtypes(slot, mod_map))))

    owners, collisions = {}, set()
    for entry in TABLE.values():
        for tokens in variants(entry):
            mod_map = mods(entry, tokens)
            layout = operand_layout(entry, mod_map)
            shape = tuple(accepts(s, mod_map) for s, _, n in layout for _ in range(n))
            key = (entry.family, frozenset(t for t in tokens if t), shape)
            first = owners.setdefault(key, entry.name)
            if first != entry.name:
                collisions.add(tuple(sorted((first, entry.name))))

    assert collisions == {
        ("cp_async_ca_ignore_src", "cp_async_ca_src_size"),
        ("cp_async_cg_ignore_src", "cp_async_cg_src_size"),
    }, f"expected exactly the src-size/ignore-src pairs to collapse, got {sorted(collisions)}"


def test_ptx_stub_up_to_date():
    """The checked-in tvm.script.tirx stub must match the generator."""
    from tvm.backend.cuda.ptx import gen_stubs

    stub = gen_stubs.STUB_PATH
    assert stub.read_text(encoding="utf-8") == gen_stubs.generate(), (
        "python/tvm/script/tirx.pyi is stale; regenerate with "
        "`python -m tvm.backend.cuda.ptx.gen_stubs -o python/tvm/script/tirx.pyi`"
    )


@requires_nvcc
def test_ptxas_gate_rejects_invalid():
    """Honesty check: the gate path must actually reject bad instructions."""
    bogus = '__device__ void f(unsigned x) { asm volatile("totally.bogus.instr %0;" : : "r"(x)); }'
    with pytest.raises(Exception, match="bogus|error"):
        _assert_ptxas_ok(bogus, rdc=True)


@requires_nvcc
def test_ptx_sampled_helpers_assemble():
    """Fast tier: a seeded sample of every family's variants assembles."""
    import random

    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, renderings

    rng = random.Random(20260802)
    by_arch = {}
    for entry in TABLE.values():
        arch = entry.cert_arch or PTX_ARCH
        rendered = list(renderings(entry))
        for i in rng.sample(range(len(rendered)), min(48, len(rendered))):
            _, _, source = render_variant(entry, *_as_render_args(rendered[i]))
            by_arch.setdefault(arch, []).append(source.replace("__forceinline__ ", ""))
    for arch, sources in by_arch.items():
        _assert_ptxas_ok("\n".join([_CERT_PRELUDE, *sources]), rdc=True, arch=arch)


_CERT_SHARDS = 32


@pytest.mark.skipif(
    not os.environ.get("PTX_CERT"),
    reason="full-table ptxas certification; run with PTX_CERT=1 after changing the table",
)
@requires_nvcc
@pytest.mark.parametrize("shard", range(_CERT_SHARDS))
def test_ptx_all_helpers_certify(shard):
    """Certification tier: EVERY legal variant assembles under ptxas.

    One scoped exception: an OPEN immediate operand (role="imm" with neither
    literal nor choices) has no domain to enumerate, so its entries are
    certified at the enumeration's sample values (imm_combos' open_samples,
    default "0"). For those entries this proves the instruction SHAPE
    assembles, not the caller's particular constant -- a facility limit of
    sampling an open domain, not a property of the table.

    Sharded so pytest-xdist can spread the nvcc work::

        PTX_CERT=1 pytest -n 16 -k certify tests/python/tirx/codegen/test_ptx_dialect.py

    Stride slicing keeps shards balanced (ld dominates the variant count).
    Variants are grouped by their family's arch floor and each group is
    assembled at that arch: below an instruction's floor ptxas rejects legal
    variants, and believing it would delete real coverage.
    __forceinline__ must be stripped and -rdc used: unreferenced inline
    device functions are silently dropped before ptxas ever sees them.
    """
    from tvm.backend.cuda.ptx.render import render_variant
    from tvm.backend.cuda.ptx.table import TABLE, renderings

    by_arch = {}
    covered = 0
    for index, (entry, rendering) in enumerate(
        (TABLE[name], r) for name in sorted(TABLE) for r in renderings(TABLE[name])
    ):
        if index % _CERT_SHARDS == shard:
            covered += 1
            arch = entry.cert_arch or PTX_ARCH
            _, _, src = render_variant(entry, *_as_render_args(rendering))
            by_arch.setdefault(arch, []).append(src.replace("__forceinline__ ", ""))
    assert covered, "empty shard: lower _CERT_SHARDS"
    for arch, sources in by_arch.items():
        _assert_ptxas_ok("\n".join([_CERT_PRELUDE, *sources]), rdc=True, arch=arch)


@requires_nvcc
def test_ptx_nvcc_smoke():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        if tx == 0:
            val = T.local_scalar("uint32")
            T.ptx.ld.global_.acquire.gpu.b32(val, A.ptr_to([0]))
            T.ptx.st.shared__cta.b32(smem.ptr_to([0]), val)
            T.ptx.red.relaxed.gpu.global_.add.u32(B.ptr_to([0]), T.uint32(1))
            T.ptx.prefetch.global_.L2(A.ptr_to([16]))
        T.cuda.cta_sync()
        B[tx] = smem[tx % 4]

    _assert_ptxas_ok(_cuda_source(kernel))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="CUDA GPU not available")
def test_ptx_ld_st_gpu_roundtrip():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        val = T.local_scalar("uint32")
        T.ptx.ld.global_.acquire.gpu.b32(val, A.ptr_to([tx]))
        T.ptx.st.release.gpu.global_.b32(B.ptr_to([tx]), val)

    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=TARGET, tir_pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        a_np = np.arange(32, dtype=np.uint32) + 100
        b_np = np.zeros(32, dtype=np.uint32)
        a = tvm.runtime.tensor(a_np, device=dev)
        b = tvm.runtime.tensor(b_np, device=dev)
        mod(a, b)
        np.testing.assert_array_equal(b.numpy(), a_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
