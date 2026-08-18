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

import itertools
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
    from tvm.backend.cuda.codegen.registry import CODEGEN_REGISTRY
    from tvm.backend.cuda.ptx.table import TABLE, escape_token

    assert hasattr(T, "ptx")
    for entry in TABLE.values():
        op = Op.get(entry.op_name)  # raises if unregistered
        assert op.get_attr("TCallEffectKind") is not None, entry.name
        # The printer name is the attribute path a program types, so it is the
        # *escaped* family: `and`/`or`/`not` (ISA 9.7.8) are Python keywords and
        # print as `T.ptx.and_`. Identity for every other family.
        family = escape_token(entry.family)  # several entries may share a mnemonic
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


@requires_nvcc
def test_ptx_ld_s32_wide_destination_codegen():
    """A signed scalar load may sign-extend into a wider destination register."""

    @T.prim_func
    def kernel(a_ptr: T.handle, out_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "int32")
        Out = T.match_buffer(out_ptr, (32,), "int64")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        wide = T.local_scalar("int64")
        T.ptx.ld.global_.s32(wide, A.ptr_to([tx]))
        Out[tx] = wide

    src = _cuda_source(kernel)
    assert "tvm_builtin_ptx_ld_global_s32_s64(int64_t& __d" in src
    assert 'asm volatile("ld.global.s32 %0, [%1];" : "=l"(__d)' in src
    assert "cvt." not in src
    _assert_ptxas_ok(src)


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


@requires_nvcc
def test_ptx_predicated_destination_preserves_old_value():
    @T.prim_func
    def kernel(a_ptr: T.handle, out_ptr: T.handle):
        A = T.match_buffer(a_ptr, (1,), "float32")
        Out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        value: T.float32 = T.float32(0)
        pred: T.uint32 = T.cast(tx == 0, "uint32")
        T.ptx.ld.global_.f32(value, A.ptr_to([0]), pred=pred, preserve_dst=True)
        T.ptx.ex2.approx.ftz.f32(value, value, pred=pred, preserve_dst=True)
        Out[tx] = value

    src = _cuda_source(kernel)
    assert "tvm_builtin_ptx_ld_global_f32_pred_keep" in src
    assert "tvm_builtin_ptx_ex2_approx_ftz_f32_pred_keep" in src
    assert '"+f"(__d)' in src
    _assert_ptxas_ok(src)


@requires_nvcc
def test_ptx_predicated_destination_is_undefined_by_default():
    @T.prim_func
    def kernel(a_ptr: T.handle, out_ptr: T.handle):
        A = T.match_buffer(a_ptr, (1,), "float32")
        Out = T.match_buffer(out_ptr, (32,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        value = T.local_scalar("float32")
        pred: T.uint32 = T.cast(tx == 0, "uint32")
        T.ptx.ld.global_.f32(value, A.ptr_to([0]), pred=pred)
        Out[tx] = T.if_then_else(pred != 0, value, T.float32(0))

    src = _cuda_source(kernel)
    assert "tvm_builtin_ptx_ld_global_f32_pred_undef" in src
    assert '"=f"(__d)' in src
    assert '"+f"(__d)' not in src
    _assert_ptxas_ok(src)


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


def test_ptx_integer_arithmetic_dispatch():
    """The 9.7.1 integer lines share five mnemonics with the floating-point ones.

    Nothing in the call names the entry: `T.ptx.add` reaches three candidates
    (integer, single/double, half) and the written type token is what resolves
    them, exactly as ptxas resolves them. `.wide` goes further -- it is a
    separate entry per mnemonic because its result is "twice as wide as a and
    b", a dtype no token in the instruction names.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "int32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        si = T.local_scalar("int32")
        sw = T.local_scalar("int64")
        fv = T.local_scalar("float32")
        cnt = T.local_scalar("uint32")
        T.ptx.add.s32(si, A[0], A[1])  # integer line
        T.ptx.add.rn.f32(fv, fv, fv)  # floating-point line, same mnemonic
        T.ptx.mul.lo.s32(si, si, A[2])  # .lo keeps the operand width
        T.ptx.mul.wide.s32(sw, si, A[3])  # .wide doubles the destination
        T.ptx.mad.wide.s32(sw, si, A[0], sw)  # ... and mad's accumulator too
        T.ptx.popc.b32(cnt, si)  # .b32 source, .u32 destination
        T.ptx.dp4a.s32.s32(si, si, si, si)
        A[tx % 4] = si + T.int32(sw) + T.int32(cnt)

    src = _cuda_source(kernel)
    # Same mnemonic, different entries, each emitting its own ISA line.
    assert "add.s32 %0, %1, %2;" in src
    assert "add.rn.f32 %0, %1, %2;" in src
    assert "mul.lo.s32 %0, %1, %2;" in src
    # The derived destination is a 64-bit "l" register while the sources stay
    # 32-bit "r" -- the whole reason `.wide` is its own entry.
    assert '"=l"(__d) : "r"(__a), "r"(__b)' in src
    assert "mul.wide.s32 %0, %1, %2;" in src
    assert "mad.wide.s32 %0, %1, %2, %3;" in src
    assert "popc.b32 %0, %1;" in src
    assert "dp4a.s32.s32 %0, %1, %2, %3;" in src

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # A derived dtype is enforced at trace time like any other: `.wide.s32`
    # writes 64 bits, so a 32-bit destination is rejected before codegen.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must have dtype int64"):

        @T.prim_func
        def narrow_wide_dst(out: T.Buffer((1,), "int32")):
            T.device_entry()
            T.ptx.mul.wide.s32(out[0], T.int32(2), T.int32(3))

    # `.sat` is a syntax line of its own, not a free qualifier: the check
    # rejects it where the ISA does not spell it.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="hi.s32"):

        @T.prim_func
        def sat_on_lo(out: T.Buffer((1,), "int32")):
            T.device_entry()
            T.ptx.mad.lo.sat.s32(out[0], T.int32(2), T.int32(3), T.int32(4))


def test_ptx_floating_point_dispatch():
    """ISA 9.7.3's lines, including three mnemonics shared with the integer group.

    `mad`, `div` and `abs` each name an integer entry and a floating-point one;
    as with `add` above, only the written tokens choose between them. The rest
    of the section is here for its shapes: a `.pred` destination (testp), the
    f64 approximations that exist only with a mandatory `.ftz` (9.7.3.14 and
    9.7.3.17), and the lone approximation with no `.ftz` at all (tanh).
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        f = T.local_scalar("float32")
        d = T.local_scalar("float64")
        si = T.local_scalar("int32")
        isnan = T.local_scalar("uint32")
        T.ptx.mad.rn.f32(f, A[0], A[1], A[2])  # fp line
        T.ptx.mad.lo.s32(si, si, si, si)  # integer line, same mnemonic
        T.ptx.div.approx.ftz.f32(f, f, A[3])  # fp divide
        T.ptx.div.s32(si, si, si)  # integer divide, same mnemonic
        T.ptx.abs.ftz.f32(f, f)  # fp abs
        T.ptx.abs.s32(si, si)  # integer abs, same mnemonic
        T.ptx.copysign.f32(f, A[0], f)
        T.ptx.sqrt.rn.f64(d, d)
        T.ptx.rsqrt.approx.ftz.f64(d, d)  # ISA 9.7.3.17
        T.ptx.rcp.approx.ftz.f64(d, d)  # ISA 9.7.3.14
        T.ptx.sin.approx.f32(f, f)
        T.ptx.cos.approx.ftz.f32(f, f)
        T.ptx.lg2.approx.f32(f, f)
        T.ptx.tanh.approx.f32(f, f)
        T.ptx.testp.notanumber.f32(isnan, f)
        A[tx % 4] = f + T.float32(d) + T.float32(si) + T.float32(isnan)

    src = _cuda_source(kernel)
    for text in (
        "mad.rn.f32 %0, %1, %2, %3;",
        "mad.lo.s32 %0, %1, %2, %3;",
        "div.approx.ftz.f32 %0, %1, %2;",
        "div.s32 %0, %1, %2;",
        "abs.ftz.f32 %0, %1;",
        "abs.s32 %0, %1;",
        "copysign.f32 %0, %1, %2;",
        "sqrt.rn.f64 %0, %1;",
        "rsqrt.approx.ftz.f64 %0, %1;",
        "rcp.approx.ftz.f64 %0, %1;",
        "sin.approx.f32 %0, %1;",
        "cos.approx.ftz.f32 %0, %1;",
        "lg2.approx.f32 %0, %1;",
        "tanh.approx.f32 %0, %1;",
    ):
        assert text in src, text
    # The predicate result crosses the C boundary as a uint32 materialized by
    # selp, with the real .pred register living inside the asm block.
    assert ".reg .pred pd0; testp.notanumber.f32 pd0, %1; selp.b32 %0, 1, 0, pd0;" in src

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # sqrt has no f64 approximation at any spelling, unlike rcp (9.7.3.14).
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="only on the .f32 line"):

        @T.prim_func
        def sqrt_approx_f64(out: T.Buffer((1,), "float64")):
            T.device_entry()
            T.ptx.sqrt.approx.f64(out[0], T.float64(2.0))

    # ... and rcp's f64 approximation is unreachable without its mandatory .ftz.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="rcp.approx.ftz.f64"):

        @T.prim_func
        def rcp_approx_f64(out: T.Buffer((1,), "float64")):
            T.device_entry()
            T.ptx.rcp.approx.f64(out[0], T.float64(2.0))

    # One of .approx/.full/.rnd is required: the bare mnemonic names no line.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="missing required modifier"):

        @T.prim_func
        def div_without_mode(out: T.Buffer((1,), "float32")):
            T.device_entry()
            T.ptx.div.f32(out[0], T.float32(1.0), T.float32(2.0))


def test_ptx_half_precision_dispatch():
    """ISA 9.7.4, whose every mnemonic is shared with a wider-precision entry.

    `abs` is the extreme case: three entries answer to it (integer, f32/f64,
    half), and only the type token says which. The section also has the two
    qualifiers no same-precision line carries -- fma's `.relu` and `.oob` --
    and ex2's split over whether `.ftz` is mandatory or unspellable.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        h = T.local_scalar("uint16")  # .f16 / .bf16 carrier
        p = T.local_scalar("uint32")  # .f16x2 / .bf16x2 carrier
        f = T.local_scalar("float32")
        si = T.local_scalar("int32")
        T.ptx.add.rn.sat.f16(h, h, h)  # existing half entry, beside the new ones
        T.ptx.fma.rn.relu.f16x2(p, p, p, p)  # .relu: no same-precision line has it
        T.ptx.fma.rn.oob.bf16x2(p, p, p, p)  # .oob likewise
        T.ptx.fma.rn.f32(f, f, f, f)  # same mnemonic, single precision
        T.ptx.abs.ftz.f16(h, h)  # three-way family: half ...
        T.ptx.abs.ftz.f32(f, f)  # ... single ...
        T.ptx.abs.s32(si, si)  # ... and integer
        T.ptx.neg.bf16x2(p, p)
        T.ptx.tanh.approx.f16x2(p, p)
        T.ptx.ex2.approx.f16(h, h)
        T.ptx.ex2.approx.ftz.bf16(h, h)  # .ftz mandatory on this line
        A[tx % 4] = T.uint32(p) + T.uint32(h) + T.uint32(si) + T.uint32(f)

    src = _cuda_source(kernel)
    for text in (
        "add.rn.sat.f16 %0, %1, %2;",
        "fma.rn.relu.f16x2 %0, %1, %2, %3;",
        "fma.rn.oob.bf16x2 %0, %1, %2, %3;",
        "fma.rn.f32 %0, %1, %2, %3;",
        "abs.ftz.f16 %0, %1;",
        "abs.ftz.f32 %0, %1;",
        "abs.s32 %0, %1;",
        "neg.bf16x2 %0, %1;",
        "tanh.approx.f16x2 %0, %1;",
        "ex2.approx.f16 %0, %1;",
        "ex2.approx.ftz.bf16 %0, %1;",
    ):
        assert text in src, text

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # The `.oob` line spells neither .ftz nor .sat beside it.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="oob line spells no"):

        @T.prim_func
        def oob_with_ftz(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.fma.rn.oob.ftz.f16(out[0], T.uint16(0), T.uint16(0), T.uint16(0))

    # .sat and .relu are two clampings on two different syntax lines.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="separate syntax lines"):

        @T.prim_func
        def sat_and_relu(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.fma.rn.sat.relu.f16(out[0], T.uint16(0), T.uint16(0), T.uint16(0))

    # ex2's bf16 line without its mandatory .ftz names no syntax line.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="mandatorily"):

        @T.prim_func
        def ex2_bf16_no_ftz(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.ex2.approx.bf16(out[0], T.uint16(0))

    # ... and the bf16 abs/neg lines spell no .ftz at all.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="takes no .ftz"):

        @T.prim_func
        def abs_bf16_ftz(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.abs.ftz.bf16(out[0], T.uint16(0))


def test_ptx_mixed_precision_dispatch():
    """ISA 9.7.5, which adds no instruction of its own.

    Its three subsections are a fourth syntax line of add/sub/fma:
    `op{.rnd}{.sat}.f32.atype`, where a second type token names a 16-bit source
    converted to .f32 before the operation. That is the `srctype` slot, so
    these forms are reached through the same entries as the same-precision
    ones, and what selects a line is only whether that second token is written.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        f = T.local_scalar("float32")
        h = T.local_scalar("uint16")  # the .f16/.bf16 source carrier
        T.ptx.fma.rn.f32.f16(f, h, h, A[0])  # both sources converted
        T.ptx.add.f32.bf16(f, h, f)  # `{.rnd}` omitted: the bare mixed line
        T.ptx.sub.rn.sat.f32.f16(f, h, f)
        T.ptx.add.rn.f32(f, f, f)  # same entry, same-precision line
        A[tx % 4] = f

    src = _cuda_source(kernel)
    for text in (
        "fma.rn.f32.f16 %0, %1, %2, %3;",
        "add.f32.bf16 %0, %1, %2;",
        "sub.rn.sat.f32.f16 %0, %1, %2;",
        "add.rn.f32 %0, %1, %2;",
    ):
        assert text in src, text
    # Two carriers in one helper: the converted source is 16-bit, the rest f32.
    assert '"=f"(__d) : "h"(__a), "f"(__b)' in src

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # The mixed line spells no .ftz, unlike the .f32 line it shares an entry with.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="takes no .ftz"):

        @T.prim_func
        def mixed_with_ftz(out: T.Buffer((1,), "float32")):
            T.device_entry()
            T.ptx.add.rn.ftz.f32.f16(out[0], T.uint16(0), T.float32(0))

    # A converted source exists only on the .f32 line -- there is no f64 form.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="only exists on the .f32"):

        @T.prim_func
        def mixed_f64(out: T.Buffer((1,), "float64")):
            T.device_entry()
            T.ptx.add.rn.f64.f16(out[0], T.uint16(0), T.float64(0))

    # And `mul` is the one mnemonic of the four with no mixed line at all, so
    # its entry declares no srctype slot and the token does not resolve.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def mul_mixed(out: T.Buffer((1,), "float32")):
            T.device_entry()
            T.ptx.mul.rn.f32.f16(out[0], T.uint16(0), T.float32(0))


def test_ptx_comparison_selection_dispatch():
    """ISA 9.7.6, the section whose results and selectors are predicates.

    Four mnemonics, eight entries: `setp` alone is four, because `{.BoolOp}`
    adds an operand and `[|q]` adds a destination, and those are two
    independent shape choices rather than optional tokens. Nothing in a call
    names the entry -- the arity and the operand classes pick it, exactly as
    ptxas picks the syntax line.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "int32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        p = T.local_scalar("uint32")
        q = T.local_scalar("uint32")
        d = T.local_scalar("uint32")
        f = T.local_scalar("float32")
        T.ptx.setp.lt.s32(p, A[0], A[1])  # one destination
        T.ptx.setp.lt.s32(p, q, A[0], A[1])  # ... two: `p|q`, chosen by arity
        T.ptx.setp.lt.and_.s32(p, A[0], A[1], T.ptx.pred(q))  # ... plus a BoolOp
        T.ptx.setp.gt.or_.s32(p, q, A[2], A[3], T.ptx.pred(d))  # ... and both
        T.ptx.set.lt.u32.f32(d, f, f)  # writes a value, not a predicate
        T.ptx.selp.b32(d, d, p, T.ptx.pred(q))  # predicate selects
        T.ptx.slct.ftz.b32.f32(d, d, p, f)  # a sign selects
        A[tx % 4] = T.int32(p + q + d)

    src = _cuda_source(kernel)
    for text in (
        "setp.lt.s32 pd0, %1, %2;",
        "setp.lt.s32 pd0|pd1, %2, %3;",
        "setp.lt.and.s32 pd0, %1, %2, ps0;",
        "setp.gt.or.s32 pd0|pd1, %2, %3, ps0;",
        "set.lt.u32.f32 %0, %1, %2;",
        "selp.b32 %0, %1, %2, ps0;",
        "slct.ftz.b32.f32 %0, %1, %2, %3;",
    ):
        assert text in src, text

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # Each source type takes its own operator set, and ptxas agrees with every
    # rejection below (probed at sm_90 before they were written into the check).
    # lo/ls/hi/hs are unsigned-only alternates for lt/le/gt/ge.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="signed"):

        @T.prim_func
        def unsigned_op_on_signed(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.setp.lo.s32(out[0], T.int32(1), T.int32(2))

    # The unordered comparisons and the NaN predicates are floating point only.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="floating-point comparison"):

        @T.prim_func
        def float_op_on_integer(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.setp.nan.u32(out[0], T.uint32(1), T.uint32(2))

    # A bit-size type has no ordering, only equality.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="only with eq/ne"):

        @T.prim_func
        def ordered_on_bitsize(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.setp.lt.b32(out[0], T.uint32(1), T.uint32(2))

    # .ftz applies only to .f32 comparisons ...
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="only to .f32 comparisons"):

        @T.prim_func
        def ftz_off_f32(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.setp.eq.ftz.s32(out[0], T.int32(1), T.int32(2))

    # ... and on slct, only to the .f32 selector line.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="f32 selector line"):

        @T.prim_func
        def ftz_on_s32_selector(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.slct.ftz.b32.s32(out[0], T.uint32(1), T.uint32(2), T.int32(3))


def test_ptx_half_comparison_dispatch():
    """ISA 9.7.7, the half twin of 9.7.6 -- same two mnemonics, different grids.

    Two things separate it from the section it shares names with. `setp`'s
    destination shape is decided by the type rather than chosen: the scalar
    lines spell one predicate, the packed lines only `p|q`, and there the pair
    is the two lanes rather than a result and its complement. And `set` pairs
    its two type tokens on a grid, so which (dtype, stype) combinations exist
    is a fact of the section, not a product of two slots.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        h = T.local_scalar("uint16")  # f16 / bf16 carrier
        x2 = T.local_scalar("uint32")  # f16x2 / bf16x2 carrier
        p = T.local_scalar("uint32")
        q = T.local_scalar("uint32")
        d32 = T.local_scalar("uint32")
        T.ptx.setp.eq.f16(p, h, h)  # scalar type: one destination
        T.ptx.setp.lt.ftz.f16x2(p, q, x2, x2)  # packed type: the pair, per lane
        T.ptx.setp.eq.and_.bf16(p, h, h, T.ptx.pred(q))
        T.ptx.setp.gt.or_.bf16x2(p, q, x2, x2, T.ptx.pred(d32))
        T.ptx.set.lt.u32.f16(d32, h, h)  # integer answer to a half compare
        T.ptx.set.gt.f16.f32(h, T.float32(1.0), T.float32(2.0))  # ... and back
        T.ptx.set.eq.f16x2.f16x2(x2, x2, x2)  # packed both sides
        A[tx % 4] = p + q + d32 + T.uint32(h) + x2

    src = _cuda_source(kernel)
    for text in (
        "setp.eq.f16 pd0, %1, %2;",
        "setp.lt.ftz.f16x2 pd0|pd1, %2, %3;",
        "setp.eq.and.bf16 pd0, %1, %2, ps0;",
        "setp.gt.or.bf16x2 pd0|pd1, %2, %3, ps0;",
        "set.lt.u32.f16 %0, %1, %2;",
        "set.gt.f16.f32 %0, %1, %2;",
        "set.eq.f16x2.f16x2 %0, %1, %2;",
    ):
        assert text in src, text

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # Every rejection below was probed against ptxas before being written into
    # the check. The (dtype, stype) grid: a packed source needs a 32-bit answer.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="no half-precision syntax"):

        @T.prim_func
        def bad_pair(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.set.eq.u16.f16x2(out[0], T.uint32(0), T.uint32(0))

    # `.ftz` follows the source precision: a bf16 source has none to flush.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="does not take it"):

        @T.prim_func
        def ftz_bf16_source(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.set.eq.ftz.u32.bf16(out[0], T.uint16(0), T.uint16(0))

    # ... and no bf16 destination takes it either.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="destination takes no"):

        @T.prim_func
        def ftz_bf16_dest(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.set.eq.ftz.bf16.f32(out[0], T.float32(0), T.float32(0))

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="spells no .ftz"):

        @T.prim_func
        def setp_ftz_bf16(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.setp.eq.ftz.bf16(out[0], T.uint16(0), T.uint16(0))

    # The unordered comparisons need a floating-point source.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="integer source"):

        @T.prim_func
        def unordered_on_integer(out: T.Buffer((1,), "uint16")):
            T.device_entry()
            T.ptx.set.equ.f16.s32(out[0], T.int32(0), T.int32(0))

    # 9.7.7 spells no unsigned alternates at all, so `lo` never resolves here.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def unsigned_alternate(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.setp.lo.f16(out[0], T.uint16(0), T.uint16(0))


def test_ptx_logic_shift_dispatch():
    """ISA 9.7.8, including the three mnemonics that are Python keywords.

    `and`, `or` and `not` cannot be attributes as they stand, so the surface
    spells them `and_`/`or_`/`not_` and the escape is carried through both
    directions -- typing, and the name the printer emits. The round-trip
    assertion below is what proves the second half: a script that printed
    `T.ptx.and(...)` would not re-parse at all.

    The section also puts `.pred` in the *type* slot rather than on a single
    operand, so a helper can hold three bridges around one instruction.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (4,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        d = T.local_scalar("uint32")
        p = T.local_scalar("uint32")
        q = T.local_scalar("uint32")
        s = T.local_scalar("int32")
        T.ptx.and_.b32(d, A[0], A[1])
        T.ptx.or_.b32(d, d, A[2])
        T.ptx.xor.b32(d, d, A[3])
        T.ptx.and_.pred(p, T.ptx.pred(q), T.ptx.pred(d))  # .pred as the type
        T.ptx.not_.pred(p, T.ptx.pred(p))
        T.ptx.not_.b32(d, d)
        T.ptx.cnot.b32(d, d)
        T.ptx.lop3.b32(d, d, d, d, 0x80)  # a & b & c, by look-up table
        T.ptx.lop3.or_.b32(d, p, d, d, d, 0xFE, T.ptx.pred(q))  # d|p pair
        T.ptx.shf.l.clamp.b32(d, d, d, T.uint32(4))  # funnel shift
        T.ptx.shl.b32(d, d, T.uint32(2))
        T.ptx.shr.s32(s, s, T.uint32(1))  # signed: fills with the sign bit
        A[tx % 4] = d + p + q + T.uint32(s)

    src = _cuda_source(kernel)
    for text in (
        "and.b32 %0, %1, %2;",
        "or.b32 %0, %1, %2;",
        "xor.b32 %0, %1, %2;",
        "and.pred pd0, ps0, ps1;",
        "not.pred pd0, ps0;",
        "not.b32 %0, %1;",
        "cnot.b32 %0, %1;",
        "lop3.b32 %0, %1, %2, %3, 128;",
        "lop3.or.b32 %0|pd0, %2, %3, %4, 254, ps0;",
        "shf.l.clamp.b32 %0, %1, %2, %3;",
        "shl.b32 %0, %1, %2;",
        "shr.s32 %0, %1, %2;",
    ):
        assert text in src, text

    # The load-bearing one: `and`/`or`/`not` have to survive being printed.
    script = kernel.script()
    assert "T.ptx.and_(" in script and "T.ptx.not_(" in script
    reparsed = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # cnot has no predicate line -- ptxas: "Unexpected instruction types
    # specified for 'cnot'" -- so the token is not in its slot at all.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def cnot_pred(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.cnot.pred(out[0], T.uint32(0))

    # lop3's BoolOp line stops at .or/.and; ptxas rejects .xor outright.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def lop3_xor(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.lop3.xor.b32(out[0], T.uint32(0), T.uint32(0), T.uint32(0), 0x80)

    # shl is untyped -- a left shift zero-fills whatever the bits mean, so
    # there is no signed line (unlike shr, which needs to know how to fill).
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def shl_signed(out: T.Buffer((1,), "int32")):
            T.device_entry()
            T.ptx.shl.s32(out[0], T.int32(1), T.uint32(2))

    # Open immediates may survive tracing so explicitly-unrolled expressions
    # can specialize, but a runtime LUT byte still has no register form and is
    # rejected at CUDA codegen.
    @T.prim_func
    def lut_runtime(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (1,), "uint32")
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            d = T.local_scalar("uint32")
            T.ptx.lop3.b32(d, A[0], A[0], A[0], A[0])

    with pytest.raises((ValueError, tvm.error.InternalError), match="compile-time constants"):
        _cuda_source(lut_runtime)

    @T.prim_func
    def lut_unrolled(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (1,), "uint32")
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            d = T.local_scalar("uint32")
            for i in T.unroll(2):
                T.ptx.lop3.b32(d, A[0], A[0], A[0], i * 128)

    unrolled_src = _cuda_source(lut_unrolled)
    assert "lop3.b32 %0, %1, %2, %3, 0;" in unrolled_src
    assert "lop3.b32 %0, %1, %2, %3, 128;" in unrolled_src


def test_ptx_data_movement_dispatch():
    """ISA 9.7.9's newly registered instructions, end to end.

    The section's own difficulty is that its shapes vary more than its
    qualifiers: `mov` shares a mnemonic with ten vector pack/unpack entries and
    is told apart by arity alone, `shfl.sync` and `multimem` each split on
    whether a bracketed part of the syntax line is written, and several
    operands are immediates that live in the instruction text.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (8,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        d = T.local_scalar("uint32")
        p = T.local_scalar("uint32")
        v = T.local_scalar("uint32")
        pol = T.local_scalar("uint64")
        gen = T.local_scalar("uint64")
        T.ptx.mov.b32(d, A[0])  # scalar mov: two operands
        T.ptx.mov.pred(p, T.ptx.pred(d))  # ... and its predicate line
        T.ptx.prmt.b32(d, d, d, A[1])  # generic byte permute
        T.ptx.prmt.b32.f4e(d, d, d, A[2])  # ... and a mode-driven one
        T.ptx.ldu.global_.b32(v, A.ptr_to([3]))
        T.ptx.shfl_sync.idx.b32(d, d, T.uint32(0), T.uint32(31), T.uint32(0xFFFFFFFF))
        T.ptx.shfl_sync.up.b32(d, p, d, T.uint32(1), T.uint32(0), T.uint32(0xFFFFFFFF))
        T.ptx.isspacep.global_(p, A.ptr_to([4]))
        T.ptx.cvta.to.global_.u64(gen, A.ptr_to([5]))
        T.ptx.cvt_pack.sat.u16.s32(d, T.int32(1), T.int32(2))
        T.ptx.cvt_pack.sat.u8.s32.b32(d, T.int32(1), T.int32(2), d)
        T.ptx.createpolicy.fractional.L2__evict_last.b64(pol)
        T.ptx.createpolicy.cvt.L2.b64(pol, pol)
        T.ptx.applypriority.global_.L2__evict_normal(A.ptr_to([6]))
        T.ptx.discard.global_.L2(A.ptr_to([7]))
        T.ptx.prefetchu.L1(A.ptr_to([0]))
        T.ptx.multimem_ld_reduce.add.u32(v, A.ptr_to([0]))
        T.ptx.multimem_red.relaxed.gpu.add.u32(A.ptr_to([0]), v)
        smem[tx % 4] = d + p + v
        A[tx % 8] = smem[tx % 4] + T.uint32(gen) + T.uint32(pol)

    src = _cuda_source(kernel)
    for text in (
        "mov.b32 %0, %1;",
        "mov.pred pd0, ps0;",
        "prmt.b32 %0, %1, %2, %3;",
        "prmt.b32.f4e %0, %1, %2, %3;",
        "ldu.global.b32 %0, [%1];",
        "shfl.sync.idx.b32 %0, %1, %2, %3, %4;",
        "shfl.sync.up.b32 %0|pd0, %2, %3, %4, %5;",
        "isspacep.global pd0, %1;",
        "cvta.to.global.u64 %0, %1;",
        "cvt.pack.sat.u16.s32 %0, %1, %2;",
        "cvt.pack.sat.u8.s32.b32 %0, %1, %2, %3;",
        "createpolicy.fractional.L2::evict_last.b64 %0;",
        "createpolicy.cvt.L2.b64 %0, %1;",
        "applypriority.global.L2::evict_normal [%0], 128;",
        "discard.global.L2 [%0], 128;",
        "prefetchu.L1 [%0];",
        "multimem.ld_reduce.add.u32 %0, [%1];",
        "multimem.red.relaxed.gpu.add.u32 [%0], %1;",
    ):
        assert text in src, text

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # Every rejection below was probed against ptxas before it was written into
    # a check. multimem pairs an ordering semantic with a scope ...
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="go together"):

        @T.prim_func
        def sem_without_scope(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            v = T.local_scalar("uint32")
            T.ptx.multimem_ld_reduce.relaxed.add.u32(v, A.ptr_to([0]))

    # ... and `.weak` is the line that has neither.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="takes no scope"):

        @T.prim_func
        def weak_with_scope(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            v = T.local_scalar("uint32")
            T.ptx.multimem_ld_reduce.weak.gpu.add.u32(v, A.ptr_to([0]))

    # The op x type table: `.add` is the row that takes .s32 but not .s64.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match=r"\.add takes"):

        @T.prim_func
        def add_s64(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "int64")
            T.device_entry()
            v = T.local_scalar("int64")
            T.ptx.multimem_ld_reduce.add.s64(v, A.ptr_to([0]))

    # The scalar float line has no lone half: a width has to reach 32 bits.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="the scalar line takes"):

        @T.prim_func
        def scalar_f16(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint16")
            T.device_entry()
            v = T.local_scalar("uint16")
            T.ptx.multimem_ld_reduce.add.f16(v, A.ptr_to([0]))

    # `.acc::f32` raises an accumulation, so it needs something to accumulate.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="applies to .add"):

        @T.prim_func
        def acc_on_min(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            v = T.local_scalar("uint32")
            T.ptx.multimem_ld_reduce.min.acc__f32.f16x2(v, A.ptr_to([0]))

    # st.async's mmio line is system-scoped only.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="requires .sys"):

        @T.prim_func
        def mmio_gpu(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptx.st_async.mmio.release.gpu.global_.u32(A.ptr_to([0]), T.uint32(1))


def test_ptx_parallel_sync_dispatch():
    """ISA 9.7.14's warp-level primitives and the atom shapes beside `.op`.

    The section is where predicates are most load-bearing: they are sources
    (bar.red's `c`), destinations (vote, elect), and both at once. It is also
    where one mnemonic carries the most shapes -- `atom` now has the `.op`
    line, `.cas`, `.exch`, the half-precision adds and two vector lines, all
    resolved by tokens and arity alone.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (8,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        d = T.local_scalar("uint32")
        p = T.local_scalar("uint32")
        m = T.local_scalar("uint32")
        h = T.local_scalar("uint16")
        FULL = T.uint32(0xFFFFFFFF)
        T.ptx.vote_sync.all.pred(p, T.ptx.pred(d), FULL)
        T.ptx.vote_sync.ballot.b32(m, T.ptx.pred(d), FULL)
        T.ptx.match.any.sync.b32(m, m, FULL)
        T.ptx.match.all.sync.b32(m, p, m, FULL)  # the |p twin
        T.ptx.activemask.b32(m)
        T.ptx.elect_sync(d, p, FULL)
        T.ptx.redux_sync.add.u32(d, d, FULL)
        T.ptx.redux_sync.and_.b32(d, d, FULL)
        T.ptx.atom.global_.add.u32(d, A.ptr_to([0]), T.uint32(1))  # the .op line
        T.ptx.atom.global_.cas.b32(d, A.ptr_to([1]), d, d)  # ... and .cas
        T.ptx.atom.global_.exch.b32(d, A.ptr_to([2]), d)  # ... and .exch
        T.ptx.atom.global_.add.noftz.f16(h, A.ptr_to([3]), h)
        T.ptx.bar.red.popc.u32(d, T.uint32(0), T.ptx.pred(p))
        T.ptx.bar.red.and_.pred(p, T.uint32(0), T.ptx.pred(p))
        A[tx % 8] = d + p + m + T.uint32(h)

    src = _cuda_source(kernel)
    for text in (
        "vote.sync.all.pred pd0, ps0, %2;",
        "vote.sync.ballot.b32 %0, ps0, %2;",
        "match.any.sync.b32 %0, %1, %2;",
        "match.all.sync.b32 %0|pd0, %2, %3;",
        "activemask.b32 %0;",
        "elect.sync %0|pd0, %2;",
        "redux.sync.add.u32 %0, %1, %2;",
        "redux.sync.and.b32 %0, %1, %2;",
        "atom.global.add.u32 %0, [%1], %2;",
        "atom.global.cas.b32 %0, [%1], %2, %3;",
        "atom.global.exch.b32 %0, [%1], %2;",
        "atom.global.add.noftz.f16 %0, [%1], %2;",
        "bar.red.popc.u32 %0, %1, ps0;",
        "bar.red.and.pred pd0, %1, ps0;",
    ):
        assert text in src, text

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)

    # Every rejection below was probed against ptxas first. redux pairs the
    # arithmetic ops with a signed type and the bitwise ones with the untyped
    # word -- they are two syntax lines, so the token does not even resolve.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def redux_add_b32(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.redux_sync.add.b32(out[0], T.uint32(1), T.uint32(0xFFFFFFFF))

    # atom's vector lines bound the width by the element: a packed pair stops
    # at .v4, and only a lone half reaches .v8.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="already a 32-bit pair"):

        @T.prim_func
        def packed_v8(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            v = T.local_scalar("uint32")
            T.ptx.atom.global_.add.noftz.v8.f16x2(
                v, v, v, v, v, v, v, v, A.ptr_to([0]), v, v, v, v, v, v, v, v
            )

    # red.async's op groups: `.add` reaches 64 bits, the bitwise ops do not.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match=r"\.and takes"):

        @T.prim_func
        def red_async_and_u64(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint64")
            T.device_entry()
            v = T.local_scalar("uint64")
            T.ptx.red_async.relaxed.cluster.shared__cluster.mbarrier__complete_tx__bytes.and_.u64(
                A.ptr_to([0]), v, A.ptr_to([0])
            )


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


@requires_nvcc
def test_ptx_tcgen05_mma_block_size_form():
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
            T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32"](
                tmem, desc, desc, idesc, tmem, tmem, T.ptx.pred(flag)
            )
        A[tx] = A[tx]

    src = _cuda_source(kernel)
    assert "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32" in src
    _assert_ptxas_ok(src, arch="sm_100a")

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="mxf4.*block32"):

        @T.prim_func
        def invalid_mxf4_block16():
            T.device_entry()
            tmem = T.local_scalar("uint32")
            desc = T.local_scalar("uint64")
            idesc = T.local_scalar("uint32")
            flag = T.local_scalar("uint32")
            T.ptx["tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block16"](
                tmem, desc, desc, idesc, tmem, tmem, T.ptx.pred(flag)
            )


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


def test_ptx_sink_lane_codegen_and_roundtrip():
    """`T.ptx.SINK` renders the ISA's `_` and survives print/parse.

    A sunk lane has no C parameter, so the printed call is *shorter* than the
    one that was written -- the marker is the only thing that can say a lane
    was there at all, and without it reparsing would land on a different
    arity.
    """

    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            hi = T.local_scalar("uint32")
            packed = T.local_scalar("uint64")
            T.ptx.mov.b64(T.ptx.SINK, hi, packed)
        A[tx] = A[tx]

    src = _cuda_source(kernel)
    assert "mov.b64 {_, %0}, %1;" in src
    assert "_sink_d0" in src
    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)


def test_ptx_sink_rejected_where_the_isa_has_no_underscore():
    """Which operands take `_` is read off each syntax line, never derived.

    Not from the direction: `ld` sinks a destination it does not write and
    `st` sinks a source it does not store, so "is it written" answers nothing.
    Not from the family either: mov's *pack* shape has a scalar destination
    and a vector source, and the ISA gives neither of them a sink.
    """
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="not sinkable here"):

        @T.prim_func
        def sink_a_source():
            T.device_entry()
            packed = T.local_scalar("uint64")
            lo = T.local_scalar("uint32")
            hi = T.local_scalar("uint32")
            T.ptx.mov.b64(packed, lo, T.ptx.SINK)
            T.evaluate(hi)

    # ISA 9.7.9.4: "provided that at least one element is a scalar register".
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must be a real register"):

        @T.prim_func
        def sink_every_lane():
            T.device_entry()
            packed = T.local_scalar("uint64")
            T.ptx.mov.b64(T.ptx.SINK, T.ptx.SINK, packed)


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

    def render(name, predicated=False, dtypes=None, imms=None, **by_name):
        entry = TABLE[name]
        return render_variant(entry, tokens_for(entry, **by_name), predicated, dtypes, imms)[2]

    # tokens_for is what lets the goldens below name their modifiers. Positional
    # tuples silently shift when a slot is inserted, which is the one edit this
    # table invites; naming makes that a loud error instead.
    assert tokens_for(TABLE["ld"], space="global", type="b32") == (
        ("", "", "", "global", "", "", "", "", "", "b32")
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
    assert render("ld", dtypes=("int64", "uint64"), space="global", type="s32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_ld_global_s32_s64"
        "(int64_t& __d, const void* __addr) {\n"
        '  asm volatile("ld.global.s32 %0, [%1];" : "=l"(__d) : "l"(__addr) : "memory");\n'
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

    # Integer arithmetic (ISA 9.7.1). The plain shape first: an integer line
    # sharing the `add` mnemonic with the floating-point entry, resolved apart
    # by its type token alone.
    assert render("add_int", sat="sat", type="s32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_add_int_sat_s32"
        "(int32_t& __d, int32_t __a, int32_t __b) {\n"
        '  asm volatile("add.sat.s32 %0, %1, %2;" : "=r"(__d) : "r"(__a), "r"(__b));\n'
        "}\n"
    )
    # A dtype the instruction text does not name: `.wide`'s destination is
    # "twice as wide as a and b", so one written token (.s16) types the sources
    # while the result is the derived .s32 -- 16-bit "h" inputs, a 32-bit "r"
    # output. mad.wide derives c the same way, which is why its accumulator is
    # the wide type and its multiplicands are not.
    assert render("mul_wide", mode="wide", type="s16") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mul_wide_wide_s16"
        "(int32_t& __d, int16_t __a, int16_t __b) {\n"
        '  asm volatile("mul.wide.s16 %0, %1, %2;" : "=r"(__d) : "h"(__a), "h"(__b));\n'
        "}\n"
    )
    assert render("mad_wide", mode="wide", type="u32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mad_wide_wide_u32"
        "(uint64_t& __d, uint32_t __a, uint32_t __b, uint64_t __c) {\n"
        '  asm volatile("mad.wide.u32 %0, %1, %2, %3;" : "=l"(__d) : "r"(__a), '
        '"r"(__b), "l"(__c));\n'
        "}\n"
    )
    # A destination the ISA types outright: popc counts a 64-bit source into a
    # .u32 result, so the two operands have unrelated widths.
    assert render("popc", type="b64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_popc_b64"
        "(uint32_t& __d, uint64_t __a) {\n"
        '  asm volatile("popc.b64 %0, %1;" : "=r"(__d) : "l"(__a));\n'
        "}\n"
    )
    # Two written type tokens, and an accumulator derived from both: c and d
    # are .u32 only when atype and btype both are, so this mixed pair makes
    # them .s32 (ISA 9.7.1.24).
    assert render("dp4a", atype="u32", btype="s32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_dp4a_u32_s32"
        "(int32_t& __d, uint32_t __a, int32_t __b, int32_t __c) {\n"
        '  asm volatile("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(__d) : "r"(__a), '
        '"r"(__b), "r"(__c));\n'
        "}\n"
    )
    # Five operands under the ISA's own names: the destination is `f`, and `d`
    # is the field-length input -- the one family where `d` is not the result.
    assert render("bfi", type="b64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_bfi_b64"
        "(uint64_t& __f, uint64_t __a, uint64_t __b, uint32_t __c, uint32_t __d) {\n"
        '  asm volatile("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(__f) : "l"(__a), "l"(__b), '
        '"r"(__c), "r"(__d));\n'
        "}\n"
    )
    # bmsk's destination is written .b32 but declared u32: ptxas takes no float
    # register anywhere in this instruction, unlike every other .bN family (see
    # the entry's note). So there is no f32 twin of this helper to render.
    assert render("bmsk", mode="clamp", type="b32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_bmsk_clamp_b32"
        "(uint32_t& __d, uint32_t __a, uint32_t __b) {\n"
        '  asm volatile("bmsk.clamp.b32 %0, %1, %2;" : "=r"(__d) : "r"(__a), "r"(__b));\n'
        "}\n"
    )

    # Floating point (ISA 9.7.3). A `.pred` destination: the instruction writes
    # a predicate register, which no asm constraint can bind, so the helper
    # declares one inside the block and materializes 0/1 through selp on the
    # way out. Still exactly one instruction -- the rest is the boundary.
    assert render("testp", op="notanumber", type="f32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_testp_notanumber_f32"
        "(uint32_t& __p, float __a) {\n"
        '  asm volatile("{ .reg .pred pd0; testp.notanumber.f32 pd0, %1; '
        'selp.b32 %0, 1, 0, pd0; }" : "=r"(__p) : "f"(__a));\n'
        "}\n"
    )
    # `.full` is a divide mode beside the rounding modes, not a qualifier on
    # top of one -- the ISA requires exactly one of .approx/.full/.rnd, so the
    # entry fuses them into a single mandatory slot.
    assert render("div_f", mode="approx", ftz="ftz", type="f32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_div_f_approx_ftz_f32"
        "(float& __d, float __a, float __b) {\n"
        '  asm volatile("div.approx.ftz.f32 %0, %1, %2;" : "=f"(__d) : "f"(__a), "f"(__b));\n'
        "}\n"
    )
    # mad shares fma's grid and `_check_farith`, and shares the `mad` mnemonic
    # with the two integer entries; `.rn` is what tells them apart.
    assert render("mad_f", rnd="rn", sat="sat", type="f32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_mad_f_rn_sat_f32"
        "(float& __d, float __a, float __b, float __c) {\n"
        '  asm volatile("mad.rn.sat.f32 %0, %1, %2, %3;" : "=f"(__d) : "f"(__a), '
        '"f"(__b), "f"(__c));\n'
        "}\n"
    )
    # ISA 9.7.3.14 is its own subsection because it is a different computation,
    # but syntactically it is one cell of rcp's grid -- reachable only with the
    # mandatory .ftz its syntax line spells (see `_check_rcp`).
    assert render("rcp", mode="approx", ftz="ftz", type="f64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_rcp_approx_ftz_f64"
        "(double& __d, double __value) {\n"
        '  asm volatile("rcp.approx.ftz.f64 %0, %1;" : "=d"(__d) : "d"(__value));\n'
        "}\n"
    )

    # Half precision (ISA 9.7.4). fma's two extra clampings, on the packed
    # type: `.relu` and `.oob` exist on no same-precision line, which is what
    # keeps the half fma out of the f32/f64 entry. A packed pair rides one
    # 32-bit register, so every operand is uint32.
    assert render("fma_half", rnd="rn", oob="oob", relu="relu", type="f16x2") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_fma_half_rn_oob_relu_f16x2"
        "(uint32_t& __d, uint32_t __a, uint32_t __b, uint32_t __c) {\n"
        '  asm volatile("fma.rn.oob.relu.f16x2 %0, %1, %2, %3;" : "=r"(__d) : "r"(__a), '
        '"r"(__b), "r"(__c));\n'
        "}\n"
    )
    # ex2's bf16 line spells `.ftz` mandatorily while its f16 line does not
    # offer it -- the reason these are a separate entry from the .f32 ex2.
    assert render("ex2_half", mode="approx", ftz="ftz", type="bf16x2") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_ex2_half_approx_ftz_bf16x2"
        "(uint32_t& __d, uint32_t __value) {\n"
        '  asm volatile("ex2.approx.ftz.bf16x2 %0, %1;" : "=r"(__d) : "r"(__value));\n'
        "}\n"
    )
    # The scalar half type is the other carrier: a 16-bit register on "h".
    assert render("abs_half", ftz="ftz", type="f16") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_abs_half_ftz_f16"
        "(uint16_t& __d, uint16_t __a) {\n"
        '  asm volatile("abs.ftz.f16 %0, %1;" : "=h"(__d) : "h"(__a));\n'
        "}\n"
    )

    # Mixed precision (ISA 9.7.5) is not a separate instruction but a fourth
    # syntax line of add/sub/fma: a second type token names a 16-bit source
    # that is converted to .f32 before the operation. So one helper carries two
    # carriers at once -- "h" for the converted sources, "f" for the rest.
    # fma converts both a and b (`.abtype`); add/sub convert only a.
    assert render("fma", rnd="rn", sat="sat", type="f32", srctype="bf16") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_fma_rn_sat_f32_bf16"
        "(float& __d, uint16_t __a, uint16_t __b, float __c) {\n"
        '  asm volatile("fma.rn.sat.f32.bf16 %0, %1, %2, %3;" : "=f"(__d) : "h"(__a), '
        '"h"(__b), "f"(__c));\n'
        "}\n"
    )

    # Comparison and selection (ISA 9.7.6). setp's `p|q`: ONE operand position
    # holding two predicate destinations, joined by the ISA's own separator
    # rather than a comma. Each half is a real register with its own selp on
    # the way out -- q carries the Boolean applied to the complement of the
    # compare, so it is a second result, not a restatement of p.
    assert render("setp_pq", cmp="lt", type="s32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_setp_pq_lt_s32"
        "(uint32_t& __p, uint32_t& __q, int32_t __a, int32_t __b) {\n"
        '  asm volatile("{ .reg .pred pd0; .reg .pred pd1; setp.lt.s32 pd0|pd1, %2, %3; '
        'selp.b32 %0, 1, 0, pd0; selp.b32 %1, 1, 0, pd1; }" : "=r"(__p), "=r"(__q) '
        ': "r"(__a), "r"(__b));\n'
        "}\n"
    )
    # A predicate *source* rides the same carrier in the other direction: setp
    # in, then the instruction. selp is the plainest case of it.
    assert render("selp", type="f64") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_selp_f64"
        "(double& __d, double __a, double __b, uint32_t __c) {\n"
        '  asm volatile("{ .reg .pred ps0; setp.ne.b32 ps0, %3, 0; '
        'selp.f64 %0, %1, %2, ps0; }" : "=d"(__d) : "d"(__a), "d"(__b), "r"(__c));\n'
        "}\n"
    )
    # `set` writes a value rather than a predicate, so its two type tokens are
    # independent: an f32 comparison landing 0xffffffff in a u32 destination.
    assert render("set", cmp="lt", dtype="u32", stype="f32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_set_lt_u32_f32"
        "(uint32_t& __d, float __a, float __b) {\n"
        '  asm volatile("set.lt.u32.f32 %0, %1, %2;" : "=r"(__d) : "f"(__a), "f"(__b));\n'
        "}\n"
    )
    # slct likewise: the selected value and the number whose sign selects it
    # are separately typed.
    assert render("slct", ftz="ftz", dtype="b32", ctype="f32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_slct_ftz_b32_f32"
        "(uint32_t& __d, uint32_t __a, uint32_t __b, float __c) {\n"
        '  asm volatile("slct.ftz.b32.f32 %0, %1, %2, %3;" : "=r"(__d) : "r"(__a), '
        '"r"(__b), "f"(__c));\n'
        "}\n"
    )

    # Half-precision comparison (ISA 9.7.7). The packed setp reuses the pipe
    # pair, but its two halves mean something else than 9.7.6's: p and q are
    # the two lanes' comparisons, not a result and its complement.
    assert render("setp_half_pq", cmp="lt", ftz="ftz", type="f16x2") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_setp_half_pq_lt_ftz_f16x2"
        "(uint32_t& __p, uint32_t& __q, uint32_t __a, uint32_t __b) {\n"
        '  asm volatile("{ .reg .pred pd0; .reg .pred pd1; '
        "setp.lt.ftz.f16x2 pd0|pd1, %2, %3; selp.b32 %0, 1, 0, pd0; "
        'selp.b32 %1, 1, 0, pd1; }" : "=r"(__p), "=r"(__q) : "r"(__a), "r"(__b));\n'
        "}\n"
    )
    # set's two type tokens run in either direction: a half-valued answer to an
    # integer comparison ...
    assert render("set_half", cmp="lt", dtype="f16", stype="s32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_set_half_lt_f16_s32"
        "(uint16_t& __d, int32_t __a, int32_t __b) {\n"
        '  asm volatile("set.lt.f16.s32 %0, %1, %2;" : "=h"(__d) : "r"(__a), "r"(__b));\n'
        "}\n"
    )
    # ... and an integer-valued answer to a half one, where `.ftz` is legal
    # because the *source* is the f16 (see `_FTZ_SET_STYPES`).
    assert render("set_half", cmp="gt", ftz="ftz", dtype="u32", stype="f16") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_set_half_gt_ftz_u32_f16"
        "(uint32_t& __d, uint16_t __a, uint16_t __b) {\n"
        '  asm volatile("set.gt.ftz.u32.f16 %0, %1, %2;" : "=r"(__d) : "h"(__a), "h"(__b));\n'
        "}\n"
    )

    # Logic and shift (ISA 9.7.8). `.pred` as an instruction type, rather than
    # as one operand's class: three bridges -- two setp in, one selp out --
    # wrapped around a single instruction that never touches the carriers.
    assert render("and", type="pred") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_and_pred"
        "(uint32_t& __d, uint32_t __a, uint32_t __b) {\n"
        '  asm volatile("{ .reg .pred pd0; .reg .pred ps0; .reg .pred ps1; '
        "setp.ne.b32 ps0, %1, 0; setp.ne.b32 ps1, %2, 0; and.pred pd0, ps0, ps1; "
        'selp.b32 %0, 1, 0, pd0; }" : "=r"(__d) : "r"(__a), "r"(__b));\n'
        "}\n"
    )
    # lop3's BoolOp form: the pipe pair holding two DIFFERENT register classes.
    # `d` is an ordinary b32 output bound straight to %0 while `p` rides the
    # predicate bridge, so the pair renders `%0|pd0` -- the mechanism groups
    # the text and leaves each half its own constraint. The LUT byte is an open
    # immediate, baked into the text and into the helper name.
    assert render("lop3_bool", boolop="and", type="b32", imms=("128",)) == (
        "__forceinline__ __device__ void tvm_builtin_ptx_lop3_bool_and_b32_128"
        "(uint32_t& __d, uint32_t& __p, uint32_t __a, uint32_t __b, uint32_t __c, "
        "uint32_t __q) {\n"
        '  asm volatile("{ .reg .pred pd0; .reg .pred ps0; setp.ne.b32 ps0, %5, 0; '
        'lop3.and.b32 %0|pd0, %2, %3, %4, 128, ps0; selp.b32 %1, 1, 0, pd0; }" '
        ': "=r"(__d), "=r"(__p) : "r"(__a), "r"(__b), "r"(__c), "r"(__q));\n'
        "}\n"
    )
    # The shift amount is a 32-bit value "regardless of the instruction type",
    # so a 16-bit shl still takes a uint32 there.
    assert render("shl", type="b16") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_shl_b16"
        "(uint16_t& __d, uint16_t __a, uint32_t __b) {\n"
        '  asm volatile("shl.b16 %0, %1, %2;" : "=h"(__d) : "h"(__a), "r"(__b));\n'
        "}\n"
    )

    # Data movement (ISA 9.7.9). shfl.sync's `d|p`: a pipe pair whose halves
    # are DIFFERENT register classes -- an ordinary b32 result bound straight
    # to %0, and an in-range predicate that rides the bridge.
    assert render("shfl_sync_p", mode="up", type="b32") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_shfl_sync_p_up_b32"
        "(uint32_t& __d, uint32_t& __p, uint32_t __a, uint32_t __b, uint32_t __c, "
        "uint32_t __membermask) {\n"
        '  asm volatile("{ .reg .pred pd0; shfl.sync.up.b32 %0|pd0, %2, %3, %4, %5; '
        'selp.b32 %1, 1, 0, pd0; }" : "=r"(__d), "=r"(__p) : "r"(__a), "r"(__b), '
        '"r"(__c), "r"(__membermask));\n'
        "}\n"
    )
    # multimem's vector line: `.v4` of `.f16` is one 64-bit access spread over
    # four half registers, and `.acc::f32` widens only the accumulation.
    assert render("multimem_ld_reduce_f_vec", op="add", acc="acc::f32", vec="v4", type="f16") == (
        "__forceinline__ __device__ void "
        "tvm_builtin_ptx_multimem_ld_reduce_f_vec_add_acc__f32_v4_f16"
        "(uint16_t& __d0, uint16_t& __d1, uint16_t& __d2, uint16_t& __d3, "
        "const void* __addr) {\n"
        '  asm volatile("multimem.ld_reduce.add.acc::f32.v4.f16 {%0, %1, %2, %3}, [%4];" '
        ': "=h"(__d0), "=h"(__d1), "=h"(__d2), "=h"(__d3) : "l"(__addr) : "memory");\n'
        "}\n"
    )
    # A table-owned immediate: the ISA fixes applypriority's size at 128, so it
    # is in the text and the helper has no parameter for it at all.
    assert render("applypriority", space="global", level="L2::evict_normal") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_applypriority_global_L2__evict_normal"
        "(const void* __addr) {\n"
        '  asm volatile("applypriority.global.L2::evict_normal [%0], 128;" :  : "l"(__addr)'
        ' : "memory");\n'
        "}\n"
    )
    # A caller-chosen immediate: tensormap.replace's field3 new_val must be a
    # constant, so each legal value is its own helper, named for it.
    assert render(
        "tensormap_replace_elemtype",
        mode="tile",
        field="elemtype",
        space="global",
        width="b1024",
        type="b32",
        imms=("7",),
    ) == (
        "__forceinline__ __device__ void "
        "tvm_builtin_ptx_tensormap_replace_elemtype_tile_elemtype_global_b1024_b32_7"
        "(const void* __addr) {\n"
        '  asm volatile("tensormap.replace.tile.elemtype.global.b1024.b32 [%0], 7;" :  : '
        '"l"(__addr) : "memory");\n'
        "}\n"
    )

    # Parallel synchronization (ISA 9.7.14). elect.sync's `d|p` is the one
    # pipe pair the ISA makes mandatory -- ptxas rejects a bare destination --
    # and its halves are two different register classes.
    assert render("elect_sync") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_elect_sync"
        "(uint32_t& __d, uint32_t& __p, uint32_t __membermask) {\n"
        '  asm volatile("{ .reg .pred pd0; elect.sync %0|pd0, %2; '
        'selp.b32 %1, 1, 0, pd0; }" : "=r"(__d), "=r"(__p) : "r"(__membermask));\n'
        "}\n"
    )
    # atom.cas is the four-value shape, and the only atom line reaching 128
    # bits in every position -- the "q" constraint the C boundary needs.
    assert render("atom_cas", space="global", op="cas", type="b128") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_atom_cas_global_cas_b128"
        "(__uint128_t& __d, const void* __addr, __uint128_t __compare, "
        "__uint128_t __value) {\n"
        '  asm volatile("atom.global.cas.b128 %0, [%1], %2, %3;" : "=q"(__d) : '
        '"l"(__addr), "q"(__compare), "q"(__value) : "memory");\n'
        "}\n"
    )
    # bar.red reduces a predicate across a barrier, so a predicate crosses the
    # boundary in BOTH directions around one instruction: setp on the way in,
    # selp on the way out.
    assert render("bar_red_pred", action="red", op="and", type="pred") == (
        "__forceinline__ __device__ void tvm_builtin_ptx_bar_red_pred_red_and_pred"
        "(uint32_t& __d, uint32_t __a, uint32_t __c) {\n"
        '  asm volatile("{ .reg .pred pd0; .reg .pred ps0; setp.ne.b32 ps0, %2, 0; '
        'bar.red.and.pred pd0, %1, ps0; selp.b32 %0, 1, 0, pd0; }" : "=r"(__d) : '
        '"r"(__a), "r"(__c) : "memory");\n'
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
    assert len(call.args) == 2 + 1 + 8 + 1  # operands + pred + slot tokens + marker
    out = tvm.tirx.decl_buffer((1,), "uint32", name="out", scope="local")
    call = T.ptx.ld.global_.b32(out[0], global_ptr, pred=flag)
    assert str(call.args[-1]).strip('"') == "pred"
    call = T.ptx.ld.global_.b32(out[0], global_ptr, pred=flag, preserve_dst=True)
    assert str(call.args[-1]).strip('"') == "pred,keep"
    with pytest.raises(ValueError, match="requires pred"):
        T.ptx.ld.global_.b32(out[0], global_ptr, preserve_dst=True)
    with pytest.raises(ValueError, match="requires a written destination"):
        T.ptx.st.release.gpu.global_.b32(global_ptr, val, pred=flag, preserve_dst=True)


# fp16/bf16 dtypes bring in __half / __nv_bfloat16 and their bit-cast helpers.
_CERT_PRELUDE = "#include <cstdint>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>"

_ASM_RE = re.compile(r'asm(?: volatile)?\("(.*?)"\s*:', re.S)
_BLOCK_RE = re.compile(r"^\{ (?P<body>.*) \}$")
# The asm block's sanctioned non-instructions: `render.BRIDGE`'s register
# declarations and the conversions that move a value between the class the ISA
# names and the carrier inline asm can bind, plus `@p`'s own guard. Never
# semantics -- see BRIDGE.
#
# Matched in FULL, not by opcode prefix, because an opcode is no longer a
# discriminator: `setp` and `selp` are registered instructions (ISA 9.7.6), so
# a helper's one real instruction can carry the same mnemonic as the bridge
# statements around it. What separates them is the shape the bridge always
# has -- it names a bridge-local register (`p`, `ps<n>`, `pd<n>`, `raw_<slot>`)
# and compares or selects against the literals the conversion is made of, while
# a registered instruction's operands are `%<n>` throughout. Matching the whole
# statement also makes this stricter than the prefix form it replaces: a stray
# conversion in some other shape is now a failure instead of being peeled.
_BOUNDARY_RE = re.compile(
    r"|".join(
        (
            r"\.reg \.pred (?:p|ps\d+|pd\d+);",  # @p guard / pred bridge declarations
            r"\.reg \.b8 raw_\w+;",  # b8 bridge declaration
            r"setp\.ne\.b32 (?:p|ps\d+), %\d+, 0;",  # @p guard, pred_src conversion in
            r"selp\.b32 %\d+, 1, 0, pd\d+;",  # pred_dst materialization out
            r"cvt\.u8\.u16 raw_\w+, %\d+;",  # b8 conversion in
            r"cvt\.u16\.u8 %\d+, raw_\w+;",  # b8 conversion out
        )
    )
)


def _as_render_args(rendering):
    """`renderings` yields (tokens, dtypes, predicated, imms); render_variant
    takes (tokens, predicated, dtypes, imms)."""
    tokens, dtypes, predicated, imms, sinks = rendering
    return tokens, predicated, dtypes, imms, sinks


def _addr_offset_samples(entry):
    """Small certification axis for address immediates, separate from modifier products."""
    from tvm.backend.cuda.ptx.table import renderings

    enabled = [
        logical_slot
        for logical_slot, slot in enumerate(s for s in entry.operands if s.kind == "addr")
        if slot.allow_imm_offset
    ]
    if not enabled:
        return ()
    representative = _as_render_args(next(iter(renderings(entry))))
    samples = [
        (representative, ((logical_slot, offset),))
        for logical_slot in enabled
        for offset in (16, -16)
    ]
    if len(enabled) > 1:
        samples.append(
            (
                representative,
                tuple(
                    (logical_slot, 16 if index % 2 == 0 else -16)
                    for index, logical_slot in enumerate(enabled)
                ),
            )
        )
    return tuple(samples)


def _sole_instruction(asm_text):
    """The single PTX statement in ``asm_text``, or None if it is not exactly one.

    The sanctioned boundary conversions are peeled first: ``@p``'s guard,
    pred_src's setp, pred_dst's selp. They convert values at the block
    boundary; they never add a second instruction.
    """
    m = _BLOCK_RE.match(asm_text)
    if m:
        stmts = [f"{part.strip()};" for part in m.group("body").split(";") if part.strip()]
        core = [st for st in stmts if not _BOUNDARY_RE.fullmatch(st)]
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
        for tokens, dtypes, predicated, imms, sinks in renderings(entry):
            opcode, helper, source = render_variant(entry, tokens, predicated, dtypes, imms, sinks)
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
            # The address coercion `cvta` is a separate IR node, so no OTHER
            # instruction's helper may smuggle one in. The cvta entries
            # themselves are the exception, being that instruction: the guard
            # keys off the mnemonic rather than the table name so that every
            # syntax line of it (`cvta.to.space` and `cvta.space`) is covered.
            assert "cvta" not in source or entry.ptx_name == "cvta", (
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
        for tokens, dtypes, predicated, imms, sinks in renderings(entry):
            opcode, helper, source = render_variant(entry, tokens, predicated, dtypes, imms, sinks)
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
        for args, addr_offsets in _addr_offset_samples(entry):
            _, helper, _ = render_variant(entry, *args, addr_offsets=addr_offsets)
            assert helper not in names, f"address-offset helper name collision: {helper}"
            names.add(helper)
    assert total == 200018  # update when the table grows


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
        for tokens, dtypes, predicated, imms, sinks in renderings(entry):
            _, _, source = render_variant(entry, tokens, predicated, dtypes, imms, sinks)
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
        for args, addr_offsets in _addr_offset_samples(entry):
            _, _, source = render_variant(entry, *args, addr_offsets=addr_offsets)
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
    baseline = (
        (TABLE[name], _as_render_args(rendering), ())
        for name in sorted(TABLE)
        for rendering in renderings(TABLE[name])
    )
    address_samples = (
        (TABLE[name], args, addr_offsets)
        for name in sorted(TABLE)
        for args, addr_offsets in _addr_offset_samples(TABLE[name])
    )
    for index, (entry, args, addr_offsets) in enumerate(itertools.chain(baseline, address_samples)):
        if index % _CERT_SHARDS == shard:
            covered += 1
            arch = entry.cert_arch or PTX_ARCH
            _, _, src = render_variant(entry, *args, addr_offsets=addr_offsets)
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
