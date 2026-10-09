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
# pylint: disable=invalid-name, too-many-arguments
"""CUDA, PTX, and NVSHMEM TIR intrinsic builders."""

from __future__ import annotations

from enum import Enum

import tvm_ffi

from tvm import tirx
from tvm.ir import Attrs, Call, Op, StringImm, const
from tvm.ir.op import _init_op_api, _make_op_api
from tvm.ir.type import PointerType, PrimType
from tvm.tirx.op import access_ptr, bitwise_and, call_intrin
from tvm.tirx.operator.intrinsics._common import (
    CP_ASYNC_BULK_CACHE_HINT as _CP_ASYNC_BULK_CACHE_HINT,
)
from tvm.tirx.operator.intrinsics._common import MBARRIER_ARRIVE_SCOPE as _MBARRIER_ARRIVE_SCOPE
from tvm.tirx.operator.intrinsics._common import MBARRIER_ARRIVE_SEM as _MBARRIER_ARRIVE_SEM
from tvm.tirx.operator.intrinsics._common import MBARRIER_ARRIVE_SPACE as _MBARRIER_ARRIVE_SPACE
from tvm.tirx.operator.intrinsics._common import NVSHMEM_CMP as _NVSHMEM_CMP
from tvm.tirx.operator.intrinsics._common import NVSHMEM_SIG_OP as _NVSHMEM_SIG_OP
from tvm.tirx.operator.intrinsics._common import TCGEN05_CTA_GROUP as _TCGEN05_CTA_GROUP

tir = tirx

########################################################
# CUDA native builtins
########################################################


def cuda_iket_mark(name, payload=None):
    """Create an NVIDIA IKET marker annotation."""
    if payload is not None:
        return call_intrin("", "tirx.cuda.iket_mark", name, payload)
    return call_intrin("", "tirx.cuda.iket_mark", name)


def cuda_iket_range_start(name, payload=None):
    """Create an NVIDIA IKET token-range start annotation."""
    if payload is not None:
        return call_intrin("uint32", "tirx.cuda.iket_range_start", name, payload)
    return call_intrin("uint32", "tirx.cuda.iket_range_start", name)


def cuda_iket_range_end(token, payload=None):
    """Create an NVIDIA IKET token-range end annotation."""
    if payload is not None:
        return call_intrin("", "tirx.cuda.iket_range_end", token, payload)
    return call_intrin("", "tirx.cuda.iket_range_end", token)


def cuda_iket_range_push(name, payload=None):
    """Create an NVIDIA IKET stack-range push annotation."""
    if payload is not None:
        return call_intrin("", "tirx.cuda.iket_range_push", name, payload)
    return call_intrin("", "tirx.cuda.iket_range_push", name)


def cuda_iket_official_event(event_id, source_code="", payload=None):
    """Create an NVIDIA IKET official range-end event."""
    if payload is not None:
        return call_intrin(
            "uint32", "tirx.cuda.iket_official_event", event_id, source_code, payload
        )
    return call_intrin("uint32", "tirx.cuda.iket_official_event", event_id, source_code)


def cuda_func_call(func_name, *args, ty=None, span=None):
    """Call a CUDA function with its source code as the final operand.

    ``args`` contains the function arguments followed by the source string.
    ``ty`` specifies an explicit result; omitted results are void.
    """
    return Call("tirx.cuda.func_call", [func_name, *args], ty=ty, span=span)


def cuda_warp_reduce(value, op, width=32, *, ty=None, span=None):
    """Warp-level butterfly shuffle-XOR reduction.

    Reduces ``value`` across ``width`` adjacent lanes using the specified
    operation.  Codegen emits ``log2(width)`` steps of
    ``__shfl_xor_sync(0xFFFFFFFF, val, mask)`` with descending XOR masks.

    Parameters
    ----------
    value : Expr
        The per-thread scalar value to reduce.

    op : str
        Reduction operation: ``"sum"``, ``"max"``, or ``"min"``.

    width : int
        Number of lanes participating in each reduction group.
        Must be a power of two in [2, 32].  Defaults to 32 (full warp).

    Returns
    -------
    call : Expr
        The reduced value (same dtype as *value*).
    """
    return Call(
        "tirx.cuda.warp_reduce",
        [value, op, width],
        ty=ty,
        span=span,
    )


def cuda_warp_sum(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "sum", width)``."""
    return cuda_warp_reduce(value, "sum", width)


def cuda_warp_max(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "max", width)``."""
    return cuda_warp_reduce(value, "max", width)


def cuda_warp_min(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "min", width)``."""
    return cuda_warp_reduce(value, "min", width)


def cuda_cta_reduce(value, op, num_warps, scratch, *, ty=None, span=None):
    """CTA-wide reduction via warp shuffle + shared memory.

    Two-step reduction: (1) intra-warp shuffle reduction, (2) warp-0
    collects per-warp partials from ``scratch``, reduces, broadcasts via
    ``__syncthreads()``.  All CTA threads must participate.

    Parameters
    ----------
    value : Expr
        Per-thread scalar value to reduce.

    op : str
        Reduction operation: ``"sum"``, ``"max"``, or ``"min"``.

    num_warps : int
        Number of warps in the CTA.  Must be a power of two in [1, 32].

    scratch : Var
        Data pointer to shared-memory scratch space (>= num_warps elements).

    Returns
    -------
    call : Expr
        The reduced value broadcast to all threads (same dtype as *value*).
    """
    return Call(
        "tirx.cuda.cta_reduce",
        [value, op, num_warps, scratch],
        ty=ty,
        span=span,
    )


def cuda_cta_sum(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "sum", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "sum", num_warps, scratch)


def cuda_cta_max(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "max", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "max", num_warps, scratch)


def cuda_cta_min(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "min", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "min", num_warps, scratch)


_WAIT_UNTIL_SCOPE = ("cta", "cluster", "gpu", "sys")
# Global only, and that is the whole surface a declared word needs. A protocol
# that waits inside a CTA or a cluster has `mbarrier`, which is the hardware's
# own primitive for it and which the checker already models by generation; a
# polled flag in shared memory would be a worse spelling of the same thing. The
# shared-memory reads that look like a protocol in practice -- fetching a TMEM
# allocation address out of a mailbox -- are ordinary reads after a barrier,
# with no spin and no predicate, so they are not protocols at all.
_WAIT_UNTIL_SPACE = ("global",)
_WAIT_UNTIL_DTYPE = ("int32", "uint32", "int64", "uint64")
# Widths, so a requested PTX type can be checked against the word it accesses.
# Which spellings an instruction actually takes is the instruction table's
# business: `add` has no `.b32` form, so a bit-typed arrival is rejected there
# rather than listed as illegal here.
_WAIT_UNTIL_PTX_TYPES = {
    "b32": 32,
    "s32": 32,
    "u32": 32,
    "b64": 64,
    "s64": 64,
    "u64": 64,
    # A 128-bit word is a 16-byte naturally aligned span, moved by the `.b128`
    # forms of `ld`/`st` (PTX ISA 8.3, sm_70). It carries a value no scalar
    # predicate can test, so `wait` rejects it; `store` and `load` take it.
    "b128": 128,
}


def _validate_wait_until_attrs(scope, space, ptx_type=None):
    """The attributes a declared synchronization-word wait still carries."""
    if scope not in _WAIT_UNTIL_SCOPE:
        raise ValueError(f"invalid scope={scope!r}; expected one of {_WAIT_UNTIL_SCOPE}")
    if space not in _WAIT_UNTIL_SPACE:
        detail = (
            "; a protocol that waits within a CTA or cluster belongs on an "
            "`mbarrier`, which the checker models by generation"
            if space == "shared"
            else ""
        )
        raise ValueError(f"invalid space={space!r}; expected one of {_WAIT_UNTIL_SPACE}{detail}")
    if ptx_type is not None and ptx_type not in _WAIT_UNTIL_PTX_TYPES:
        raise ValueError(
            f"invalid ptx_type={ptx_type!r}; expected one of {tuple(sorted(_WAIT_UNTIL_PTX_TYPES))}"
        )


def _reject_wide_word_for_predicate(ptx_type, what):
    """A predicate tests one scalar, so a 128-bit word cannot be waited on.

    The exit value of a 16-byte word is not a number the loop can compare, and
    the checker's record of a declared word's writes holds one scalar per
    write. A kernel that has to read one spells the `ld` itself.
    """
    if ptx_type is not None and _WAIT_UNTIL_PTX_TYPES[ptx_type] > 64:
        raise ValueError(
            f"{what} does not take a {_WAIT_UNTIL_PTX_TYPES[ptx_type]}-bit word: "
            "its exit value is not a scalar a predicate can test; read it with "
            "a plain `ld` instead"
        )


def cuda_wait_until(
    dst,
    ptr,
    predicate,
    scope="gpu",
    space="global",
    ptx_type=None,
    backoff_ns=None,
    *,
    ty=None,
    span=None,
):
    """Read the global word at ``ptr`` into ``dst`` until ``predicate`` holds,
    and leave the exit value there.

    Waiting on an address this way is also what declares it a synchronization
    word: the checker judges every access to that address against the protocol
    the wait names, rather than as an ordinary pair of memory accesses.

    ``dst`` is an initialized thread-local scalar; its current value is tested
    first, so an already satisfied predicate performs no load. ``predicate`` is
    a trace-time callable taking the current value, or the boolean expression
    itself. It is re-evaluated on every iteration, so it may test ``dst``
    against a loop-carried scalar such as a barrier's phase: the loop body only
    loads, and nothing it does can move that scalar.

    The wait always synchronizes with the contributions that made the
    predicate hold, so data those threads published elsewhere is visible when
    it returns. It polls ``ld.relaxed.<scope>`` and closes with one
    ``ld.acquire.<scope>`` into a discarded register, which is the cheapest
    spelling of that edge rather than a separate mode: paying acquire on every
    poll costs more, and closing with an ``acq_rel``/``sc`` fence instead costs
    far more, because the fence loses the loop's fast-path exit.

    There is no way to ask for less. A wait whose exit value is the whole
    message does take an edge it has no use for, and the two such waits in the
    kernel corpus were measured against this form on three shapes: every ratio
    landed inside the band that identical code measured against itself, and
    the two smaller shapes disagreed on the sign. So a relaxed mode would buy
    nothing here, while what it asks for is a promise made at the call site --
    that nothing the word guards is read afterwards -- which the call site
    cannot show and the next edit can silently break.

    ``T.nvshmem.wait_until`` shares this name deliberately: both block until a
    value satisfies a condition. They differ in what they wait on and how the
    condition is written -- that one names a symmetric object across PEs and
    takes an enumerated comparison; this one names an address in this device's
    global memory and takes a predicate. A protocol that waits within a CTA or
    cluster belongs on an ``mbarrier`` instead, which is why ``space`` admits
    only ``global``.

    ``backoff_ns`` puts a ``__nanosleep`` before each retry, as a contended
    wait is ordinarily written. It goes before the load, so a predicate that
    holds on entry still performs no load and no sleep, and a wait whose first
    poll succeeds pays nothing. A kernel that spells the backoff itself writes
    ``ld`` once and then waits, which is the same instruction sequence.

    The backoff is the only thing a wait carries besides its own load, and it
    stays a scalar for a reason: it runs every iteration, touches no memory,
    and cannot move what the predicate reads, so it changes nothing the
    checker concludes. A timeout that has to print and trap is not that, and
    belongs to a loop the kernel writes itself.
    """
    scope = _static_str(scope)
    space = _static_str(space)
    ptx_type = _static_str(ptx_type) if ptx_type is not None else None
    _validate_wait_until_attrs(scope, space, ptx_type)
    _reject_wide_word_for_predicate(ptx_type, "wait_until")
    if tirx.is_tensor_var(dst):
        dst = dst[0]
    condition = tirx.convert(predicate(dst) if callable(predicate) else predicate)
    return Call(
        "tirx.cuda.wait_until",
        [
            dst,
            ptr,
            condition,
            scope,
            space,
            ptx_type or "",
            tirx.convert(0 if backoff_ns is None else backoff_ns),
        ],
        ty=ty,
        span=span,
    )


def _validate_mbarrier_arrive_attrs(sem, scope, space, remote):
    if (sem == "") != (scope == ""):
        raise ValueError("mbarrier.arrive sem and scope must be specified together")
    if sem not in _MBARRIER_ARRIVE_SEM:
        raise ValueError(f"invalid sem={sem!r}; expected one of {_MBARRIER_ARRIVE_SEM}")
    if scope not in _MBARRIER_ARRIVE_SCOPE:
        raise ValueError(f"invalid scope={scope!r}; expected one of {_MBARRIER_ARRIVE_SCOPE}")
    if space not in _MBARRIER_ARRIVE_SPACE:
        raise ValueError(f"invalid space={space!r}; expected one of {_MBARRIER_ARRIVE_SPACE}")
    if remote is not None and space != "shared::cluster":
        raise ValueError("remote mbarrier.arrive requires space='shared::cluster'")


_cp_async_raw = _make_op_api(Op.get("tirx.s_tir.cp_async_raw"), __name__)


def ptx_cp_async_legacy(
    dst_ptr,
    dst_offset,
    src_ptr,
    src_offset,
    cp_size,
    *,
    elem_dtype="int8",
    ty=None,
    span=None,
):
    """Fold element offsets into the raw cp.async pointer operands.

    ``elem_dtype`` scales element offsets independently of the call's result ``ty``.
    """
    dst_ptr = _wrap_or_fold_access_ptr(dst_ptr, dst_offset, elem_dtype)
    src_ptr = _wrap_or_fold_access_ptr(src_ptr, src_offset, elem_dtype)
    return _cp_async_raw(dst_ptr, 0, src_ptr, 0, cp_size, ty=ty, span=span)


def _is_static_unicast_cta_mask(cta_mask):
    if isinstance(cta_mask, int):
        return cta_mask == 0 or cta_mask & (cta_mask - 1) == 0
    if isinstance(cta_mask, tirx.IntImm):
        value = int(cta_mask)
        return value == 0 or value & (value - 1) == 0
    return False


def cuda_mov_sreg(bits, reg_name, *, ty=None, span=None):
    """TVM intrinsic to tvm instrinsics to fetch PTX pre-defined registers

    Parameters
    ----------
    bits : int
        The number of bits of the register.

    reg_name : str
        The name of the register.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return Call(
        "tirx.cuda.mov_sreg",
        [bits, reg_name],
        ty=ty,
        span=span,
    )


_legacy_mma = _make_op_api(Op.get("tirx.ptx_legacy.mma"), __name__)


def ptx_legacy_mma(
    shape,
    a_layout,
    b_layout,
    a_dtype,
    b_dtype,
    c_dtype,
    a_ptr,
    a_offset,
    b_ptr,
    b_offset,
    acc_ptr,
    c_offset,
    saturate,
    operator=None,
    *,
    ty=None,
    span=None,
):
    """Construct legacy MMA with element offsets and an optional bit operator."""
    args = [
        shape,
        a_layout,
        b_layout,
        a_dtype,
        b_dtype,
        c_dtype,
        a_ptr,
        a_offset,
        b_ptr,
        b_offset,
        acc_ptr,
        c_offset,
        saturate,
    ]
    if operator is not None:
        args.append(operator)
    return _legacy_mma(*args, ty=ty, span=span)


def bmma_sync(
    fragment_d,
    index_d,
    fragment_a,
    index_a,
    fragment_b,
    index_b,
    fragment_c,
    index_c,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for tensor core bmma_sync operators

    Parameters
    ----------
    fragment_d : Var
        The bwmma fragment_d.

    index_d : Expr
        The fragment_d index.

    fragment_a : Var
        The bwmma fragment_a.

    index_a : Expr
        The fragment_a index.

    fragment_b : Var
        The bwmma fragment_b.

    index_b : Expr
        The fragment_b index.

    fragment_c : Var
        The bwmma fragment_c.

    index_c : Expr
        The fragment_c index.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin(
        ty,
        "tirx.cuda.bmma_sync",
        fragment_d,
        index_d,
        fragment_a,
        index_a,
        fragment_b,
        index_b,
        fragment_c,
        index_c,
        span=span,
    )


def mma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """Store the result of PTX MMA into a destination pointer."""

    return call_intrin(dtype, "tirx.mma_store", m, n, dst_ptr, src_ptr, src_offset, dst_stride)


_mma_store_legacy = _make_op_api(Op.get("tirx.mma_store_legacy"), __name__)


def mma_store_legacy(m, n, dst_ptr, src_ptr, src_offset, dst_stride, *, ty=None, span=None):
    """Store MMA registers using explicit pointer and element-offset operands."""
    return _mma_store_legacy(
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
        ty=ty,
        span=span,
    )


def mma_fill(dtype, local_size, local_ptr, offset):
    """Zero-initialize an MMA accumulation register."""

    return call_intrin(dtype, "tirx.mma_fill", local_size, local_ptr, offset)


_mma_fill_legacy = _make_op_api(Op.get("tirx.mma_fill_legacy"), __name__)


def mma_fill_legacy(local_size, local_ptr, offset, *, ty=None, span=None):
    """Initialize MMA registers using an explicit element offset."""
    return _mma_fill_legacy(local_size, local_ptr, offset, ty=ty, span=span)


_PTX_TO_NUMPY_DTYPE = {
    "fp16": "float16",
    "fp32": "float32",
    "fp64": "float64",
    "bf16": "bfloat16",
    "tf32": "float32",
    "s8": "int8",
    "u8": "uint8",
    "s32": "int32",
    "s4": "int4",
    "u4": "uint4",
    "b1": "int1",
    "b16": "uint16",
    "e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
}


def _ptx_to_numpy_dtype(dtype_str):
    """Map a PTX-abbreviation or numpy dtype string to a numpy dtype string
    suitable for ``access_ptr`` (which scales the offset by the element
    bit width). Unknown strings pass through unchanged so a caller may also
    pass an already-numpy dtype."""
    s = dtype_str if isinstance(dtype_str, str) else str(dtype_str)
    return _PTX_TO_NUMPY_DTYPE.get(s, s)


def _wrap_or_fold_access_ptr(ptr, offset, elem_dtype):
    """Wrap ``ptr`` with ``access_ptr`` unless it already is one.

    Several s_tir tensor intrinsics already pass ``buffer.access_ptr(...)``
    (an ``access_ptr`` Call) for the pointer argument. Naively wrapping
    that again yields a nested ``access_ptr(... access_ptr(...) ...)``
    whose ``args[0]`` is a Call rather than a Var, which crashes the
    lowering rule (Downcast<Var> at intrin_rule.cc) and several s_tir
    passes that assume a raw buffer var. Detect that case and fold the
    outer offset into the inner one.
    """

    is_access_ptr_call = (
        isinstance(ptr, Call) and isinstance(ptr.op, Op) and ptr.op.name == "tirx.access_ptr"
    )
    if is_access_ptr_call:
        # Inner Call already wraps the buffer var. Reuse its inner var and
        # inner access element type, and add the
        # outer offset (which is in `elem_dtype` units, same convention as
        # the inner since both come from the same buffer).
        inner_args = ptr.args
        inner_var = inner_args[0]
        inner_offset = inner_args[1]
        rw_mask = inner_args[3]
        return Call(
            "tirx.access_ptr",
            [inner_var, inner_offset + offset, 1, rw_mask],
            ty=ptr.ty,
            attrs=ptr.attrs,
            ty_args=ptr.ty_args,
            span=ptr.span,
        )
    return access_ptr(elem_dtype, ptr, offset, 1, 1)


_legacy_ldmatrix = _make_op_api(Op.get("tirx.ptx_legacy.ldmatrix"), __name__)


def ptx_legacy_ldmatrix(
    trans,
    num,
    dtype,
    local_ptr,
    local_offset,
    smem_ptr,
    smem_offset,
    *,
    ty=None,
    span=None,
):
    """Load a matrix with explicit pointer and element-offset operands.

    The legacy lowering uses the result ``ty`` for its element width, including
    the transposed int8 gather form. ``dtype`` is the PTX instruction type token.
    """
    return _legacy_ldmatrix(
        trans,
        num,
        dtype,
        local_ptr,
        local_offset,
        smem_ptr,
        smem_offset,
        ty=ty,
        span=span,
    )


@tvm_ffi.register_object("tirx.cuda.TCGen05InstrDescriptorAttrs")
class TCGen05InstrDescriptorAttrs(Attrs):
    """Static options for the dense tcgen05 instruction descriptor."""


@tvm_ffi.register_object("tirx.cuda.TCGen05InstrDescriptorBlockScaledAttrs")
class TCGen05InstrDescriptorBlockScaledAttrs(Attrs):
    """Static options for the block-scaled tcgen05 instruction descriptor."""


_encode_instr_descriptor = _make_op_api(
    Op.get("tirx.cuda.tcgen05_encode_instr_descriptor"), __name__
)
_encode_instr_descriptor_block_scaled = _make_op_api(
    Op.get("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled"), __name__
)


def cuda_tcgen05_encode_instr_descriptor(
    desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_groups=1,
    neg_a=False,
    neg_b=False,
    sat_d=False,
    is_sparse=False,
):
    """TVM intrinsic to create instruction descriptor for tcgen05 MMA without block scaling

    Parameters
    ----------
    desc : Expr
        The pointer to the instruction descriptor.

    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    M : int
        The size of non-reduction dimension of Matrix A.

    N : int
        The size of non-reduction dimension of Matrix B.

    K : int
        The size of reduction dimension of Matrix A/B.

    trans_a : bool
        Whether the multiplicand matrix A is transposed.
        True for M/N major, False for K major.

    trans_b : bool
        Whether the multiplicand matrix B is transposed.
        True for M/N major, False for K major.

    n_cta_groups : int
        The number of CTA groups involved in the MMA operation.

    neg_a : bool
        Whether to negate the multiplicand matrix A.

    neg_b : bool
        Whether to negate the multiplicand matrix B.

    sat_d : bool
        Whether to saturate the resultant matrix D.

    is_sparse : bool
        Whether the MMA operation is sparse.
    """
    _choice("n_cta_groups", n_cta_groups, _TCGEN05_CTA_GROUP)
    return _encode_instr_descriptor(
        desc,
        d_dtype=d_dtype,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        M=M,
        N=N,
        K=K,
        trans_a=trans_a,
        trans_b=trans_b,
        n_cta_groups=n_cta_groups,
        neg_a=neg_a,
        neg_b=neg_b,
        sat_d=sat_d,
        is_sparse=is_sparse,
    )


def cuda_tcgen05_encode_instr_descriptor_block_scaled(
    desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    M,
    N,
    K,
    trans_a,
    trans_b,
    n_cta_groups=1,
    neg_a=False,
    neg_b=False,
    is_sparse=False,
):
    """TVM intrinsic to create instruction descriptor for tcgen05 MMA with block scaling

    Parameters
    ----------
    desc : Expr
        The pointer to the instruction descriptor.

    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    sfa_dtype : str
        The datatype of scale factor matrix A.

    sfb_dtype : str
        The datatype of scale factor matrix B.

    M : int
        The size of non-reduction dimension of Matrix A.

    N : int
        The size of non-reduction dimension of Matrix B.

    K : int
        The size of reduction dimension of Matrix A/B.

    trans_a : bool
        Whether the multiplicand matrix A is transposed.
        True for M/N major, False for K major.

    trans_b : bool
        Whether the multiplicand matrix B is transposed.
        True for M/N major, False for K major.

    n_cta_groups : int
        The number of CTA groups involved in the MMA operation.

    neg_a : bool
        Whether to negate the multiplicand matrix A.

    neg_b : bool
        Whether to negate the multiplicand matrix B.

    is_sparse : bool
        Whether the MMA operation is sparse.
    """
    _choice("n_cta_groups", n_cta_groups, _TCGEN05_CTA_GROUP)
    return _encode_instr_descriptor_block_scaled(
        desc,
        d_dtype=d_dtype,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sfa_dtype=sfa_dtype,
        sfb_dtype=sfb_dtype,
        M=M,
        N=N,
        K=K,
        trans_a=trans_a,
        trans_b=trans_b,
        n_cta_groups=n_cta_groups,
        neg_a=neg_a,
        neg_b=neg_b,
        is_sparse=is_sparse,
    )


def _choice(name: str, value, options):
    """Validate `value` is one of `options`. Raise a clear ValueError otherwise.

    Symbolic values (Var, non-constant Expr) are accepted without
    validation; specialization later replaces them with concrete values
    that the C-side intrinsic body re-checks.
    """
    if isinstance(value, str):
        concrete = value
    elif isinstance(value, StringImm):
        concrete = value.value
    else:
        # Concrete int / IntImm value: validate.
        try:
            concrete = int(value)
        except (TypeError, ValueError):
            return  # symbolic; defer check
    if concrete not in options:
        raise ValueError(f"invalid {name}={concrete!r}; expected one of {tuple(options)}")


def _static_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, StringImm):
        return value.value
    return None


# See top-of-file imports for `_FENCE_SEM` etc. (re-exported from _common).
# Note: TCGEN05_LDST_SHAPES values must stay in sync with the tcgen05.ld/.st
# shape tokens in backend/cuda/ptx/table.py.


def timer_init_cuda(
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    num_groups,
    group_id,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for initializing the CUDA profiler, and store profiling result in a buffer.

    Parameters
    ----------
    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    num_groups: int
        The number of groups in the profiler.

    group_id: Expr
        The group id of the current thread.

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call(
        "tirx.timer_init_cuda",
        [profiler_buffer, profiler_tag, profiler_write_offset, num_groups, group_id],
        ty=ty,
        span=span,
    )


def timer_start_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for starting the timer for profiling a specific event, and storing profiling result in a buffer.

    Parameters
    ----------
    event_type: Enum
        The event to profile.

    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    profiler_write_stride: int
        The stride to advance in buffer in the next write.

    leader_cond: Expr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.timer_start_cuda",
        [
            event_type.value if isinstance(event_type, Enum) else event_type,
            profiler_buffer,
            profiler_tag,
            profiler_write_offset,
            profiler_write_stride,
            leader_cond,
        ],
        ty=ty,
        span=span,
    )


def timer_end_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for ending the timer for profiling a specific event, and storing profiling result in a buffer.

    Parameters
    ----------
    event_type: Enum
        The event to profile.

    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    profiler_write_stride: int
        The stride to advance in buffer in the next write.

    leader_cond: Expr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.timer_end_cuda",
        [
            event_type.value if isinstance(event_type, Enum) else event_type,
            profiler_buffer,
            profiler_tag,
            profiler_write_offset,
            profiler_write_stride,
            leader_cond,
        ],
        ty=ty,
        span=span,
    )


def timer_finalize_cuda(
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
    *,
    ty=None,
    span=None,
):
    """TVM intrinsic for finalizing the CUDA profiler, and store profiling result in a buffer.

    Parameters
    ----------
    profiler_buffer: Var
        The buffer to store the profiling result.

    profiler_tag: Var
        Buffer of length 1 storing the base tag of the current thread.

    profiler_write_offset: Var
        Buffer of length 1 storing the offset in buffer to write the next
        profiling result for the current thread.

    profiler_write_stride: int
        The stride to advance in buffer in the next write.

    leader_cond: Expr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call(
        "tirx.timer_finalize_cuda",
        [profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond],
        ty=ty,
        span=span,
    )


def cuda_atomic_add(res_addr, value, *, ty=None, span=None):
    """TVM intrinsic to call cuda atomic add instruction

    Parameters
    ----------
    res_addr : Expr
        The result address.

    value: Expr
        The value to add.

    Returns
    -------
    call : Expr
        The call expression.
    """
    value = tir.convert(value)
    return Call(
        "tirx.cuda.atomic_add",
        [res_addr, value],
        ty=ty,
        span=span,
    )


def cuda_sm100_2sm_leader_smem_addr(ptr):
    """Return the SM100 2SM leader CTA shared-address operand.

    The input is a generic pointer to shared memory.
    """
    return bitwise_and(globals()["cvta_generic_to_shared"](ptr), const(0xFEFFFFFF, dtype="uint32"))


_PTX_CVT_TYPES = {
    "u8",
    "u16",
    "u32",
    "u64",
    "s8",
    "s16",
    "s32",
    "s64",
    "bf16",
    "f16",
    "f32",
    "f64",
    "f16x2",
    "bf16x2",
    "tf32",
    "e4m3x2",
    "e5m2x2",
    "e2m1x2",
    "e2m3x2",
    "e3m2x2",
    "e4m3x4",
    "e5m2x4",
    "e2m1x4",
    "e2m3x4",
    "e3m2x4",
    "ue8m0x2",
    "s2f6x2",
}
_PTX_CVT_ROUNDING = {"", "rni", "rzi", "rmi", "rpi", "rn", "rz", "rm", "rp", "rna", "rs"}
_PTX_CVT_SCALED = {"", "n2::ue8m0"}
_PTX_CVT_RETURN_TYPE = {
    "u8": "uint8",
    "s8": "int8",
    "u16": "uint16",
    "s16": "int16",
    "u32": "uint32",
    "s32": "int32",
    "u64": "uint64",
    "s64": "int64",
    "f32": "float32",
    "f64": "float64",
    "f16": "uint16",
    "bf16": "uint16",
    "e4m3x2": "uint16",
    "e5m2x2": "uint16",
    "e2m1x2": "uint8",
    "e2m3x2": "uint16",
    "e3m2x2": "uint16",
    "ue8m0x2": "uint16",
    "s2f6x2": "uint16",
    "tf32": "uint32",
    "f16x2": "uint32",
    "bf16x2": "uint32",
    "e2m1x4": "uint16",
    "e4m3x4": "uint32",
    "e5m2x4": "uint32",
    "e2m3x4": "uint32",
    "e3m2x4": "uint32",
}


_PTX_CACHE_POLICY = {
    "evict_normal": 0x1000000000000000,
    "evict_first": 0x12F0000000000000,
    "evict_last": 0x14F0000000000000,
}


def _resolve_cache_policy(cache_hint, cache_policy, choices=_CP_ASYNC_BULK_CACHE_HINT):
    _choice("cache_hint", cache_hint, choices)
    if cache_policy is not None:
        return cache_policy, True
    if cache_hint:
        if cache_hint not in _PTX_CACHE_POLICY:
            raise ValueError(
                f"Unsupported built-in cache policy {cache_hint!r}; pass cache_policy explicitly"
            )
        return const(_PTX_CACHE_POLICY[cache_hint], dtype="uint64"), True
    return const(0, dtype="uint64"), False


def _ptx_vec_len(vec):
    return int(vec[1:]) if vec else 1


def _normalize_ptx_ld_dst(dst, vec, op_name):
    if dst is None:
        if vec:
            raise ValueError(f"vec {op_name} requires dst")
        return [], 0
    if isinstance(dst, list | tuple):
        if not vec:
            raise ValueError(f"{op_name} scatter dst requires vec")
        vec_len = _ptx_vec_len(vec)
        if len(dst) != vec_len:
            raise ValueError(f"{op_name} scatter dst length must match {vec}: got {len(dst)}")
        return list(dst), vec_len
    return [dst], 1


def _validate_ptx_address(addr, space, op_name):
    """Validate pointer and raw shared-memory address forms."""
    addr_ty = getattr(addr, "ty", None)
    if isinstance(addr_ty, PointerType):
        return
    if isinstance(addr_ty, PrimType):
        if addr_ty.dtype == "uint32":
            if not str(space).startswith("shared"):
                raise ValueError(f"{op_name} uint32 address requires shared state space")
            return
        if addr_ty.dtype.startswith(("int", "uint")):
            raise ValueError(
                f"{op_name} integer address must be uint32 in shared state space, "
                f"got {addr_ty.dtype}"
            )


def cuda_atomic_cas(ptr, old_val, new_val, *, ty=None, span=None):
    """TVM intrinsic to call cuda atomic cas instruction

    Parameters
    ----------
    ptr: Expr
        The pointer to the memory location.

    old_val: Expr
        The old value.

    new_val: Expr
        The new value.

    Returns
    -------
    call : Expr
        The call expression.
    """
    old_val = tir.convert(old_val)
    return Call(
        "tirx.cuda.atomic_cas",
        [ptr, old_val, new_val],
        ty=ty,
        span=span,
    )


########################################################
# NVSHMEM builtins
########################################################


def nvshmem_my_pe(*, ty=None, span=None):
    """TVM intrinsic to call nvshmem_my_pe()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call("tirx.nvshmem.my_pe", [], ty=ty, span=span)


def nvshmem_n_pes(*, ty=None, span=None):
    """TVM intrinsic to call nvshmem_n_pes()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call("tirx.nvshmem.n_pes", [], ty=ty, span=span)


def nvshmem_getmem_nbi(dst, src, nelems, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_getmem_nbi()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: Expr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per thread.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.nvshmem.getmem_nbi",
        [dst, src, nelems, pe],
        ty=ty,
        span=span,
    )


def nvshmem_putmem_nbi(dst, src, nelems, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_putmem_nbi()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address of the destination data object.

    src: Expr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per thread.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call(
        "tirx.nvshmem.putmem_nbi",
        [dst, src, nelems, pe],
        ty=ty,
        span=span,
    )


def nvshmem_getmem_nbi_warp(dst, src, nelems, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_getmem_nbi_warp()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: Expr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per warp.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.nvshmem.getmem_nbi_warp",
        [dst, src, nelems, pe],
        ty=ty,
        span=span,
    )


def nvshmem_putmem_nbi_warp(dst, src, nelems, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_putmem_nbi_warp()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address of the destination data object.

    src: Expr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per warp.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call(
        "tirx.nvshmem.putmem_nbi_warp",
        [dst, src, nelems, pe],
        ty=ty,
        span=span,
    )


def nvshmem_getmem_nbi_block(dst, src, nelems, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_getmem_nbi_block()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: Expr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per block.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.nvshmem.getmem_nbi_block",
        [dst, src, nelems, pe],
        ty=ty,
        span=span,
    )


def nvshmem_putmem_nbi_block(dst, src, nelems, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_putmem_nbi_block()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address of the destination data object.

    src: Expr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per block.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call(
        "tirx.nvshmem.putmem_nbi_block",
        [dst, src, nelems, pe],
        ty=ty,
        span=span,
    )


def nvshmem_signal_op(sig_addr, signal, sig_op, pe, *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_signal_op()

    Parameters
    ----------
    sig_addr: Expr
        The pointer to the symmetric address of the signal word to be updated, must be uint64_t*.

    signal: uint64_t
        The value used to update sig_addr.

    sig_op: str
        Operation used to update sig_addr with signal, typical sig_op values are "set" and "add".

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """

    _choice("sig_op", sig_op, _NVSHMEM_SIG_OP)
    return Call(
        "tirx.nvshmem.signal_op",
        [sig_addr, signal, sig_op, pe],
        ty=ty,
        span=span,
    )


def nvshmem_wait_until(ivar, cmp, cmp_value, type="uint64_t", *, ty=None, span=None):
    """TVM intrinsic to call nvshmem_wait_until()

    Parameters
    ----------
    ivar: Expr
        The pointer to the symmetric address of a remotely accessible data object, must be TYPE*.

    cmp: str
        The compare operator that compares ivar with cmp_value.

    cmp_value: TYPE
        The value to be compared with ivar.

    type: str
        The TYPE of ivar and cmp_value.

    Returns
    -------
    call : Expr
        The call expression.
    """

    _choice("cmp", cmp, _NVSHMEM_CMP)
    return Call(
        "tirx.nvshmem.wait_until",
        [ivar, cmp, cmp_value, type],
        ty=ty,
        span=span,
    )


def nvshmem_quiet(*, ty=None, span=None):
    """TVM intrinsic to call nvshmem_quiet()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call("tirx.nvshmem.quiet", [], ty=ty, span=span)


def nvshmem_putmem_signal_nbi(
    dst, src, nelems, sig_addr, signal, sig_op, pe, *, ty=None, span=None
):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: Expr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per thread.

    sig_addr: Expr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.nvshmem.putmem_signal_nbi",
        [dst, src, nelems, sig_addr, signal, sig_op, pe],
        ty=ty,
        span=span,
    )


def nvshmem_putmem_signal_nbi_warp(
    dst, src, nelems, sig_addr, signal, sig_op, pe, *, ty=None, span=None
):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi_warp()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: Expr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per warp.

    sig_addr: Expr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.nvshmem.putmem_signal_nbi_warp",
        [dst, src, nelems, sig_addr, signal, sig_op, pe],
        ty=ty,
        span=span,
    )


def nvshmem_putmem_signal_nbi_block(
    dst, src, nelems, sig_addr, signal, sig_op, pe, *, ty=None, span=None
):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi_block()

    Parameters
    ----------
    dst: Expr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: Expr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per block.

    sig_addr: Expr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : Expr
        The call expression.
    """  # noqa: E501

    return Call(
        "tirx.nvshmem.putmem_signal_nbi_block",
        [dst, src, nelems, sig_addr, signal, sig_op, pe],
        ty=ty,
        span=span,
    )


def nvshmem_fence(*, ty=None, span=None):
    """TVM intrinsic to call nvshmem_fence()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call("tirx.nvshmem.fence", [], ty=ty, span=span)


def nvshmem_barrier_all(*, ty=None, span=None):
    """TVM intrinsic to call nvshmem_barrier_all()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return Call("tirx.nvshmem.barrier_all", [], ty=ty, span=span)


def _op_api_factory(op, module_name):
    from tvm.tirx.op import _make_registered_op_api

    return _make_registered_op_api(op, module_name)


# Canonical Op builders also supply the historical direct-import aliases.


@tvm_ffi.register_object("tirx.cuda.TensorMapEncodeTiledAttr")
class TensorMapEncodeTiledAttr(Attrs):
    """Descriptor dtype and fixed options for tiled tensor-map encoding."""

    def __init__(
        self,
        descriptor_dtype,
        rank,
        interleave=0,
        swizzle=0,
        l2_promotion=0,
        oob_fill=0,
        force_cu_dtype=-1,
    ):
        self.__init_handle_by_constructor__(
            tvm_ffi.get_global_func("tirx.cuda.TensorMapEncodeTiledAttr"),
            descriptor_dtype,
            rank,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
            force_cu_dtype,
        )


def tensormap_encode_tiled(
    *args,
    descriptor_dtype,
    rank,
    interleave=0,
    swizzle=0,
    l2_promotion=0,
    oob_fill=0,
    force_cu_dtype=-1,
    span=None,
):
    """Encode a tiled tensor map using runtime pointers and shape operands.

    Arguments are the descriptor and data pointers, global dimensions (rank),
    byte strides (rank - 1), box dimensions (rank), and element strides (rank).
    The dtype describes the final descriptor units, including any promotion.
    CUDA-host codegen encodes directly; other hosts use the runtime packed call.
    """
    if not 1 <= rank <= 5 or len(args) != 4 * rank + 1:
        raise ValueError("tensormap_encode_tiled requires rank 1..5 and 4 * rank + 1 operands")
    return Call(
        "tirx.cuda.tensormap_encode_tiled",
        args,
        attrs=TensorMapEncodeTiledAttr(
            descriptor_dtype, rank, interleave, swizzle, l2_promotion, oob_fill, force_cu_dtype
        ),
        ty="int32",
        span=span,
    )


_init_op_api("tirx.cuda", __name__)
for _name in Op.list_op_names():
    if _name.startswith("tirx.cuda."):
        _suffix = _name.removeprefix("tirx.cuda.")
        if "." not in _suffix:
            globals().setdefault("cuda_" + _suffix, globals()[_suffix])
