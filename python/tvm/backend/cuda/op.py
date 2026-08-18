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

from tvm import tirx
from tvm.ir import Call, Op
from tvm.ir.type import PointerType, PrimType
from tvm.runtime import const
from tvm.tirx.op import bitwise_and, call_intrin, tvm_access_ptr
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


def cuda_iket_range_pop():
    """Create an NVIDIA IKET stack-range pop annotation."""
    return call_intrin("", "tirx.cuda.iket_range_pop")


def cuda_iket_sentinel_token(name):
    """Create a no-op NVIDIA IKET range token for warp-uniform control flow."""
    return call_intrin("uint32", "tirx.cuda.iket_sentinel_token", name)


def cuda_iket_official_event(event_id, source_code="", payload=None):
    """Create an NVIDIA IKET official range-end event."""
    if payload is not None:
        return call_intrin(
            "uint32", "tirx.cuda.iket_official_event", event_id, source_code, payload
        )
    return call_intrin("uint32", "tirx.cuda.iket_official_event", event_id, source_code)


def cuda_func_call(func_name, *args, source_code, return_type="void"):
    """TVM intrinsic to call a CUDA function. Source code is provided as a string.

    Parameters
    ----------
    func_name: str
        The name of the CUDA function.

    args: Expr
        The arguments to the CUDA function.

    source_code: str
        The source code of the CUDA function.

    return_type: str
        The return type of the CUDA function.
    """
    return call_intrin(return_type, "tirx.cuda.func_call", func_name, *args, source_code)


def cuda_warp_reduce(value, op, width=32):
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
    return call_intrin(value.ty, "tirx.cuda.warp_reduce", value, op, width)


def cuda_warp_sum(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "sum", width)``."""
    return cuda_warp_reduce(value, "sum", width)


def cuda_warp_max(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "max", width)``."""
    return cuda_warp_reduce(value, "max", width)


def cuda_warp_min(value, width=32):
    """Convenience wrapper: ``cuda_warp_reduce(value, "min", width)``."""
    return cuda_warp_reduce(value, "min", width)


def cuda_cta_reduce(value, op, num_warps, scratch):
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
    return call_intrin(value.ty, "tirx.cuda.cta_reduce", value, op, num_warps, scratch)


def cuda_cta_sum(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "sum", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "sum", num_warps, scratch)


def cuda_cta_max(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "max", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "max", num_warps, scratch)


def cuda_cta_min(value, num_warps, scratch):
    """Convenience wrapper: ``cuda_cta_reduce(value, "min", num_warps, scratch)``."""
    return cuda_cta_reduce(value, "min", num_warps, scratch)


def cuda_warp_sync():
    """TVM intrinsic to synchronize threads within the current warp.

    This lowers to a CUDA `__syncwarp()` call.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.warp_sync")


def cuda_cta_sync():
    """TVM intrinsic to call CUDA syncthreads (block-wide barrier)

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.cta_sync")


def cuda_grid_sync():
    """TVM intrinsic to call CUDA grid-wide sync (cooperative groups)

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.grid_sync")


def cuda_cluster_sync():
    """TVM intrinsic to call CUDA cluster-wide barrier sync

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.cluster_sync")


def cuda_thread_rank():
    """TVM intrinsic that returns ``cooperative_groups::thread_rank()``
    for the enclosing CTA -- the linear thread index within the block.

    Useful for building "single thread of CTA" predicates without
    referencing user-declared scope_id vars. For example, the idiomatic
    mbarrier.init leader predicate is::

        T.cuda.thread_rank() == 0

    Returns
    -------
    call : Expr
        The call expression (``int32``).
    """
    return call_intrin("int32", "tirx.cuda.thread_rank")


def cuda_half2float(src):
    """TVM intrinsic to convert half to float

    Parameters
    ----------
    src : Expr
        Source pointer.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("float32", "tirx.cuda.half2float", src)


def cuda_bfloat162float(src):
    """TVM intrinsic to convert bfloat16 to float

    Parameters
    ----------
    src : Expr
        Source pointer.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("float32", "tirx.cuda.bfloat162float", src)


def cuda_float22half2(dst, src):
    """TVM intrinsic to convert float2 to half2 with rounding

    Parameters
    ----------
    dst : Expr
        Destination pointer.

    src : Expr
        Source pointer.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.float22half2", dst, src)


def cuda_trap_when_assert_failed(cond):
    """TVM intrinsic to trap when assertion failed (cond == false)

    Parameters
    ----------
    cond : Expr
        Condition to check.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.trap_when_assert_failed", cond)


def cuda_runtime_instr_desc(desc, sf_id):
    """TVM intrinsic to update runtime instruction descriptor

    Parameters
    ----------
    desc : Expr
        Pointer to the descriptor (uint32*).

    sf_id : Expr
        The subfragment id.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.runtime_instr_desc", desc, sf_id)


def cuda_half8tofloat8(src_addr, dst_addr):
    """TVM intrinsic to convert 8 half2s to 8 float2s

    Parameters
    ----------
    src_addr : Expr
        Source pointer.

    dst_addr : Expr
        Destination pointer.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.half8tofloat8", src_addr, dst_addr)


def cuda_float8tohalf8(src_addr, dst_addr):
    """TVM intrinsic to convert 8 float2s to 8 half2s

    Parameters
    ----------
    src_addr : Expr
        Source pointer.

    dst_addr : Expr
        Destination pointer.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.float8tohalf8", src_addr, dst_addr)


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


def cuda_mbarrier_wait(bar, phase):
    """Retry ``mbarrier.try_wait.parity.acquire.cta`` until it returns true.

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    phase : int
        The phase of the barrier.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.mbarrier_wait", bar, phase)


def cuda_mbarrier_wait_acquire_cluster(bar, phase):
    """``mbarrier.try_wait.parity.acquire.cluster`` retry loop.

    Cluster-scope acquire wait — used to wait on a barrier that a remote CTA in
    the cluster arrives on (a group cluster wait).

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    phase : int
        The phase of the barrier.
    """
    return call_intrin("", "tirx.cuda.mbarrier_wait_acquire_cluster", bar, phase)


def ptx_cp_async_legacy(*all_args):
    """Legacy ``ptx_cp_async`` API taking explicit src/dst offsets.

    Signature: ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)``.
    Offsets are folded into the pointers via ``tvm_access_ptr`` and the call
    lowers through the raw ``tirx.s_tir.cp_async_raw`` op.

    ``T.s_tir.cp_async_raw.legacy`` runs through ``_dtype_forward`` which
    prepends a ``dtype=`` kwarg as a leading positional. The dtype names
    the *element* type of the buffer (offsets are in elements of that
    dtype, not bytes), so this function accepts either 5 or 6 positional
    args.
    """
    args = list(all_args)
    elem_dtype = "int8"
    if len(args) == 6:
        # Leading positional is the buffer element dtype, used to scale
        # offsets correctly when folding via ``tvm_access_ptr``.
        elem_dtype = args.pop(0)
    if len(args) != 5:
        raise ValueError(
            f"ptx_cp_async_legacy expects 5 args (or 6 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
        )
    dst_ptr, dst_offset, src_ptr, src_offset, cp_size = args
    dst_ptr = _wrap_or_fold_access_ptr(dst_ptr, dst_offset, elem_dtype)
    src_ptr = _wrap_or_fold_access_ptr(src_ptr, src_offset, elem_dtype)
    # The raw 5-arg Call InjectPTXAsyncCopy emits; offsets are already folded.
    return call_intrin(elem_dtype, "tirx.s_tir.cp_async_raw", dst_ptr, 0, src_ptr, 0, cp_size)


def _is_static_unicast_cta_mask(cta_mask):
    if isinstance(cta_mask, int):
        return cta_mask == 0 or cta_mask & (cta_mask - 1) == 0
    if isinstance(cta_mask, tirx.IntImm):
        value = int(cta_mask)
        return value == 0 or value & (value - 1) == 0
    return False


def cuda_elect_sync():
    """TVM intrinsic to call elect.sync"""
    return call_intrin("uint32", "tirx.cuda.elect_sync")


def cuda_mov_sreg(bits, reg_name):
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
    return call_intrin("int" + str(bits), "tirx.cuda.mov_sreg", bits, reg_name)


def ptx_legacy_mma(*all_args, operator=None):
    """Legacy ``ptx_mma`` API.

    Signature: ``(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
    multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
    c_index, saturate, operator=None)``. The accumulator is reused as
    both input and output (no separate ``d``/``c`` slot). Translation:

    * ``a_dtype, b_dtype, c_dtype`` → fork ``a_type, b_type, c_type``
      (and reuse ``c_dtype`` as fork ``d_type`` since the accumulator
      dtype is the output dtype here).
    * ``(a_ptr, a_offset)`` and ``(b_ptr, b_offset)`` → folded via
      :func:`tvm_access_ptr`.
    * ``(accumulator, c_index)`` → folded; passed for both ``d_ptr`` and
      ``c_ptr`` since the accumulator is reused as the output.

    ``T.ptx_legacy.mma`` runs through ``_dtype_forward`` which prepends a
    ``dtype=`` kwarg as a leading positional, so this function accepts
    either 13 or 14 positional args.
    """
    args = list(all_args)
    # ``T.ptx_legacy.mma(..., dtype="...")`` has the dtype prepended by
    # ``_dtype_forward``; strip it here.
    if len(args) in (14, 15):
        _ = args.pop(0)
    if len(args) == 14:
        # operator passed positionally as the trailing arg.
        operator = args.pop()
    if len(args) != 13:
        raise ValueError(
            f"ptx_legacy_mma expects 13-15 positional args (with optional "
            f"leading ``call_dtype`` from dtype= kwarg and optional trailing "
            f"``operator``); got {len(all_args)}"
        )
    (
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
    ) = args
    # Emit tirx.ptx_legacy.mma directly with separate (ptr_var, offset)
    # pairs. codegen_cuda.cc uses C pointer arithmetic ``ptr + offset``
    # so element offsets stay element-accurate, and lower_warp_memory
    # rewrites the offset's group component to a thread-local index.
    call_args = [
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
        call_args.append(operator)
    return call_intrin("", "tirx.ptx_legacy.mma", *call_args)


def mma_store(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """Store the result of PTX MMA into a destination pointer."""

    return call_intrin(dtype, "tirx.mma_store", m, n, dst_ptr, src_ptr, src_offset, dst_stride)


def mma_store_legacy(dtype, m, n, dst_ptr, src_ptr, src_offset, dst_stride):
    """mma_store with apache-style pointer/offset semantics."""

    return call_intrin(
        dtype,
        "tirx.mma_store_legacy",
        m,
        n,
        dst_ptr,
        src_ptr,
        src_offset,
        dst_stride,
    )


def mma_fill(dtype, local_size, local_ptr, offset):
    """Zero-initialize an MMA accumulation register."""

    return call_intrin(dtype, "tirx.mma_fill", local_size, local_ptr, offset)


def mma_fill_legacy(dtype, local_size, local_ptr, offset):
    """mma_fill with apache-style pointer/offset semantics."""

    return call_intrin(dtype, "tirx.mma_fill_legacy", local_size, local_ptr, offset)


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
    suitable for ``tvm_access_ptr`` (which scales the offset by the element
    bit width). Unknown strings pass through unchanged so a caller may also
    pass an already-numpy dtype."""
    s = dtype_str if isinstance(dtype_str, str) else str(dtype_str)
    return _PTX_TO_NUMPY_DTYPE.get(s, s)


def _wrap_or_fold_access_ptr(ptr, offset, elem_dtype):
    """Wrap ``ptr`` with ``tvm_access_ptr`` unless it already is one.

    Several s_tir tensor intrinsics already pass ``buffer.access_ptr(...)``
    (an ``tvm_access_ptr`` Call) for the pointer argument. Naively wrapping
    that again yields a nested ``tvm_access_ptr(... access_ptr(...) ...)``
    whose ``args[1]`` is a Call rather than a Var, which crashes the
    lowering rule (Downcast<Var> at intrin_rule.cc) and several s_tir
    passes that assume a raw buffer var. Detect that case and fold the
    outer offset into the inner one.
    """

    is_access_ptr_call = (
        isinstance(ptr, Call) and isinstance(ptr.op, Op) and ptr.op.name == "tirx.tvm_access_ptr"
    )
    if is_access_ptr_call:
        # Inner Call already wraps the buffer var. Reuse its inner var and
        # inner element dtype (the marker type_annotation), and add the
        # outer offset (which is in `elem_dtype` units, same convention as
        # the inner since both come from the same buffer).
        inner_args = ptr.args
        inner_marker = inner_args[0]
        inner_var = inner_args[1]
        inner_offset = inner_args[2]
        rw_mask = inner_args[4]
        return call_intrin(
            ptr.ty,
            "tirx.tvm_access_ptr",
            inner_marker,
            inner_var,
            inner_offset + offset,
            1,
            rw_mask,
        )
    return tvm_access_ptr(elem_dtype, ptr, offset, 1, 1)


def ptx_legacy_ldmatrix(*all_args):
    """Legacy ``ptx_ldmatrix`` API taking explicit offsets.

    Signature: ``(trans, num, dtype, local_ptr, local_offset, smem_ptr,
    smem_offset)``. Offsets are folded into the pointers via
    ``tvm_access_ptr``.

    ``T.ptx_legacy.ldmatrix`` runs through ``_dtype_forward`` which
    prepends a ``dtype=`` kwarg as a leading positional naming the buffer
    element type — offsets are in elements of that dtype, not bytes, so
    we forward it to ``tvm_access_ptr`` for correct scaling.
    """
    if len(all_args) == 8:
        elem_dtype, trans, num, dtype, local_ptr, local_offset, smem_ptr, smem_offset = all_args
    elif len(all_args) == 7:
        trans, num, dtype, local_ptr, local_offset, smem_ptr, smem_offset = all_args
        elem_dtype = "int8"
    else:
        raise ValueError(
            f"ptx_legacy_ldmatrix expects 7 args (or 8 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
        )
    # Call.dtype carries the buffer element type so codegen can pick the
    # int8+trans manual-loop fallback (ldmatrix can't transpose int8).
    return call_intrin(
        elem_dtype,
        "tirx.ptx_legacy.ldmatrix",
        trans,
        num,
        dtype,
        local_ptr,
        local_offset,
        smem_ptr,
        smem_offset,
    )


def cuda_wgmma_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    """TVM intrinsic to create memory descriptor for wgmma instructions

    Parameters
    ----------
    desc : Expr
        The pointer to the shared memory descriptor.

    addr : Expr
        The address of the matrix.

    ldo : Expr
        The leading dimension offset.

    sdo : Expr
        The stride dimension offset.

    swizzle : int
        The swizzle value (CUtensorMapSwizzle_enum).
    """
    return call_intrin(
        "", "tirx.cuda.wgmma_encode_matrix_descriptor", desc, addr, ldo, sdo, swizzle
    )


def cuda_wgmma_noop_barrier(reg):
    """TVM intrinsic to call "" : "+{format}"(reg)::"memory"

    Parameters
    ----------
    reg : Expr
        The register to fence.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.wgmma_noop_barrier", reg)


def cuda_tcgen05_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    """TVM intrinsic to create memory descriptor for tcgen05 instructions

    Parameters
    ----------
    desc : Expr
        The pointer to the shared memory descriptor.

    addr : Expr
        The address of the matrix.

    ldo : Expr
        The leading dimension offset.

    sdo : Expr
        The stride dimension offset.

    swizzle : int
        The swizzle value (CUtensorMapSwizzle_enum).
    """
    return call_intrin(
        "", "tirx.cuda.tcgen05_encode_matrix_descriptor", desc, addr, ldo, sdo, swizzle
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
    return call_intrin(
        "",
        "tirx.cuda.tcgen05_encode_instr_descriptor",
        desc,
        d_dtype,
        a_dtype,
        b_dtype,
        M,
        N,
        K,
        trans_a,
        trans_b,
        n_cta_groups,
        neg_a,
        neg_b,
        sat_d,
        is_sparse,
    )


def cuda_tcgen05_encode_instr_descriptor_block_scaled(
    desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    sfa_tmem_addr,
    sfb_tmem_addr,
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

    sfa_tmem_addr : Expr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : Expr
        The address of the scale factor matrix B in tensor memory, should be uint32_t.

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
    return call_intrin(
        "",
        "tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled",
        desc,
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        sfa_tmem_addr,
        sfb_tmem_addr,
        M,
        N,
        K,
        trans_a,
        trans_b,
        n_cta_groups,
        neg_a,
        neg_b,
        is_sparse,
    )


def _choice(name: str, value, options):
    """Validate `value` is one of `options`. Raise a clear ValueError otherwise.

    Symbolic values (Var, non-constant Expr) are accepted without
    validation; specialization later replaces them with concrete values
    that the C-side intrinsic body re-checks.
    """
    if isinstance(value, str):
        concrete = value
    elif isinstance(value, tirx.StringImm):
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
    if isinstance(value, tirx.StringImm):
        return value.value
    return None


# See top-of-file imports for `_FENCE_SEM` etc. (re-exported from _common).
# Note: TCGEN05_LDST_SHAPES values must stay in sync with the tcgen05.ld/.st
# shape tokens in backend/cuda/ptx/table.py.


def timer_init_cuda(profiler_buffer, profiler_tag, profiler_write_offset, num_groups, group_id):
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

    return call_intrin(
        "void",
        "tirx.timer_init_cuda",
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        num_groups,
        group_id,
    )


def timer_start_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
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

    return call_intrin(
        "void",
        "tirx.timer_start_cuda",
        event_type.value,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
    )


def timer_end_cuda(
    event_type,
    profiler_buffer,
    profiler_tag,
    profiler_write_offset,
    profiler_write_stride,
    leader_cond,
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

    return call_intrin(
        "void",
        "tirx.timer_end_cuda",
        event_type.value,
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
    )


def timer_finalize_cuda(
    profiler_buffer, profiler_tag, profiler_write_offset, profiler_write_stride, leader_cond
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

    return call_intrin(
        "void",
        "tirx.timer_finalize_cuda",
        profiler_buffer,
        profiler_tag,
        profiler_write_offset,
        profiler_write_stride,
        leader_cond,
    )


def cuda_atomic_add(res_addr, value):
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
    return call_intrin(value.ty, "tirx.cuda.atomic_add", res_addr, value)


def cuda_thread_fence():
    """TVM intrinsic to call cuda thread fence instruction

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.thread_fence")


def cuda_warpgroup_sync(bar_no):
    """TVM intrinsic to synchronize a CUDA warpgroup via a named barrier.

    Parameters
    ----------
    bar_no : Expr
        The named barrier id to use for the warpgroup.

    Notes
    -----
    Synchronizes 128 threads in a warpgroup using `bar.sync bar_no, 128`.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.warpgroup_sync", bar_no)


def cuda_syncthreads_and(cond):
    """TVM intrinsic to call cuda syncthreads_and instruction

    Parameters
    ----------
    cond: Expr
        The condition.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("int64", "tirx.cuda.syncthreads_and", cond)


def cuda_syncthreads_or(cond):
    """TVM intrinsic to call cuda syncthreads_or instruction

    Parameters
    ----------
    cond: Expr
        The condition.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("int64", "tirx.cuda.syncthreads_or", cond)


def cuda_nano_sleep(time):
    """TVM intrinsic to call cuda nano sleep instruction

    Parameters
    ----------
    time: Expr
        The time to sleep.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.nano_sleep", time)


def cuda_printf(fmt, *args):
    """TVM intrinsic to call cuda printf instruction

    Parameters
    ----------
    fmt: str
        The format string.

    *args: list
        The arguments to the format string.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.printf", fmt, *args)


def cuda_ldg(addr, dtype, *, dst=None, vec=""):
    """TVM intrinsic to call CUDA C++ ``__ldg()``.

    Parameters
    ----------
    addr : Expr
        The memory address to load.

    dtype : str
        The data type of the loaded value.

    dst : Expr or tuple[Expr], optional
        Destination pointers for vector loads.

    vec : str
        CUDA vector width. Use ``"v2"`` or ``"v4"`` together with tuple/list
        ``dst``.

    Returns
    """
    if dst is None:
        if vec:
            raise ValueError("vector cuda.ldg requires dst")
        return call_intrin(dtype, "tirx.cuda.ldg", addr, dtype)
    if vec not in ("v2", "v4"):
        raise ValueError(f"vector cuda.ldg expects vec in {{'v2', 'v4'}}, got {vec!r}")
    if not isinstance(dst, list | tuple):
        raise ValueError("vector cuda.ldg requires tuple/list dst")
    vec_len = int(vec[1:])
    if len(dst) != vec_len:
        raise ValueError(f"cuda.ldg dst length must match {vec}: got {len(dst)}")
    return call_intrin("", "tirx.cuda.ldg", *dst, addr, dtype, vec, vec_len)


def cuda_fdividef(x, y):
    """TVM intrinsic to call CUDA C++ ``__fdividef`` fast float division."""
    return call_intrin("float32", "tirx.cuda.fdividef", x, y)


def cuda_get_tmem_addr(addr, row_offset, col_offset):
    """TVM intrinsic to call cuda tmem address calculation

    Parameters
    ----------
    addr: Expr
        The memory address to calculate.

    row_offset: Expr
        The row offset to calculate.

    col_offset: Expr
        The column offset to calculate.

    Returns
    -------
    call : Expr
        The call expression.
    """
    return call_intrin("uint32", "tirx.cuda.get_tmem_addr", addr, row_offset, col_offset)


def cuda_cvta_generic_to_shared(ptr):
    """Convert a generic pointer to a shared-memory address (uint32).

    Wraps ``__cvta_generic_to_shared(ptr)``. Used by op-wrappers that
    precompute the shared-memory address at the wrapper layer instead of
    inside the asm helper body.
    """
    return call_intrin("uint32", "tirx.cuda.cvta_generic_to_shared", ptr)


def cuda_smem_addr_from_uint64(cluster_addr):
    """Narrow a 64-bit cluster-mapped SMEM address to a 32-bit SMEM address.

    Wraps ``static_cast<unsigned int>(cluster_addr)``. Used by
    cp.async.bulk.shared::cluster.* op-wrappers.
    """
    return call_intrin("uint32", "tirx.cuda.smem_addr_from_uint64", cluster_addr)


def cuda_sm100_2sm_leader_smem_addr(ptr):
    """Return the SM100 2SM leader CTA shared-address operand.

    The input is a generic pointer to shared memory.
    """
    return bitwise_and(cuda_cvta_generic_to_shared(ptr), const(0xFEFFFFFF, dtype="uint32"))


def cuda_any_sync(mask, pred):
    """TVM intrinsic for PTX warp-wide any predicate (__any_sync)

    Parameters
    ----------
    mask : Expr
        The thread mask (uint32).
    pred : Expr
        The predicate value (int32).

    Returns
    -------
    call : Expr
        The call expression returning 1 if any thread in mask has pred != 0.
    """
    return call_intrin("int32", "tirx.cuda.any_sync", mask, pred)


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


def cuda_uint_as_float(bits):
    return call_intrin("float32", "tirx.cuda.uint_as_float", bits)


def cuda_float_as_uint(x):
    return call_intrin("uint32", "tirx.cuda.float_as_uint", x)


def cuda_ballot_sync(mask, pred):
    return call_intrin("uint32", "tirx.cuda.ballot_sync", mask, pred)


def cuda_ffs_u32(value):
    return call_intrin("int32", "tirx.cuda.ffs_u32", value)


def cuda_reduce_add_sync_u32(mask, value):
    return call_intrin("uint32", "tirx.cuda.reduce_add_sync_u32", mask, value)


def cuda_reduce_min_sync_u32(mask, value):
    return call_intrin("uint32", "tirx.cuda.reduce_min_sync_u32", mask, value)


def cuda_clock64():
    return call_intrin("uint64", "tirx.cuda.clock64")


def cuda_make_float2(x, y):
    return call_intrin("uint64", "tirx.cuda.make_float2", x, y)


def cuda_float2_x(packed):
    return call_intrin("float32", "tirx.cuda.float2_x", packed)


def cuda_float2_y(packed):
    return call_intrin("float32", "tirx.cuda.float2_y", packed)


def cuda_fmul2_rn(a, b):
    return call_intrin("uint64", "tirx.cuda.fmul2_rn", a, b)


def cuda_fadd2_rn(a, b):
    return call_intrin("uint64", "tirx.cuda.fadd2_rn", a, b)


def cuda_float22bfloat162_rn(v0, v1):
    return call_intrin("uint32", "tirx.cuda.float22bfloat162_rn", v0, v1)


def cuda_float22bfloat162_rn_from_float2(packed):
    return call_intrin("uint32", "tirx.cuda.float22bfloat162_rn_from_float2", packed)


def cuda_bfloat1622float2(packed):
    return call_intrin("uint64", "tirx.cuda.bfloat1622float2", packed)


def cuda_hmin2(a, b):
    return call_intrin("uint32", "tirx.cuda.hmin2", a, b)


def cuda_hmax2(a, b):
    return call_intrin("uint32", "tirx.cuda.hmax2", a, b)


def cuda_fp8x4_e4m3_from_float4(x, y, z, w):
    return call_intrin("uint32", "tirx.cuda.fp8x4_e4m3_from_float4", x, y, z, w)


def cuda_atomic_cas(ptr, old_val, new_val):
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
    return call_intrin(old_val.ty, "tirx.cuda.atomic_cas", ptr, old_val, new_val)


########################################################
# NVSHMEM builtins
########################################################


def nvshmem_my_pe():
    """TVM intrinsic to call nvshmem_my_pe()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return call_intrin("int32", "tirx.nvshmem.my_pe")


def nvshmem_n_pes():
    """TVM intrinsic to call nvshmem_n_pes()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return call_intrin("int32", "tirx.nvshmem.n_pes")


def nvshmem_getmem_nbi(dst, src, nelems, pe):
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

    return call_intrin("", "tirx.nvshmem.getmem_nbi", dst, src, nelems, pe)


def nvshmem_putmem_nbi(dst, src, nelems, pe):
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

    return call_intrin("", "tirx.nvshmem.putmem_nbi", dst, src, nelems, pe)


def nvshmem_getmem_nbi_warp(dst, src, nelems, pe):
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

    return call_intrin("", "tirx.nvshmem.getmem_nbi_warp", dst, src, nelems, pe)


def nvshmem_putmem_nbi_warp(dst, src, nelems, pe):
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

    return call_intrin("", "tirx.nvshmem.putmem_nbi_warp", dst, src, nelems, pe)


def nvshmem_getmem_nbi_block(dst, src, nelems, pe):
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

    return call_intrin("", "tirx.nvshmem.getmem_nbi_block", dst, src, nelems, pe)


def nvshmem_putmem_nbi_block(dst, src, nelems, pe):
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

    return call_intrin("", "tirx.nvshmem.putmem_nbi_block", dst, src, nelems, pe)


def nvshmem_signal_op(sig_addr, signal, sig_op, pe):
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
    return call_intrin("", "tirx.nvshmem.signal_op", sig_addr, signal, sig_op, pe)


def nvshmem_wait_until(ivar, cmp, cmp_value, type="uint64_t"):
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
    return call_intrin("", "tirx.nvshmem.wait_until", ivar, cmp, cmp_value, type)


def nvshmem_quiet():
    """TVM intrinsic to call nvshmem_quiet()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.quiet")


def nvshmem_putmem_signal_nbi(dst, src, nelems, sig_addr, signal, sig_op, pe):
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

    return call_intrin(
        "", "tirx.nvshmem.putmem_signal_nbi", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_putmem_signal_nbi_warp(dst, src, nelems, sig_addr, signal, sig_op, pe):
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

    return call_intrin(
        "", "tirx.nvshmem.putmem_signal_nbi_warp", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_putmem_signal_nbi_block(dst, src, nelems, sig_addr, signal, sig_op, pe):
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

    return call_intrin(
        "", "tirx.nvshmem.putmem_signal_nbi_block", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_fence():
    """TVM intrinsic to call nvshmem_fence()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.fence")


def nvshmem_barrier_all():
    """TVM intrinsic to call nvshmem_barrier_all()

    Returns
    -------
    call : Expr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.barrier_all")
