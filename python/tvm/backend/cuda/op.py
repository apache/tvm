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
from tvm.ir import Op, PrimExpr
from tvm.runtime import const
from tvm.tirx.expr import Call
from tvm.tirx.op import bitwise_and, call_intrin, tvm_access_ptr
from tvm.tirx.operator.intrinsics._common import CLUSTER_BARRIER_SEM as _CLUSTER_BARRIER_SEM
from tvm.tirx.operator.intrinsics._common import (
    CP_ASYNC_BULK_CACHE_HINT as _CP_ASYNC_BULK_CACHE_HINT,
)
from tvm.tirx.operator.intrinsics._common import CP_ASYNC_BULK_RED_OP as _CP_ASYNC_BULK_RED_OP
from tvm.tirx.operator.intrinsics._common import CP_ASYNC_CACHE_HINT as _CP_ASYNC_CACHE_HINT
from tvm.tirx.operator.intrinsics._common import CP_ASYNC_FILL_MODE as _CP_ASYNC_FILL_MODE
from tvm.tirx.operator.intrinsics._common import CP_ASYNC_PREFETCH_SIZE as _CP_ASYNC_PREFETCH_SIZE
from tvm.tirx.operator.intrinsics._common import F32X2_ROUND as _F32X2_ROUND
from tvm.tirx.operator.intrinsics._common import FENCE_PROXY_ASYNC_SPACE as _FENCE_PROXY_ASYNC_SPACE
from tvm.tirx.operator.intrinsics._common import FENCE_SCOPE as _FENCE_SCOPE
from tvm.tirx.operator.intrinsics._common import FENCE_SEM as _FENCE_SEM
from tvm.tirx.operator.intrinsics._common import LDMATRIX_DTYPE as _LDMATRIX_DTYPE
from tvm.tirx.operator.intrinsics._common import LDMATRIX_NUM as _LDMATRIX_NUM
from tvm.tirx.operator.intrinsics._common import NVSHMEM_CMP as _NVSHMEM_CMP
from tvm.tirx.operator.intrinsics._common import NVSHMEM_SIG_OP as _NVSHMEM_SIG_OP
from tvm.tirx.operator.intrinsics._common import TCGEN05_CP_DECOMPRESS as _TCGEN05_CP_DECOMPRESS
from tvm.tirx.operator.intrinsics._common import TCGEN05_CP_MULTICAST as _TCGEN05_CP_MULTICAST
from tvm.tirx.operator.intrinsics._common import TCGEN05_CP_SHAPES as _TCGEN05_CP_SHAPES
from tvm.tirx.operator.intrinsics._common import TCGEN05_CTA_GROUP as _TCGEN05_CTA_GROUP
from tvm.tirx.operator.intrinsics._common import TCGEN05_LDST_SHAPES as _TCGEN05_LDST_SHAPES

tir = tirx

########################################################
# CUDA native builtins
########################################################


def cuda_func_call(func_name, *args, source_code, return_type="void"):
    """TVM intrinsic to call a CUDA function. Source code is provided as a string.

    Parameters
    ----------
    func_name: str
        The name of the CUDA function.

    args: PrimExpr
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
    value : PrimExpr
        The per-thread scalar value to reduce.

    op : str
        Reduction operation: ``"sum"``, ``"max"``, or ``"min"``.

    width : int
        Number of lanes participating in each reduction group.
        Must be a power of two in [2, 32].  Defaults to 32 (full warp).

    Returns
    -------
    call : PrimExpr
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
    value : PrimExpr
        Per-thread scalar value to reduce.

    op : str
        Reduction operation: ``"sum"``, ``"max"``, or ``"min"``.

    num_warps : int
        Number of warps in the CTA.  Must be a power of two in [1, 32].

    scratch : Var
        Data pointer to shared-memory scratch space (>= num_warps elements).

    Returns
    -------
    call : PrimExpr
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


def cuda_copy_bytes(dst, src, num_bytes):
    """Typed load/store copy of ``num_bytes`` bytes.

    Copies ``num_bytes`` bytes from ``src`` to ``dst`` using a single
    typed load/store instruction.  Codegen selects the appropriate C++
    vector type (``uint4``, ``uint2``, ``unsigned int``, etc.).

    Parameters
    ----------
    dst : Var
        Destination pointer.

    src : Var
        Source pointer.

    num_bytes : int
        Number of bytes to copy.  Must be one of {1, 2, 4, 8, 16}.

    Returns
    -------
    call : PrimExpr
        A void call expression.
    """
    return call_intrin("void", "tirx.cuda.copy_bytes", dst, src, num_bytes)


def cuda_copy_128b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 16)`` — copies 128 bits."""
    return cuda_copy_bytes(dst, src, 16)


def cuda_copy_64b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 8)`` — copies 64 bits."""
    return cuda_copy_bytes(dst, src, 8)


def cuda_copy_32b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 4)`` — copies 32 bits."""
    return cuda_copy_bytes(dst, src, 4)


def cuda_copy_16b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 2)`` — copies 16 bits."""
    return cuda_copy_bytes(dst, src, 2)


def cuda_copy_8b(dst, src):
    """Convenience wrapper: ``cuda_copy_bytes(dst, src, 1)`` — copies 8 bits."""
    return cuda_copy_bytes(dst, src, 1)


def cuda_warp_sync():
    """TVM intrinsic to synchronize threads within the current warp.

    This lowers to a CUDA `__syncwarp()` call.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.warp_sync")


def cuda_cta_sync():
    """TVM intrinsic to call CUDA syncthreads (block-wide barrier)

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.cta_sync")


def cuda_grid_sync():
    """TVM intrinsic to call CUDA grid-wide sync (cooperative groups)

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.grid_sync")


def cuda_cluster_sync():
    """TVM intrinsic to call CUDA cluster-wide barrier sync

    Returns
    -------
    call : PrimExpr
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
    call : PrimExpr
        The call expression (``int32``).
    """
    return call_intrin("int32", "tirx.cuda.thread_rank")


def cuda_half2float(src):
    """TVM intrinsic to convert half to float

    Parameters
    ----------
    src : PrimExpr
        Source pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("float32", "tirx.cuda.half2float", src)


def cuda_bfloat162float(src):
    """TVM intrinsic to convert bfloat16 to float

    Parameters
    ----------
    src : PrimExpr
        Source pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("float32", "tirx.cuda.bfloat162float", src)


def cuda_float22half2(dst, src):
    """TVM intrinsic to convert float2 to half2 with rounding

    Parameters
    ----------
    dst : PrimExpr
        Destination pointer.

    src : PrimExpr
        Source pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.float22half2", dst, src)


def cuda_trap_when_assert_failed(cond):
    """TVM intrinsic to trap when assertion failed (cond == false)

    Parameters
    ----------
    cond : PrimExpr
        Condition to check.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.trap_when_assert_failed", cond)


def cuda_runtime_instr_desc(desc, sf_id):
    """TVM intrinsic to update runtime instruction descriptor

    Parameters
    ----------
    desc : PrimExpr
        Pointer to the descriptor (uint32*).

    sf_id : PrimExpr
        The subfragment id.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.runtime_instr_desc", desc, sf_id)


def cuda_half8tofloat8(src_addr, dst_addr):
    """TVM intrinsic to convert 8 half2s to 8 float2s

    Parameters
    ----------
    src_addr : PrimExpr
        Source pointer.

    dst_addr : PrimExpr
        Destination pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.half8tofloat8", src_addr, dst_addr)


def cuda_float8tohalf8(src_addr, dst_addr):
    """TVM intrinsic to convert 8 float2s to 8 half2s

    Parameters
    ----------
    src_addr : PrimExpr
        Source pointer.

    dst_addr : PrimExpr
        Destination pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.float8tohalf8", src_addr, dst_addr)


def ptx_mma_sp(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
    metadata,
    meta_index,
    sparse_selector,
    saturate,
):
    """TVM intrinsic for sparse tensor core ptx instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of multiplicand fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment B.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    metadata : Expr
        The metadata of operand.

    meta_index : Expr
        The metadata index of operand.

    sparse_selector : Expr
        The sparse selector indicating the thread that stores the metadata.

    saturate : bool
        The optional saturation at the output.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        "tirx.ptx.mma_sp",
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
        metadata,
        meta_index,
        sparse_selector,
        saturate,
    )


def ptx_cp_async_bulk(
    dtype, shared_ptr, shared_offset, global_ptr, global_offset, bytes, barrier_id
):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async.bulk
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk

    Parameters
    ----------
    dtype : str
       The data type of the result.

    shared_ptr : Var
        The shared memory pointer variable.

    shared_offset : Expr
        The offset of shared memory pointer.

    global_ptr : Var
        The global memory pointer variable.

    global_offset : Expr
        The offset of global memory pointer.

    bytes : int
        The data size to copy.

    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin(
        dtype,
        "tirx.ptx.cp_async_bulk",
        shared_ptr,
        shared_offset,
        global_ptr,
        global_offset,
        bytes,
        barrier_id,
    )


def ptx_cp_async_bulk_shared_to_cluster(dst_ptr, src_ptr, size, mbar):
    """PTX cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes

    Asynchronous bulk copy from executing CTA's shared memory to a remote
    CTA's shared memory within the same cluster.

    Parameters
    ----------
    dst_ptr : PrimExpr
        Destination pointer in shared::cluster address space (remote CTA).

    src_ptr : PrimExpr
        Source pointer in shared::cta address space (local CTA).

    size : PrimExpr
        Number of bytes to copy (must be multiple of 16).

    mbar : PrimExpr
        Mbarrier address in shared::cluster space for completion signaling,
        usually produced by ``T.ptx.map_shared_rank``.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.cp_async_bulk_shared_to_cluster", dst_ptr, src_ptr, size, mbar)


def ptx_cp_async_mbarrier_arrive(barrier_id):
    """TVM intrinsic for ptx async copy barrier using cp.async.mbarrier.arrive
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive

    Parameters
    ----------
    barrier_id : int
        The ID of the barrier shared memory pointer.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.cp_async_mbarrier_arrive", barrier_id)


def ptx_fence(sem: str, scope: str):
    """TVM intrinsic for PTX fence instruction.

    Generates: fence.{sem}.{scope};

    Parameters
    ----------
    sem : str
        The semantics of the fence. One of "sc", "acq_rel".
    scope : str
        The scope of the fence. One of "cta", "cluster", "gpu", "sys".

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("sem", sem, _FENCE_SEM)
    _choice("scope", scope, _FENCE_SCOPE)
    return call_intrin("", "tirx.ptx.fence", sem, scope)


def ptx_fence_proxy_async(space: str = ""):
    """TVM intrinsic for PTX fence.proxy.async instruction.

    Generates: fence.proxy.async[.{space}];

    Parameters
    ----------
    space : str
        The address space qualifier. One of "", "global", "shared::cta", "shared::cluster".
        Empty string means no qualifier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("space", space, _FENCE_PROXY_ASYNC_SPACE)
    return call_intrin("", "tirx.ptx.fence_proxy_async", space)


def ptx_mbarrier_init(bar, thread_count):
    """TVM intrinsic to call mbarrier.init.shared::cta.b64

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    thread_count : int
        The number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.mbarrier_init", bar, thread_count)


def ptx_mbarrier_arrive(bar, cta_id=None, pred=None, count=None):
    """TVM intrinsic to call
        mbarrier.arrive.shared::cta.b64
    or
        @p mapa.shared::cluster.u32
        @p mbarrier.arrive.shared::cluster.b64 [, count]

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    cta_id : Optional[PrimExpr]
        The cta id.

    pred : Optional[PrimExpr]
        The predicate to guard the operation.

    count : Optional[PrimExpr]
        Explicit arrival count operand for the cross-CTA (cluster) form. When
        ``None`` the implicit count-of-1 form is emitted; when given, emits
        ``mbarrier.arrive.shared::cluster.b64 _, [addr], count``.
    """
    if cta_id is None and pred is None:
        return call_intrin("", "tirx.ptx.mbarrier_arrive", bar)
    assert cta_id is not None and pred is not None
    if count is None:
        return call_intrin("", "tirx.ptx.mbarrier_arrive", bar, cta_id, pred)
    return call_intrin("", "tirx.ptx.mbarrier_arrive", bar, cta_id, pred, count)


def ptx_mbarrier_arrive_cluster_count(bar, cta_id, count):
    """Cross-CTA ``mbarrier.arrive`` on CTA ``cta_id`` with an explicit count.

    Convenience for an already-elected thread: emits
    ``@p mapa.shared::cluster.u32`` + ``@p mbarrier.arrive.shared::cluster.b64 _,
    [addr], count`` with the guard defaulted to 1.
    """
    return call_intrin("", "tirx.ptx.mbarrier_arrive", bar, cta_id, True, count)


def ptx_mbarrier_arrive_expect_tx(bar, byte_count, cta_id=None, pred=None):
    """TVM intrinsic to call
        mbarrier.arrive_expect_tx.shared::cta.b64
    or
        @p mapa.shared::cluster.u32
        @p mbarrier.arrive_expect_tx.shared::cluster.b64

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    byte_count : int
        Increases the tx count of the mbarrier object to track completion of
        addtional async transactions.

    cta_id : Optional[PrimExpr]
        The cta id.

    pred : Optional[PrimExpr]
        The predicate to guard the operation.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if cta_id is None and pred is None:
        return call_intrin("", "tirx.ptx.mbarrier_arrive_expect_tx", bar, byte_count)
    assert cta_id is not None
    # Cross-CTA expect_tx from an already-elected thread: default the guard to 1
    # (the caller has elected a single lane), so callers can pass cta_id alone.
    if pred is None:
        pred = True
    return call_intrin("", "tirx.ptx.mbarrier_arrive_expect_tx", bar, byte_count, cta_id, pred)


def ptx_mbarrier_try_wait(bar, phase):
    """TVM intrinsic to call mbarrier.try_wait.parity repeatedly until it returns true

    Parameters
    ----------
    bar : Var
        The pointer to barrier variable.

    phase : int
        The phase of the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.mbarrier_try_wait", bar, phase)


def ptx_mbarrier_try_wait_acquire_cluster(bar, phase):
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
    return call_intrin("", "tirx.ptx.mbarrier_try_wait_acquire_cluster", bar, phase)


def ptx_mbarrier_try_wait_once(bar, phase, ticks):
    """TVM intrinsic for one-shot non-blocking ``mbarrier.try_wait.parity``.

    Returns ``1`` if the requested parity has been reached and ``0`` otherwise.
    This is intended for bounded debug waits; production waits should use
    :func:`ptx_mbarrier_try_wait`.
    """
    return call_intrin("uint32", "tirx.ptx.mbarrier_try_wait_once", bar, phase, ticks)


def ptx_bar_arrive(name_bar_id, thread_count):
    """TVM intrinsic to call bar.arrive a, b

    Parameters
    ----------
    name_bar_id : int
        The ID of the named barrier.

    thread_count : int
        The number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.bar_arrive", name_bar_id, thread_count)


def ptx_bar_sync(name_bar_id, thread_count):
    """TVM intrinsic to call bar.sync a, {b}

    Parameters
    ----------
    name_bar_id : int
        The ID of the named barrier.

    thread_count : int
        The number of threads expected to arrive at the barrier.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.bar_sync", name_bar_id, thread_count)


def ptx_cp_async(
    dst_ptr,
    src_ptr,
    cp_size,
    *,
    cache_hint="",
    cache_policy=None,
    prefetch_size=-1,
    predicate=-1,
    fill_mode="",
):
    """TVM intrinsic for ptx async copy from global to shared memory using cp.async
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async

    Dispatches to one of three PTX-form-aligned ops:

    * ``ptx_cp_async_src_size`` for ``fill_mode == "zero"`` (zero-fill via
      ``src_size = pred ? cp_size : 0``).
    * ``ptx_cp_async_ignore_src`` for a non-empty ``predicate`` with no
      fill_mode (``setp+@p`` guards the asm).
    * ``ptx_cp_async_plain`` for the no-predicate / no-fill_mode case.

    Parameters
    ----------
    shared_ptr : PrimExpr
        The pointer to the shared memory.

    global_ptr : PrimExpr
        The pointer to the global memory.

    cp_size : int
        The data size to copy.

    cache_hint : str["evict_last", "evict_first", "evict_normal", ""]
        The cache hint.

    prefetch_size : int[-1, 64, 128, 256]
        The prefetch size.

    predicate : PrimExpr
        The predicate to guard the operation.

    fill_mode : str["zero", ""]
        The fill mode.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    _choice("prefetch_size", prefetch_size, _CP_ASYNC_PREFETCH_SIZE)
    _choice("fill_mode", fill_mode, _CP_ASYNC_FILL_MODE)
    return call_intrin(
        "",
        "tirx.ptx.cp_async",
        dst_ptr,
        src_ptr,
        cp_size,
        cache_policy,
        int(has_cache_policy),
        prefetch_size,
        predicate,
        fill_mode,
    )


def ptx_cp_async_legacy(*all_args):
    """Legacy ``ptx_cp_async`` API taking explicit src/dst offsets.

    Signature: ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)``.
    Offsets are folded into the pointers via ``tvm_access_ptr`` then
    dispatched to fork-native :func:`ptx_cp_async`.

    ``T.ptx.cp_async_legacy`` runs through ``_dtype_forward`` which
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
    dst_ptr = tvm_access_ptr(elem_dtype, dst_ptr, dst_offset, 1, 1)
    src_ptr = tvm_access_ptr(elem_dtype, src_ptr, src_offset, 1, 1)
    return ptx_cp_async(dst_ptr, src_ptr, cp_size)


def ptx_cp_async_commit_group():
    """TVM intrinsic for ptx async copy commit
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.cp_async_commit_group")


def ptx_cp_async_wait_group(num=0):
    """TVM intrinsic for ptx async copy wait
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group

    Parameters
    ----------
    num : int, optional
        The number of the most recent uncommitted pending cp.async groups to wait.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.cp_async_wait_group", num)


def ptx_cp_async_bulk_tensor_global_to_cluster(
    dim, dst_ptr, bar, tensormap_addr, cta_mask, cta_group, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.async.bulk.tensor.dim.shared::cluster.global.tile.mbarrier::complete_tx::bytes

    Parameters
    ----------
    dim : int
        The dimension of the source tensor.

    dst_ptr : PrimExpr
        The destination pointer to the shared memory.

    bar : PrimExpr
        The pointer to mbarrier variable.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cta_mask : int
        The mask of the cta for multicast.

    cta_group : int
        Must be either 1 or 2.
        If set to 1, mbarrier must be in the shared memory of the same CTA as the shared memory destination
        If set to 2, mbarrier can be in shared memory of either the same CTA as the shared memory destination
                     or the shared memory of the peer CTA.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        specifies the starting coordinates in the tensor data in the global memory

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_global_to_cluster",
            dim,
            dst_ptr,
            bar,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_tensor_global_to_cluster",
        dim,
        dst_ptr,
        bar,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster(
    dim, dst_ptr, bar, tensormap_addr, cta_mask, cta_group, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call
    cp.async.bulk.tensor.dim.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes

    Parameters
    ----------
    dim : int
        The dimension of the source tensor.

    dst_ptr : PrimExpr
        The destination pointer to the shared memory.

    bar : PrimExpr
        The pointer to mbarrier variable.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cta_mask : int
        The mask of the cta for multicast.

    cta_group : int
        Must be either 1 or 2.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        The TMA coordinates followed by the 4 gather row indices.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_tile_gather4_global_to_cluster",
            dim,
            dst_ptr,
            bar,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_tensor_tile_gather4_global_to_cluster",
        dim,
        dst_ptr,
        bar,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_shared_to_global(
    dim, src_ptr, tensormap_addr, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.async.bulk.tensor.dim.global.shared::cta.tile.bulk_group

    Parameters
    ----------
    dim : int
        The dimension of the copy tensor.

    src_ptr : PrimExpr
        The source pointer to the shared memory.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        specifies the starting coordinates in the tensor data in the global memory

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_shared_to_global",
            dim,
            src_ptr,
            tensormap_addr,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_tensor_shared_to_global",
        dim,
        src_ptr,
        tensormap_addr,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_global_to_cluster_prefetch(
    dim, tensormap_addr, cache_hint, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.async.bulk.prefetch.tensor.dim.L2.global.tile

    Parameters
    ----------
    dim : int
        The dimension of the source tensor.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cache_hint : str
        The cache hint.

    coords : List[PrimExpr]
        specifies the starting coordinates in the tensor data in the global memory

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy, *coords = coords
        return call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_global_to_cluster_prefetch",
            dim,
            tensormap_addr,
            cache_hint,
            has_cache_policy,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_tensor_global_to_cluster_prefetch",
        dim,
        tensormap_addr,
        cache_policy,
        int(has_cache_policy),
        *coords,
    )


def ptx_cp_async_bulk_tensor_shared_to_global_reduce(
    dim, src_ptr, tensormap_addr, cache_hint, red_op, *coords, cache_policy=None
):
    """TVM intrinsic to call cp.reduce.async.bulk.tensor.dim.dst.src.redOp

    Parameters
    ----------
    dim : int
        The dimension of the copy tensor.

    src_ptr : PrimExpr
        The source pointer to the shared memory.

    tensormap_addr : PrimExpr
        The generic address of the tensor map object.

    cache_hint: str
        The cache hint.

    red_op: str
        The reduction operator.

    coords: List[PrimExpr]
        The coordinates of the tensor.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    if isinstance(cache_hint, PrimExpr):
        has_cache_policy = red_op
        red_op, *coords = coords
        _choice("red_op", red_op, _CP_ASYNC_BULK_RED_OP)
        return call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_shared_to_global_reduce",
            dim,
            src_ptr,
            tensormap_addr,
            cache_hint,
            has_cache_policy,
            red_op,
            *coords,
        )
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    _choice("red_op", red_op, _CP_ASYNC_BULK_RED_OP)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_tensor_shared_to_global_reduce",
        dim,
        src_ptr,
        tensormap_addr,
        cache_policy,
        int(has_cache_policy),
        red_op,
        *coords,
    )


def ptx_cp_async_bulk_commit_group():
    """TVM intrinsic to call cp.async.bulk.tensor.commit_group

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.cp_async_bulk_commit_group")


def ptx_cp_async_bulk_wait_group(n=0, read=True):
    """TVM intrinsic to call cp.async.bulk.tensor.wait_group

    Parameters
    ----------
    n : int
        The number of the most recent uncommitted pending cp.async groups to wait.

    read : bool
        Whether the wait is for read.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.cp_async_bulk_wait_group", n, read)


def ptx_barrier_cluster_arrive(sem="", aligned=True):
    """TVM intrinsic to call barrier.cluster.arrive{.sem}{.aligned}

    Parameters
    ----------
    sem : str
        Either release or relaxed or empty string.

    aligned : bool
        Whether all threads in the warp must execute the same instruction.
    """
    _choice("sem", sem, _CLUSTER_BARRIER_SEM)
    return call_intrin("", "tirx.ptx.barrier_cluster_arrive", sem, aligned)


def ptx_barrier_cluster_wait(acquire=False, aligned=True):
    """TVM intrinsic to call barrier.cluster.wait{.acquire}{.aligned}

    Parameters
    ----------
    acquire : bool
        The memory synchronization

    aligned : bool
        Whether all threads in the warp must execute the same instruction.
    """
    return call_intrin("", "tirx.ptx.barrier_cluster_wait", acquire, aligned)


def ptx_clc_try_cancel(handle, mbar):
    """TVM intrinsic to call clusterlaunchcontrol.try_cancel.

    Async-requests cancelling the next cluster's launch (work-stealing): writes the
    16B response handle to smem and signals ``mbar`` (complete_tx, multicast to both
    cluster CTAs).

    Parameters
    ----------
    handle : PrimExpr
        Pointer to the 16B (uint4) smem response handle.

    mbar : PrimExpr
        Pointer to the mbarrier signalled when the handle lands.
    """
    return call_intrin("", "tirx.ptx.clc_try_cancel", handle, mbar)


def ptx_clc_query_cancel(handle):
    """TVM intrinsic to call clusterlaunchcontrol.query_cancel.

    Decodes the response handle written by :func:`ptx_clc_try_cancel`. Returns the
    cancelled cluster's first ``ctaid.x``, or ``0xFFFFFFFF`` when no work was stolen.

    Parameters
    ----------
    handle : PrimExpr
        Pointer to the 16B (uint4) smem response handle.
    """
    return call_intrin("uint32", "tirx.ptx.clc_query_cancel", handle)


def ptx_elect_sync():
    """TVM intrinsic to call elect.sync"""
    return call_intrin("uint32", "tirx.ptx.elect_sync")


def ptx_fence_mbarrier_init():
    """TVM intrinsic for PTX fence.mbarrier_init.release.cluster instruction.

    Generates: fence.mbarrier_init.release.cluster;

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.fence_mbarrier_init")


def ptx_fetch_register(bits, reg_name):
    """TVM intrinsic to tvm instrinsics to fetch PTX pre-defined registers

    Parameters
    ----------
    bits : int
        The number of bits of the register.

    reg_name : str
        The name of the register.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int" + str(bits), "tirx.ptx.fetch_register", bits, reg_name)


def ptx_mma(
    shape,
    a_layout,
    b_layout,
    d_type,
    a_type,
    b_type,
    c_type,
    d_ptrs,
    a_ptrs,
    b_ptrs,
    c_ptrs=None,
    saturate=False,
    bit_op=None,
):
    """TVM intrinsic for ptx tensor core mma instructions.
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma

    Each per-thread register of every operand is addressed by its OWN pointer
    (one ``void*`` per b32/f32 register), so the register fragments need not be
    contiguous in the register file. ``d_ptrs`` / ``a_ptrs`` / ``b_ptrs`` /
    ``c_ptrs`` are lists of one pointer per 32-bit register (b32 for
    fp16/bf16/tf32/int8 multiplicands, f32/f64 for the accumulator), enumerated
    in the fixed PTX register order (see the gemm dispatch /
    ``tests/python/tirx-base/test_tir_ptx_mma.py``).

    Within one b32 register the packed elements (e.g. 2 fp16 along k_pack)
    must stay contiguous (stride 1); only the b32 registers themselves may be
    scattered.

    Parameters
    ----------
    shape : str
        The shape of mma fragment.

    a_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    b_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    d_type : str
        The data type of result fragment D.

    a_type : str
        The data type of multiplicand fragment A.

    b_type : str
        The data type of multiplicand fragment B.

    c_type : str
        The data type of accumulator fragment C.

    d_ptrs : List[PrimExpr]
        One pointer per result-fragment D register, in PTX order.

    a_ptrs : List[PrimExpr]
        One pointer per multiplicand-A register, in PTX order.

    b_ptrs : List[PrimExpr]
        One pointer per multiplicand-B register, in PTX order.

    c_ptrs : Optional[List[PrimExpr]]
        One pointer per accumulator-C register, in PTX order. ``None`` (the
        default) means the accumulator is not used (beta == 0): codegen feeds
        a literal 0 for each C slot.

    saturate : bool
        The optional saturation at the output.

    bit_op : Optional[Literal["xor", "and"]]
        The 1-bit operator (for the b1 subbyte form). ``None`` means unused.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    d_ptrs = list(d_ptrs)
    a_ptrs = list(a_ptrs)
    b_ptrs = list(b_ptrs)
    has_c = c_ptrs is not None
    c_ptrs = list(c_ptrs) if has_c else []

    # Encode group register counts as leading attrs so codegen can slice the
    # flat pointer tail. ``no_c_ptr`` mirrors the legacy IntImm(0) sentinel.
    no_c_ptr = not has_c
    # Flattened pointer list: D regs, A regs, B regs, then C regs (if any).
    ptrs = [*d_ptrs, *a_ptrs, *b_ptrs, *c_ptrs]

    base = [
        "",
        "tirx.ptx.mma",
        shape,
        a_layout,
        b_layout,
        d_type,
        a_type,
        b_type,
        c_type,
        len(d_ptrs),
        len(a_ptrs),
        len(b_ptrs),
        len(c_ptrs),
        no_c_ptr,
        *ptrs,
        saturate,
    ]
    if bit_op is None:
        return call_intrin(*base)
    return call_intrin(*base, bit_op)


def ptx_mma_legacy(*all_args, operator=None):
    """Legacy ``ptx_mma`` API.

    Signature: ``(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
    multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
    c_index, saturate, operator=None)``. The accumulator is reused as
    both input and output (no separate ``d``/``c`` slot), unlike
    fork-native :func:`ptx_mma` which distinguishes them. Translation:

    * ``a_dtype, b_dtype, c_dtype`` → fork ``a_type, b_type, c_type``
      (and reuse ``c_dtype`` as fork ``d_type`` since the accumulator
      dtype is the output dtype here).
    * ``(a_ptr, a_offset)`` and ``(b_ptr, b_offset)`` → folded via
      :func:`tvm_access_ptr`.
    * ``(accumulator, c_index)`` → folded; passed for both ``d_ptr`` and
      ``c_ptr`` since the accumulator is reused as the output.

    ``T.ptx.mma.legacy`` runs through ``_dtype_forward`` which prepends a
    ``dtype=`` kwarg as a leading positional, so this function accepts
    either 13 or 14 positional args.
    """
    args = list(all_args)
    # ``T.ptx.mma.legacy(..., dtype="...")`` has the dtype prepended by
    # ``_dtype_forward``; strip it here.
    if len(args) in (14, 15):
        _ = args.pop(0)
    if len(args) == 14:
        # operator passed positionally as the trailing arg.
        operator = args.pop()
    if len(args) != 13:
        raise ValueError(
            f"ptx_mma_legacy expects 13-15 positional args (with optional "
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
    # Emit tirx.ptx_mma_legacy directly with separate (ptr_var, offset)
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
    return call_intrin("", "tirx.ptx.mma_legacy", *call_args)


def ptx_mma_sp_legacy(*all_args):
    """Legacy ``ptx_mma_sp`` API.

    Signature: ``(shape, A_layout, B_layout, A_dtype, B_dtype, C_dtype,
    multiplicand_a, a_index, multiplicand_b, b_index, accumulator,
    c_index, metadata, meta_index, sparse_selector, saturate)``.

    ``T.ptx.mma_sp.legacy`` runs through ``_dtype_forward`` which prepends
    a ``dtype=`` kwarg as a leading positional, so this function accepts
    either 16 or 17 positional args.
    """
    args = list(all_args)
    if len(args) == 17:
        _ = args.pop(0)
    if len(args) != 16:
        raise ValueError(
            f"ptx_mma_sp_legacy expects 16 args (or 17 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
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
        meta_ptr,
        meta_offset,
        sparse_selector,
        saturate,
    ) = args
    return ptx_mma_sp(
        c_dtype,
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
        meta_ptr,
        meta_offset,
        sparse_selector,
        saturate,
    )


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


def ptx_ldmatrix(trans, num, dtype, smem_ptr, *dst_handles):
    """TVM intrinsic for ldmatrix.sync.aligned.m8n8.x{num}{.trans}.shared.{dtype}.

    Mirrors the PTX ISA destination form: each output register is a separate
    operand. Pass ``T.address_of(buf[idx])`` (or ``buf.ptr_to([idx])``) for
    each destination — the slots may be non-contiguous.

    Parameters
    ----------
    trans : bool
        Apply the ``.trans`` modifier.
    num : int
        One of 1, 2, 4 — number of m8n8 fragments.
    dtype : str
        ``"b16"`` (4 bytes per fragment register) or ``"b8"`` (2 bytes per).
    smem_ptr : PrimExpr
        Generic pointer to source shared memory.
    *dst_handles : PrimExpr
        N pointer-to-uint32 destinations, where
        ``N = num if dtype == "b16" else num // 2``.

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
    """
    _choice("num", num, _LDMATRIX_NUM)
    _choice("dtype", dtype, _LDMATRIX_DTYPE)
    # _LDMATRIX_DTYPE entries carry leading dot (".b16" / ".b8").
    dtype_bare = dtype.lstrip(".") if isinstance(dtype, str) else dtype
    n_regs = int(num) if dtype_bare == "b16" else int(num) // 2
    if len(dst_handles) != n_regs:
        raise ValueError(
            f"ldmatrix .x{int(num)}.{dtype_bare} expects {n_regs} destination "
            f"handles, got {len(dst_handles)}"
        )
    return call_intrin("", "tirx.ptx.ldmatrix", trans, num, dtype, smem_ptr, *dst_handles)


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
            "handle",
            "tirx.tvm_access_ptr",
            inner_marker,
            inner_var,
            inner_offset + offset,
            1,
            rw_mask,
        )
    return tvm_access_ptr(elem_dtype, ptr, offset, 1, 1)


def ptx_ldmatrix_legacy(*all_args):
    """Legacy ``ptx_ldmatrix`` API taking explicit offsets.

    Signature: ``(trans, num, dtype, local_ptr, local_offset, smem_ptr,
    smem_offset)``. Offsets are folded into the pointers via
    ``tvm_access_ptr`` and dispatched to the fork-native
    :func:`ptx_ldmatrix`.

    ``T.ptx.ldmatrix_legacy`` runs through ``_dtype_forward`` which
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
            f"ptx_ldmatrix_legacy expects 7 args (or 8 with dtype= kwarg "
            f"prepended); got {len(all_args)}"
        )
    # Call.dtype carries the buffer element type so codegen can pick the
    # int8+trans manual-loop fallback (ldmatrix can't transpose int8).
    return call_intrin(
        elem_dtype,
        "tirx.ptx.ldmatrix_legacy",
        trans,
        num,
        dtype,
        local_ptr,
        local_offset,
        smem_ptr,
        smem_offset,
    )


def ptx_stmatrix(trans, num, dtype, smem_ptr, *src_handles, shape="m8n8", space="shared"):
    """TVM intrinsic for ``stmatrix.sync.aligned.shape.x{num}{.trans}.space.{dtype}``.

    Mirrors :func:`ptx_ldmatrix`: each source register is a separate operand.
    Pass ``T.address_of(buf[idx])`` (or ``buf.ptr_to([idx])``) for each
    source — the slots may be non-contiguous.

    Parameters
    ----------
    trans : bool
        Apply the ``.trans`` modifier (required for ``shape == "m16n8"``).
    num : int
        One of 1, 2, 4 — number of m8n8 fragments per warp.
    dtype : str
        ``".b16"`` (4 bytes per fragment register) or ``".b8"`` (2 bytes per).
    smem_ptr : PrimExpr
        Destination pointer in shared memory.
    *src_handles : PrimExpr
        ``num`` pointer-to-uint32 sources.
    shape : str, keyword-only, default "m8n8"
        ``"m8n8"`` or ``"m16n8"``.
    space : str, keyword-only, default "shared"
        ``"shared"`` or ``"shared::cta"``.

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-stmatrix
    """
    _choice("num", num, _LDMATRIX_NUM)
    _choice("dtype", dtype, _LDMATRIX_DTYPE)
    if shape not in ("m8n8", "m16n8"):
        raise ValueError(f"Unsupported stmatrix shape {shape!r}")
    if space not in ("shared", "shared::cta"):
        raise ValueError(f"Unsupported stmatrix state space {space!r}")
    if shape == "m16n8" and not trans:
        raise ValueError("stmatrix .m16n8 requires .trans")
    n_regs = int(num)
    if len(src_handles) != n_regs:
        dtype_bare = dtype.lstrip(".") if isinstance(dtype, str) else dtype
        raise ValueError(
            f"stmatrix .x{int(num)}.{dtype_bare} expects {n_regs} source "
            f"handles, got {len(src_handles)}"
        )
    return call_intrin(
        "", "tirx.ptx.stmatrix", trans, num, dtype, shape, space, smem_ptr, *src_handles
    )


def ptx_wgmma_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    """TVM intrinsic to create memory descriptor for wgmma instructions

    Parameters
    ----------
    desc : PrimExpr
        The pointer to the shared memory descriptor.

    addr : PrimExpr
        The address of the matrix.

    ldo : PrimExpr
        The leading dimension offset.

    sdo : PrimExpr
        The stride dimension offset.

    swizzle : int
        The swizzle value (CUtensorMapSwizzle_enum).
    """
    return call_intrin("", "tirx.ptx.wgmma_encode_matrix_descriptor", desc, addr, ldo, sdo, swizzle)


def ptx_wgmma_noop_barrier(reg):
    """TVM intrinsic to call "" : "+{format}"(reg)::"memory"

    Parameters
    ----------
    reg : PrimExpr
        The register to fence.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.wgmma_noop_barrier", reg)


def ptx_wgmma_mma_async_ss(
    descA, descB, *accums, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD
):
    """TVM intrinsic to call wgmma.mma_async.sync.aligned.shape.dtype.atype.btype over 2 smem operators

    Parameters
    ----------
    M : int
        The number of rows in matrix A and D.

    N : int
        The number of columns in matrix B and D.

    K : int
        The number of columns in matrix A and rows in matrix B.

    in_dtype : str
        The data type of the input matrices.

    out_type : str
        The data type of the output matrices.

    transA : bool
        True for M/N major, False for K major.

    transB : bool
        True for M/N major, False for K major.

    scaleA : float
        The scaling factor for matrix A.

    scaleB : float
        The scaling factor for matrix B.

    scaleD : PrimExpr
        True: D = A * B + D, False: D = A * B.

    descA : PrimExpr
        The SMEM descriptor of matrix A

    descB : PrimExpr
        The SMEM descriptor of matrix B

    accums : list
        The accumulators registers.
    """  # noqa: E501
    return call_intrin(
        "",
        "tirx.ptx.wgmma_mma_async_ss",
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        transA,
        transB,
        scaleA,
        scaleB,
        scaleD,
        descA,
        descB,
        *accums,
    )


def ptx_wgmma_mma_async_rs(
    descB, *reg_list, M, N, K, in_dtype, out_dtype, transA, transB, scaleA, scaleB, scaleD
):
    """TVM intrinsic to call wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
        When A is in register and B is in shared memory

    Parameters
    ----------
    M : int
        The number of rows in matrix A and D.

    N : int
        The number of columns in matrix B and D.

    K : int
        The number of columns in matrix A and rows in matrix B.

    in_dtype : str
        The data type of the input matrices.

    out_type : str
        The data type of the output matrices.

    transA : bool
        True for M/N major, False for K major.

    transB : bool
        True for M/N major, False for K major.

    scaleA : float
        The scaling factor for matrix A.

    scaleB : float
        The scaling factor for matrix B.

    scaleD : PrimExpr
        True: D = A * B + D, False: D = A * B.

    descB : PrimExpr
        The SMEM descriptor of matrix B

    reg_list : list
        The A registers and accumulators registers.
    """
    return call_intrin(
        "",
        "tirx.ptx.wgmma_mma_async_rs",
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        transA,
        transB,
        scaleA,
        scaleB,
        scaleD,
        descB,
        *reg_list,
    )


def ptx_wgmma_fence():
    """TVM intrinsic to call wgmma.fence.sync.aligned

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.wgmma_fence")


def ptx_wgmma_commit_group():
    """TVM intrinsic to call wgmma.commit_group.sync.aligned

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.wgmma_commit_group")


def ptx_wgmma_wait_group(n):
    """TVM intrinsic to call wgmma.wait_group.sync.aligned

    Parameters
    ----------
    n : int
        The number of the most recent uncommitted pending wgmma groups to wait.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.wgmma_wait_group", n)


def ptx_setmaxnreg(inc: bool, reg_count):
    """TVM intrinsic to call setmaxnreg.action.sync.aligned.u32 imm-reg-count

    Parameters
    ----------
    inc : bool
        True to increase the register count, False to decrease.

    reg_count : int
        The register count.
    """
    return call_intrin("", "tirx.ptx.setmaxnreg", inc, reg_count)


def ptx_tcgen05_alloc(dst_ptr, n_cols, cta_group=1):
    """TVM intrinsic to call tcgen05.alloc.cta_group.sync.aligned
        Dynamically allocates the number of cols in tensor memory, and write
        the address of allocated memory to shared memory.

    Parameters
    ----------
    dst_ptr : Var
        The pointer to the destination shared memory.

    n_cols : int
        The number of columns to allocate in tensor memory.
        Must be a multiple of 32 and a power of 2, and within the range [32, 512].

    cta_group : int
        The number of CTA groups involved in the allocation.
        If cta_group=1, one warp from CTA performs the allocation. Else, if cta_group=2,
        one warp from each of the peer CTAs perform the allocation.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx.tcgen05_alloc", dst_ptr, n_cols, cta_group)


def ptx_tcgen05_dealloc(taddr, n_cols, cta_group=1):
    """TVM intrinsic to call tcgen05.dealloc.cta_group.sync.aligned
        Deallocates the tensor memory specified by the tensor memory address taddr.

    Parameters
    ----------
    taddr : PrimExpr
        The address of previously allocated tensor memory, should be uint32_t.

    n_cols : int
        The number of columns to deallocate in tensor memory.
        Must be a multiple of 32 and a power of 2, and within the range [32, 512].

    cta_group : int
        The number of CTA groups involved in the deallocation.
        If cta_group=1, one warp from CTA performs the deallocation. Else, if cta_group=2,
        one warp from each of the peer CTAs perform the deallocation.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx.tcgen05_dealloc", taddr, n_cols, cta_group)


def ptx_tcgen05_relinquish_alloc_permit(cta_group=1):
    """TVM intrinsic to call tcgen05.relinquish_alloc_permit.cta_group.sync.aligned
        The CTA of the executing thread is relinquishing the right to allocate
        Tensor Memory after calling this op.

    Parameters
    ----------
    cta_group : int
        The number of CTA groups involved in relinquishing.
        If cta_group=1, one warp from CTA performs the relinquishing. Else, if cta_group=2,
        one warp from each of the peer CTAs perform the relinquishing.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx.tcgen05_relinquish_alloc_permit", cta_group)


def ptx_tcgen05_encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle):
    """TVM intrinsic to create memory descriptor for tcgen05 instructions

    Parameters
    ----------
    desc : PrimExpr
        The pointer to the shared memory descriptor.

    addr : PrimExpr
        The address of the matrix.

    ldo : PrimExpr
        The leading dimension offset.

    sdo : PrimExpr
        The stride dimension offset.

    swizzle : int
        The swizzle value (CUtensorMapSwizzle_enum).
    """
    return call_intrin(
        "", "tirx.ptx.tcgen05_encode_matrix_descriptor", desc, addr, ldo, sdo, swizzle
    )


def ptx_tcgen05_encode_instr_descriptor(
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
    desc : PrimExpr
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
        "tirx.ptx.tcgen05_encode_instr_descriptor",
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


def ptx_tcgen05_encode_instr_descriptor_block_scaled(
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
    desc : PrimExpr
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

    sfa_tmem_addr : PrimExpr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : PrimExpr
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
        "tirx.ptx.tcgen05_encode_instr_descriptor_block_scaled",
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


def ptx_tcgen05_mma(
    d_tmem_addr,
    a_operand,
    b_desc,
    i_desc,
    *disable_output_lane,
    d_dtype,
    a_dtype,
    b_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
    scale_input_d=0,
    pred=None,
):
    """TVM intrinsic to call tcgen05.mma.cta_group.kind without block scaling.

    Parameters
    ----------
    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. The inline asm tests
        `enable_input_d != 0`: zero means D = A*B, non-zero means D = A*B + D.

    scale_input_d : int
        The optional scaling factor to scale input matrix D.
        D = A*B+D * (2 ^ - scale-input-d)

    disable_output_lane : list
        The lanes that should not be updated in the resultant matrix D.

    pred : Optional[PrimExpr]
        Runtime ``uint32`` instruction-level predicate. When given, emit
        ``@p_issue tcgen05.mma...`` with ``p_issue = (pred != 0)``. Preserves
        PTX-level predicate semantics (single predicated SASS instruction).
    """

    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)

    # default value for disable_output_lane
    if len(disable_output_lane) == 0:
        disable_output_lane = [0] * (4 if cta_group == 1 else 8)

    args = [
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
    ]
    if pred is not None:
        args.append(pred)
    return call_intrin("", "tirx.ptx.tcgen05_mma", *args)


def ptx_tcgen05_mma_block_scale(
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    i_desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
):
    """TVM intrinsic to call tcgen05.mma.cta_group.kind.block_scale
        Performs matrix multiplication with block scaling:
        (A * scale_A)  * (B * scale_B) + D

    Parameters
    ----------
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

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    sfa_tmem_addr : PrimExpr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : PrimExpr
        The address of the scale factor matrix B in tensor memory, should be uint32_t.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. Zero means D = A*B,
        non-zero means D = A*B + D.
    """

    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin(
        "",
        "tirx.ptx.tcgen05_mma_block_scale",
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


def ptx_tcgen05_mma_sp(
    d_tmem_addr,
    a_operand,
    b_desc,
    sp_tmem_addr,
    i_desc,
    *disable_output_lane,
    d_dtype,
    a_dtype,
    b_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
    scale_input_d=0,
):
    """TVM intrinsic to call tcgen05.mma.sp.cta_group.kind without block scaling.

    Parameters
    ----------
    d_dtype : str
        The datatype of resultant matrix D.

    a_dtype : str
        The datatype of multiplicand matrix A.

    b_dtype : str
        The datatype of multiplicand matrix B.

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    sp_tmem_addr : PrimExpr
        The address of the metadata of sparse matrix in tensor memory, should be uint32_t.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. The inline asm tests
        `enable_input_d != 0`: zero means D = A*B, non-zero means D = A*B + D.

    scale_input_d : int
        The optional scaling factor to scale input matrix D.
        D = A*B+D * (2 ^ - scale-input-d)

    disable_output_lane : list
        The lanes that should not be updated in the resultant matrix D.
    """

    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)

    # default value for disable_output_lane
    if len(disable_output_lane) == 0:
        disable_output_lane = [0] * (4 if cta_group == 1 else 8)

    return call_intrin(
        "",
        "tirx.ptx.tcgen05_mma_sp",
        d_dtype,
        a_dtype,
        b_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sp_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
        scale_input_d,
        *disable_output_lane,
    )


def ptx_tcgen05_mma_sp_block_scale(
    d_tmem_addr,
    a_operand,
    b_desc,
    sfa_tmem_addr,
    sfb_tmem_addr,
    sp_tmem_addr,
    i_desc,
    *,
    d_dtype,
    a_dtype,
    b_dtype,
    sfa_dtype,
    sfb_dtype,
    use_a_tmem,
    cta_group,
    enable_input_d=1,
):
    """TVM intrinsic to call tcgen05.mma.sp.cta_group.kind.block_scale
        Performs sparse matrix multiplication with block scaling:
        (A * scale_A)  * (B * scale_B) + D

    Parameters
    ----------
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

    d_tmem_addr : PrimExpr
        The address of the resultant matrix D in tensor memory, should be uint32_t.

    a_operand : PrimExpr
        Either the matrix descriptor of multiplicand matrix A in shared memory,
        or the address of the multiplicand matrix A in tensor memory (uint32_t).

    b_desc : PrimExpr
        The matrix descriptor of multiplicand matrix B in shared memory.

    sfa_tmem_addr : PrimExpr
        The address of the scale factor matrix A in tensor memory, should be uint32_t.

    sfb_tmem_addr : PrimExpr
        The address of the scale factor matrix B in tensor memory, should be uint32_t.

    sp_tmem_addr : PrimExpr
        The address of the metadata of sparse matrix in tensor memory, should be uint32_t.

    i_desc : PrimExpr
        The instruction descriptor of the MMA operation.

    use_a_tmem : bool
        Whether the multiplicand matrix A is in tensor memory.

    cta_group : int
        The number of CTA groups involved in the MMA operation.

    enable_input_d : PrimExpr
        Scale operand for the input accumulator C/D. Zero means D = A*B,
        non-zero means D = A*B + D.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin(
        "",
        "tirx.ptx.tcgen05_mma_sp_block_scale",
        d_dtype,
        a_dtype,
        b_dtype,
        sfa_dtype,
        sfb_dtype,
        d_tmem_addr,
        a_operand,
        b_desc,
        sfa_tmem_addr,
        sfb_tmem_addr,
        sp_tmem_addr,
        i_desc,
        use_a_tmem,
        cta_group,
        enable_input_d,
    )


def ptx_tcgen05_fence_before_thread_sync():
    """TVM intrinsic to call tcgen05.fence::before_thread_sync
    Orders all prior asynchronous tcgen05 operations relative to subsequent operations.
    """
    return call_intrin("", "tirx.ptx.tcgen05_fence_before_thread_sync")


def ptx_tcgen05_fence_after_thread_sync():
    """TVM intrinsic to call tcgen05.fence::after_thread_sync
    Orders all subsequent asynchronous tcgen05 operations relative to previous operations.
    """
    return call_intrin("", "tirx.ptx.tcgen05_fence_after_thread_sync")


def _choice(name: str, value, options):
    """Validate `value` is one of `options`. Raise a clear ValueError otherwise.

    Symbolic values (Var, non-constant PrimExpr) are accepted without
    validation; specialization later replaces them with concrete values
    that the C-side intrinsic body re-checks.
    """
    # Concrete int / IntImm value: validate.
    try:
        concrete = int(value)
    except (TypeError, ValueError):
        return  # symbolic; defer check
    if concrete not in options:
        raise ValueError(f"invalid {name}={concrete!r}; expected one of {tuple(options)}")


# See top-of-file imports for `_FENCE_SEM` etc. (re-exported from _common).
# Note: TCGEN05_LDST_SHAPES values must stay in sync with the shape branches
# of codegen_ptx_tcgen05_ld/_st in intrinsics/cuda/tcgen05.py.


def ptx_tcgen05_cp(
    taddr, src_desc, *, shape, cta_group=1, multicast="", decompress="", row=0, col=0
):
    """TVM intrinsic for the Blackwell `tcgen05.cp` PTX instruction.

    The emitted PTX is::

        tcgen05.cp.cta_group::{cta_group}.{shape}[.{multicast}][.{decompress}] [taddr], src_desc;

    Each keyword argument maps 1:1 to a PTX token: read the call and you
    know what instruction is emitted.

    Parameters
    ----------
    taddr : PrimExpr
        Destination tensor-memory address (uint32). Callers typically pass
        ``tmem_base + column_offset_in_uint32s`` directly. Use the optional
        ``row`` / ``col`` keyword arguments only when the address needs
        runtime row/col composition via ``get_tmem_addr`` (high 16 bits row,
        low 16 bits col).

    src_desc : PrimExpr
        The 64-bit shared-memory matrix descriptor.

    shape : str
        One of ``"32x128b"``, ``"4x256b"``, ``"128x128b"``, ``"128x256b"``,
        ``"64x128b"``.

    cta_group : int
        1 or 2.

    multicast : str
        One of ``""``, ``"warpx4"``, ``"warpx2::02_13"``, ``"warpx2::01_23"``.
        ``"32x128b"`` requires ``"warpx4"``; ``"64x128b"`` requires one of the
        ``warpx2::*`` values; other shapes require ``""``.

    decompress : str
        Trailing PTX suffix for fp4/fp6 → fp8 on-the-fly decompression.
        One of ``""``, ``"b8x16.b4x16_p64"``, ``"b8x16.b6x16_p32"``.

    row, col : PrimExpr
        Optional row/col offsets added to ``taddr`` at runtime. Default 0.
    """
    _choice("shape", shape, _TCGEN05_CP_SHAPES)
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    _choice("multicast", multicast, _TCGEN05_CP_MULTICAST)
    _choice("decompress", decompress, _TCGEN05_CP_DECOMPRESS)
    if shape == "32x128b" and multicast != "warpx4":
        raise ValueError(f"shape=32x128b requires multicast='warpx4', got {multicast!r}")
    if shape == "64x128b" and multicast not in ("warpx2::02_13", "warpx2::01_23"):
        raise ValueError(f"shape=64x128b requires multicast in warpx2::*, got {multicast!r}")
    if shape in ("128x128b", "128x256b", "4x256b") and multicast != "":
        raise ValueError(f"shape={shape} requires multicast='', got {multicast!r}")

    return call_intrin(
        "",
        "tirx.ptx.tcgen05_cp",
        taddr,
        src_desc,
        shape,
        cta_group,
        multicast,
        decompress,
        row,
        col,
    )


def ptx_tcgen05_shift(taddr, cta_group=1):
    """TVM intrinsic to call tcgen05.shift.cta_group.down
        Asynchronously shift down the rows of the matrix in Tensor Memory for a warp.

    Parameters
    ----------
    taddr : PrimExpr
        The address of matrix in tensor memory, should be uint32_t.

    cta_group : int
        The number of CTA groups involved in the shift.
        If cta_group=1, shift operation is performed in the Tensor Memory of current CTA.
        Else, shift operation is performed in the Tensor Memory of both the current CTA and
        the peer CTA.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    return call_intrin("", "tirx.ptx.tcgen05_shift", taddr, cta_group)


def ptx_tcgen05_ld(src_addr, *regs, shape, num, row=0, col=0, pack=False):
    """TVM intrinsic for tcgen05.ld.sync.aligned — async collective load from TMEM.

    Emits ``tcgen05.ld.sync.aligned.{shape}.x{num}[.pack::16b].b32 {regs}, [addr];``

    Parameters
    ----------
    src_addr : PrimExpr
        Tensor-memory source address (uint32).

    regs : list[PrimExpr]
        Destination registers. Count depends on shape x num.

    shape : str
        One of ``"16x32bx2"``, ``"16x64b"``, ``"16x128b"``, ``"16x256b"``, ``"32x32b"``.

    num : int
        Repeat factor along the columns. Power-of-two in [1, 128].

    row, col : PrimExpr
        Optional TMEM row/col offsets added to ``src_addr`` at runtime (row must be
        a multiple of 32). Default 0.

    pack : bool
        Pack two 16-bit chunks into a single 32-bit register.
    """
    _choice("shape", shape, _TCGEN05_LDST_SHAPES)
    return call_intrin("", "tirx.ptx.tcgen05_ld", src_addr, row, col, shape, num, pack, *regs)


def ptx_tcgen05_st(dst_addr, *regs, shape, num, row=0, col=0, unpack=False):
    """TVM intrinsic for tcgen05.st.sync.aligned — async collective store to TMEM.

    Emits ``tcgen05.st.sync.aligned.{shape}.x{num}[.unpack::16b].b32 [addr], {regs};``

    Parameters
    ----------
    dst_addr : PrimExpr
        Tensor-memory destination address (uint32).

    regs : list[PrimExpr]
        Source registers. Count depends on shape x num.

    shape : str
        One of ``"16x32bx2"``, ``"16x64b"``, ``"16x128b"``, ``"16x256b"``, ``"32x32b"``.

    num : int
        Repeat factor along the columns. Power-of-two in [1, 128].

    row, col : PrimExpr
        Optional TMEM row/col offsets added to ``dst_addr`` at runtime (row must be
        a multiple of 32). Default 0.

    unpack : bool
        Unpack a 32-bit register into two 16-bit chunks.
    """
    _choice("shape", shape, _TCGEN05_LDST_SHAPES)
    return call_intrin("", "tirx.ptx.tcgen05_st", dst_addr, row, col, shape, num, unpack, *regs)


def ptx_tcgen05_wait_ld():
    """TVM intrinsic to call tcgen05.wait::ld.sync.aligned
    Wait for the completion of all prior async tcgen05.ld operations.
    """
    return call_intrin("", "tirx.ptx.tcgen05_wait_ld")


def ptx_tcgen05_wait_st():
    """TVM intrinsic to call tcgen05.wait::st.sync.aligned
    Wait for the completion of all prior async tcgen05.st operations.
    """
    return call_intrin("", "tirx.ptx.tcgen05_wait_st")


def ptx_tcgen05_commit(bar, cta_group=1, cta_mask=0, *, pred=None):
    """TVM intrinsic to call tcgen05.commit.cta_group

    Parameters
    ----------
    bar : PrimExpr
        The pointer to mbarrier variable.

    cta_group: int
        The number of CTA groups involved in previous tcgen05 operations.

    cta_mask : int
        The mask of the CTAs in the cluster, used for multicast.

    pred : Optional[PrimExpr]
        Runtime ``uint32`` predicate. When given, emit
        ``@p tcgen05.commit...`` with ``p = (pred != 0)``. This preserves
        PTX-level instruction predicate semantics (single predicated
        instruction in SASS), distinct from a C-level ``if`` branch.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    _choice("cta_group", cta_group, _TCGEN05_CTA_GROUP)
    args = [bar, cta_group, cta_mask]
    if pred is not None:
        args.append(pred)
    return call_intrin("", "tirx.ptx.tcgen05_commit", *args)


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

    group_id: PrimExpr
        The group id of the current thread.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin(
        "handle",
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

    leader_cond: PrimExpr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "handle",
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

    leader_cond: PrimExpr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "handle",
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

    leader_cond: PrimExpr
        The condition to check if the current thread is the leader.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin(
        "handle",
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
    res_addr : PrimExpr
        The result address.

    value: PrimExpr
        The value to add.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    value = tir.convert(value)
    return call_intrin(value.ty, "tirx.cuda.atomic_add", res_addr, value)


def cuda_thread_fence():
    """TVM intrinsic to call cuda thread fence instruction

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.thread_fence")


def cuda_warpgroup_sync(bar_no):
    """TVM intrinsic to synchronize a CUDA warpgroup via a named barrier.

    Parameters
    ----------
    bar_no : PrimExpr
        The named barrier id to use for the warpgroup.

    Notes
    -----
    Synchronizes 128 threads in a warpgroup using `bar.sync bar_no, 128`.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.warpgroup_sync", bar_no)


def cuda_syncthreads_and(cond):
    """TVM intrinsic to call cuda syncthreads_and instruction

    Parameters
    ----------
    cond: PrimExpr
        The condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int64", "tirx.cuda.syncthreads_and", cond)


def cuda_syncthreads_or(cond):
    """TVM intrinsic to call cuda syncthreads_or instruction

    Parameters
    ----------
    cond: PrimExpr
        The condition.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("int64", "tirx.cuda.syncthreads_or", cond)


def cuda_nano_sleep(time):
    """TVM intrinsic to call cuda nano sleep instruction

    Parameters
    ----------
    time: PrimExpr
        The time to sleep.

    Returns
    -------
    call : PrimExpr
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
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.cuda.printf", fmt, *args)


def cuda_ldg(addr, dtype):
    """TVM intrinsic to call CUDA C++ __ldg() function

    Parameters
    ----------
    addr : PrimExpr
        The memory address to load.

    dtype : str
        The data type of the loaded value.

    Returns
    """
    return call_intrin(dtype, "tirx.cuda.ldg", addr, dtype)


def cuda_get_tmem_addr(addr, row_offset, col_offset):
    """TVM intrinsic to call cuda tmem address calculation

    Parameters
    ----------
    addr: PrimExpr
        The memory address to calculate.

    row_offset: PrimExpr
        The row offset to calculate.

    col_offset: PrimExpr
        The column offset to calculate.

    Returns
    -------
    call : PrimExpr
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


def cuda_sm100_tma_2sm_mbarrier_addr(bar):
    """Compute the SM100 2SM TMA mbarrier shared-address operand."""
    return bitwise_and(cuda_cvta_generic_to_shared(bar), const(0xFEFFFFFF, dtype="uint32"))


def ptx_exp2(x):
    """TVM intrinsic for PTX fast exp2 approximation (ex2.approx.ftz.f32)

    Parameters
    ----------
    x : PrimExpr
        The float32 input value.

    Returns
    -------
    call : PrimExpr
        The call expression returning 2^x (approximate).
    """
    return call_intrin("float32", "tirx.ptx.exp2", x)


def ptx_rcp(x):
    """TVM intrinsic for PTX fast reciprocal approximation (rcp.approx.ftz.f32)

    Parameters
    ----------
    x : PrimExpr
        The float32 input value.

    Returns
    -------
    call : PrimExpr
        The call expression returning 1/x (approximate).
    """
    return call_intrin("float32", "tirx.ptx.rcp", x)


def ptx_any_sync(mask, pred):
    """TVM intrinsic for PTX warp-wide any predicate (__any_sync)

    Parameters
    ----------
    mask : PrimExpr
        The thread mask (uint32).
    pred : PrimExpr
        The predicate value (int32).

    Returns
    -------
    call : PrimExpr
        The call expression returning 1 if any thread in mask has pred != 0.
    """
    return call_intrin("int32", "tirx.ptx.any_sync", mask, pred)


def ptx_reduce3_max_f32(a, b, c):
    """TVM intrinsic to call 3-input max.f32 PTX instruction (sm_100a+)

    Parameters
    ----------
    a, b, c : PrimExpr
        The three float32 values to compare.

    Returns
    -------
    call : PrimExpr
        The call expression returning max(a, b, c).
    """
    return call_intrin("float32", "tirx.ptx.reduce3_max_f32", a, b, c)


def ptx_reduce3_min_f32(a, b, c):
    """TVM intrinsic to call 3-input min.f32 PTX instruction (sm_100a+)

    Parameters
    ----------
    a, b, c : PrimExpr
        The three float32 values to compare.

    Returns
    -------
    call : PrimExpr
        The call expression returning min(a, b, c).
    """
    return call_intrin("float32", "tirx.ptx.reduce3_min_f32", a, b, c)


def _ptx_binary_arith(op_name, dtype, d, a, b, *, rounding="rn", ftz=False, sat=False):
    """Shared helper for add/sub/mul over (f32 | f32x2 | f64), DPS form."""
    _choice("rounding", rounding, _F32X2_ROUND)
    if dtype == "f64" and (ftz or sat):
        raise ValueError(f"PTX {op_name}.f64 does not accept .ftz or .sat")
    if dtype == "f32x2" and sat:
        raise ValueError(f"PTX {op_name}.f32x2 does not accept .sat")
    return call_intrin(
        "",
        f"tirx.ptx.{op_name}_{dtype}",
        d,
        a,
        b,
        rounding,
        int(ftz),
        int(sat),
    )


def _ptx_fma(dtype, d, a, b, c, *, rounding="rn", ftz=False, sat=False):
    """Shared helper for fma over (f32 | f32x2 | f64), DPS form."""
    _choice("rounding", rounding, _F32X2_ROUND)
    if dtype == "f64" and (ftz or sat):
        raise ValueError("PTX fma.f64 does not accept .ftz or .sat")
    if dtype == "f32x2" and sat:
        raise ValueError("PTX fma.f32x2 does not accept .sat")
    return call_intrin(
        "",
        f"tirx.ptx.fma_{dtype}",
        d,
        a,
        b,
        c,
        rounding,
        int(ftz),
        int(sat),
    )


def ptx_add_f32(d_addr, a, b, *, rounding="rn", ftz=False, sat=False):
    """PTX ``add{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("add", "f32", d_addr, a, b, rounding=rounding, ftz=ftz, sat=sat)


def ptx_add_f32x2(d_addr, a, b, *, rounding="rn", ftz=False):
    """PTX ``add{.rnd}{.ftz}.f32x2 [d_addr], a, b`` — DPS form.

    a, b are packed-as-uint64 register operands (2 fp32 each).
    """
    return _ptx_binary_arith("add", "f32x2", d_addr, a, b, rounding=rounding, ftz=ftz)


def ptx_add_f64(d_addr, a, b, *, rounding="rn"):
    """PTX ``add{.rnd}.f64 [d_addr], a, b`` — DPS form (no .ftz / .sat)."""
    return _ptx_binary_arith("add", "f64", d_addr, a, b, rounding=rounding)


def ptx_sub_f32(d_addr, a, b, *, rounding="rn", ftz=False, sat=False):
    """PTX ``sub{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("sub", "f32", d_addr, a, b, rounding=rounding, ftz=ftz, sat=sat)


def ptx_sub_f32x2(d_addr, a, b, *, rounding="rn", ftz=False):
    """PTX ``sub{.rnd}{.ftz}.f32x2 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("sub", "f32x2", d_addr, a, b, rounding=rounding, ftz=ftz)


def ptx_sub_f64(d_addr, a, b, *, rounding="rn"):
    """PTX ``sub{.rnd}.f64 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("sub", "f64", d_addr, a, b, rounding=rounding)


def ptx_mul_f32(d_addr, a, b, *, rounding="rn", ftz=False, sat=False):
    """PTX ``mul{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("mul", "f32", d_addr, a, b, rounding=rounding, ftz=ftz, sat=sat)


def ptx_mul_f32x2(d_addr, a, b, *, rounding="rn", ftz=False):
    """PTX ``mul{.rnd}{.ftz}.f32x2 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("mul", "f32x2", d_addr, a, b, rounding=rounding, ftz=ftz)


def ptx_mul_f64(d_addr, a, b, *, rounding="rn"):
    """PTX ``mul{.rnd}.f64 [d_addr], a, b`` — DPS form."""
    return _ptx_binary_arith("mul", "f64", d_addr, a, b, rounding=rounding)


def ptx_fma_f32(d_addr, a, b, c, *, rounding="rn", ftz=False, sat=False):
    """PTX ``fma{.rnd}{.ftz}{.sat}.f32 [d_addr], a, b, c`` — DPS form."""
    return _ptx_fma("f32", d_addr, a, b, c, rounding=rounding, ftz=ftz, sat=sat)


def ptx_fma_f32x2(d_addr, a, b, c, *, rounding="rn", ftz=False):
    """PTX ``fma{.rnd}{.ftz}.f32x2 [d_addr], a, b, c`` — DPS form.

    a, b, c are packed-as-uint64 register operands.
    """
    return _ptx_fma("f32x2", d_addr, a, b, c, rounding=rounding, ftz=ftz)


def ptx_fma_f64(d_addr, a, b, c, *, rounding="rn"):
    """PTX ``fma{.rnd}.f64 [d_addr], a, b, c`` — DPS form."""
    return _ptx_fma("f64", d_addr, a, b, c, rounding=rounding)


def ptx_max_f32(a, b, *, ftz=False, nan=False):
    """TVM intrinsic for PTX ``max{.ftz}{.NaN}.f32 d, a, b``.

    2-operand form (distinct from :func:`ptx_reduce3_max_f32` which is the
    3-operand SM_100+ form). ``.NaN`` qualifier propagates NaN inputs to
    the output; without it, NaN inputs are silently ignored.

    Parameters
    ----------
    a, b : PrimExpr
        Float32 inputs.
    ftz : bool
        If True, flush subnormals to zero (``.ftz``).
    nan : bool
        If True, propagate NaN inputs (``.NaN``).
    """
    return call_intrin("float32", "tirx.ptx.max_f32", a, b, int(ftz), int(nan))


def ptx_griddepcontrol_wait():
    """TVM intrinsic for PTX ``griddepcontrol.wait`` (sm_90+).

    Blocks the current grid until prerequisite grids signalled via
    :func:`ptx_griddepcontrol_launch_dependents` have finished. Acts as a
    full memory barrier.
    """
    return call_intrin("", "tirx.ptx.griddepcontrol_wait")


def ptx_griddepcontrol_launch_dependents():
    """TVM intrinsic for PTX ``griddepcontrol.launch_dependents`` (sm_90+).

    Signals that the current grid has reached a point where dependent
    grids may begin execution.
    """
    return call_intrin("", "tirx.ptx.griddepcontrol_launch_dependents")


_PTX_LD_SCOPE = {"cta", "cluster", "gpu", "sys"}
_PTX_LD_SPACE = {"global", "shared", "shared::cta", "shared::cluster", "local"}
_PTX_LD_VOLATILE_SPACE = _PTX_LD_SPACE | {"const"}
_PTX_LD_TYPE = {"b32", "u32", "u64", "s32", "f32"}
_PTX_LD_COP = {"", "ca", "cg", "cs", "lu", "cv"}
_PTX_MEM_SCOPE = {"", "cta", "cluster", "gpu", "sys"}
_PTX_MEM_SPACE = {"global", "shared", "shared::cta", "shared::cluster"}
_PTX_SCALAR_TYPE = {"b32", "b64", "u32", "u64", "s32", "s64", "f32", "f64"}
_PTX_RED_OP = {"and", "or", "xor", "add", "inc", "dec", "min", "max"}
_PTX_ATOM_OP = {"and", "or", "xor", "exch", "add", "inc", "dec", "min", "max"}
_PTX_ST_VEC = {"", "v2", "v4", "v8"}
_PTX_ST_COP = {"", "wb", "cg", "cs", "wt"}
_PTX_PREFETCH_TENSORMAP_SPACE = {"", "const", "param"}
_PTX_SCALAR_RETURN_TYPE = {
    "b32": "uint32",
    "u32": "uint32",
    "s32": "int32",
    "b64": "uint64",
    "u64": "uint64",
    "s64": "int64",
    "f32": "float32",
    "f64": "float64",
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


def ptx_ld_acquire(addr, return_type, ptx_type, *, scope="gpu", space="global"):
    """TVM intrinsic for scalar PTX ``ld.acquire.scope{.ss}.type`` loads.

    This wrapper covers the scalar no-cache-policy/no-vector instances of the
    PTX ISA ``ld.acquire`` form. ``scope``, state ``space``, PTX ``type`` and
    TVM ``return_type`` are explicit so callers can request either raw-bit or
    typed loads.

    Parameters
    ----------
    addr : PrimExpr
        The memory address to load.

    return_type : str
        TVM dtype returned by the load.

    ptx_type : str
        PTX type suffix such as ``"b32"``, ``"u64"``, or ``"s32"``.

    scope : str
        PTX memory scope: ``"cta"``, ``"cluster"``, ``"gpu"``, or ``"sys"``.

    space : str
        PTX state space suffix.

    Returns
    -------
    call : PrimExpr
        The loaded value.
    """
    _choice("scope", scope, _PTX_LD_SCOPE)
    _choice("space", space, _PTX_LD_SPACE)
    _choice("ptx_type", ptx_type, _PTX_LD_TYPE)
    return call_intrin(
        return_type, "tirx.ptx.ld_acquire", addr, return_type, ptx_type, scope, space
    )


def ptx_ld(
    addr,
    return_type,
    ptx_type,
    *,
    weak=False,
    space="global",
    cop="",
    cache_hint="",
    cache_policy=None,
):
    """TVM intrinsic for scalar PTX ``ld{.weak}{.ss}{.cop}{.level::cache_hint}.type``.

    This wrapper covers scalar no-prefetch/no-vector instances of the weak
    generic load form.
    """
    _choice("space", space, _PTX_LD_SPACE | {"const", "param::entry", "param::func"})
    _choice("cop", cop, _PTX_LD_COP)
    _choice("ptx_type", ptx_type, _PTX_LD_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        return_type,
        "tirx.ptx.ld",
        addr,
        cache_policy,
        return_type,
        int(bool(weak)),
        space,
        cop,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_ld_volatile(addr, return_type, ptx_type, *, space="global"):
    """TVM intrinsic for scalar PTX ``ld.volatile{.ss}.type`` loads.

    This wrapper covers scalar no-prefetch/no-vector instances.
    """
    _choice("space", space, _PTX_LD_VOLATILE_SPACE)
    _choice("ptx_type", ptx_type, _PTX_LD_TYPE)
    return call_intrin(return_type, "tirx.ptx.ld_volatile", addr, return_type, ptx_type, space)


def ptx_ld_global_acquire(res, addr):
    """TVM intrinsic to call the legacy ptx ld.global.acquire helper.

    Parameters
    ----------
    res : PrimExpr
        The result of the load.

    addr : PrimExpr
        The memory address to load.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return call_intrin("", "tirx.ptx.ld_global_acquire", res, addr)


def ptx_red_scalar(
    address,
    value,
    *,
    sem="",
    scope="",
    space="global",
    op,
    ptx_type,
    cache_hint="",
    cache_policy=None,
):
    _choice("scope", scope, _PTX_MEM_SCOPE)
    _choice("space", space, _PTX_MEM_SPACE)
    _choice("op", op, _PTX_RED_OP)
    _choice("ptx_type", ptx_type, _PTX_SCALAR_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(
        cache_hint, cache_policy, _CP_ASYNC_CACHE_HINT
    )
    if sem not in ("", "relaxed", "release"):
        raise ValueError(f"Unsupported PTX red sem {sem!r}")
    return call_intrin(
        "",
        "tirx.ptx.red_scalar",
        address,
        value,
        cache_policy,
        sem,
        scope,
        space,
        op,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_atom_scalar(
    address,
    value,
    *,
    sem="",
    scope="",
    space="global",
    op,
    ptx_type,
    cache_hint="",
    cache_policy=None,
):
    _choice("scope", scope, _PTX_MEM_SCOPE)
    _choice("space", space, _PTX_MEM_SPACE)
    _choice("op", op, _PTX_ATOM_OP)
    _choice("ptx_type", ptx_type, _PTX_SCALAR_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    if sem not in ("", "relaxed", "acquire", "release", "acq_rel"):
        raise ValueError(f"Unsupported PTX atom sem {sem!r}")
    return call_intrin(
        _PTX_SCALAR_RETURN_TYPE[ptx_type],
        "tirx.ptx.atom_scalar",
        address,
        value,
        cache_policy,
        sem,
        scope,
        space,
        op,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_st(
    address,
    *values,
    weak=False,
    space="shared",
    cop="",
    vec="",
    ptx_type,
    cache_hint="",
    cache_policy=None,
):
    _choice("space", space, _PTX_MEM_SPACE | {"local", "param::func"})
    _choice("cop", cop, _PTX_ST_COP)
    _choice("vec", vec, _PTX_ST_VEC)
    _choice("ptx_type", ptx_type, _PTX_SCALAR_TYPE)
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.st",
        address,
        *values,
        cache_policy,
        int(bool(weak)),
        space,
        cop,
        vec,
        ptx_type,
        int(has_cache_policy),
    )


def ptx_st_bulk(ptr, num_bytes, *, weak=False, space="shared::cta"):
    if space not in ("", "shared::cta"):
        raise ValueError(f"Unsupported PTX st.bulk space {space!r}")
    return call_intrin("", "tirx.ptx.st_bulk", ptr, num_bytes, int(bool(weak)), space)


def ptx_prefetch_tensormap(tensormap_addr, space=""):
    _choice("space", space, _PTX_PREFETCH_TENSORMAP_SPACE)
    return call_intrin("", "tirx.ptx.prefetch_tensormap", tensormap_addr, space)


def ptx_mbarrier_test_wait_parity(barrier, phase, *, sem="", scope="", space="shared::cta"):
    if sem not in ("", "acquire", "relaxed"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity sem {sem!r}")
    if scope not in ("", "cta", "cluster"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity scope {scope!r}")
    if bool(sem) != bool(scope):
        raise ValueError("mbarrier.test_wait.parity sem and scope must be set together")
    if space not in ("shared", "shared::cta"):
        raise ValueError(f"Unsupported mbarrier.test_wait.parity space {space!r}")
    return call_intrin(
        "uint32", "tirx.ptx.mbarrier_test_wait_parity", barrier, phase, sem, scope, space
    )


def ptx_cp_async_bulk_g2s_cta(
    dst_ptr,
    src_ptr,
    num_bytes,
    mbarrier_ptr,
    *,
    cache_hint="",
    cache_policy=None,
    ignore_oob=False,
    ignore_bytes_left=0,
    ignore_bytes_right=0,
):
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_g2s_cta",
        dst_ptr,
        src_ptr,
        num_bytes,
        ignore_bytes_left,
        ignore_bytes_right,
        mbarrier_ptr,
        cache_policy,
        int(has_cache_policy),
        int(bool(ignore_oob)),
    )


def ptx_cp_async_bulk_g2s_cluster(
    dst_ptr,
    src_ptr,
    num_bytes,
    mbarrier_ptr,
    *,
    cache_hint="",
    cache_policy=None,
    multicast=False,
    cta_mask=0,
):
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_g2s_cluster",
        dst_ptr,
        src_ptr,
        num_bytes,
        mbarrier_ptr,
        cta_mask,
        cache_policy,
        int(has_cache_policy),
        int(bool(multicast)),
    )


def ptx_cp_async_bulk_s2s_cluster(dst_ptr, src_ptr, num_bytes, mbarrier):
    return call_intrin(
        "", "tirx.ptx.cp_async_bulk_s2s_cluster", dst_ptr, src_ptr, num_bytes, mbarrier
    )


def ptx_cp_async_bulk_s2g(
    dst_ptr, src_ptr, num_bytes, *, cache_hint="", cache_policy=None, cp_mask=False, byte_mask=0
):
    cache_policy, has_cache_policy = _resolve_cache_policy(cache_hint, cache_policy)
    return call_intrin(
        "",
        "tirx.ptx.cp_async_bulk_s2g",
        dst_ptr,
        src_ptr,
        num_bytes,
        byte_mask,
        cache_policy,
        int(has_cache_policy),
        int(bool(cp_mask)),
    )


def ptx_fns_b32(mask, base, offset):
    return call_intrin("uint32", "tirx.ptx.fns_b32", mask, base, offset)


def ptx_add_rn_f32_bf16(acc, x):
    return call_intrin("float32", "tirx.ptx.add_rn_f32_bf16", acc, x)


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


def ptx_map_shared_rank(ptr, rank):
    """TVM intrinsic to call ptx map_shared_rank instruction

    Parameters
    ----------
    ptr: PrimExpr
        The generic pointer to the local shared memory, handle type

    rank: int
        The rank of the distributed shared memory.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return ptx_mapa(ptr, rank, space="", ptx_type="u64", return_type="uint64")


def ptx_mapa(ptr, rank, *, space="", ptx_type="u64", return_type="uint64"):
    """TVM intrinsic for PTX ``mapa{.space}.type d, a, b``."""
    if space not in ("", "shared::cluster"):
        raise ValueError(f"Unsupported mapa space {space!r}")
    if ptx_type not in ("u32", "u64"):
        raise ValueError(f"Unsupported mapa type {ptx_type!r}")
    return call_intrin(return_type, "tirx.ptx.mapa", ptr, rank, space, ptx_type, return_type)


def cuda_atomic_cas(ptr, old_val, new_val):
    """TVM intrinsic to call cuda atomic cas instruction

    Parameters
    ----------
    ptr: PrimExpr
        The pointer to the memory location.

    old_val: PrimExpr
        The old value.

    new_val: PrimExpr
        The new value.

    Returns
    -------
    call : PrimExpr
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
    call : PrimExpr
        The call expression.
    """

    return call_intrin("int32", "tirx.nvshmem.my_pe")


def nvshmem_n_pes():
    """TVM intrinsic to call nvshmem_n_pes()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("int32", "tirx.nvshmem.n_pes")


def nvshmem_getmem_nbi(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_getmem_nbi()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: PrimExpr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per thread.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin("", "tirx.nvshmem.getmem_nbi", dst, src, nelems, pe)


def nvshmem_putmem_nbi(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_putmem_nbi()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the destination data object.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per thread.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.putmem_nbi", dst, src, nelems, pe)


def nvshmem_getmem_nbi_warp(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_getmem_nbi_warp()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: PrimExpr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per warp.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin("", "tirx.nvshmem.getmem_nbi_warp", dst, src, nelems, pe)


def nvshmem_putmem_nbi_warp(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_putmem_nbi_warp()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the destination data object.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per warp.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.putmem_nbi_warp", dst, src, nelems, pe)


def nvshmem_getmem_nbi_block(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_getmem_nbi_block()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be updated.

    src: PrimExpr
        The pointer to the symmetric address of the source data object.

    nelems: int
        The number of bytes to get per block.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin("", "tirx.nvshmem.getmem_nbi_block", dst, src, nelems, pe)


def nvshmem_putmem_nbi_block(dst, src, nelems, pe):
    """TVM intrinsic to call nvshmem_putmem_nbi_block()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the destination data object.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of the data object to be copied.

    nelems: int
        The number of bytes to put per block.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.putmem_nbi_block", dst, src, nelems, pe)


def nvshmem_signal_op(sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_signal_op()

    Parameters
    ----------
    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal word to be updated, must be uint64_t*.

    signal: uint64_t
        The value used to update sig_addr.

    sig_op: str
        Operation used to update sig_addr with signal, typical sig_op values are "set" and "add".

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    _choice("sig_op", sig_op, _NVSHMEM_SIG_OP)
    return call_intrin("", "tirx.nvshmem.signal_op", sig_addr, signal, sig_op, pe)


def nvshmem_wait_until(ivar, cmp, cmp_value, type="uint64_t"):
    """TVM intrinsic to call nvshmem_wait_until()

    Parameters
    ----------
    ivar: PrimExpr
        The pointer to the symmetric address of a remotely accessible data object, must be TYPE*.

    cmp: str
        The compare operator that compares ivar with cmp_value.

    cmp_value: TYPE
        The value to be compared with ivar.

    type: str
        The TYPE of ivar and cmp_value.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    _choice("cmp", cmp, _NVSHMEM_CMP)
    return call_intrin("", "tirx.nvshmem.wait_until", ivar, cmp, cmp_value, type)


def nvshmem_quiet():
    """TVM intrinsic to call nvshmem_quiet()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.quiet")


def nvshmem_putmem_signal_nbi(dst, src, nelems, sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per thread.

    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "", "tirx.nvshmem.putmem_signal_nbi", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_putmem_signal_nbi_warp(dst, src, nelems, sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi_warp()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per warp.

    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "", "tirx.nvshmem.putmem_signal_nbi_warp", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_putmem_signal_nbi_block(dst, src, nelems, sig_addr, signal, sig_op, pe):
    """TVM intrinsic to call nvshmem_putmem_signal_nbi_block()

    Parameters
    ----------
    dst: PrimExpr
        The pointer to the symmetric address of the data object to be updated on the remote PE.

    src: PrimExpr
        The pointer to the symmetric address or host/device address of data object containing the data to be copied.

    nelems: int
        The number of bytes to put per block.

    sig_addr: PrimExpr
        The pointer to the symmetric address of the signal data object to be updated on the remote PE as a signal, must be uint64_t*.

    signal: uint64_t
        The unsigned 64-bit value that is used for updating the remote sig_addr signal data object.

    sig_op: str
        Signal operator that represents the type of update to be performed on the remote sig_addr signal data object.

    pe: int
        The PE number of the remote PE.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """  # noqa: E501

    return call_intrin(
        "", "tirx.nvshmem.putmem_signal_nbi_block", dst, src, nelems, sig_addr, signal, sig_op, pe
    )


def nvshmem_fence():
    """TVM intrinsic to call nvshmem_fence()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.fence")


def nvshmem_barrier_all():
    """TVM intrinsic to call nvshmem_barrier_all()

    Returns
    -------
    call : PrimExpr
        The call expression.
    """

    return call_intrin("", "tirx.nvshmem.barrier_all")
