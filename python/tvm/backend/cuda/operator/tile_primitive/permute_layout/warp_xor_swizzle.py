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

"""CUDA permute_layout dispatch: warp register-staged in-place transpose with
optional per-lane XOR-swizzle to avoid SMEM bank conflicts on the write phase.

The dispatcher reasons about the **layout's shard**, not the buffer's
declared shape (the two can differ — a buffer with ``shape=(PIPE, M, K)``
may carry a layout whose shard has more dims internally, with grouping
mapping shard segments onto buffer dims).  Concretely:

    src_sliced = src.layout.slice(src.shape, region).canonicalize()
    dst_sliced = dst.layout.slice(dst.shape, region).canonicalize()
    # If the two sliced shards have different structures (which is common —
    # a linear layout collapses to 1D under canon while a transposed one
    # keeps its multi-dim structure), regroup src to dst's shape.
    if src_sliced.shard != dst_sliced.shard:
        src_sliced, _ = src_sliced.group(dst.shard.extents)
    extent  = [int(it.extent) for it in dst_sliced.shard]   # iteration shape
    src_str = [int(it.stride) for it in src_sliced.shard]
    dst_str = [int(it.stride) for it in dst_sliced.shard]

The algorithm:

    regs[P]
    for r in 0..P:
        j  = r XOR ((lane >> SHIFT) & MASK)
        i  = lane + j * 32                             # flat logical index
        idx = decompose(i, extent)                     # iter multi-dim index
        regs[r] = src[project(idx, src.shape, slice_starts)]
    warp_sync()
    for r in 0..P:
        j  = r XOR ((lane >> SHIFT) & MASK)
        i  = lane + j * 32
        idx = decompose(i, extent)
        dst[project(idx, dst.shape, slice_starts)] = regs[r]
    warp_sync()

where ``project`` mixed-radix-folds the iter shard dims back onto the
buffer's iterated slice dims (so the emit's index matches buf.shape rank,
which TIR's BufferLoad/Store requires).

SHIFT and MASK are chosen by simulating the bank pattern at the **shard
granularity** (where strides are affine), trying k = 0, 1, …, log2(P)
and picking the smallest k that makes both phases bank-conflict-free.

Correctness rests on:

* For each lane, ``r ↦ r XOR const`` is a bijection on ``[0, P)``.
* Therefore (lane, r) ↔ flat over [0, V).
* Both layouts are verified bijections on the slice (every logical
  position has a unique byte offset under that layout).
* The mixed-radix projection from iter shard idx to buf coord is exactly
  what TIR's BufferLoad does internally when buf.shape rank < shard rank
  — so iter shard's strides and the buffer-indexed byte offset agree.
"""

from __future__ import annotations

import math

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, BufferRegion, IntImm, PrimFunc
from tvm.tirx.layout import TileLayout, _flatten_coord
from tvm.tirx.operator.tile_primitive import DispatchContext, fail, register_dispatch
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..common import get_indices, get_st_extent

# ---------- helpers ----------------------------------------------------------


def _as_buffer_and_region(arg):
    """Normalize a Buffer or BufferRegion to (buffer, start_list, extent_list)."""
    if isinstance(arg, Buffer):
        buf = arg
        extent = list(buf.shape)
        st = [0] * len(extent)
    elif isinstance(arg, BufferRegion):
        buf = arg.buffer
        st, extent = get_st_extent(arg)
    else:
        raise TypeError(f"unexpected permute_layout arg type: {type(arg)}")
    return buf, list(st), list(extent)


def _as_int(x):
    """Return int(x) if x is int-like, else None."""
    if isinstance(x, int):
        return x
    if isinstance(x, IntImm):
        return int(x.value)
    if hasattr(x, "value") and isinstance(x.value, int):
        return int(x.value)
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _layout_shard_int(layout):
    """Return (extents, strides) as int lists from a TileLayout's shard, or (None, None)."""
    if not isinstance(layout, TileLayout):
        return None, None
    extents, strides = [], []
    for it in layout.shard:
        e = _as_int(it.extent)
        s = _as_int(it.stride)
        if e is None or s is None:
            return None, None
        extents.append(e)
        strides.append(s)
    return extents, strides


def _decompose_row_major(i, extent):
    out, rem = [], i
    for e in reversed(extent):
        out.append(rem % e)
        rem //= e
    return list(reversed(out))


def _eval_offset(idx, strides):
    return sum(i * s for i, s in zip(idx, strides))


def _check_bijection(extent, strides):
    """Iteration extents + strides define a bijection on [0, V)?"""
    V = math.prod(extent)
    seen = set()
    for i in range(V):
        off = _eval_offset(_decompose_row_major(i, extent), strides)
        if off in seen:
            return False
        seen.add(off)
    return len(seen) == V


def _bank_free(extent, strides, dtype_bytes, P, k):
    """For every register slot r ∈ [0, P), do the 32 lanes hit 32 distinct banks?"""
    T, BANKS, BANK_W = 32, 32, 4
    shift = 5 - k
    mask = (1 << k) - 1
    for r in range(P):
        seen = set()
        for lane in range(T):
            j = r ^ ((lane >> shift) & mask)
            flat = lane + j * T
            idx = _decompose_row_major(flat, extent)
            off_bytes = _eval_offset(idx, strides) * dtype_bytes
            bank = (off_bytes // BANK_W) % BANKS
            if bank in seen:
                return False
            seen.add(bank)
    return True


def _choose_xor_k(extent, src_strides, dst_strides, dtype_bytes, P):
    max_k = int(math.log2(P)) if P > 0 else 0
    for k in range(max_k + 1):
        if _bank_free(extent, src_strides, dtype_bytes, P, k) and _bank_free(
            extent, dst_strides, dtype_bytes, P, k
        ):
            return k
    return None


# ---------- validator + dispatch impl ---------------------------------------


def _gather(op_call):
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_arg, src_arg = op_call.args[0], op_call.args[1]
    src_buf, src_st, src_ext = _as_buffer_and_region(src_arg)
    dst_buf, dst_st, dst_ext = _as_buffer_and_region(dst_arg)
    return src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext


def _why_reject(op_call, sctx):
    if not sctx.is_warp:
        return f"scope {sctx.scope_kind!r} is not 'warp'"
    if "threadIdx.y" in sctx.launch_params or "threadIdx.z" in sctx.launch_params:
        return "multi-dim threadIdx is not supported"

    src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext = _gather(op_call)

    if src_buf.dtype != dst_buf.dtype:
        return f"dtype mismatch: dst={dst_buf.dtype} vs src={src_buf.dtype}"

    src_ext_i = [_as_int(e) for e in src_ext]
    dst_ext_i = [_as_int(e) for e in dst_ext]
    if None in src_ext_i or None in dst_ext_i:
        return "extents must be compile-time integers"
    if src_ext_i != dst_ext_i:
        return f"slice shape mismatch: src={src_ext_i} vs dst={dst_ext_i}"

    dtype_bytes = DataType(src_buf.dtype).bits // 8
    if dtype_bytes not in (1, 2, 4, 8, 16):
        return f"unsupported dtype byte width: {dtype_bytes}"

    if not isinstance(src_buf.layout, TileLayout):
        return "src buffer's layout is not a plain TileLayout"
    if not isinstance(dst_buf.layout, TileLayout):
        return "dst buffer's layout is not a plain TileLayout"

    # Slice + canonicalize both layouts.  The result's shard describes the
    # iteration domain; runtime starts (like ``ks``) are folded into the
    # layout's offset, separate from the shard's affine part.
    src_region = [(s, s + e) for s, e in zip(src_st, src_ext)]
    dst_region = [(s, s + e) for s, e in zip(dst_st, dst_ext)]
    src_sliced = src_buf.layout.slice(list(src_buf.shape), src_region)
    dst_sliced = dst_buf.layout.slice(list(dst_buf.shape), dst_region)
    if src_sliced is None or dst_sliced is None:
        return "layout.slice failed"
    src_sliced = src_sliced.canonicalize()
    dst_sliced = dst_sliced.canonicalize()

    # Iteration shape: regroup dst onto the iterated buf dims; the result's
    # shard may stay finer than iter_buf_extents (one buf dim ↔ several shard
    # dims via seps), which is fine.  Then regroup src to match dst's shard
    # extents exactly so both phases share the same iteration index space.
    iter_buf_extents = [e for e in src_ext_i if e != 1]
    try:
        dst_grouped, dst_seps = dst_sliced.group(iter_buf_extents)
        src_grouped, _ = src_sliced.group([int(it.extent) for it in dst_grouped.shard])
    except Exception as e:
        return f"layout.group failed: {e}"

    dst_ext_, dst_str_ = _layout_shard_int(dst_grouped)
    src_ext_, src_str_ = _layout_shard_int(src_grouped)
    if dst_ext_ is None or src_ext_ is None:
        return "regrouped layout shard contains non-integer extent/stride"
    if src_ext_ != dst_ext_:
        return f"src shard {src_ext_} doesn't match dst shard {dst_ext_} after regrouping"

    extent = dst_ext_
    V = math.prod(extent)
    T = 32
    if V == 0 or V % T != 0:
        return f"volume {V} not divisible by warp size {T}"
    P = V // T
    if P == 0 or (P & (P - 1)) != 0 or P > T:
        return f"per-thread count {P} must be power of 2 in [1, {T}]"

    if not _check_bijection(extent, src_str_):
        return "src layout (regrouped) is not a bijection on the slice"
    if not _check_bijection(extent, dst_str_):
        return "dst layout is not a bijection on the slice"
    return None


def _impl(op_call, sctx):
    src_buf, src_st, src_ext, dst_buf, dst_st, dst_ext = _gather(op_call)
    src_ext_i = [_as_int(e) for e in src_ext]

    src_region = [(s, s + e) for s, e in zip(src_st, src_ext)]
    dst_region = [(s, s + e) for s, e in zip(dst_st, dst_ext)]
    src_sliced = src_buf.layout.slice(list(src_buf.shape), src_region).canonicalize()
    dst_sliced = dst_buf.layout.slice(list(dst_buf.shape), dst_region).canonicalize()

    iter_buf_extents = [e for e in src_ext_i if e != 1]
    dst_grouped, dst_seps = dst_sliced.group(iter_buf_extents)
    src_grouped, _ = src_sliced.group([int(it.extent) for it in dst_grouped.shard])

    extent, dst_str_ = _layout_shard_int(dst_grouped)
    _, src_str_ = _layout_shard_int(src_grouped)
    V = math.prod(extent)
    P = V // 32
    dtype_bytes = DataType(src_buf.dtype).bits // 8

    k_opt = _choose_xor_k(extent, src_str_, dst_str_, dtype_bytes, P)
    if k_opt is None:
        fail(f"no XOR-bits k ∈ [0, log2(P)={int(math.log2(P))}] makes both phases bank-free")

    shift = 5 - k_opt
    mask = (1 << k_opt) - 1

    iter_buf_dims = [i for i, e in enumerate(src_ext_i) if e != 1]
    seps = list(dst_seps)

    def _project(iter_idx, st_list):
        buf_idx = list(st_list)
        for bi in range(len(seps) - 1):
            lo, hi = seps[bi], seps[bi + 1]
            flat = _flatten_coord(iter_idx[lo:hi], extent[lo:hi])
            buf_idx[iter_buf_dims[bi]] = st_list[iter_buf_dims[bi]] + flat
        return tuple(buf_idx)

    tid_x = sctx.launch_params["threadIdx.x"]
    dtype = src_buf.dtype

    # Shared 32/64b: base ptr + stride offset avoids buf[] flatten IMAD path.
    direct = (
        dtype_bytes in (4, 8)
        and "shared" in str(src_buf.scope())
        and "shared" in str(dst_buf.scope())
    )
    ptx_t = f"b{dtype_bytes * 8}"
    # ld/st only move bits, so use an unsigned container of the matching width:
    # ``ptx.ld(..., bN)`` rejects a float return dtype, and the permute is a pure
    # byte shuffle, so a float32/float64 tile loads/stores correctly as uint.
    bits_dtype = f"uint{dtype_bytes * 8}"

    def _iter_off(iter_idx, strides):
        return sum(iter_idx[d] * strides[d] for d in range(len(strides)))

    # fmt: off
    if direct:
        @T.prim_func
        def impl():
            warp_size = T.meta_var(32)
            lane_id = T.meta_var(tid_x % warp_size)
            regs = T.alloc_buffer((P,), bits_dtype, scope="local")
            base_src = T.meta_var(src_buf.ptr_to(list(src_st)))
            base_dst = T.meta_var(dst_buf.ptr_to(list(dst_st)))
            # Phase 1: read via L_src
            for r in T.unroll(0, P):
                j = T.meta_var(r ^ ((lane_id >> shift) & mask))
                flat = T.meta_var(lane_id + j * warp_size)
                iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                off = T.meta_var(_iter_off(iter_idx, src_str_))
                ptr = T.meta_var(T.ptr_byte_offset(base_src, off * dtype_bytes, dtype))
                regs[r] = T.ptx.ld(ptr, bits_dtype, ptx_t, space="shared")
            T.cuda.warp_sync()
            # Phase 2: write via L_dst
            for r in T.unroll(0, P):
                j = T.meta_var(r ^ ((lane_id >> shift) & mask))
                flat = T.meta_var(lane_id + j * warp_size)
                iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                off = T.meta_var(_iter_off(iter_idx, dst_str_))
                ptr = T.meta_var(T.ptr_byte_offset(base_dst, off * dtype_bytes, dtype))
                T.evaluate(T.ptx.st(ptr, regs[r], space="shared", ptx_type=ptx_t))
            T.cuda.warp_sync()
    else:
        @T.prim_func
        def impl():
            warp_size = T.meta_var(32)
            lane_id = T.meta_var(tid_x % warp_size)
            regs = T.alloc_buffer((P,), dtype, scope="local")
            # Phase 1: read via L_src
            for r in T.unroll(0, P):
                j = T.meta_var(r ^ ((lane_id >> shift) & mask))
                flat = T.meta_var(lane_id + j * warp_size)
                iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                src_idx = T.meta_var(_project(iter_idx, src_st))
                regs[r] = src_buf[tuple(src_idx)]
            T.cuda.warp_sync()
            # Phase 2: write via L_dst
            for r in T.unroll(0, P):
                j = T.meta_var(r ^ ((lane_id >> shift) & mask))
                flat = T.meta_var(lane_id + j * warp_size)
                iter_idx = T.meta_var(get_indices(flat, [0] * len(extent), extent))
                dst_idx = T.meta_var(_project(iter_idx, dst_st))
                dst_buf[tuple(dst_idx)] = regs[r]
            T.cuda.warp_sync()
    # fmt: on
    return impl


# === Variant: permute_layout/warp_xor_swizzle (priority=20) ============
#
# When: warp scope; matching dst/src dtype + slice shape; both buffers carry
# a plain TileLayout; after slice + canonicalize (and regrouping src to dst's
# structure if needed), the iteration extents form a power-of-2 ≤32 elements
# per lane; both layouts are bijections on the slice; and there exists an
# XOR-bits ``k`` that makes both phases bank-conflict-free.
#
# Buffer ``shape`` rank does NOT need to equal layout ``shard`` rank — the
# dispatcher uses the layout shard for iteration (after slice+canon) and
# projects back onto ``buf.shape`` via mixed-radix grouping for the emit.
#
# Before (TilePrimitiveCall):
#     with T.warp():
#         # SFA_smem: u32 (PIPE, BLK_SFA//32, 32), layout shard 4D
#         #   (PIPE, BLK_SFA//128, 4, 32) strides (BLK_SFA, 128, 32, 1)
#         # SFA_post: same shape; layout shard 4D, strides (BLK_SFA, 128, 1, 4)
#         Tx.permute_layout(SFA_post[ks, :, :], SFA_smem[ks, :, :])
#
# After (BLK_SFA=128, P=4, k=2, shift=3):
#     lane_id = threadIdx.x % 32
#     regs = T.alloc_buffer((4,), "uint32", scope="local")
#     for r in T.unroll(4):
#         j = r ^ ((lane_id >> 3) & 0x3)
#         flat = lane_id + j * 32
#         (g, l) = decompose(flat, extent=[4, 32])
#         regs[r] = src[ks, g, l]
#     T.cuda.warp_sync()
#     for r in T.unroll(4):
#         j = r ^ ((lane_id >> 3) & 0x3)
#         flat = lane_id + j * 32
#         (g, l) = decompose(flat, extent=[4, 32])
#         dst[ks, g, l] = regs[r]
#     T.cuda.warp_sync()
@register_dispatch(
    "permute_layout",
    "cuda",
    variant="warp_xor_swizzle",
    priority=20,
)
def permute_layout_dispatch(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    reason = _why_reject(op, sctx)
    if reason is not None:
        fail(reason)
    return _impl(op, sctx)


__all__ = [
    "_bank_free",
    "_check_bijection",
    "_choose_xor_k",
    "_decompose_row_major",
    "_eval_offset",
    "permute_layout_dispatch",
]
