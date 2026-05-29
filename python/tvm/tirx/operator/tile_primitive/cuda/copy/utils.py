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
"""Shared helpers for copy operator dispatches on CUDA targets."""

from collections.abc import Iterable

import tvm
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import fail
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import get_st_extent, get_vec_len, match_scope, validate_copy_op


def _is_valid_smem_tmem_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
    """Validate smem->tmem copy operation.

    The new tcgen05.cp.32x128b.warpx4 dispatch requires the destination tmem
    buffer to declare warpx4 broadcast as ``R[4 : 32@TLane]``. The legacy
    128-row dispatch path (no replica) goes through a separate code path and
    is not handled here.
    """
    dst_region, src_region = op_call.args[:2]
    src: Buffer = src_region.buffer
    dst: Buffer = dst_region.buffer
    if not (src.scope().startswith("shared") and dst.scope() == "tmem"):
        return (False, f"expected shared->tmem, got {src.scope()}->{dst.scope()}")
    if not (src.layout and dst.layout):
        return (False, "both buffers must have layouts")
    if dst.allocated_addr is None:
        return (False, "tmem buffer must have allocated_addr")
    # Require warpx4 router on TMEM side so this dispatch only handles the
    # 32x128b.warpx4 case; other shapes (128x256b/128x128b etc.) fall back
    # to the legacy dispatch.
    rep = dst.layout.replica
    if not (
        len(rep) == 1
        and int(rep[0].extent) == 4
        and int(rep[0].stride) == 32
        and "TLane" in str(rep[0].axis)
    ):
        return (False, f"requires R[4:32@TLane] on tmem, got replica={list(rep)}")
    return (True, None)


def _single_thread_exec(op_call: TilePrimitiveCall, sctx: DispatchContext):
    """Predicate: exec scope must be single-thread."""
    exec_scope = sctx.scope_kind
    ok = exec_scope == "thread"
    return (ok, None if ok else f"expected thread exec_scope, got {exec_scope}")


DEFAULT_ALLOWED_PAIRS: tuple[tuple[str, str], ...] = (
    ("global", "shared*"),
    ("shared*", "global"),
    ("global", "local"),
    ("local", "global"),
    ("shared*", "local"),
    ("local", "shared*"),
)


def _scope_allowed(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    allowed_pairs: Iterable[tuple[str, str]] = DEFAULT_ALLOWED_PAIRS,
):
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = (op_call.dst, op_call.src)
    src_scope = src_buffer_region.buffer.scope()
    dst_scope = dst_buffer_region.buffer.scope()
    ok = any(
        (
            match_scope(src_scope, src_pat) and match_scope(dst_scope, dst_pat)
            for src_pat, dst_pat in allowed_pairs
        )
    )
    if not ok:
        allowed_str = ", ".join((f"{a}->{b}" for a, b in allowed_pairs))
        return (
            False,
            f"unsupported memory scopes src={src_scope} dst={dst_scope}; allowed: {allowed_str}",
        )
    return (True, None)


def _is_valid_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
    return (validate_copy_op(op_call, sctx), "validate_copy_op failed")


def _vec_len_possible(op_call: TilePrimitiveCall, sctx: DispatchContext):
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = (op_call.dst, op_call.src)
    if sctx.is_cta:
        tx = sctx.launch_params["threadIdx.x"].dom.extent
    elif sctx.is_thread:
        tx = 1
    else:
        return (False, f"unsupported exec_scope {sctx.scope_kind} for vec_len")
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is None:
        vec_len = get_vec_len(
            dst_buffer_region,
            src_buffer_region,
            [
                128 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
                64 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
                32 // tvm.runtime.DataType(src_buffer_region.buffer.dtype).bits,
                1,
            ],
            thread_cnt=tx,
        )
    if vec_len is None:
        return (False, "no valid vector length; check alignment/extents/thread-count")
    return (True, None)


def copy_default_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc | None:
    """Schedule copy operation
    The implementation serves as a fallback for copy operations that uses a single thread
    to move data element by element.
    """
    op_call = TilePrimitiveCall.downcast(op_call)
    dst_buffer_region, src_buffer_region = (op_call.dst, op_call.src)
    src: Buffer = src_buffer_region.buffer
    dst: Buffer = dst_buffer_region.buffer
    src_st, src_extent = get_st_extent(src_buffer_region)
    dst_st, dst_extent = get_st_extent(dst_buffer_region)

    def copy(dst, src):
        dst_indices = [i for i in range(len(dst.shape)) if dst_extent[i] != 1]
        src_indices = [i for i in range(len(src.shape)) if src_extent[i] != 1]
        assert len(dst_indices) == len(src_indices)
        copy_extents = [dst_extent[i] for i in dst_indices]

        def get_dst_coord(lvs):
            if isinstance(lvs, tvm.tirx.Var):
                lvs = [lvs]
            coord = [dst_st[i] for i in range(len(dst.shape))]
            for i, lv in enumerate(lvs):
                coord[dst_indices[i]] += lv
            return coord

        def get_src_coord(lvs):
            if isinstance(lvs, tvm.tirx.Var):
                lvs = [lvs]
            coord = [src_st[i] for i in range(len(src.shape))]
            for i, lv in enumerate(lvs):
                coord[src_indices[i]] += lv
            return coord

        with Tx.grid(*copy_extents) as lvs:
            Tx.buffer_store(dst, src[tuple(get_src_coord(lvs))], get_dst_coord(lvs))

    if sctx.is_cta:
        tx = sctx.launch_params["threadIdx.x"].dom.extent
        assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

        @Tx.prim_func(check_well_formed=False)
        def impl():
            for tid_x in Tx.thread_binding(tx, "threadIdx.x"):
                if tid_x == 0:
                    copy(dst, src)
            if dst.scope().startswith("shared"):
                Tx.tvm_storage_sync("shared")
    elif sctx.is_thread:

        @Tx.prim_func(check_well_formed=False)
        def impl():
            copy(dst, src)
    else:
        fail(f"unsupported exec_scope {sctx.scope_kind}")
    return impl
