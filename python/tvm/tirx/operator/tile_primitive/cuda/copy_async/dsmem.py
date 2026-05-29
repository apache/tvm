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

"""copy_async dispatch variant: dsmem (shared::cta -> shared::cluster)."""

import functools
import operator

import tvm
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.operator.tile_primitive import (
    DispatchContext,
    fail,
    predicate,
    register_dispatch,
)
from tvm.tirx.stmt import TilePrimitiveCall

from ..common import validate_copy_op
from ..exec_scope_utils import single_thread
from .utils import find_contiguous_region, to_tile_layout


def _is_shared_to_shared(op_call: TilePrimitiveCall) -> bool:
    """Check if both src and dst are in shared memory."""
    op_call = TilePrimitiveCall.downcast(op_call)
    src_scope = op_call.src.buffer.scope()
    dst_scope = op_call.dst.buffer.scope()
    return src_scope.startswith("shared") and dst_scope.startswith("shared")


def copy_dsmem_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Implement shared-to-shared cross-CTA copy using cp.async.bulk.

    Uses cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
    to copy data from the executing CTA's shared memory to a remote CTA's shared
    memory within the same cluster.

    The copy region is decomposed into contiguous byte chunks based on layout
    analysis of both src and dst buffers. Non-contiguous dimensions are iterated
    over, emitting one cp.async.bulk instruction per contiguous chunk.
    """
    op_call = TilePrimitiveCall.downcast(op_call)

    # Extract config
    remote_cta_id = op_call.config.get("remote_cta_id", None)
    if remote_cta_id is None:
        fail("remote_cta_id not set in config")
    mbar = op_call.config.get("mbar", None)
    if mbar is None:
        fail("mbar not set in config")

    # Extract buffer regions
    dst_buffer_region = op_call.dst
    src_buffer_region = op_call.src
    src_buf: Buffer = src_buffer_region.buffer
    dst_buf: Buffer = dst_buffer_region.buffer

    src_st = [r.min for r in src_buffer_region.region]
    src_ext = [r.extent for r in src_buffer_region.region]
    dst_st = [r.min for r in dst_buffer_region.region]
    dst_ext = [r.extent for r in dst_buffer_region.region]

    dtype_bytes = tvm.DataType(src_buf.dtype).bits // 8

    # Get tile layouts for both buffers
    src_tile_layout = to_tile_layout(src_buf.layout, src_buf.shape)
    dst_tile_layout = to_tile_layout(dst_buf.layout, dst_buf.shape)

    # Slice layouts to copy region
    src_region_tuples = [(src_st[i], src_st[i] + src_ext[i]) for i in range(len(src_st))]
    sliced_src = src_tile_layout.slice([s for s in src_buf.shape], src_region_tuples)
    if sliced_src is None:
        fail("Cannot slice src layout for DSMEM copy")

    dst_region_tuples = [(dst_st[i], dst_st[i] + dst_ext[i]) for i in range(len(dst_st))]
    sliced_dst = dst_tile_layout.slice([s for s in dst_buf.shape], dst_region_tuples)
    if sliced_dst is None:
        fail("Cannot slice dst layout for DSMEM copy")

    # Group src layout by region extents, then group dst by src's shard extents
    # This creates 1:1 shard correspondence between the two layouts
    grouped_src, src_seps = sliced_src.canonicalize().group(src_ext)
    src_shard_extents = [s.extent for s in grouped_src.shard]
    grouped_dst, dst_seps = sliced_dst.canonicalize().group(src_shard_extents)

    # Find contiguous regions in both layouts
    src_contig_indices, _ = find_contiguous_region(grouped_src)
    dst_contig_indices, _ = find_contiguous_region(grouped_dst)

    # Intersect: walk from innermost outward, include only matching shard indices
    shared_contig_indices = []
    for s_idx, d_idx in zip(src_contig_indices, dst_contig_indices):
        if s_idx != d_idx:
            break
        shared_contig_indices.append(s_idx)

    # Compute chunk size
    if shared_contig_indices:
        chunk_elements = functools.reduce(
            operator.mul, [grouped_src.shard[i].extent for i in shared_contig_indices], 1
        )
    else:
        chunk_elements = 1

    chunk_bytes = chunk_elements * dtype_bytes
    if chunk_bytes < 16 or chunk_bytes % 16 != 0:
        fail(
            f"Layouts not compatible for bulk DSMEM copy: "
            f"chunk_bytes={chunk_bytes} (need >= 16 and multiple of 16)"
        )

    # Build iteration space over non-contiguous (outer) shards
    shared_contig_set = set(shared_contig_indices)
    outer_shard_indices = [i for i in range(len(grouped_src.shard)) if i not in shared_contig_set]
    outer_extents = [grouped_src.shard[i].extent for i in outer_shard_indices]
    outer_src_strides = [grouped_src.shard[i].stride for i in outer_shard_indices]
    outer_dst_strides = [grouped_dst.shard[i].stride for i in outer_shard_indices]

    # Helper to compute element offsets from loop variables (called via Tx.meta_var)
    def compute_offsets(loop_vars):
        if len(outer_extents) == 1:
            lvs = [loop_vars]
        else:
            lvs = list(loop_vars)
        src_off = 0
        dst_off = 0
        for j, v in enumerate(lvs):
            src_off = src_off + v * outer_src_strides[j]
            dst_off = dst_off + v * outer_dst_strides[j]
        return src_off, dst_off

    src_tile = to_tile_layout(src_buf.layout, src_buf.shape)
    dst_tile = to_tile_layout(dst_buf.layout, dst_buf.shape)

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def impl():
        # Map mbar to remote CTA (complete_tx targets the destination's mbar)
        remote_mbar = Tx.ptx.map_shared_rank(mbar, remote_cta_id)

        if not outer_extents:
            # Single contiguous chunk — no iteration needed
            src_ptr = src_buf.ptr_to(src_st)
            cluster_dst = Tx.ptx.map_shared_rank(dst_buf.ptr_to(dst_st), remote_cta_id)
            Tx.ptx.cp_async.bulk.s2c(cluster_dst, src_ptr, chunk_bytes, remote_mbar)
        else:
            for loop_vars in Tx.grid(*outer_extents):
                src_elem_offset, dst_elem_offset = Tx.meta_var(compute_offsets(loop_vars))

                src_buf_w = Tx.decl_buffer(
                    src_buf.shape, src_buf.dtype, src_buf.data,
                    elem_offset=src_buf.elem_offset + src_elem_offset,
                    scope=src_buf.scope(),
                    layout=src_tile,
                )
                dst_buf_w = Tx.decl_buffer(
                    dst_buf.shape, dst_buf.dtype, dst_buf.data,
                    elem_offset=dst_buf.elem_offset + dst_elem_offset,
                    scope=dst_buf.scope(),
                    layout=dst_tile,
                )

                src_ptr = src_buf_w.ptr_to(src_st)
                cluster_dst = Tx.ptx.map_shared_rank(dst_buf_w.ptr_to(dst_st), remote_cta_id)
                Tx.ptx.cp_async.bulk.s2c(cluster_dst, src_ptr, chunk_bytes, remote_mbar)
    # fmt: on

    return impl


# === Variant: copy_async/dsmem (priority=10) ===
#
# When: valid async copy at single-thread scope where both src and dst are in
# shared memory. Used for intra-cluster DSMEM copies (shared::cta -> shared::cluster).
#
# Before (TilePrimitiveCall):
#     Tx.copy_async(
#         dst_smem[0:128, 0:64],
#         src_smem[0:128, 0:64],
#         config={"mbar": mbar, "remote_cta_id": cta_id}
#     )
#
# After (emits cp.async.bulk.shared::cluster.shared::cta):
#   cluster_dst = mapa(dst_smem.ptr, cta_id)
#   cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
#       [cluster_dst], [src_smem.ptr], size, [mbar]
@register_dispatch(
    "copy_async",
    "cuda",
    variant="dsmem",
    priority=10,
    when=[
        predicate(
            "validate_copy_op", lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op")
        ),
        predicate(
            "single_thread",
            lambda op, sctx: (
                single_thread(op, sctx),
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread",
            ),
        ),
        predicate(
            "is_shared_to_shared",
            lambda op, sctx: (_is_shared_to_shared(op), "not a shared-to-shared copy"),
        ),
    ],
)
def copy_async_dispatch_dsmem(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return copy_dsmem_impl(op, sctx)
