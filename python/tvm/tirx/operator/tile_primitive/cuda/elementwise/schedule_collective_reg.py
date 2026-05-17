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

"""Schedule B: tile-local collective (scope > thread + local buffer + layout).

Generic over arity — iterates ``plan.srcs``. Two sub-paths:

  full   : every buffer-region covers its full buffer; flatten via
           ``decl_buffer((local_total,), ...)`` and iterate the linear index.
  sliced : at least one region is partial; ``buf.local(*shape)`` per buffer
           + multi-dim get_indices per element.
"""

from __future__ import annotations

import functools
import operator

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, fail

from ..common import get_indices, get_st_extent, get_thread_cnt
from ..layout_utils import get_local_region
from ._common import (
    basic_layout_checks,
    buffer_regions,
    compute_dtype_of,
    infer_vec_len,
    is_full_region,
    sigs_equal,
    slice_and_sig,
)
from .schema import OpSpec


def validate_tile_local(spec: OpSpec):
    """Predicate factory: scope in {warp,warpgroup,cta}; all bufs local + layout; sig match."""

    def _check(op: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
        if sctx.scope_kind not in ["warp", "warpgroup", "cta"]:
            return False, f"tile_local requires warp/warpgroup/cta, got {sctx.scope_kind}"
        plan, msg = spec.parse(op)
        if msg is not None or plan is None:
            return False, msg

        if plan.dst.buffer.scope() != "local":
            return False, f"dst scope must be local, got {plan.dst.buffer.scope()}"
        for s in plan.srcs:
            if s.buf_region is None:
                continue
            buf = s.buf_region.buffer
            if buf.scope() != "local":
                return False, "src buffer must be in local scope"

        # tile_local handles three sub-shapes depending on layouts:
        #   (a) all dst + buffer-srcs carry NON-trivial layouts -> shape/sig must match
        #   (b) some buf has trivial (flat thread-private) layout while others have
        #       non-trivial collective layouts -> thread-asymmetric view, e.g.
        #       GEMM epilogue cast `dst_flat[no*8:no*8+8] = cast(src_wg[128, 8])`.
        # We accept both; the emit function picks the right view per buf.
        def _is_nontrivial(buf):
            return buf.layout is not None and not buf.layout.is_trivial()

        any_nontrivial = _is_nontrivial(plan.dst.buffer) or any(
            s.buf_region is not None and _is_nontrivial(s.buf_region.buffer) for s in plan.srcs
        )
        if not any_nontrivial:
            return False, "tile_local requires at least one buf with non-trivial layout"

        if spec.check_extras is not None:
            ok, why = spec.check_extras(plan.extras, compute_dtype_of(plan))
            if not ok:
                return False, why

        a = Analyzer()
        # Only enforce shape/sig equality across the buffers with NON-trivial layouts.
        # Trivially-laid-out (flat thread-private) buffers are validated separately.
        if _is_nontrivial(plan.dst.buffer):
            for s in plan.srcs:
                if (
                    s.buf_region is None
                    or s.index_fn is not None
                    or not _is_nontrivial(s.buf_region.buffer)
                ):
                    continue
                if not basic_layout_checks(s.buf_region, plan.dst, a, disallow_swizzle=True):
                    return False, "shape/layout mismatch between src and dst"

        # Region-level layout constraints — only on bufs with non-trivial layouts.
        for br in buffer_regions(plan):
            if not _is_nontrivial(br.buffer):
                continue
            st, ext = get_st_extent(br)
            layout = br.buffer.layout
            for it in layout.shard:
                if it.axis.is_thread() and a.can_prove_equal(it.stride, 0):
                    return False, "thread axis with zero stride unsupported"
            replica = getattr(layout, "replica", None) or []
            if any(it.axis.is_thread() for it in replica):
                return False, "thread axis in replica unsupported"
            if get_local_region(layout, br.buffer.shape, st, ext) is None:
                return False, "invalid region for tile_local"

        # Layout signatures must agree across all bufs with non-trivial layouts.
        sigs = []
        if _is_nontrivial(plan.dst.buffer):
            sigs.append(slice_and_sig(plan.dst)[3])
        for s in plan.srcs:
            if (
                s.buf_region is not None
                and _is_nontrivial(s.buf_region.buffer)
                and s.index_fn is None
            ):
                sigs.append(slice_and_sig(s.buf_region)[3])
        if not sigs_equal(a, *sigs):
            return False, "layout signature mismatch"

        # Launch-thread consistency: pick any buf with non-trivial layout as anchor.
        anchor_br = (
            plan.dst
            if _is_nontrivial(plan.dst.buffer)
            else next(
                s.buf_region
                for s in plan.srcs
                if s.buf_region is not None and _is_nontrivial(s.buf_region.buffer)
            )
        )
        _, _, anchor_sliced, _ = slice_and_sig(anchor_br)
        thr_extents = [it.extent for it in anchor_sliced.shard if it.axis.is_thread()]
        expected = functools.reduce(operator.mul, thr_extents, 1)
        actual = get_thread_cnt(sctx)
        if thr_extents and not a.can_prove_equal(expected, actual):
            return False, f"thread count mismatch: expected {expected} got {actual}"
        return True, None

    return _check


def emit_tile_local(op_call: TilePrimitiveCall, spec: OpSpec, sctx: DispatchContext) -> PrimFunc:
    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        fail(msg or "parse failed")

    # Try vector intrinsic emit first (e.g. packed_f32x2 for sm100 f32 op).
    if spec.vec_emit_factory is not None:
        impl = spec.vec_emit_factory(op_call, plan, sctx, vec_len=2)
        if impl is not None:
            return impl

    # If any buffer lacks layout, we can't use the fast "full" flat path
    # uniformly — fall through to sliced which handles per-buf views.
    has_flat_buf = (plan.dst.buffer.layout is None or plan.dst.buffer.layout.is_trivial()) or any(
        s.buf_region is not None
        and (s.buf_region.buffer.layout is None or s.buf_region.buffer.layout.is_trivial())
        for s in plan.srcs
    )
    full = (
        not has_flat_buf
        and is_full_region(plan.dst)
        and all(s.buf_region is None or is_full_region(s.buf_region) for s in plan.srcs)
    )
    if full:
        return _emit_full(op_call, spec, plan)
    return _emit_sliced(op_call, spec, sctx, plan)


# -----------------------------------------------------------------------------
# Full-region: flatten each local buffer to (local_total,) and iterate linear idx.
# -----------------------------------------------------------------------------
def _emit_full(op_call: TilePrimitiveCall, spec, plan) -> PrimFunc:
    dst = plan.dst.buffer
    dst_st, dst_ext = get_st_extent(plan.dst)
    dst_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_ext)
    if not dst_info:
        fail("dst layout not supported for tile_local (full)")
    _, _, dst_local_ext = dst_info
    local_total = functools.reduce(operator.mul, dst_local_ext, 1)

    # vec_len: use op_call.config or infer from local_total alignment.
    vec_len = op_call.config.get("vec_len", None)
    if vec_len is None:
        a = Analyzer()
        ele = DataType(dst.dtype).bits
        for s in plan.srcs:
            if s.buf_region is not None:
                ele = max(ele, DataType(s.buf_region.buffer.dtype).bits)
        for v in [128 // ele, 64 // ele, 32 // ele, 1]:
            if v > 0 and a.can_prove_equal(local_total % v, 0):
                vec_len = v
                break
    assert vec_len is not None

    compute = spec.compute
    extras = plan.extras
    srcs = plan.srcs

    # Pre-extract the underlying buffers for buffer-region srcs (None for scalars).
    src_buffers = [s.buf_region.buffer if not s.is_scalar else None for s in srcs]

    @Tx.prim_func(check_well_formed=False)
    def impl():
        with Tx.thread():
            base_dst = Tx.decl_buffer((local_total,), dst.dtype, dst.data, scope=dst.scope())
            # Hoist one flat decl per buffer src.
            bases = Tx.meta_var(
                [
                    None
                    if b is None
                    else Tx.decl_buffer((local_total,), b.dtype, b.data, scope=b.scope())
                    for b in src_buffers
                ]
            )
            for s in Tx.serial(0, local_total // vec_len):
                for vec in Tx.vectorized(vec_len):
                    idx = Tx.meta_var(s * vec_len + vec)
                    src_vals = Tx.meta_var(
                        [
                            src.scalar if src.is_scalar else bases[i][idx]
                            for i, src in enumerate(srcs)
                        ]
                    )
                    base_dst[idx] = Tx.cast(compute(src_vals, extras, dst.dtype), dst.dtype)

    return impl


# -----------------------------------------------------------------------------
# Sliced-region: buf.local(*shape) per buffer + multi-dim index decomp.
# -----------------------------------------------------------------------------
def _emit_sliced(op_call: TilePrimitiveCall, spec, sctx: DispatchContext, plan) -> PrimFunc:
    thread_cnt = get_thread_cnt(sctx)
    assert thread_cnt is not None

    dst = plan.dst.buffer
    dst_st, dst_ext = get_st_extent(plan.dst)

    # Pick an anchor buf (the one with layout) to determine per-thread element count.
    if dst.layout is not None and not dst.layout.is_trivial():
        anchor_info = get_local_region(dst.layout, list(dst.shape), dst_st, dst_ext)
        if not anchor_info:
            fail("dst layout not supported for tile_local (sliced)")
    else:
        anchor_info = None
        for src in plan.srcs:
            if src.buf_region is not None and src.buf_region.buffer.layout is not None:
                b = src.buf_region.buffer
                st, ext = get_st_extent(src.buf_region)
                anchor_info = get_local_region(b.layout, b.shape, st, ext)
                if anchor_info is not None:
                    break
        if anchor_info is None:
            fail("no anchor with valid layout for tile_local (sliced)")
    _, _, anchor_local_ext = anchor_info
    local_total = functools.reduce(operator.mul, anchor_local_ext, 1)

    vec_len = infer_vec_len(op_call, plan, thread_cnt=thread_cnt, fallback_to_scalar=True)
    if vec_len is None:
        fail("could not infer vec_len for tile_local (sliced)")

    # Per-buf access info: ("layout", local_info) for layout-bearing bufs,
    # or ("flat", (None, region_st, region_ext)) for bufs without layout.
    dst_has_layout = dst.layout is not None and not dst.layout.is_trivial()
    if dst_has_layout:
        dst_local_shape, dst_local_st, dst_local_ext = (
            anchor_info
            if anchor_info[0] is not None
            else get_local_region(dst.layout, list(dst.shape), dst_st, dst_ext)
        )
    else:
        dst_local_shape = None
        dst_local_st = dst_st
        dst_local_ext = dst_ext

    per_src_info: list = []
    for src in plan.srcs:
        if src.buf_region is None:
            per_src_info.append(None)
            continue
        b = src.buf_region.buffer
        st, ext = get_st_extent(src.buf_region)
        if b.layout is not None and not b.layout.is_trivial():
            info = get_local_region(b.layout, b.shape, st, ext)
            if not info:
                fail("src layout not supported for tile_local (sliced)")
            per_src_info.append(("layout", info))
        else:
            per_src_info.append(("flat", (None, st, ext)))

    compute = spec.compute
    extras = plan.extras
    srcs = plan.srcs
    src_buffers = [s.buf_region.buffer if not s.is_scalar else None for s in srcs]

    if dst_has_layout:

        @Tx.prim_func(check_well_formed=False)
        def impl():
            with Tx.thread():
                dst_view = dst.local(*dst_local_shape)
                src_views = Tx.meta_var(
                    [
                        None
                        if per_src_info[i] is None or per_src_info[i][0] == "flat"
                        else src_buffers[i].local(*per_src_info[i][1][0])
                        for i in range(len(srcs))
                    ]
                )
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        idx_dst = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                        src_vals = Tx.meta_var(
                            [
                                src.scalar
                                if src.is_scalar
                                else (
                                    src_views[i][
                                        tuple(
                                            get_indices(
                                                fused,
                                                per_src_info[i][1][1],
                                                per_src_info[i][1][2],
                                            )
                                        )
                                    ]
                                    if per_src_info[i][0] == "layout"
                                    else src_buffers[i][
                                        tuple(
                                            get_indices(
                                                fused,
                                                per_src_info[i][1][1],
                                                per_src_info[i][1][2],
                                            )
                                        )
                                    ]
                                )
                                for i, src in enumerate(srcs)
                            ]
                        )
                        dst_view[tuple(idx_dst)] = Tx.cast(
                            compute(src_vals, extras, dst.dtype), dst.dtype
                        )

    else:
        # dst is trivially laid out (flat thread-private) — index it directly.
        @Tx.prim_func(check_well_formed=False)
        def impl():
            with Tx.thread():
                src_views = Tx.meta_var(
                    [
                        None
                        if per_src_info[i] is None or per_src_info[i][0] == "flat"
                        else src_buffers[i].local(*per_src_info[i][1][0])
                        for i in range(len(srcs))
                    ]
                )
                for s in Tx.serial(0, local_total // vec_len):
                    for vec in Tx.vectorized(vec_len):
                        fused = Tx.meta_var(s * vec_len + vec)
                        idx_dst = Tx.meta_var(get_indices(fused, dst_local_st, dst_local_ext))
                        src_vals = Tx.meta_var(
                            [
                                src.scalar
                                if src.is_scalar
                                else (
                                    src_views[i][
                                        tuple(
                                            get_indices(
                                                fused,
                                                per_src_info[i][1][1],
                                                per_src_info[i][1][2],
                                            )
                                        )
                                    ]
                                    if per_src_info[i][0] == "layout"
                                    else src_buffers[i][
                                        tuple(
                                            get_indices(
                                                fused,
                                                per_src_info[i][1][1],
                                                per_src_info[i][1][2],
                                            )
                                        )
                                    ]
                                )
                                for i, src in enumerate(srcs)
                            ]
                        )
                        dst[tuple(idx_dst)] = Tx.cast(
                            compute(src_vals, extras, dst.dtype), dst.dtype
                        )

    return impl
