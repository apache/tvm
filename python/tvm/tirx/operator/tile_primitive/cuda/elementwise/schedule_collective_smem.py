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

"""Schedule C: shared-buffer fused-tid distribution (scope > thread).

Generic over arity — iterates ``plan.srcs`` and delegates math to
``spec.compute``.
"""

from __future__ import annotations

from tvm.arith.analyzer import Analyzer
from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, fail

from ..common import get_indices, get_st_extent, get_thread_cnt
from ._common import (
    basic_layout_checks,
    compute_dtype_of,
    emit_scope_sync,
    fetch_src_value,
    infer_vec_len,
    n_elements,
    sigs_equal,
    slice_and_sig,
    tid_in_scope_expr,
)
from .schema import OpSpec


def validate_shared(spec: OpSpec):
    """Predicate factory: scope in {thread,warp,warpgroup,cta}; all bufs in shared*."""

    def _check(op: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
        if sctx.scope_kind not in ["thread", "warp", "warpgroup", "cta"]:
            return False, f"unsupported scope {sctx.scope_kind}"
        plan, msg = spec.parse(op)
        if msg is not None or plan is None:
            return False, msg

        if not plan.dst.buffer.scope().startswith("shared"):
            return False, f"dst must be shared*, got {plan.dst.buffer.scope()}"
        if plan.dst.buffer.layout is None:
            return False, "dst must have layout"
        for s in plan.srcs:
            if s.buf_region is None:
                continue
            buf = s.buf_region.buffer
            if not buf.scope().startswith("shared"):
                return False, "src buffer must be shared*"
            if buf.layout is None:
                return False, "src buffer must have layout"

        if spec.check_extras is not None:
            ok, why = spec.check_extras(plan.extras, compute_dtype_of(plan))
            if not ok:
                return False, why

        a = Analyzer()
        for s in plan.srcs:
            if s.buf_region is None or s.index_fn is not None:
                # Skip shape check for broadcasting srcs (have custom index_fn).
                continue
            if not basic_layout_checks(s.buf_region, plan.dst, a, disallow_swizzle=False):
                return False, "shape/layout mismatch between src and dst"

        sigs = [slice_and_sig(plan.dst)[3]]
        for s in plan.srcs:
            if s.buf_region is not None and s.index_fn is None:
                sigs.append(slice_and_sig(s.buf_region)[3])
        if not sigs_equal(a, *sigs):
            return False, "layout signature mismatch"
        return True, None

    return _check


def emit_shared(op_call: TilePrimitiveCall, spec: OpSpec, sctx: DispatchContext) -> PrimFunc:
    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        fail(msg or "parse failed")

    dst = plan.dst.buffer
    dst_st, dst_ext = get_st_extent(plan.dst)
    total = n_elements(plan.dst)
    thread_cnt = get_thread_cnt(sctx)
    if thread_cnt is None:
        fail(f"unsupported scope {sctx.scope_kind} for shared emit")
    assert "threadIdx.y" not in sctx.launch_params and "threadIdx.z" not in sctx.launch_params

    vec_len = infer_vec_len(op_call, plan, thread_cnt=thread_cnt, fallback_to_scalar=True)
    if vec_len is None:
        fail("could not infer vec_len for shared emit")

    compute = spec.compute
    srcs = plan.srcs
    extras = plan.extras
    sync = emit_scope_sync(sctx.scope_kind)

    def _tid():
        return tid_in_scope_expr(sctx, thread_cnt)

    @Tx.prim_func(check_well_formed=False)
    def impl():
        tid = _tid()
        for s in Tx.serial(0, Tx.ceildiv(total, vec_len * thread_cnt)):
            for vec in Tx.vectorized(vec_len):
                fused = Tx.meta_var(s * vec_len * thread_cnt + tid * vec_len + vec)
                if fused < total:
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_ext))
                    src_vals = Tx.meta_var(
                        [fetch_src_value(src, fused, dst_idx, dst_st, dst_ext) for src in srcs]
                    )
                    dst[tuple(dst_idx)] = Tx.cast(compute(src_vals, extras, dst.dtype), dst.dtype)
        sync()

    return impl
