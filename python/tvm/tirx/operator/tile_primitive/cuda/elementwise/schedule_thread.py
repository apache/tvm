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

"""Schedule A: per-thread vectorized serial loop (scope == thread).

Generic over arity — iterates ``plan.srcs`` without knowing about
unary/binary/cast/fma. The op-specific math is delegated to ``spec.compute``.
"""

from __future__ import annotations

from tvm.script import tirx as Tx
from tvm.tirx import PrimFunc, TilePrimitiveCall
from tvm.tirx.operator.tile_primitive import DispatchContext, fail

from ..common import get_indices, get_st_extent
from ._common import (
    compute_dtype_of,
    fetch_src_value,
    infer_vec_len,
    n_elements,
)
from .schema import OpSpec


def validate_per_thread(spec: OpSpec):
    """Predicate factory for ``per_thread``:

    Accepts:
      (a) scope == thread + all buf-region srcs in local scope
      (b) scope > thread (warp/warpgroup/cta) + all buf-region srcs in local
          scope AND all have trivial layouts (i.e. flat thread-private regs,
          no collective tile semantics — each thread independently runs the
          loop on its own private copy). Used by e.g. tests where binary is
          called at cta scope on flat local bufs.
    """

    def _check(op: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
        plan, msg = spec.parse(op)
        if msg is not None or plan is None:
            return False, msg
        if plan.dst.buffer.scope() != "local":
            return False, f"dst scope must be local, got {plan.dst.buffer.scope()}"
        for s in plan.srcs:
            if s.buf_region is not None and s.buf_region.buffer.scope() != "local":
                return False, "all buffer-region srcs must be in local scope"

        if not sctx.is_thread:
            # Path (b): allowed only if all bufs are trivial (no non-trivial layout).
            if sctx.scope_kind not in ("warp", "warpgroup", "cta"):
                return False, f"per_thread unsupported scope {sctx.scope_kind}"
            dst_lay = plan.dst.buffer.layout
            if dst_lay is not None and not dst_lay.is_trivial():
                return False, "non-trivial dst layout — use tile_local instead"
            for s in plan.srcs:
                if s.buf_region is None:
                    continue
                lay = s.buf_region.buffer.layout
                if lay is not None and not lay.is_trivial():
                    return False, "non-trivial src layout — use tile_local instead"

        if spec.check_extras is not None:
            ok, why = spec.check_extras(plan.extras, compute_dtype_of(plan))
            if not ok:
                return False, why
        return True, None

    return _check


def emit_per_thread(op_call: TilePrimitiveCall, spec: OpSpec, sctx: DispatchContext) -> PrimFunc:
    plan, msg = spec.parse(op_call)
    if msg is not None or plan is None:
        fail(msg or "parse failed")
    dst = plan.dst.buffer
    dst_st, dst_ext = get_st_extent(plan.dst)
    total = n_elements(plan.dst)
    vec_len = infer_vec_len(op_call, plan, thread_cnt=1, fallback_to_scalar=False)
    if vec_len is None:
        fail("could not infer vec_len for per_thread")

    # Try vector intrinsic emit first (e.g. add.<rm>.ftz.f32x2 for sm100 f32).
    # Carries PTX-level attrs (rounding_mode etc.) that scalar `a+b` cannot.
    if spec.vec_emit_factory is not None:
        impl = spec.vec_emit_factory(op_call, plan, sctx, vec_len)
        if impl is not None:
            return impl

    compute = spec.compute
    srcs = plan.srcs
    extras = plan.extras

    @Tx.prim_func(check_well_formed=False)
    def impl():
        with Tx.thread():
            for s in Tx.serial(0, total // vec_len):
                for vec in Tx.vectorized(vec_len):
                    fused = Tx.meta_var(s * vec_len + vec)
                    dst_idx = Tx.meta_var(get_indices(fused, dst_st, dst_ext))
                    # Build src expressions in Python (Tx.meta_var binds the
                    # list at meta-time so it isn't parsed as an IR alloc).
                    src_vals = Tx.meta_var(
                        [fetch_src_value(src, fused, dst_idx, dst_st, dst_ext) for src in srcs]
                    )
                    dst[tuple(dst_idx)] = Tx.cast(compute(src_vals, extras, dst.dtype), dst.dtype)

    return impl
