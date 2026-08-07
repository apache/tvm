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

"""``copy_async`` dispatch for ``global → shared`` via ``cp.async``
(SASS: ``LDGSTS``).

Shares the partition / layout-alignment algorithm with
``cuda/copy/vec_auto_gmem_smem.py`` (sync ``T.copy`` global ↔ shared); differs at
emit time only:

* direction: ``cp.async`` is global → shared only (hardware restriction).
* cp_size: PTX ``cp.async`` only accepts 4 / 8 / 16 bytes, so the vec-width
  candidate set is restricted to ``{32, 64, 128}`` bits.
* emit: one ``T.ptx["cp.async..."]`` per slice instead of the synchronous
  ``T.cuda.copy_{vec_bits}b(dst, src)``.
* config: ``prefetch_size`` lands in the instruction spelling, ``predicate``
  rides ``pred=`` (@p), and ``fill_mode="zero"`` selects the src-size arity
  (zero-filling the tail).  This covers predicated zero-fill loads such as
  FlashMLA RoPE KV staging while preserving the same layout partitioner.
* config: ``direct=True`` is an explicit thread-scope fast path for callers
  that already selected a physically contiguous 4/8/16-byte slice.  It emits
  one ``cp.async`` from the region starts and bypasses the synthesized
  partitioner, matching hand-written per-thread copies.

Note: ``cp.async`` does **not** sync at emit time — caller is responsible
for ``commit_group`` / ``wait_group`` / ``cta_sync`` plumbing around the
async pipeline.
"""

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.operator.tile_primitive.dispatcher import (
    predicate,
    register_dispatch,
)
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..copy._common import (
    _TID_AXIS_FOR_SCOPE,
    _thread_cnt,
    align_layouts_gs,
)
from ..copy.utils import _is_valid_copy, _scope_allowed
from ..copy.vec_auto_reg import _all_threads_active, _axis_decl, _ptr_off
from ..layout_utils import recompose_swizzle

# cp.async is unidirectional: global → shared.
_LDGSTS_PAIRS = [("global", "shared*")]
# cp.async cp_size ∈ {4, 8, 16} bytes ⇒ vec_bits ∈ {32, 64, 128}.
_LDGSTS_VEC_BITS = (128, 64, 32)


def _emit_cp_async(dst_ptr, src_ptr, cp_size, prefetch_size, predicate_expr, fill_mode):
    """Emit one ptx cp.async for the ldgsts configuration.

    fill_mode="zero" is the src-size arity (src-size = pred ? cp-size : 0,
    zero-filling the tail); a bare predicate rides pred= (@p); otherwise the
    plain form. The instruction spelling carries the prefetch qualifier.
    """
    from tvm.tirx.op import if_then_else

    cop = "cg" if int(cp_size) == 16 else "ca"
    pref = "" if int(prefetch_size) == -1 else f".L2::{int(prefetch_size)}B"
    chain = f"cp.async.{cop}.shared.global{pref}"
    is_default = (isinstance(predicate_expr, int) and predicate_expr == -1) or (
        hasattr(predicate_expr, "value") and int(predicate_expr.value) == -1
    )
    has_pred = not is_default
    if fill_mode == "zero":
        src_size = T.cast(if_then_else(predicate_expr != 0, int(cp_size), 0), "uint32")
        T.evaluate(T.ptx[chain](dst_ptr, src_ptr, int(cp_size), src_size))
    elif has_pred:
        T.evaluate(T.ptx[chain](dst_ptr, src_ptr, int(cp_size), pred=predicate_expr))
    else:
        T.evaluate(T.ptx[chain](dst_ptr, src_ptr, int(cp_size)))


def _config_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, _IntImm):
        return bool(int(value.value))
    return bool(value)


def _divides_thread_cnt_ldgsts(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Mirror of ``gmem_smem._divides_thread_cnt``: reject copies whose
    region element count doesn't divide ``thread_cnt`` (and reject
    ``thread_cnt=0`` scopes outright). See that docstring for rationale."""
    op_call = TilePrimitiveCall.downcast(op_call)
    thread_cnt = _thread_cnt(sctx)
    if thread_cnt <= 0:
        return False, f"degenerate thread_cnt={thread_cnt} (scope has empty intra)"
    g_br = op_call.src if op_call.src.buffer.scope() == "global" else op_call.dst
    n_elements = 1
    for r in g_br.region:
        ext = r.extent
        try:
            n_elements *= int(ext)
        except (TypeError, ValueError):
            return False, f"non-constant region extent {ext}"
    if n_elements % thread_cnt != 0:
        return False, (f"region size {n_elements} not divisible by thread_cnt={thread_cnt}")
    return True, None


def _is_ldgsts(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_target("cuda"):
        return False, "non-cuda target"
    if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
        return False, f"unsupported exec_scope {sctx.scope_kind}"
    for check in (
        lambda: _all_threads_active(sctx),
        lambda: _is_valid_copy(op_call, sctx),
        lambda: _scope_allowed(op_call, sctx, allowed_pairs=_LDGSTS_PAIRS),
        lambda: _divides_thread_cnt_ldgsts(op_call, sctx),
    ):
        ok, msg = check()
        if not ok:
            return False, msg
    return True, None


def _emit_ldgsts(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    # Predicate above guarantees src is global, dst is shared.
    g_buf, g_br = src, op_call.src
    s_buf, s_br = dst, op_call.dst

    elem_bits = DataType(src.dtype).bits
    prefetch_size = op_call.config.get("prefetch_size", -1)
    predicate_expr = op_call.config.get("predicate", -1)
    fill_mode = op_call.config.get("fill_mode", "")

    if _config_bool(op_call.config.get("direct", False)):
        if sctx.scope_kind != "thread":
            raise ValueError("ldgsts direct=True is only valid in thread scope")
        n_elements = 1
        for r in s_br.region:
            try:
                n_elements *= int(r.extent)
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"ldgsts direct=True requires constant extent, got {r.extent}"
                ) from err
        cp_size = n_elements * elem_bits // 8
        if n_elements * elem_bits % 8 != 0 or cp_size not in (4, 8, 16):
            raise ValueError(
                f"ldgsts direct=True requires a 4/8/16-byte region, got {n_elements} "
                f"elements of {elem_bits} bits"
            )
        s_start = [r.min for r in s_br.region]
        g_start = [r.min for r in g_br.region]

        @T.prim_func(check_well_formed=False)
        def impl():
            _emit_cp_async(
                s_buf.ptr_to(s_start),
                g_buf.ptr_to(g_start),
                cp_size,
                prefetch_size,
                predicate_expr,
                fill_mode,
            )

        return impl

    g_region = [(r.min, r.min + r.extent) for r in g_br.region]
    s_region = [(r.min, r.min + r.extent) for r in s_br.region]

    thread_cnt = _thread_cnt(sctx)

    with sctx.target:
        g_p, s_p, vec_len = align_layouts_gs(
            g_buf.layout,
            g_buf.shape,
            g_region,
            s_buf.layout,
            s_buf.shape,
            s_region,
            elem_bits,
            thread_cnt,
            vec_bits_candidates=_LDGSTS_VEC_BITS,
        )
        s_apply_layout = recompose_swizzle(s_buf.layout, s_p)

    vec_bits = vec_len * elem_bits
    cp_size = vec_bits // 8  # cp.async cp_size is in bytes
    if cp_size not in (4, 8, 16):
        # align_layouts_gs already restricted candidates to _LDGSTS_VEC_BITS,
        # so reaching here means no candidate worked at all.
        from tvm.tirx.operator.tile_primitive.dispatcher import fail

        fail(f"ldgsts: cannot find a cp.async-compatible vec_len for elem_bits={elem_bits}")

    # Mirror vec_auto_gmem_smem.py: build 3D `(f, tid, 0)` against
    # `[total_outer, thread_cnt, vec_len]` and let `s_p.apply(coord, shape)`
    # flatten + resplit into whatever multi-iter T / outer-iter structure
    # `align_layouts_gs` picked. Emit is oblivious to how many shard iters
    # cover T.
    n_elements = 1
    for it in s_p.shard:
        n_elements *= int(it.extent)
    assert n_elements % (thread_cnt * vec_len) == 0, (
        f"partition produced {n_elements} elements but thread_cnt({thread_cnt}) * "
        f"vec_len({vec_len}) = {thread_cnt * vec_len} doesn't divide it"
    )
    total_outer = n_elements // (thread_cnt * vec_len)
    apply_shape = [
        _IntImm("int32", total_outer),
        _IntImm("int32", thread_cnt),
        _IntImm("int32", vec_len),
    ]

    s_zero = [0] * len(s_buf.shape)
    g_zero = [0] * len(g_buf.shape)

    tid_axis_name = _TID_AXIS_FOR_SCOPE[sctx.scope_kind] if thread_cnt > 1 else None

    def _decl_tid():
        if tid_axis_name is not None:
            return _axis_decl(tid_axis_name, sctx)
        return _IntImm("int32", 0)

    v0 = _IntImm("int32", 0)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        tid = _decl_tid()
        for f in T.unroll(total_outer):
            g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]
            s_off = s_apply_layout.apply(f, tid, v0, shape=apply_shape)["m"]
            s_ptr = _ptr_off(s_buf.ptr_to(s_zero), s_off)
            g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
            _emit_cp_async(s_ptr, g_ptr, cp_size, prefetch_size, predicate_expr, fill_mode)
        # cp.async is caller-synced — no cta_sync here (commit_group /
        # wait_group / cta_sync are the caller's responsibility).
    # fmt: on
    return impl


@register_dispatch(
    "copy_async",
    "cuda",
    variant="ldgsts",
    priority=20,
    when=[predicate("ldgsts_applicable", _is_ldgsts)],
)
def copy_schedule_ldgsts(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit_ldgsts(op_call, sctx)
