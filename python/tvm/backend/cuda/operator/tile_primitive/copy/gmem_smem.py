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

"""Copy dispatch for ``global ↔ shared`` (no register side).

There's no per-thread register side to inherit a partition from — both sides
are cross-thread storage. The partition is synthesized from the surrounding
scope context (warp / warpgroup / cta / thread): ``thread_cnt`` is derived
from ``sctx.intra`` and each thread takes ``n_elements / thread_cnt``
consecutive fused-index slots. Layout / partition algorithm lives in
``_common.py`` and is shared with ``ldgsts.py``.
"""

import tvm
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx import Var as _TirVar
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.operator.tile_primitive.dispatcher import (
    predicate,
    register_dispatch,
)
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ._common import (
    _TID_AXIS_FOR_SCOPE,
    _thread_cnt,
    align_layouts_gs,
    copy_ptx_form,
    copy_ptx_ld_return_type,
)
from ._swizzle_iter import (
    emit_init,
    emit_iter_offset,
    get_swizzle,
    try_recognize,
)
from .reg import _all_threads_active, _axis_decl, _ptr_off
from .utils import _is_valid_copy, _scope_allowed

_GMEM_SMEM_PAIRS = [
    ("global", "shared*"),
    ("shared*", "global"),
]


def _divides_thread_cnt(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    """Reject copies whose region element count does not divide ``thread_cnt``.

    Without this guard the emit's ``[outer, T, vec]`` partition has no
    integer solution: either every thread gets fractional work, or
    ``thread_cnt=0`` (degenerate scope) hits a modulo-by-zero. Both cases
    indicate a poorly-shaped copy (e.g. 1024-thread CTA writing a 64-elem
    tail) that this dispatch refuses to paper over with a slow scalar emit.
    """
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


def _is_gmem_smem(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_target("cuda"):
        return False, "non-cuda target"
    if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
        return False, f"unsupported exec_scope {sctx.scope_kind}"
    for check in (
        lambda: _all_threads_active(sctx),
        lambda: _is_valid_copy(op_call, sctx),
        lambda: _scope_allowed(op_call, sctx, allowed_pairs=_GMEM_SMEM_PAIRS),
        lambda: _divides_thread_cnt(op_call, sctx),
    ):
        ok, msg = check()
        if not ok:
            return False, msg
    return True, None


def _emit_gmem_smem(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    if src.scope() == "global":
        g_buf, g_br, s_buf, s_br = src, op_call.src, dst, op_call.dst
        g_is_src = True
    else:
        g_buf, g_br, s_buf, s_br = dst, op_call.dst, src, op_call.src
        g_is_src = False

    g_region = [(r.min, r.min + r.extent) for r in g_br.region]
    s_region = [(r.min, r.min + r.extent) for r in s_br.region]

    elem_bits = DataType(src.dtype).bits
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
        )

    # vec_len=1 is the scalar fallback — uses the same unified
    # [outer x thread x vec] coord scheme below.

    vec_bits = vec_len * elem_bits
    num_bytes = vec_bits // 8
    vec, ptx_type = copy_ptx_form(num_bytes)

    # Partition guarantees ``prod(s_p.shard.extents) == prod(g_p.shard.extents)
    # == n_elements`` (the total transfer count). Express the per-thread
    # per-round address as a 3D coord ``(f, tid, 0)`` against shape
    # ``[total_outer, thread_cnt, vec_len]``, and let ``layout.apply`` flatten
    # it through whatever multi-iter T / outer-iter structure ``align_layouts_gs``
    # picked. This makes the emit oblivious to how many iters the partition
    # split T or outer across.
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

    # Walk shard from the vec iter backward to find the prefix that covers
    # the T region exactly (∏ext == thread_cnt). The iters consumed are T
    # iters; the leading prefix is the outer iter list — handed to
    # ``try_recognize`` so the swizzle fast path can decide whether the
    # outer iter strides match a pattern it can lower to signed_strides.
    if thread_cnt > 1:
        acc, _i = 1, len(s_p.shard) - 2
        while _i >= 0 and acc < thread_cnt:
            _ext = int(s_p.shard[_i].extent)
            if acc * _ext > thread_cnt:
                break
            acc *= _ext
            _i -= 1
        outer_iters_s = list(s_p.shard[: _i + 1]) if acc == thread_cnt else []
    else:
        outer_iters_s = list(s_p.shard[:-1])

    # SwizzleLayout on s_buf: try the closed-form signed-strides pattern
    # (precomputed once per thread, then per-iter is a sum of register
    # adds); fall back to per-iter ``swizzle.apply`` (one full XOR +
    # decompose per iter). Closure picked at parse time so the TIRx parser
    # doesn't AST-evaluate a "dead" ternary branch.
    swizzle = get_swizzle(s_buf.layout)
    swizzle_pattern = None
    if swizzle is not None and outer_iters_s:
        if tid_axis_name is not None:
            _tid_placeholder = _TirVar(tid_axis_name, "int32")
        else:
            _tid_placeholder = _IntImm("int32", 0)
        s_off_template = s_p.apply(
            _IntImm("int32", 0),
            _tid_placeholder,
            _IntImm("int32", 0),
            shape=apply_shape,
        )["m"]
        # Bind the tid placeholder's range so the (C1) analyzer check can
        # discharge ``bit_bj(s_off // C) == 0`` for high bj's. Outer iter
        # stride here is ``thread_cnt * vec_len`` ⇒ bj ∈ [log2(thread_cnt),
        # ...]; without bounds the analyzer can't prove the lane's high bits
        # are 0 and rejects.
        var_bounds = {}
        if tid_axis_name is not None:
            var_bounds[_tid_placeholder] = tvm.ir.Range.from_min_extent(0, thread_cnt)
        swizzle_pattern = try_recognize(
            swizzle,
            [int(it.extent) for it in outer_iters_s],
            [int(it.stride) for it in outer_iters_s],
            s_off_template,
            var_bounds=var_bounds or None,
        )

    class _SwizzleState:
        def __init__(self):
            self.signed_strides = None
            self.base_off = None

    state = _SwizzleState()

    def _decl_tid():
        if tid_axis_name is not None:
            return _axis_decl(tid_axis_name, sctx)
        return _IntImm("int32", 0)

    def _setup_swizzle(tid):
        if swizzle_pattern is None:
            return
        s_off_resolved = s_p.apply(
            _IntImm("int32", 0),
            tid,
            _IntImm("int32", 0),
            shape=apply_shape,
        )["m"]
        state.signed_strides, state.base_off = emit_init(
            swizzle_pattern,
            s_off_resolved,
        )

    if swizzle_pattern is not None:

        def _s_off(f, s_lin):
            return emit_iter_offset(
                swizzle_pattern,
                state.signed_strides,
                state.base_off,
                f,
            )
    elif swizzle is not None:
        _sw = swizzle

        def _s_off(f, s_lin):
            return _sw.apply(s_lin)["m"]
    else:

        def _s_off(f, s_lin):
            return s_lin

    v0 = _IntImm("int32", 0)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        tid = _decl_tid()
        _setup_swizzle(tid)
        tmp = T.alloc_local((vec_len,), src.dtype)
        tmp_ptr = tmp.ptr_to([0])
        # NB: pass typed ptr_to(...) directly to _ptr_off; caching in a
        # local var turns it into void* + offset = byte arithmetic →
        # misaligned vector ops.
        #
        # Use a serial TIR loop and let ptxas unroll downstream. Mirrors
        # the reg.py rationale in commit ac7ecf70f0: explicit ``T.unroll``
        # materializes the per-iter scratch (s_lin/g_lin/s_off/s_ptr/g_ptr)
        # as N copies of each ``alignas(64)`` declaration. For large
        # ``total_outer`` (e.g. thread-scope fp32 swizzled copies of 32x256
        # at vec=4 ⇒ 2048 iters; ldgsts test4 ⇒ ~4k iters once both
        # g2s/s2g sites add up) this floods the kernel and nvcc times out.
        for f in range(total_outer):
            s_lin = s_p.apply(f, tid, v0, shape=apply_shape)["m"]
            g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]
            s_off = _s_off(f, s_lin)
            s_ptr = _ptr_off(s_buf.ptr_to(s_zero), s_off)
            g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
            if g_is_src:
                T.ptx.ld(
                    g_ptr,
                    copy_ptx_ld_return_type(ptx_type),
                    ptx_type,
                    dst=tmp_ptr,
                    space="global",
                    vec=vec,
                )
                T.ptx.st(
                    s_ptr, src=tmp_ptr, space="shared", vec=vec, ptx_type=ptx_type
                )
            else:
                T.ptx.ld(
                    s_ptr,
                    copy_ptx_ld_return_type(ptx_type),
                    ptx_type,
                    dst=tmp_ptr,
                    space="shared",
                    vec=vec,
                )
                T.ptx.st(
                    g_ptr, src=tmp_ptr, space="global", vec=vec, ptx_type=ptx_type
                )
    # fmt: on
    return impl


@register_dispatch(
    "copy",
    "cuda",
    variant="gmem_smem",
    priority=10,
    when=[predicate("gmem_smem_applicable", _is_gmem_smem)],
)
def copy_schedule_gmem_smem(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit_gmem_smem(op_call, sctx)
