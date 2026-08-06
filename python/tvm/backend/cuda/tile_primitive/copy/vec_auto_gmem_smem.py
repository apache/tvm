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

from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ..layout_utils import recompose_swizzle
from ._common import (
    _TID_AXIS_FOR_SCOPE,
    _thread_cnt,
    align_layouts_gs,
    copy_ptxd_form,
)
from .utils import _is_valid_copy, _scope_allowed
from .vec_auto_reg import _all_threads_active, _axis_decl, _ptr_off

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
        s_apply_layout = recompose_swizzle(s_buf.layout, s_p)

    # vec_len=1 is the scalar fallback — uses the same unified
    # [outer x thread x vec] coord scheme below.

    vec_bits = vec_len * elem_bits
    num_bytes = vec_bits // 8
    tail, lanes, reg_dtype = copy_ptxd_form(num_bytes)
    # Chains are built here: a Python string bound inside the traced body is
    # not something the parser can carry.
    ld_g, st_s = f"ld.global.{tail}", f"st.shared.{tail}"
    ld_s, st_g = f"ld.shared.{tail}", f"st.global.{tail}"

    # Express the per-thread per-round address as a 3D coord ``(f, tid, 0)`` vs
    # ``[total_outer, thread_cnt, vec_len]``; ``layout.apply`` flattens the rest.
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
        # The scratch only shuttles bits, so it is allocated in the PTX
        # container type rather than the element type.
        tmp = T.alloc_local((lanes,), reg_dtype)
        # Pass typed ptr_to(...) directly to _ptr_off (caching → byte math,
        # misaligned vec ops); keep a serial loop, T.unroll floods the kernel.
        for f in range(total_outer):
            g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]
            s_off = s_apply_layout.apply(f, tid, v0, shape=apply_shape)["m"]
            s_ptr = _ptr_off(s_buf.ptr_to(s_zero), s_off)
            g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
            if g_is_src:
                T.ptxd[ld_g](*[tmp[i] for i in range(lanes)], g_ptr)
                T.ptxd[st_s](s_ptr, *[tmp[i] for i in range(lanes)])
            else:
                T.ptxd[ld_s](*[tmp[i] for i in range(lanes)], s_ptr)
                T.ptxd[st_g](g_ptr, *[tmp[i] for i in range(lanes)])
    # fmt: on
    return impl
