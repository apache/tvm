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
``cuda/copy/gmem_smem.py`` (sync ``T.copy`` global ↔ shared); differs at
emit time only:

* direction: ``cp.async`` is global → shared only (hardware restriction).
* cp_size: PTX ``cp.async`` only accepts 4 / 8 / 16 bytes, so the vec-width
  candidate set is restricted to ``{32, 64, 128}`` bits.
* emit: ``T.evaluate(T.ptx.cp_async(dst, src, cp_size))`` instead of the
  synchronous ``T.cuda.copy_{vec_bits}b(dst, src)``.

Note: ``cp.async`` does **not** sync at emit time — caller is responsible
for ``commit_group`` / ``wait_group`` / ``cta_sync`` plumbing around the
async pipeline.
"""

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

from ..copy._common import (
    _TID_AXIS_FOR_SCOPE,
    _thread_cnt,
    align_layouts_gs,
)
from ..copy._swizzle_iter import (
    emit_init,
    emit_iter_offset,
    get_swizzle,
    try_recognize,
)
from ..copy.reg import _all_threads_active, _axis_decl, _ptr_off
from ..copy.utils import _is_valid_copy, _scope_allowed

# cp.async is unidirectional: global → shared.
_LDGSTS_PAIRS = [("global", "shared*")]
# cp.async cp_size ∈ {4, 8, 16} bytes ⇒ vec_bits ∈ {32, 64, 128}.
_LDGSTS_VEC_BITS = (128, 64, 32)


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
            vec_bits_candidates=_LDGSTS_VEC_BITS,
        )

    vec_bits = vec_len * elem_bits
    cp_size = vec_bits // 8  # cp.async cp_size is in bytes
    if cp_size not in (4, 8, 16):
        # align_layouts_gs already restricted candidates to _LDGSTS_VEC_BITS,
        # so reaching here means no candidate worked at all.
        from tvm.tirx.operator.tile_primitive.dispatcher import fail

        fail(f"ldgsts: cannot find a cp.async-compatible vec_len for elem_bits={elem_bits}")

    # Mirror gmem_smem.py: build 3D `(f, tid, 0)` against
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

    # T-iters-walk-back to recover outer_iters_s for the fast-path
    # recognizer. Same trick as gmem_smem.py.
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
        swizzle_pattern = try_recognize(
            swizzle,
            [int(it.extent) for it in outer_iters_s],
            [int(it.stride) for it in outer_iters_s],
            s_off_template,
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
        for f in T.unroll(total_outer):
            s_lin = s_p.apply(f, tid, v0, shape=apply_shape)["m"]
            g_lin = g_p.apply(f, tid, v0, shape=apply_shape)["m"]
            s_off = _s_off(f, s_lin)
            s_ptr = _ptr_off(s_buf.ptr_to(s_zero), s_off)
            g_ptr = _ptr_off(g_buf.ptr_to(g_zero), g_lin)
            T.evaluate(T.ptx.cp_async(s_ptr, g_ptr, cp_size))
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
