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

"""Non-ldmatrix copy dispatch for register ↔ memory.

This file owns every copy where one side is per-thread local (``R`` =
register). That R side carries the partition: its ``TileLayout`` ``shard``
has thread-axis iters telling us which thread owns which logical coordinate.
The other side (``S``) can be ``shared*`` or ``global`` — the algorithm is
identical either way.

Slice/canonicalize both sides, align via perm+group, then emit a per-thread
vectorized copy loop. Direction-symmetric: covers R2S / S2R / R2G / G2R.
"""

import tvm
from tvm.arith import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as Tx
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx import Var as _TirVar
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TileLayout
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import TilePrimitiveCall

from ._common import _alignment_ok
from ._swizzle_iter import (
    emit_fallback_offset,
    emit_init,
    emit_iter_offset,
    get_swizzle,
    try_recognize,
)
from .utils import _is_valid_copy, _scope_allowed


def _extract_tile(layout, region):
    """Strip swizzle off ``layout`` so we can perm/group it as a TileLayout.

    ``region`` is the per-axis ``(start, end)`` pair list — we only consume
    its extents when ``layout`` is a bare ``SwizzleLayout`` (rebuilding a
    trivial TileLayout for it). Plain ``TileLayout`` / ``ComposeLayout``
    don't need the extent, so symbolic regions are fine for them.
    """
    if isinstance(layout, ComposeLayout):
        return layout.tile_layout
    if isinstance(layout, SwizzleLayout):
        # TODO: keep swizzle info around for later (addressing in emit).
        extents = [int(end - start) for (start, end) in region]
        return TileLayout(S[tuple(extents)])
    return layout


_REG_PAIRS = [
    ("local", "shared*"),
    ("shared*", "local"),
    ("local", "global"),
    ("global", "local"),
]
_SCOPE_RANK = {"thread": 0, "warp": 1, "warpgroup": 2, "cta": 3}
_VALID_R_SUBSCOPES = {"thread", "warp", "warpgroup"}


def _all_threads_active(sctx: DispatchContext) -> tuple[bool, str | None]:
    if sctx.scope_kind == "thread":
        return True, None
    required: dict[str, int] = {}
    if sctx.scope_kind in ("warp", "warpgroup", "cta"):
        required["laneid"] = 32
    if sctx.scope_kind == "warpgroup":
        required["wid_in_wg"] = 4
    if sctx.scope_kind == "cta":
        tx_iv = sctx.launch_params.get("threadIdx.x")
        if tx_iv is None:
            return False, "cta scope missing threadIdx.x launch_params"
        try:
            required["warpid"] = int(tx_iv.dom.extent) // 32
        except (TypeError, ValueError):
            return False, f"non-static threadIdx.x extent: {tx_iv.dom.extent}"
    for axis_name, expected in required.items():
        if axis_name not in sctx.intra:
            return False, f"sctx.intra missing {axis_name!r}"
        ext_raw, off_raw = sctx.intra[axis_name]
        try:
            ext, off = int(ext_raw), int(off_raw)
        except (TypeError, ValueError):
            return False, f"non-static range for {axis_name}: ({ext_raw}, {off_raw})"
        if ext != expected or off != 0:
            return False, f"{axis_name} narrowed to [{off}, {off + ext}) vs full [0, {expected})"
    return True, None


def _r_side_layout_valid(
    op_call: TilePrimitiveCall, sctx: DispatchContext
) -> tuple[bool, str | None]:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    r_buf = src if src.scope() == "local" else dst
    layout = r_buf.layout
    if layout is None:
        return False, "R has no layout"
    if layout.is_swizzle():
        return False, "R layout is swizzle"
    if not isinstance(layout, TileLayout):
        return False, f"R layout is {type(layout).__name__}, not TileLayout"

    scope_rank = _SCOPE_RANK[sctx.scope_kind]
    seen_thread_axes: set[str] = set()
    for it in layout.shard:
        ax = it.axis
        if not ax.is_thread():
            continue
        ax_scope = ax.get_scope()
        ax_sub = ax.get_subscope()
        if ax_scope is None or ax_sub is None:
            return False, f"R thread axis {ax.name!r} missing scope/subscope"
        if ax_sub.name not in _VALID_R_SUBSCOPES:
            return False, f"R thread axis {ax.name!r} subscope={ax_sub.name!r} (not register-level)"
        if ax_scope.name not in _SCOPE_RANK or _SCOPE_RANK[ax_scope.name] > scope_rank:
            return (
                False,
                f"R thread axis {ax.name!r} scope={ax_scope.name!r} > exec {sctx.scope_kind!r}",
            )
        # TODO: lift these two; for now i = thread_value (stride=1, each axis appears once).
        if int(it.stride) != 1:
            return (
                False,
                f"R thread axis {ax.name!r} stride={int(it.stride)} != 1 (not supported yet)",
            )
        if ax.name in seen_thread_axes:
            return False, f"R thread axis {ax.name!r} appears more than once (not supported yet)"
        seen_thread_axes.add(ax.name)

    r_br = op_call.src if src.scope() == "local" else op_call.dst
    region = [(r.min, r.min + r.extent) for r in r_br.region]
    sliced = layout.slice(list(r_buf.shape), region)
    if sliced is None:
        return False, "R layout slice failed"
    analyzer = Analyzer()
    for axis, off in sliced.offset.items():
        if axis.is_thread() and not analyzer.can_prove_equal(off, 0):
            return False, f"R sliced offset on thread axis {axis.name!r} = {off}"
    return True, None


def _s_side_slice_ok(op_call: TilePrimitiveCall) -> tuple[bool, str | None]:
    """S is the non-local side (shared* or global). Slice must succeed."""
    op_call = TilePrimitiveCall.downcast(op_call)
    src_br = op_call.src
    dst_br = op_call.dst
    s_br = dst_br if src_br.buffer.scope() == "local" else src_br
    s_buf: Buffer = s_br.buffer
    layout = s_buf.layout
    if layout is None:
        return False, "S has no layout"
    region = [(r.min, r.min + r.extent) for r in s_br.region]
    if layout.slice(list(s_buf.shape), region) is None:
        return False, "S layout slice failed"
    return True, None


def _is_reg_copy(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_cuda():
        return False, "non-cuda target"
    if sctx.scope_kind not in ("thread", "warp", "warpgroup", "cta"):
        return False, f"unsupported exec_scope {sctx.scope_kind}"
    for check in (
        lambda: _all_threads_active(sctx),
        lambda: _is_valid_copy(op_call, sctx),
        lambda: _scope_allowed(op_call, sctx, allowed_pairs=_REG_PAIRS),
        lambda: _r_side_layout_valid(op_call, sctx),
        lambda: _s_side_slice_ok(op_call),
    ):
        ok, msg = check()
        if not ok:
            return False, msg
    return True, None


def _compute_perm_r(r):
    # thread axes first, then by stride descending
    def key(p):
        it = p[1]
        return (0 if it.axis.is_thread() else 1, -int(it.stride))

    return [i for i, _ in sorted(enumerate(r.shard), key=key)]


def align_layouts_raw(r_layout, r_shape, r_region, s_layout, s_shape, s_region):
    """Returns (r_p, s_p, s_seps)."""
    r = r_layout.slice(list(r_shape), r_region).canonicalize()
    s = s_layout.slice(list(s_shape), s_region).canonicalize()
    s = _extract_tile(s, s_region)
    perm = _compute_perm_r(r)
    r_shape_for_group = [int(it.extent) for it in r.shard]
    s_grp, seps = s.group(r_shape_for_group)
    s_p = s_grp.permute_by_groups(list(seps), perm)
    r_p = r.permute_dims(perm).canonicalize()
    sizes = [seps[i + 1] - seps[i] for i in range(len(seps) - 1)]
    s_seps = [0]
    for p in perm:
        s_seps.append(s_seps[-1] + sizes[p])
    return r_p, s_p, s_seps


def _split_thread_loop(r_p, s_p, s_seps):
    """Drop R's thread-axis positions and return per-R-position bundles:
    (r_iters, s_groups) — same length lists; s_groups[k] is the list of S
    iters belonging to the k-th kept R position."""
    r_iters = []
    s_groups = []
    for k, r_it in enumerate(r_p.shard):
        if r_it.axis.is_thread():
            continue
        r_iters.append(r_it)
        s_groups.append(list(s_p.shard[s_seps[k] : s_seps[k + 1]]))
    return r_iters, s_groups


def _build_atoms(r_iters, s_groups):
    """One atom per (R position, intra-group S iter): (extent, s_stride, r_mul).
    r_mul = R_stride_at_position * (product of S group extents to the right
    of this intra-position index) — i.e. how much R address advances per unit
    of this iter's loop input."""
    atoms = []
    for r_it, s_group in zip(r_iters, s_groups, strict=True):
        rs = int(r_it.stride)
        extents = [int(it.extent) for it in s_group]
        for j, s_it in enumerate(s_group):
            inner_prod = 1
            for e in extents[j + 1 :]:
                inner_prod *= e
            atoms.append((int(s_it.extent), int(s_it.stride), rs * inner_prod))
    return atoms


def _atoms_contiguous_tail_extent(atoms) -> int:
    """Like _contiguous_tail_extent but on atoms (uses s_stride for chaining)."""
    if not atoms or atoms[-1][1] != 1:
        return 0
    acc = atoms[-1][0]
    for k in range(len(atoms) - 2, -1, -1):
        if atoms[k][1] == acc:
            acc *= atoms[k][0]
        else:
            break
    return acc


def _split_atoms_for_vec(atoms, vec_len):
    """Returns outer atoms (the inner vec_len-element tail is consumed by one
    vec ld/st and dropped). Splits the boundary atom if needed."""
    outer = list(atoms)
    acc = 1
    while outer:
        ext, ss, rm = outer[-1]
        new_acc = acc * ext
        if new_acc == vec_len:
            outer.pop()
            return outer
        if new_acc > vec_len:
            inner_factor = vec_len // acc
            outer[-1] = (ext // inner_factor, ss * inner_factor, rm * inner_factor)
            return outer
        acc = new_acc
        outer.pop()
    raise ValueError(f"tail too short for vec_len {vec_len}")


def _align_layouts(op_call: TilePrimitiveCall, sctx: DispatchContext):
    op_call = TilePrimitiveCall.downcast(op_call)
    src_br = op_call.src
    dst_br = op_call.dst
    if src_br.buffer.scope() == "local":
        r_br, s_br = src_br, dst_br
    else:
        r_br, s_br = dst_br, src_br
    r_buf = r_br.buffer
    s_buf = s_br.buffer
    r_region = [(r.min, r.min + r.extent) for r in r_br.region]
    s_region = [(r.min, r.min + r.extent) for r in s_br.region]
    # Push the dispatch target so layout.canonicalize() runs scope-aware
    # fusers (e.g. laneid+wid_in_wg -> tid_in_wg).
    with sctx.target:
        return align_layouts_raw(
            r_buf.layout,
            r_buf.shape,
            r_region,
            s_buf.layout,
            s_buf.shape,
            s_region,
        )


def _make_thread_placeholders(r_p) -> dict[str, _TirVar]:
    placeholders: dict[str, _TirVar] = {}
    for it in r_p.shard:
        name = it.axis.name
        if it.axis.is_thread() and name not in placeholders:
            placeholders[name] = _TirVar(name, "int32")
    return placeholders


def _s_thread_offset(r_p, s_p, placeholders: dict[str, _TirVar]):
    """Per-thread S base offset. Coord per R position is placeholder (thread
    axis) or 0 (memory axis); apply_to_shape decomposes across s_p iters.
    Includes layout-level offsets (e.g. from slicing a non-zero S region)."""
    coord = [
        placeholders[it.axis.name] if it.axis.is_thread() else _IntImm("int32", 0)
        for it in r_p.shard
    ]
    input_shape = [int(it.extent) for it in r_p.shard]
    per_iter = s_p.apply_to_shape(coord, input_shape)
    off = _IntImm("int32", 0)
    for c, it in zip(per_iter, s_p.shard, strict=True):
        off = off + c * it.stride
    for _axis, val in s_p.offset.items():
        off = off + val
    return off


_VEC_BITS_CANDIDATES = (128, 64, 32, 16, 8)


def _vec_len_candidates(elem_bits: int) -> list[int]:
    """Widest-first element counts to try; always ends with scalar (1)."""
    out: list[int] = []
    for vb in _VEC_BITS_CANDIDATES:
        if vb < elem_bits or vb % elem_bits != 0:
            continue
        n = vb // elem_bits
        if n not in out:
            out.append(n)
    if 1 not in out:
        out.append(1)
    return out


def _choose_vec_len(elem_bits: int, atoms, r_p, s_p) -> int:
    """Widest candidate that:
      1. divides the atom contiguous-tail extent (so vec_len consecutive
         R-side regs map to vec_len contiguous S-side elements), AND
      2. keeps every per-thread / per-round address-offset term a
         multiple of vec_len, so the resulting vec ld/st pointer is
         naturally aligned to vec_bits/8 bytes.

    Only **mem-axis** strides contribute to physical address. Thread-axis
    iter strides live in partition-coord space (which thread owns which
    logical position), not in the buffer's storage space — they're
    redistributed through ``apply_to_shape`` into the mem iters and don't
    appear directly in the per-thread address. So neither r-side nor
    s-side thread-axis strides belong in the alignment check.

    The contig-tail atoms (whose extents the vec ld/st consumes) have
    stride 1 by definition; they live entirely inside the vec and
    contribute nothing to the per-round address delta. Only the
    **post-vec-split** outer atom strides matter for the per-round delta.
    """
    t = _atoms_contiguous_tail_extent(atoms)
    # Region-base offsets are real address contributions. Thread-iter
    # strides on either side are partition-virtual, not storage-physical,
    # so they don't enter the per-thread address — exclude them.
    shared_terms = list(s_p.offset.values()) + list(r_p.offset.values())
    for n in _vec_len_candidates(elem_bits):
        if n == 1:
            return n
        if t % n != 0 or t < n:
            continue
        # Post-vec-split outer atoms: these are the strides that contribute
        # to per-round address deltas after the vec consumes the inner tail.
        outer = _split_atoms_for_vec(atoms, n)
        outer_atom_terms = [a[1] for a in outer] + [a[2] for a in outer]
        if not _alignment_ok(n, outer_atom_terms + shared_terms):
            continue
        return n
    return 1


def _axis_decl(axis_name: str, sctx: DispatchContext):
    """Declare the runtime Var for one thread axis (called inside impl body).

    Each scope_id declarator emits a ``ScopeIdDef`` stmt at the current
    builder frame. ``TilePrimitiveDispatch`` re-gathers + resolves all
    ScopeIdDefs after dispatch (see ``ResolveAllScopeBinds`` in
    ``tile_primitive_dispatch.cc``), so dispatch-introduced vars are bound
    alongside kernel-declared ones.

    Extents are deferred: the kernel header is expected to declare the full
    scope-id chain (``cta_id`` / ``warpgroup_id`` / ``warp_id_in_wg`` /
    ``lane_id`` / ``thread_id`` / ``thread_id_in_wg``) — the verifier then
    fills our deferred defs from those siblings.
    """
    if axis_name == "tx":
        return sctx.launch_params["threadIdx.x"].var
    if axis_name == "laneid":
        return Tx.lane_id()
    if axis_name == "wid_in_wg":
        return Tx.warp_id_in_wg()
    if axis_name == "tid_in_wg":
        return Tx.thread_id_in_wg()
    if axis_name == "warpid":
        return Tx.warp_id()
    if axis_name == "wgid":
        return Tx.warpgroup_id()
    raise ValueError(f"unsupported thread axis {axis_name}")


def _s_thread_offset_with_vars(r_p, s_p, axis_var_map: dict):
    coord = [
        axis_var_map[it.axis.name] if it.axis.is_thread() else _IntImm("int32", 0)
        for it in r_p.shard
    ]
    input_shape = [int(it.extent) for it in r_p.shard]
    per_iter = s_p.apply_to_shape(coord, input_shape)
    off = _IntImm("int32", 0)
    for c, it in zip(per_iter, s_p.shard, strict=True):
        off = off + c * it.stride
    for _ax, val in s_p.offset.items():
        off = off + val
    return off


def _substitute_axes(s_off_template, placeholders: dict[str, _TirVar], sctx: DispatchContext):
    """Inside an impl body: declare real scope_ids and rewrite the
    placeholder-built ``s_off_template`` to use them."""
    vmap = {placeholders[name]: _axis_decl(name, sctx) for name in placeholders}
    return tvm.tirx.stmt_functor.substitute(s_off_template, vmap)


def _flat_coords(outer_atoms, flat_idx: int) -> list[int]:
    coords = []
    rem = flat_idx
    for a in reversed(outer_atoms):
        coords.append(rem % a[0])
        rem //= a[0]
    coords.reverse()
    return coords


_POINTER_OFFSET_SRC = (
    "\ntemplate <typename T>\n"
    "__forceinline__ __device__ T* tvm_builtin_pointer_offset(T* ptr, int offset) {\n"
    "    return ptr + offset;\n"
    "}\n"
)


def _ptr_off(base_ptr, off):
    return Tx.cuda.func_call(
        "tvm_builtin_pointer_offset",
        base_ptr,
        off,
        source_code=_POINTER_OFFSET_SRC,
        return_type="handle",
    )


def _outer_const_offsets(outer_atoms, flat_idx: int) -> tuple[int, int]:
    """Returns (s_offset_const, r_offset_const) for one outer-loop flat index."""
    coords = _flat_coords(outer_atoms, flat_idx)
    ds = sum(c * a[1] for c, a in zip(coords, outer_atoms))
    dr = sum(c * a[2] for c, a in zip(coords, outer_atoms))
    return ds, dr


def _emit_reg(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    if src.scope() == "local":
        r_buf, s_buf, r_is_src = src, dst, True
    else:
        r_buf, s_buf, r_is_src = dst, src, False

    with sctx.target:
        r_p, s_p, s_seps = _align_layouts(op_call, sctx)
    r_iters, s_groups = _split_thread_loop(r_p, s_p, s_seps)
    atoms = _build_atoms(r_iters, s_groups)
    elem_bits = DataType(src.dtype).bits
    vec_len = _choose_vec_len(elem_bits, atoms, r_p, s_p)
    vec_bits = vec_len * elem_bits
    outer = _split_atoms_for_vec(atoms, vec_len)
    per_thread_r_total = 1
    for it in r_iters:
        per_thread_r_total *= int(it.extent)
    per_thread_r_shape = [per_thread_r_total or 1]

    # Build the per-thread S offset OUTSIDE the impl using placeholder Vars
    # (one per thread axis). Inside the impl we'll declare the real scope_ids
    # via Tx.lane_id/Tx.thread_id_in_wg/... and substitute them in.
    placeholders = _make_thread_placeholders(r_p)
    s_off_template = _s_thread_offset(r_p, s_p, placeholders)

    # R-side base offset from slicing (e.g. ``R[i*8:i*8+8]`` ⇒ ``i*8``). The
    # canonicalize() result lives in ``r_p.offset``; sum across axes (memory
    # or thread — irrelevant once it's all on R's local stride-1 storage).
    r_off_base = _IntImm("int32", 0)
    for _ax, val in r_p.offset.items():
        r_off_base = r_off_base + val

    copy_op = getattr(Tx.cuda, f"copy_{vec_bits}b")

    total_outer = 1
    for a in outer:
        total_outer *= a[0]

    # Swizzle handling: recognize the iter-pattern on S side from the atom
    # extents/strides (atom = (extent, s_stride, r_mul); a[1] is the S-side
    # stride per outer round, equivalent to outer_iter strides in gmem_smem).
    swizzle = get_swizzle(s_buf.layout)
    swizzle_pattern = None
    if swizzle is not None:
        swizzle_pattern = try_recognize(
            swizzle,
            [a[0] for a in outer],
            [a[1] for a in outer],
            s_off_template,
        )

    class _SwizzleState:
        def __init__(self):
            self.signed_strides = None
            self.base_off = None

    state = _SwizzleState()

    def _setup_swizzle(s_off):
        if swizzle_pattern is None:
            return
        state.signed_strides, state.base_off = emit_init(swizzle_pattern, s_off)

    def _s_iter_off(f, ds, s_off):
        if swizzle_pattern is not None:
            return emit_iter_offset(swizzle_pattern, state.signed_strides, state.base_off, f)
        if swizzle is not None:
            return emit_fallback_offset(swizzle, s_off, ds)
        return s_off + ds

    # fmt: off
    s_zero_indices = [0] * len(s_buf.shape)

    @Tx.prim_func(check_well_formed=False)
    def impl():
        s_off = _substitute_axes(s_off_template, placeholders, sctx)
        _setup_swizzle(s_off)
        r_local = r_buf.local(*per_thread_r_shape)
        # Keep as a serial TIR loop and let ptxas unroll downstream. An
        # explicit ``Tx.unroll`` materializes the per-iter scratch
        # (ds/dr/s_ptr/r_ptr, swizzle ``v_<n>[]`` signed-strides) as N
        # copies of each buffer declaration; on kernels with many R↔S copy
        # sites and large ``total_outer`` (FA4 writeback) this floods the
        # function with ``alignas(64) int`` arrays and pressures registers.
        for f in range(total_outer):
            ds, dr = _outer_const_offsets(outer, f)
            s_ptr = _ptr_off(s_buf.ptr_to(s_zero_indices), _s_iter_off(f, ds, s_off))
            r_ptr = _ptr_off(r_local.ptr_to([0]), r_off_base + dr)
            if r_is_src:
                copy_op(s_ptr, r_ptr)
            else:
                copy_op(r_ptr, s_ptr)
    # fmt: on
    import os

    if os.environ.get("R2S_DUMP"):
        print("=== emitted impl ===")
        print(impl.script())
    return impl


@register_dispatch(
    "copy",
    "cuda",
    variant="reg",
    priority=10,
    when=[predicate("reg_applicable", _is_reg_copy)],
)
def copy_schedule_reg(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit_reg(op_call, sctx)
