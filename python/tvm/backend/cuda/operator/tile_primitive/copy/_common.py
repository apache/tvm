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

"""Shared partition / layout algorithm for synthesized-partition copy
dispatches (currently ``gmem_smem`` and ``ldgsts``).

``gmem_smem`` (sync ``Tx.copy`` global ↔ shared) and ``ldgsts`` (async
``Tx.copy_async`` global → shared via cp.async / SASS LDGSTS) share the
same algorithm to pick a vec-isolating + thread-distributing layout for
``G ↔ S`` copies. Only emit-time details differ (which copy instruction
to call, allowed vec widths). All the layout/partition logic lives here.
"""

from tvm import arith
from tvm.tirx.layout import ComposeLayout, Iter, S, SwizzleLayout, TileLayout
from tvm.tirx.operator.tile_primitive.registry import DispatchContext


def _alignment_ok(vec_len: int, terms) -> bool:
    """Every term must be a multiple of ``vec_len``. Constants checked
    directly; PrimExpr / symbolic terms checked via ``arith.Analyzer``.

    ``vec_len=1`` always passes (the scalar fallback). When a symbolic
    term can't be proved divisible, returns ``False`` conservatively —
    the candidate loop will then try a smaller ``vec_len``.
    """
    if vec_len <= 1:
        return True
    analyzer = arith.Analyzer()
    for t in terms:
        if isinstance(t, int):
            if t % vec_len != 0:
                return False
        else:
            if not analyzer.can_prove_equal(t % vec_len, 0):
                return False
    return True


# scope_kind → name of the scope_id that decomposes the scope into per-thread.
_TID_AXIS_FOR_SCOPE = {
    "warp": "laneid",
    "warpgroup": "tid_in_wg",
    "cta": "tx",
}


def _thread_cnt(sctx: DispatchContext) -> int:
    """Total threads active in the current scope = ∏ intra-axis extents.

    For thread scope ``sctx.intra`` is empty → returns 1.
    """
    n = 1
    for ext, _off in sctx.intra.values():
        n *= int(ext)
    return n


# -----------------------------------------------------------------------------
# Layout primitives
# -----------------------------------------------------------------------------


def _contig_group(iters: list) -> list[int]:
    """Indices (in iters) of the maximal physical-contiguous chain starting
    at the stride=1 iter, ordered stride-ascending.

    Returns [] if no stride=1 iter exists.
    """
    one_idx = next(
        (i for i, it in enumerate(iters) if int(it.stride) == 1),
        None,
    )
    if one_idx is None:
        return []
    chain = [one_idx]
    acc = int(iters[one_idx].extent)
    used = {one_idx}
    while True:
        nxt = next(
            (i for i, it in enumerate(iters) if i not in used and int(it.stride) == acc),
            None,
        )
        if nxt is None:
            break
        chain.append(nxt)
        acc *= int(iters[nxt].extent)
        used.add(nxt)
    return chain


def _try_split_vec(iters: list, vec_len: int):
    """Try to walk ``vec_len`` consecutive elements along the contig chain.

    Returns ``(new_iters, selected_positions)`` on success, ``None`` on
    failure. ``new_iters`` may contain a freshly-split iter (replacing one
    entry with its "outer" half, with the "inner" half appended at the end);
    ``selected_positions`` are positions in ``new_iters`` that together
    cover the ``vec_len`` contig elements.
    """
    chain = _contig_group(iters)
    if not chain:
        return None
    rem = vec_len
    new_iters = list(iters)
    selected: list[int] = []
    for orig_idx in chain:
        if rem == 0:
            break
        it = new_iters[orig_idx]
        ext = int(it.extent)
        if ext <= rem:
            if rem % ext != 0:
                return None
            selected.append(orig_idx)
            rem //= ext
        else:
            if ext % rem != 0:
                return None
            stride = int(it.stride)
            outer = Iter(ext // rem, stride * rem, it.axis)
            inner = Iter(rem, stride, it.axis)
            new_iters[orig_idx] = outer
            new_iters.append(inner)
            selected.append(len(new_iters) - 1)
            rem = 0
            break
    if rem != 0:
        return None
    return new_iters, selected


def _isolated_shape(iters: list, selected: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    """Build the isolated shape: each selected iter is its own segment;
    adjacent unselected iters are merged into a single segment.

    Returns ``(shape, segments)`` where ``segments[i] = (start, end)`` is
    the half-open range in ``iters`` covered by shape entry ``i``.
    """
    sel_set = set(selected)
    shape: list[int] = []
    segments: list[tuple[int, int]] = []
    cur_start = None
    cur_ext = 1
    for i, it in enumerate(iters):
        if i in sel_set:
            if cur_start is not None:
                shape.append(cur_ext)
                segments.append((cur_start, i))
                cur_start = None
                cur_ext = 1
            shape.append(int(it.extent))
            segments.append((i, i + 1))
        else:
            if cur_start is None:
                cur_start = i
            cur_ext *= int(it.extent)
    if cur_start is not None:
        shape.append(cur_ext)
        segments.append((cur_start, len(iters)))
    return shape, segments


def _vec_perm(iters: list, selected: list[int]) -> list[int]:
    """Reorder ``iters`` into ``[outer, vec]``, both ordered by stride
    descending so the stride=1 iter ends up at the very last position."""
    sel_set = set(selected)
    unsel_sorted = sorted(
        (i for i in range(len(iters)) if i not in sel_set),
        key=lambda i: -int(iters[i].stride),
    )
    sel_sorted = sorted(selected, key=lambda i: -int(iters[i].stride))
    return list(unsel_sorted) + sel_sorted


def _try_split_thread(iters: list, vec_selected: list[int], thread_cnt: int):
    """After ``_try_split_vec``, carve ``thread_cnt`` from the OUTER tail
    (smallest-stride outer iter, then towards bigger stride if needed).

    Unlike vec split, this doesn't require physical contiguity — T
    consecutive fused indices map to per-thread offsets via the layout's
    stride (which may be > 1).

    Returns ``(new_iters, thread_selected_positions)`` on success, ``None``
    on failure (outer doesn't divide T cleanly, or no outer iters left).
    """
    if thread_cnt == 1:
        return list(iters), []
    vec_set = set(vec_selected)
    outer = [i for i in range(len(iters)) if i not in vec_set]
    if not outer:
        return None
    outer_by_stride_desc = sorted(outer, key=lambda i: -int(iters[i].stride))
    rem = thread_cnt
    new_iters = list(iters)
    thread_selected: list[int] = []
    for orig_idx in reversed(outer_by_stride_desc):
        if rem == 0:
            break
        it = new_iters[orig_idx]
        ext = int(it.extent)
        if ext <= rem:
            if rem % ext != 0:
                return None
            thread_selected.append(orig_idx)
            rem //= ext
        else:
            if ext % rem != 0:
                return None
            stride = int(it.stride)
            new_iters[orig_idx] = Iter(ext // rem, stride * rem, it.axis)
            new_iters.append(Iter(rem, stride, it.axis))
            thread_selected.append(len(new_iters) - 1)
            rem = 0
            break
    if rem != 0:
        return None
    return new_iters, thread_selected


def _three_segment_perm(iters: list, t_selected: list[int], vec_selected: list[int]) -> list[int]:
    """Reorder ``iters`` into ``[outer, T, vec]`` segments. Within each
    segment, stride descending so stride=1 sits at the very end."""
    t_set = set(t_selected)
    vec_set = set(vec_selected)
    outer = sorted(
        (i for i in range(len(iters)) if i not in t_set and i not in vec_set),
        key=lambda i: -int(iters[i].stride),
    )
    t_sorted = sorted(t_selected, key=lambda i: -int(iters[i].stride))
    vec_sorted = sorted(vec_selected, key=lambda i: -int(iters[i].stride))
    return list(outer) + t_sorted + vec_sorted


def _shape_perm_for_isolated(
    shape_segments: list[tuple[int, int]], iter_perm: list[int]
) -> list[int]:
    """Given segments (one per shape entry, each = (start, end) in
    pre-perm iter positions) and the iter permutation, compute the
    corresponding shape permutation."""
    seg_of = [0] * sum(end - start for start, end in shape_segments)
    for seg_idx, (start, end) in enumerate(shape_segments):
        for k in range(start, end):
            seg_of[k] = seg_idx
    seen: set[int] = set()
    perm: list[int] = []
    for orig_idx in iter_perm:
        seg_idx = seg_of[orig_idx]
        if seg_idx not in seen:
            seen.add(seg_idx)
            perm.append(seg_idx)
    return perm


def _verify_s_tail_contig(s_p: TileLayout, vec_len: int) -> bool:
    """Check the last iters of ``s_p`` form a stride=1 contig chain whose
    extent product equals ``vec_len``."""
    iters = list(s_p.shard)
    if not iters:
        return vec_len == 1
    last = iters[-1]
    if int(last.stride) != 1:
        return False
    acc = int(last.extent)
    if acc == vec_len:
        return True
    for k in range(len(iters) - 2, -1, -1):
        it = iters[k]
        if int(it.stride) != acc:
            break
        acc *= int(it.extent)
        if acc == vec_len:
            return True
        if acc > vec_len:
            return False
    return acc >= vec_len and acc % vec_len == 0


_VEC_BITS_CANDIDATES = (128, 64, 32, 16, 8)


def _vec_len_candidates(elem_bits: int, allowed_bits: tuple | None = None) -> list[int]:
    """Vec-length candidates (in elements) for the given element width.

    ``allowed_bits`` optionally filters the per-instruction allowed widths
    (e.g. cp.async only accepts {128, 64, 32} bits = 16/8/4 bytes).
    Defaults to ``_VEC_BITS_CANDIDATES`` if not specified.
    """
    bits_tuple = allowed_bits if allowed_bits is not None else _VEC_BITS_CANDIDATES
    out: list[int] = []
    for vb in bits_tuple:
        if vb < elem_bits or vb % elem_bits != 0:
            continue
        n = vb // elem_bits
        if n not in out:
            out.append(n)
    if 1 not in out and allowed_bits is None:
        # Scalar fallback is only added for the unrestricted candidate set;
        # an instruction-specific list (cp.async etc.) keeps its strictness.
        out.append(1)
    return out


def _extract_tile(layout, region):
    """Strip swizzle so we can perm/group as a TileLayout."""
    if isinstance(layout, ComposeLayout):
        return layout.tile_layout
    if isinstance(layout, SwizzleLayout):
        extents = [int(end - start) for (start, end) in region]
        return TileLayout(S[tuple(extents)])
    return layout


def _sort_by_stride_desc(layout: TileLayout) -> TileLayout:
    """Reorder shard so list order = traversal order (outer first, stride=1
    last). Required before canonicalize() can fuse non-adjacent-but-contig
    iters."""
    iters = list(layout.shard)
    perm = sorted(range(len(iters)), key=lambda i: -int(iters[i].stride))
    if perm == list(range(len(iters))):
        return layout
    return layout.permute_dims(perm)


def _carve_tail(iters: list, chunk: int):
    """Carve ``chunk`` elements off the tail. Walk back across multiple
    iters as needed; at most one iter is split.

    Per iter (from last to first), let ``ext`` = iter extent and ``rem``
    = remaining chunk to fill:

    * ``ext == rem``: eat this iter whole, done.
    * ``ext < rem``: must divide ``rem``; eat whole, ``rem //= ext``.
    * ``ext > rem``: must divide ``ext``; split into
      ``(ext/rem, stride*rem) + (rem, stride)``, take the inner. Done.

    Returns the new iter list on success, ``None`` on failure.
    """
    if not iters or chunk <= 0:
        return None
    rem = chunk
    work = list(iters)
    for idx in range(len(work) - 1, -1, -1):
        it = work[idx]
        ext = int(it.extent)
        if ext == rem:
            return work
        if ext < rem:
            if rem % ext != 0:
                return None
            rem //= ext
            continue
        if ext % rem != 0:
            return None
        stride = int(it.stride)
        work[idx] = Iter(ext // rem, stride * rem, it.axis)
        work.insert(idx + 1, Iter(rem, stride, it.axis))
        return work
    return None


def align_layouts_gs(
    g_layout,
    g_shape,
    g_region,
    s_layout,
    s_shape,
    s_region,
    elem_bits,
    thread_cnt: int,
    vec_bits_candidates: tuple | None = None,
    debug: bool = False,
):
    """Align G and S layouts for a synthesized G↔S copy.

    Algorithm:
      1. Sort G iters by stride desc + canonicalize (fuses anything physically
         contig). Same for S.
      2. Group S by G's iter shape (per-iter shape list) + permute_by_groups
         with identity so S's groups line up with G's iters.
      3. Carve ``vec_len`` off G's tail (a single iter split if needed).
      4. Carve ``T = thread_cnt`` off the iter before vec (multi-iter walk,
         single split).
      5. Re-group S (already permuted) by the new finer per-iter shape. No
         further permute needed because steps 3-4 only refine the tail.
      6. Verify S's tail (vec segment) is physically contig.

    ``vec_bits_candidates`` optionally restricts the per-instruction allowed
    widths (e.g. cp.async only accepts {128, 64, 32} bits). When ``None``
    (default), uses the full {128, 64, 32, 16, 8} set plus a scalar (1)
    fallback.

    Returns ``(g_p, s_p, vec_len)``. ``g_p.shard`` ends as
    ``[outer iters..., T iter, vec iter]``; ``s_p.shard`` has the same iter
    count and matching iter-by-iter extents.
    """
    g = g_layout.slice(list(g_shape), g_region)
    s = s_layout.slice(list(s_shape), s_region)
    # Detect a SwizzleLayout on the S side BEFORE _extract_tile strips it.
    # vec_len must fit inside one swizzle chunk (C = 2^per_element elements);
    # otherwise the vec ld/st crosses a swizzle XOR boundary and hits the
    # wrong physical bytes mid-vec.
    s_swizzle_chunk_elems = None
    if isinstance(s_layout, ComposeLayout):
        s_swizzle_chunk_elems = 1 << int(s_layout.swizzle.per_element)
    elif isinstance(s_layout, SwizzleLayout):
        s_swizzle_chunk_elems = 1 << int(s_layout.per_element)
    g = _extract_tile(g, g_region)
    s = _extract_tile(s, s_region)

    # Only G drives the canonical form. S's iter order is derived from G's
    # pre-sort iter extents (used as the grouping shape) and then permuted
    # by G's stride-desc permutation. Independently sorting/canonicalizing
    # S would fuse iters whose strides happen to chain into one contiguous
    # range — that loses the layout's logical `(i, j) → addr` mapping for
    # layout-permuting copies (e.g. row-major GMEM → K-tiled SMEM, where
    # both layouts cover the same byte range but with different coord maps).
    g_pre_sort_extents = [int(it.extent) for it in g.shard]
    g_perm = sorted(range(len(g.shard)), key=lambda i: -int(g.shard[i].stride))
    try:
        s_grp1, seps1 = s.group(g_pre_sort_extents)
    except Exception as e:
        if debug:
            print(f"    Step-2 S.group({g_pre_sort_extents}) failed: {e}")
        return _sort_by_stride_desc(g).canonicalize(), s, 1
    # S iters keep their original strides; only the *group order* is
    # rearranged to follow G's sorted iter order. Canonicalize after the
    # permute is safe — it only fuses iters whose strides genuinely chain
    # in the post-permute order, which preserves the logical structure.
    s_aligned = s_grp1.permute_by_groups(list(seps1), g_perm).canonicalize()
    g = _sort_by_stride_desc(g).canonicalize()
    if debug:
        print(f"    g (sort+canon): shard={[(int(it.extent), int(it.stride)) for it in g.shard]}")
        print(
            f"    s (grouped+permuted by G shape {g_pre_sort_extents}): shard="
            f"{[(int(it.extent), int(it.stride)) for it in s_aligned.shard]}"
        )
        print(f"    thread_cnt: {thread_cnt}")

    dummy_axis = g.shard[-1].axis if g.shard else None

    for vec_len in _vec_len_candidates(elem_bits, vec_bits_candidates):
        if vec_len == 1:
            g_after_vec = [*list(g.shard), Iter(1, 1, dummy_axis)]
        else:
            # Swizzle chunk-size cap: vec must fit in one swizzle chunk.
            if s_swizzle_chunk_elems is not None and vec_len > s_swizzle_chunk_elems:
                if debug:
                    print(
                        f"    vec_len={vec_len}: exceeds swizzle chunk "
                        f"({s_swizzle_chunk_elems} elements)"
                    )
                continue
            g_after_vec = _carve_tail(list(g.shard), vec_len)
            if g_after_vec is None:
                if debug:
                    print(f"    vec_len={vec_len}: G vec carve failed")
                continue
        outer_for_t = g_after_vec[:-1]
        outer_after_t = _carve_tail(outer_for_t, thread_cnt) if thread_cnt > 1 else outer_for_t
        if outer_after_t is None:
            if debug:
                print(f"    vec_len={vec_len}: T={thread_cnt} carve failed")
            continue
        g_final_iters = [*outer_after_t, g_after_vec[-1]]
        new_shape = [int(it.extent) for it in g_final_iters]
        try:
            s_final_grp, _ = s_aligned.group(new_shape)
        except Exception as e:
            if debug:
                print(f"    vec_len={vec_len}: S.group({new_shape}) failed: {e}")
            continue
        g_p = TileLayout.from_iters(g_final_iters, list(g.replica), dict(g.offset))
        s_p = s_final_grp

        ok = _verify_s_tail_contig(s_p, vec_len)
        if debug:
            print(
                f"    vec_len={vec_len}: shape={new_shape}, "
                f"g_p.shard={[(int(it.extent), int(it.stride)) for it in g_p.shard]}, "
                f"s_p.shard={[(int(it.extent), int(it.stride)) for it in s_p.shard]}, "
                f"s_tail_contig={ok}"
            )
        if not ok:
            continue
        # Alignment: per-thread starting addr = base_ptr + sizeof(elem) *
        # (region_base + tid*t_stride + outer_iter_strides). For the
        # vec_bits/8 byte vector op to be naturally aligned, every one of
        # those element-count terms must be a multiple of vec_len.
        align_terms = []
        for it in s_p.shard[:-1]:
            align_terms.append(int(it.stride))
        for it in g_p.shard[:-1]:
            align_terms.append(int(it.stride))
        align_terms.extend(s_p.offset.values())
        align_terms.extend(g_p.offset.values())
        if not _alignment_ok(vec_len, align_terms):
            if debug:
                print(f"    vec_len={vec_len}: alignment check failed")
            continue
        return g_p, s_p, vec_len

    return g, s_aligned, 1


def _flat_outer_coords(outer_exts: list[int], flat_idx: int) -> list[int]:
    """Decode a row-major flat index into per-iter coords. ``outer_exts`` is
    in stride-desc order (outermost first), so the first coord changes
    slowest."""
    coords: list[int] = []
    rem = flat_idx
    for ext in reversed(outer_exts):
        coords.append(rem % ext)
        rem //= ext
    coords.reverse()
    return coords


def _outer_offsets(outer_iters_s, outer_iters_g, flat_idx):
    """Returns ``(ds, dg)``: constant offsets on S and G sides for the
    given flat outer-loop iteration."""
    outer_exts = [int(it.extent) for it in outer_iters_s]
    coords = _flat_outer_coords(outer_exts, flat_idx)
    ds = sum(c * int(it.stride) for c, it in zip(coords, outer_iters_s))
    dg = sum(c * int(it.stride) for c, it in zip(coords, outer_iters_g))
    return ds, dg
