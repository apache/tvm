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
"""Private implementation of derived :class:`Buffer` views."""

from __future__ import annotations

import functools
import re
from numbers import Integral
from typing import TYPE_CHECKING

import tvm

if TYPE_CHECKING:
    from .buffer import Buffer


def _redecl(buf: Buffer, shape, layout, *, dtype=None, elem_offset=None, addr_offset=None):
    """Re-declare a derived view over ``buf`` storage.

    Routes the buffer identity by storage kind: a ``tmem`` buffer aliases by
    column address, while every other scope carries
    ``data``/``strides``/``elem_offset`` through.
    """
    if buf.scope() == "tmem" and buf.allocated_addr is not None and len(buf.allocated_addr) > 0:
        addr = buf.allocated_addr[0]
        if addr_offset is not None:
            addr = addr + addr_offset
        return tvm.tirx.script.builder.decl_buffer(
            shape,
            buf.dtype if dtype is None else dtype,
            None,
            None,
            None,
            None,
            buf.scope(),
            buf.data_alignment,
            0,
            layout,
            allocated_addr=addr,
        )
    return tvm.tirx.script.builder.decl_buffer(
        shape,
        buf.dtype if dtype is None else dtype,
        buf.data,
        buf.strides,
        buf.elem_offset if elem_offset is None else elem_offset,
        None,
        buf.scope(),
        buf.data_alignment,
        buf.offset_factor,
        layout,
    )


def view(buf: Buffer, *args, **kwargs) -> Buffer:
    """Implement :meth:`Buffer.view`."""

    def _infer_shape(shape):
        shape = list(shape)
        if -1 in shape and shape.count(-1) == 1:
            size = functools.reduce(lambda x, y: x * y, buf.shape)
            n_size = functools.reduce(lambda x, y: x * y, [s for s in shape if s != -1], 1)
            shape[shape.index(-1)] = size // n_size
        else:
            # A PrimExpr comparison returns an EQ node, not a Python bool.
            if all(isinstance(s, int) for s in shape) and all(
                isinstance(s, int) for s in buf.shape
            ):
                assert functools.reduce(lambda x, y: x * y, shape) == functools.reduce(
                    lambda x, y: x * y, buf.shape
                ), (
                    "The shape of the buffer "
                    + str(buf.shape)
                    + " and the new shape "
                    + str(shape)
                    + " are not compatible"
                )
        return shape

    if len(args) == 1 and isinstance(args[0], str | tvm.DataType) and not kwargs:
        cast_dtype = tvm.DataType(args[0])
        cur_dtype = tvm.DataType(buf.dtype)
        if cast_dtype.bits > cur_dtype.bits:
            assert cast_dtype.bits % cur_dtype.bits == 0
            ratio = cast_dtype.bits // cur_dtype.bits
            layout = buf.layout.pack(ratio)
            shape = [s for s in buf.shape[:-1]] + [buf.shape[-1] // ratio]
            new_elem_offset = buf.elem_offset // ratio
        else:
            assert cur_dtype.bits % cast_dtype.bits == 0
            ratio = cur_dtype.bits // cast_dtype.bits
            layout = buf.layout.unpack(ratio)
            shape = [s for s in buf.shape[:-1]] + [buf.shape[-1] * ratio]
            new_elem_offset = buf.elem_offset * ratio
        return _redecl(buf, shape, layout, dtype=cast_dtype, elem_offset=new_elem_offset)

    shape = args
    assert all(
        isinstance(arg, int) or (tvm.ir.is_prim_expr(arg) and arg.ty.dtype in ["int32", "int64"])
        for arg in shape
    ), "shape must be a list of integers or Exprs with dtype int32 or int64"
    layout = kwargs.get("layout", None)
    assert set(kwargs.keys()).issubset({"layout"}), (
        f"Unsupported kwargs for view: {set(kwargs.keys()) - {'layout'}}"
    )
    if layout is None:
        shape = _infer_shape(shape)
    return _redecl(buf, shape, buf.layout if layout is None else layout)


def local(buf: Buffer, *shape, layout=None) -> Buffer:
    """Implement :meth:`Buffer.local`."""
    if not shape:
        local_layout = buf.layout.storage()
        total = functools.reduce(lambda x, y: x * y, [it.extent for it in local_layout.shard], 1)
        shape = (total,)
    return _redecl(buf, shape, buf.layout.storage() if layout is None else layout)


def permute(buf: Buffer, *dims) -> Buffer:
    """Implement :meth:`Buffer.permute`."""
    new_shape = [buf.shape[d] for d in dims]
    layout, swizzle = _surgery_parts(buf)
    grouped, seps = layout.group(list(buf.shape))
    new_layout = grouped.permute_by_groups(seps, list(dims))
    if swizzle is not None:
        new_layout = _rewrap_swizzle(new_layout, swizzle)
    return _redecl(buf, new_shape, new_layout)


def rearrange(buf: Buffer, pattern: str, /, **sizes) -> Buffer:
    """Implement :meth:`Buffer.rearrange`."""

    def _groups(side):
        out = []
        for group, bare in re.findall(r"\(([^)]*)\)|(\S+)", side.strip()):
            out.append(group.split() if group else [bare])
        return out

    if "->" not in pattern:
        raise ValueError(f"rearrange: pattern must contain '->', got {pattern!r}")
    lhs_s, rhs_s = pattern.split("->")
    lhs_groups, rhs_groups = _groups(lhs_s), _groups(rhs_s)
    if len(lhs_groups) != len(buf.shape):
        raise ValueError(
            f"rearrange: lhs has {len(lhs_groups)} groups but buffer has "
            f"{len(buf.shape)} dims (pattern {pattern!r}, shape {list(buf.shape)})"
        )
    axis_size: dict = {}
    for group, dim in zip(lhs_groups, buf.shape):
        dim_c = int(dim)
        known = {axis: int(sizes[axis]) for axis in group if axis in sizes}
        unknown = [axis for axis in group if axis not in sizes]
        product = 1
        for value in known.values():
            product *= value
        if len(unknown) == 0:
            if product != dim_c:
                raise ValueError(f"rearrange: group {group} product {product} != dim {dim_c}")
        elif len(unknown) == 1:
            if dim_c % product != 0:
                raise ValueError(
                    f"rearrange: dim {dim_c} not divisible by known product {product} "
                    f"in group {group}"
                )
            known[unknown[0]] = dim_c // product
        else:
            raise ValueError(
                f"rearrange: group {group} has >1 unknown axis; give sizes= for all but one"
            )
        axis_size.update(known)
    flat_lhs = [axis for group in lhs_groups for axis in group]
    flat_rhs = [axis for group in rhs_groups for axis in group]
    if sorted(flat_lhs) != sorted(flat_rhs):
        raise ValueError(
            f"rearrange: lhs axes {flat_lhs} != rhs axes {flat_rhs} (pattern {pattern!r})"
        )
    out = buf.view(*[axis_size[axis] for axis in flat_lhs])
    out = out.permute(*[flat_lhs.index(axis) for axis in flat_rhs])
    merged_shape = []
    for group in rhs_groups:
        size = 1
        for axis in group:
            size *= axis_size[axis]
        merged_shape.append(size)
    return out.view(*merged_shape)


def sub(buf: Buffer) -> SubIndexer:
    """Return the indexer used by :attr:`Buffer.sub`."""
    return SubIndexer(buf)


def tile(buf: Buffer, *specs) -> TileIndexer:
    """Implement :meth:`Buffer.tile`."""
    if specs and isinstance(specs[0], int | Integral):
        if len(specs) != 2:
            raise ValueError("tile(dim, factors) takes a dim and a factors tuple")
        specs = (specs,)
    normalized = []
    seen = set()
    for spec in specs:
        if not isinstance(spec, tuple | list) or len(spec) != 2:
            raise ValueError(f"tile: each spec must be (dim, factors), got {spec!r}")
        dim = _normalized_dim(buf, spec[0], "tile")
        if dim in seen:
            raise ValueError(f"tile: dim {dim} tiled more than once")
        seen.add(dim)
        factors = spec[1]
        if not isinstance(factors, tuple | list) or len(factors) < 1:
            raise ValueError(f"tile: factors for dim {dim} must be a non-empty tuple")
        normalized.append((dim, tuple(factors)))
    return TileIndexer(buf, normalized)


def chunk(buf: Buffer, spec) -> ChunkIndexer:
    """Implement :meth:`Buffer.chunk`."""
    if not isinstance(spec, tuple | list):
        raise ValueError(f"chunk: spec must be a per-dim tuple, got {spec!r}")
    if len(spec) != len(buf.shape):
        raise ValueError(f"chunk: spec length {len(spec)} != buffer rank {len(buf.shape)}")
    for dim, count in enumerate(spec):
        if count is not None and (not isinstance(count, Integral) or int(count) < 1):
            raise ValueError(f"chunk: spec[{dim}] must be None or a positive int, got {count!r}")
    return ChunkIndexer(buf, tuple(spec))


def _normalized_dim(buf: Buffer, dim, name):
    ndim = len(buf.shape)
    if dim < 0:
        dim += ndim
    if not 0 <= dim < ndim:
        raise ValueError(f"{name}: dim {dim} out of range for buffer of rank {ndim}")
    return dim


def _concrete_int(value):
    """Return ``value`` as a Python int when it is statically known."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, tvm.tirx.IntImm):
        return int(value)
    return None


def _surgery_parts(buf: Buffer):
    """Split a layout into its inner tile layout and optional swizzle params."""
    layout = buf.layout
    swizzle = None
    if isinstance(layout, tvm.tirx.layout.ComposeLayout):
        swizzle = (
            layout.per_element,
            layout.swizzle_len,
            layout.atom_len,
            layout.swizzle_inner,
        )
        layout = layout.tile_layout
    return layout, swizzle


def _rewrap_swizzle(layout, swizzle):
    if swizzle is not None:
        per_element, swizzle_len, atom_len, swizzle_inner = swizzle
        return tvm.tirx.layout.ComposeLayout(
            per_element, swizzle_len, atom_len, layout, swizzle_inner
        )
    return layout


def _dim_group_offset(group, index):
    """Return the offset of ``index`` along a grouped logical dimension."""
    if len(group) == 1:
        return index * group[0].stride
    extents = []
    for iterator in group[1:]:
        extent = iterator.extent
        if not isinstance(extent, tvm.tirx.IntImm):
            raise ValueError(
                "select/narrow: dim with multiple layout iters requires concrete iter extents"
            )
        extents.append(int(extent))
    offset = None
    remaining = index
    for pos, iterator in enumerate(group):
        inner = functools.reduce(lambda a, b: a * b, extents[pos:], 1)
        if inner == 1:
            coord = remaining
        elif isinstance(remaining, Integral):
            coord = remaining // inner
            remaining = remaining % inner
        else:
            coord = tvm.tirx.floordiv(remaining, inner)
            remaining = tvm.tirx.floormod(remaining, inner)
        term = coord * iterator.stride
        offset = term if offset is None else offset + term
    return offset


def _swizzle_offset_commutes(swizzle, extra_offset):
    """Return whether an offset may be folded outside a swizzle permutation."""
    if swizzle is None or extra_offset is None:
        return True
    per_element, swizzle_len, atom_len, _ = swizzle
    if swizzle_len == 0:
        return True
    period = 1 << (int(per_element) + int(atom_len) + swizzle_len)
    offset_c = _concrete_int(extra_offset)
    if offset_c is not None:
        return offset_c % period == 0
    from ..arith import Analyzer  # pylint: disable=import-outside-toplevel

    return Analyzer().can_prove_equal(tvm.tirx.floormod(extra_offset, period), 0)


def _rebuild_view(buf: Buffer, new_shape, new_shard, grouped, swizzle, extra_offset):
    """Rebuild a derived view while preserving its physical addresses."""
    offset_map = dict(grouped.offset.items())
    is_tmem = (
        buf.scope() == "tmem" and buf.allocated_addr is not None and len(buf.allocated_addr) > 0
    )
    if (
        not is_tmem
        and extra_offset is not None
        and not _swizzle_offset_commutes(swizzle, extra_offset)
    ):
        m_axis = tvm.tirx.layout.Axis.get("m")
        previous = offset_map.get(m_axis)
        offset_map[m_axis] = extra_offset if previous is None else previous + extra_offset
        extra_offset = None
    new_layout = tvm.tirx.layout.TileLayout.from_iters(new_shard, list(grouped.replica), offset_map)
    new_layout = _rewrap_swizzle(new_layout, swizzle)
    if is_tmem:
        addr_offset = _tmem_element_offset_to_column_offset(buf, extra_offset)
        return _redecl(buf, new_shape, new_layout, addr_offset=addr_offset)
    elem_offset = buf.elem_offset
    if extra_offset is not None:
        elem_offset = elem_offset + extra_offset
    return _redecl(buf, new_shape, new_layout, elem_offset=elem_offset)


def _tmem_element_offset_to_column_offset(buf: Buffer, element_offset):
    """Convert a TCol element offset to a physical 32-bit column offset."""
    if element_offset is None:
        return None

    bits = tvm.DataType(buf.dtype).bits
    bit_offset = element_offset * bits
    bit_offset_c = _concrete_int(bit_offset)
    if bit_offset_c is not None:
        if bit_offset_c % 32 != 0:
            raise ValueError(
                "sub: tmem TCol offset must be aligned to a physical "
                f"32-bit column; got {element_offset} elements of {buf.dtype}"
            )
        return bit_offset_c // 32

    from ..arith import Analyzer  # pylint: disable=import-outside-toplevel

    analyzer = Analyzer()
    if not analyzer.can_prove_equal(tvm.tirx.floormod(bit_offset, 32), 0):
        raise ValueError(
            "sub: tmem TCol offset must be provably aligned to a physical "
            f"32-bit column; got {element_offset} elements of {buf.dtype}"
        )
    return analyzer.simplify(tvm.tirx.floordiv(bit_offset, 32))


def _tmem_offset_axis_check(buf: Buffer, group_iters, start_c, what):
    """Check that a tmem view offset can move into ``allocated_addr``."""
    if buf.allocated_addr is None or len(buf.allocated_addr) == 0:
        return
    if start_c == 0:
        return
    for iterator in group_iters:
        axis_name = getattr(iterator.axis, "name", None) if iterator.axis is not None else None
        if axis_name != "TCol":
            raise ValueError(
                f"sub: tmem {what} with nonzero offset requires TCol-stride "
                f"layout iters; dim iter has axis {axis_name!r} (a lane-axis "
                f"offset cannot fold into the column allocated_addr)"
            )


def _view_drop(buf: Buffer, dim, index):
    """Drop one logical dimension while preserving the selected storage offset."""
    dim = _normalized_dim(buf, dim, "index")
    index_c = _concrete_int(index)
    extent_c = _concrete_int(buf.shape[dim])
    if index_c is not None and (index_c < 0 or (extent_c is not None and index_c >= extent_c)):
        raise ValueError(
            f"index {index_c} out of range for dim {dim} "
            f"of extent {extent_c if extent_c is not None else buf.shape[dim]}"
        )
    layout, swizzle = _surgery_parts(buf)
    grouped, seps = layout.group(list(buf.shape))
    iters = list(grouped.shard)
    lo, hi = seps[dim], seps[dim + 1]
    _tmem_offset_axis_check(buf, iters[lo:hi], index_c, "index")
    offset = _dim_group_offset(iters[lo:hi], index)
    new_shape = list(buf.shape[:dim]) + list(buf.shape[dim + 1 :])
    new_shard = iters[:lo] + iters[hi:]
    return _rebuild_view(buf, new_shape, new_shard, grouped, swizzle, offset)


def _view_narrow(buf: Buffer, dim, start, length):
    """Narrow one logical dimension while preserving its storage offset."""
    dim = _normalized_dim(buf, dim, "index")
    start_c = _concrete_int(start)
    length_c = _concrete_int(length)
    extent_c = _concrete_int(buf.shape[dim])
    if start_c is not None and start_c < 0:
        raise ValueError(f"sub: start {start_c} must be non-negative")
    if length_c is not None and length_c < 1:
        raise ValueError(f"sub: length {length_c} must be positive")
    if (
        start_c is not None
        and length_c is not None
        and extent_c is not None
        and start_c + length_c > extent_c
    ):
        raise ValueError(
            f"sub: range [{start_c}, {start_c + length_c}) exceeds dim {dim} extent {extent_c}"
        )
    layout, swizzle = _surgery_parts(buf)
    grouped, seps = layout.group(list(buf.shape))
    iters = list(grouped.shard)
    lo, hi = seps[dim], seps[dim + 1]
    group = iters[lo:hi]
    iter_type = tvm.tirx.layout.Iter
    if len(group) == 1:
        iterator = group[0]
        _tmem_offset_axis_check(buf, [iterator], start_c, "narrow")
        offset = start * iterator.stride
        new_group = [iter_type(length, iterator.stride, iterator.axis)]
    else:
        inner = 1
        for iterator in group[1:]:
            extent = iterator.extent
            if not isinstance(extent, tvm.tirx.IntImm):
                raise ValueError(
                    "sub: dim with multiple layout iters requires concrete iter extents"
                )
            inner *= int(extent)
        if not (isinstance(start, Integral) and isinstance(length, Integral)):
            raise ValueError(
                f"sub: dim {dim} is made of {len(group)} layout iters; "
                f"start/length must be concrete multiples of {inner}"
            )
        if start % inner != 0 or length % inner != 0:
            raise ValueError(
                f"sub: dim {dim} start/length must be multiples of the "
                f"inner iter block {inner}; got start={start} length={length}"
            )
        outer = group[0]
        _tmem_offset_axis_check(buf, [outer], start // inner, "narrow")
        offset = (start // inner) * outer.stride
        new_group = [iter_type(length // inner, outer.stride, outer.axis), *group[1:]]
    new_shape = list(buf.shape)
    new_shape[dim] = length
    new_shard = iters[:lo] + new_group + iters[hi:]
    return _rebuild_view(buf, new_shape, new_shard, grouped, swizzle, offset)


class SubIndexer:
    """Indexer returned by :attr:`Buffer.sub`."""

    def __init__(self, buffer: Buffer):
        self._buffer = buffer

    def __getitem__(self, indices) -> Buffer:
        if not isinstance(indices, tuple):
            indices = (indices,)
        buf = self._buffer
        if len(indices) > len(buf.shape):
            raise ValueError(f"sub: {len(indices)} indices for buffer of rank {len(buf.shape)}")
        dim = 0
        for item in indices:
            if isinstance(item, slice):
                step = item.step
                if step is None or (isinstance(step, Integral) and step == 1):
                    if item.start is None and item.stop is None:
                        dim += 1
                        continue
                    start = 0 if item.start is None else item.start
                    stop = buf.shape[dim] if item.stop is None else item.stop
                    buf = _view_narrow(buf, dim, start, stop - start)
                    dim += 1
                else:
                    if not isinstance(step, Integral) or step <= 0:
                        raise ValueError(f"sub: step must be a positive int, got {step!r}")
                    if item.stop is not None:
                        raise ValueError("sub: a stop bound with a step is not supported")
                    extent = buf.shape[dim]
                    if not isinstance(extent, tvm.tirx.IntImm):
                        raise ValueError("sub: a stepped slice requires a concrete dim extent")
                    extent = int(extent)
                    if extent % step != 0:
                        raise ValueError(
                            f"sub: dim extent {extent} is not divisible by step {step}"
                        )
                    start = 0 if item.start is None else item.start
                    start_c = _concrete_int(start)
                    if start_c is not None and not 0 <= start_c < step:
                        raise ValueError(
                            f"sub: stepped-slice start {start_c} must be in [0, {step})"
                        )
                    split = buf.view(*buf.shape[:dim], extent // step, step, *buf.shape[dim + 1 :])
                    buf = _view_drop(split, dim + 1, start)
                    dim += 1
            else:
                buf = _view_drop(buf, dim, item)
        return buf


class ChunkIndexer:
    """Indexer returned by :meth:`Buffer.chunk`."""

    def __init__(self, buffer: Buffer, spec):
        self._buffer = buffer
        self._spec = spec

    def __getitem__(self, picks):
        if not isinstance(picks, tuple):
            picks = (picks,)
        buf = self._buffer
        if len(picks) > len(self._spec):
            raise ValueError(f"chunk: {len(picks)} indices for a rank-{len(self._spec)} spec")
        translated = []
        for dim, pick in enumerate(picks):
            count = self._spec[dim]
            if count is None:
                translated.append(pick)
            elif isinstance(pick, slice):
                raise ValueError(
                    f"chunk: chunked dim {dim} takes a chunk index, not a slice; got {pick!r}"
                )
            else:
                size = buf.shape[dim] // int(count)
                translated.append(slice(pick * size, (pick + 1) * size))
        return buf[tuple(translated)]


class TileIndexer:
    """Indexer returned by :meth:`Buffer.tile`."""

    def __init__(self, buffer: Buffer, specs):
        self._buffer = buffer
        self._specs = specs

    def __getitem__(self, picks) -> Buffer:
        if not isinstance(picks, tuple):
            picks = (picks,)
        n_factors = sum(len(factors) for _, factors in self._specs)
        if len(picks) != n_factors:
            raise ValueError(
                f"tile: split into {n_factors} factor(s) but {len(picks)} "
                f"index(es) given (int/Expr picks, ':' keeps)"
            )
        groups, pos = [], 0
        for dim, factors in self._specs:
            groups.append((dim, factors, picks[pos : pos + len(factors)]))
            pos += len(factors)
        buf = self._buffer
        for dim, factors, dim_picks in sorted(groups, key=lambda group: group[0], reverse=True):
            buf = self._apply(buf, dim, factors, dim_picks)
        return buf

    @staticmethod
    def _is_keep(pick):
        return (
            isinstance(pick, slice)
            and pick.start is None
            and pick.stop is None
            and pick.step is None
        )

    @classmethod
    def _apply(cls, buf, dim, factors, dim_picks):
        for pick in dim_picks:
            if isinstance(pick, slice) and not cls._is_keep(pick):
                raise ValueError("tile: a factor slice must be ':' (keep whole); got a sub-slice")
        n_keep = sum(1 for pick in dim_picks if cls._is_keep(pick))
        if n_keep == len(dim_picks):
            raise ValueError(
                f"tile: dim {dim} picks no factor (all ':'); a chunk must pick "
                f"at least one factor. Use .view / .chunk for a pure reshape."
            )
        buf = buf.view(*buf.shape[:dim], *factors, *buf.shape[dim + 1 :])
        for index in reversed(range(len(dim_picks))):
            if not cls._is_keep(dim_picks[index]):
                buf = _view_drop(buf, dim + index, dim_picks[index])
        if n_keep >= 2:
            buf = buf.view(*buf.shape[:dim], -1, *buf.shape[dim + n_keep :])
        return buf
