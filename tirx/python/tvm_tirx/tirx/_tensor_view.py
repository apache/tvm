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
"""Private implementation of derived :class:`Var` views."""

from __future__ import annotations

import functools
import re
from numbers import Integral
from typing import TYPE_CHECKING

import tvm_ffi

import tvm_tirx
from tvm_tirx.runtime import MemoryScope


@functools.cache
def _tmem_scope():
    return MemoryScope("tmem")


def _is_tmem_scope(scope):
    try:
        return scope == _tmem_scope()
    except ValueError:
        # Do not cache absence: an owning backend may register its scopes later.
        return False


if TYPE_CHECKING:
    from tvm_tirx.ir import Var


def _redecl(buf: Var, shape, layout, *, dtype=None, addr_offset=None):
    """Re-declare a derived view over ``buf`` storage.

    Routes the tensor identity by storage kind: a ``tmem`` tensor aliases by
    column address, while every other scope carries its physical pointer and layout.
    """
    if _is_tmem_scope(buf.scope()):
        from tvm_tirx.backend.cuda.op import (
            _tmem_address,  # pylint: disable=import-outside-toplevel
        )
        from tvm_tirx.tirx.script.ir_builder.parser_protocol import (
            bind_,  # pylint: disable=import-outside-toplevel
        )

        addr = _tmem_address(buf)
        if addr_offset is not None:
            addr = addr + addr_offset
        ty = tvm_tirx.tirx.script.ir_builder.Tensor(
            shape,
            buf.dtype if dtype is None else dtype,
            scope="tmem",
            align=buf.data_alignment,
            layout=layout,
        )
        return bind_(tvm_tirx.ir.Call("tirx.cuda.decl_tmem", [addr], ty_args=[ty]))
    from tvm_tirx.tirx.script.ir_builder.parser_protocol import (  # pylint: disable=import-outside-toplevel
        bind_,
    )

    tensor_type = tvm_tirx.tirx.script.ir_builder.Tensor(
        shape,
        buf.dtype if dtype is None else dtype,
        scope=buf.scope(),
        align=buf.data_alignment,
        layout=layout,
    )
    return bind_(tvm_tirx.tirx.script.ir_builder.decl_tensor(buf.data, ty_args=[tensor_type]))


def _view_layout(buf):
    """Materialize implicit compact storage for layout transformations."""
    from .layout import S, TileLayout

    if buf.layout is not None:
        return buf.layout
    return TileLayout(S[tuple(buf.shape)])


def _is_memory_tile(layout, shape):
    """Whether view surgery can address each memory dimension directly."""
    from .layout import Axis, TileLayout

    return (
        isinstance(layout, TileLayout)
        and not layout.replica
        and len(layout.shard) == len(shape)
        and all(it.axis == Axis.m for it in layout.shard)
        and all(axis == Axis.m for axis in layout.offset)
        and all(
            tvm_tirx.ir.prim.expr_deep_equal(it.extent, extent)
            for it, extent in zip(layout.shard, shape)
        )
    )


def _reshape_layout(tile, shape):
    """Reshape contiguous iterator groups without changing their storage addresses."""
    from .layout import Iter, TileLayout

    analyzer = tvm_tirx.sym.Analyzer()
    old_shape = [it.extent for it in tile.shard]
    old_strides = [it.stride for it in tile.shard]
    if len(shape) == len(old_shape) and all(
        analyzer.can_prove_equal(a, b) for a, b in zip(shape, old_shape)
    ):
        return tile
    chunks = []
    for extent, stride in reversed(list(zip(old_shape, old_strides))):
        if analyzer.can_prove_equal(extent, 1):
            continue
        if chunks and analyzer.can_prove_equal(stride, chunks[-1][0] * chunks[-1][1]):
            chunks[-1] = (analyzer.simplify(chunks[-1][0] * extent), chunks[-1][1])
        else:
            chunks.append((extent, stride))
    result = [None] * len(shape)
    i = len(shape) - 1
    for extent, stride in chunks:
        product = 1
        while i >= 0:
            result[i] = analyzer.simplify(product * stride)
            product = analyzer.simplify(product * shape[i])
            i -= 1
            if analyzer.can_prove_equal(product, extent):
                break
        if not analyzer.can_prove_equal(product, extent):
            return None
    while i >= 0 and analyzer.can_prove_equal(shape[i], 1):
        result[i] = analyzer.simplify(result[i + 1] * shape[i + 1]) if i + 1 < len(shape) else 1
        i -= 1
    if i >= 0:
        return None
    return TileLayout.from_iters(
        [Iter(extent, stride, "m") for extent, stride in zip(shape, result)],
        offset=dict(tile.offset.items()),
    )


def view(buf: Var, *args, **kwargs) -> Var:
    """Implement :meth:`TensorType.view`."""

    def _infer_shape(shape):
        shape = list(shape)
        inferred = [i for i, dim in enumerate(shape) if _concrete_int(dim) == -1]
        if len(inferred) > 1:
            raise ValueError("view: at most one inferred dimension is allowed")
        if inferred:
            size = functools.reduce(lambda x, y: x * y, buf.shape)
            n_size = functools.reduce(
                lambda x, y: x * y, [s for i, s in enumerate(shape) if i != inferred[0]], 1
            )
            shape[inferred[0]] = size // n_size
        else:
            # A PrimExpr comparison returns an EQ node, not a Python bool.
            if all(isinstance(s, int) for s in shape) and all(
                isinstance(s, int) for s in buf.shape
            ):
                assert functools.reduce(lambda x, y: x * y, shape) == functools.reduce(
                    lambda x, y: x * y, buf.shape
                ), (
                    "The shape of the tensor "
                    + str(buf.shape)
                    + " and the new shape "
                    + str(shape)
                    + " are not compatible"
                )
        return shape

    from .layout import Iter, TileLayout

    source = _view_layout(buf)
    regular = _is_memory_tile(_view_layout(buf), buf.shape)
    if len(args) == 1 and isinstance(args[0], str | tvm_ffi.dtype) and not kwargs:
        cast_dtype = tvm_ffi.dtype(args[0])
        cur_dtype = tvm_ffi.dtype(buf.dtype)
        pack = cast_dtype.bits > cur_dtype.bits
        ratio = (cast_dtype.bits // cur_dtype.bits) if pack else (cur_dtype.bits // cast_dtype.bits)
        assert max(cast_dtype.bits, cur_dtype.bits) % min(cast_dtype.bits, cur_dtype.bits) == 0
        shape = [*buf.shape[:-1], buf.shape[-1] // ratio if pack else buf.shape[-1] * ratio]
        if regular:
            analyzer = tvm_tirx.sym.Analyzer()
            if not analyzer.can_prove_equal(source.shard[-1].stride, 1):
                raise ValueError("view(dtype): the last dimension must have unit stride")
            values = [
                buf.shape[-1],
                *source.offset.values(),
                *[it.stride for it in source.shard[:-1]],
            ]
            if pack and any(analyzer.can_prove(value % ratio != 0) for value in values):
                raise ValueError(
                    "view(dtype): shape, offset and strides must align to the new dtype"
                )

            def scale(value):
                return analyzer.simplify(value // ratio if pack else value * ratio)

            layout = TileLayout.from_iters(
                [
                    Iter(extent, scale(it.stride) if i + 1 < len(shape) else 1, "m")
                    for i, (extent, it) in enumerate(zip(shape, source.shard))
                ],
                offset={axis: scale(value) for axis, value in source.offset.items()},
            )
        else:
            axis = "TCol" if _is_tmem_scope(buf.scope()) else "m"
            layout = source.pack(ratio, axis) if pack else source.unpack(ratio, axis)
        return _redecl(buf, shape, None if buf.layout is None else layout, dtype=cast_dtype)

    shape = args
    assert all(
        isinstance(arg, int)
        or (tvm_tirx.ir.is_prim_expr(arg) and arg.ty.dtype in ["int32", "int64"])
        for arg in shape
    ), "shape must be a list of integers or Exprs with dtype int32 or int64"
    layout = kwargs.get("layout", None)
    assert set(kwargs.keys()).issubset({"layout"}), (
        f"Unsupported kwargs for view: {set(kwargs.keys()) - {'layout'}}"
    )
    if layout is None:
        shape = _infer_shape(shape)
    if layout is not None:
        return _redecl(buf, shape, layout)
    analyzer = tvm_tirx.sym.Analyzer()
    if not analyzer.can_prove_equal(
        functools.reduce(lambda a, b: a * b, buf.shape, 1),
        functools.reduce(lambda a, b: a * b, shape, 1),
    ):
        raise ValueError("view: source and result must have equal element counts")
    if buf.layout is None:
        return _redecl(buf, shape, None)
    if regular:
        reshaped = _reshape_layout(source, shape)
        if reshaped is not None:
            return _redecl(buf, shape, reshaped)
    return _redecl(buf, shape, source)


def permute(buf: Var, *dims) -> Var:
    """Implement :meth:`TensorType.permute`."""
    if sorted(dims) != list(range(len(buf.shape))):
        raise ValueError(f"permute: invalid dimensions {dims}")
    new_shape = [buf.shape[d] for d in dims]
    layout, swizzle = _surgery_parts(buf)
    if swizzle is None and _is_memory_tile(_view_layout(buf), buf.shape):
        grouped, seps = layout, list(range(len(buf.shape) + 1))
    else:
        grouped, seps = layout.group(list(buf.shape))
    new_layout = grouped.permute_by_groups(seps, list(dims))
    if swizzle is not None:
        new_layout = _rewrap_swizzle(new_layout, swizzle)
    return _redecl(buf, new_shape, new_layout)


def rearrange(buf: Var, pattern: str, /, **sizes) -> Var:
    """Implement :meth:`TensorType.rearrange`."""

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
            f"rearrange: lhs has {len(lhs_groups)} groups but tensor has "
            f"{len(buf.shape)} dims (pattern {pattern!r}, shape {list(buf.shape)})"
        )
    axis_size: dict = {}
    analyzer = tvm_tirx.sym.Analyzer()
    for group, dim in zip(lhs_groups, buf.shape):
        dim_c = int(analyzer.simplify(dim))
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


def sub(buf: Var) -> SubIndexer:
    """Return the indexer used by ``expr.sub``."""
    return SubIndexer(buf)


def _normalized_dim(buf: Var, dim, name):
    ndim = len(buf.shape)
    if dim < 0:
        dim += ndim
    if not 0 <= dim < ndim:
        raise ValueError(f"{name}: dim {dim} out of range for tensor of rank {ndim}")
    return dim


def _concrete_int(value):
    """Return ``value`` as a Python int when it is statically known."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, tvm_tirx.ir.IntImm):
        return int(value)
    return None


def _surgery_parts(buf: Var):
    """Split a layout into its inner tile layout and optional swizzle params."""
    layout = _view_layout(buf)
    swizzle = None
    if isinstance(layout, tvm_tirx.tirx.layout.ComposeLayout):
        swizzle = (
            layout.per_element,
            layout.swizzle_len,
            layout.atom_len,
            layout.swizzle_inner,
        )
        layout = layout.tile_layout
        # A bare swizzle describes its period, rather than the tensor extent.
        # Expand the swizzle period to the tensor domain before permuting axes.
        if layout.is_trivial():
            layout = tvm_tirx.tirx.layout.TileLayout(tvm_tirx.tirx.layout.S[tuple(buf.shape)])
    return layout, swizzle


def _rewrap_swizzle(layout, swizzle):
    if swizzle is not None:
        per_element, swizzle_len, atom_len, swizzle_inner = swizzle
        return tvm_tirx.tirx.layout.ComposeLayout(
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
        if not isinstance(extent, tvm_tirx.ir.IntImm):
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
            coord = tvm_tirx.tirx.floordiv(remaining, inner)
            remaining = tvm_tirx.tirx.floormod(remaining, inner)
        term = coord * iterator.stride
        offset = term if offset is None else offset + term
    return offset


def _rebuild_view(buf: Var, new_shape, new_shard, grouped, swizzle, extra_offset):
    """Rebuild a derived view while preserving its physical addresses."""
    offset_map = dict(grouped.offset.items())
    is_tmem = _is_tmem_scope(buf.scope())
    if not is_tmem and extra_offset is not None:
        m_axis = tvm_tirx.tirx.layout.Axis.get("m")
        previous = offset_map.get(m_axis)
        offset_map[m_axis] = extra_offset if previous is None else previous + extra_offset
        extra_offset = None
    new_layout = tvm_tirx.tirx.layout.TileLayout.from_iters(
        new_shard, list(grouped.replica), offset_map
    )
    new_layout = _rewrap_swizzle(new_layout, swizzle)
    if is_tmem:
        addr_offset = _tmem_element_offset_to_column_offset(buf, extra_offset)
        return _redecl(buf, new_shape, new_layout, addr_offset=addr_offset)
    return _redecl(buf, new_shape, new_layout)


def _tmem_element_offset_to_column_offset(buf: Var, element_offset):
    """Convert a TCol element offset to a physical 32-bit column offset."""
    if element_offset is None:
        return None

    bits = tvm_ffi.dtype(buf.dtype).bits
    bit_offset = element_offset * bits
    bit_offset_c = _concrete_int(bit_offset)
    if bit_offset_c is not None:
        if bit_offset_c % 32 != 0:
            raise ValueError(
                "sub: tmem TCol offset must be aligned to a physical "
                f"32-bit column; got {element_offset} elements of {buf.dtype}"
            )
        return bit_offset_c // 32

    from ..sym import Analyzer  # pylint: disable=import-outside-toplevel

    analyzer = Analyzer()
    if not analyzer.can_prove_equal(tvm_tirx.tirx.floormod(bit_offset, 32), 0):
        raise ValueError(
            "sub: tmem TCol offset must be provably aligned to a physical "
            f"32-bit column; got {element_offset} elements of {buf.dtype}"
        )
    return analyzer.simplify(tvm_tirx.tirx.floordiv(bit_offset, 32))


def _tmem_offset_axis_check(buf: Var, group_iters, start_c, what):
    """Check that a tmem view offset can move into the declaration address."""
    if not _is_tmem_scope(buf.scope()):
        return
    if start_c == 0:
        return
    for iterator in group_iters:
        axis_name = getattr(iterator.axis, "name", None) if iterator.axis is not None else None
        if axis_name != "TCol":
            raise ValueError(
                f"sub: tmem {what} with nonzero offset requires TCol-stride "
                f"layout iters; dim iter has axis {axis_name!r} (a lane-axis "
                f"offset cannot fold into the column address)"
            )


def _view_drop(buf: Var, dim, index):
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
    if swizzle is None and _is_memory_tile(_view_layout(buf), buf.shape):
        grouped, seps = layout, list(range(len(buf.shape) + 1))
    else:
        grouped, seps = layout.group(list(buf.shape))
    iters = list(grouped.shard)
    lo, hi = seps[dim], seps[dim + 1]
    _tmem_offset_axis_check(buf, iters[lo:hi], index_c, "index")
    offset = _dim_group_offset(iters[lo:hi], index)
    new_shape = list(buf.shape[:dim]) + list(buf.shape[dim + 1 :])
    new_shard = iters[:lo] + iters[hi:]
    return _rebuild_view(buf, new_shape, new_shard, grouped, swizzle, offset)


def _view_narrow(buf: Var, dim, start, length):
    """Narrow one logical dimension while preserving its storage offset."""
    dim = _normalized_dim(buf, dim, "index")
    analyzer = tvm_tirx.sym.Analyzer()
    start, length = analyzer.simplify(start), analyzer.simplify(length)
    start_c = _concrete_int(start)
    length_c = _concrete_int(length)
    start = start if start_c is None else start_c
    length = length if length_c is None else length_c
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
    if swizzle is None and _is_memory_tile(_view_layout(buf), buf.shape):
        grouped, seps = layout, list(range(len(buf.shape) + 1))
    else:
        grouped, seps = layout.group(list(buf.shape))
    iters = list(grouped.shard)
    lo, hi = seps[dim], seps[dim + 1]
    group = iters[lo:hi]
    iter_type = tvm_tirx.tirx.layout.Iter
    if len(group) == 1:
        iterator = group[0]
        _tmem_offset_axis_check(buf, [iterator], start_c, "narrow")
        offset = start * iterator.stride
        new_group = [iter_type(length, iterator.stride, iterator.axis)]
    else:
        inner = 1
        for iterator in group[1:]:
            extent = iterator.extent
            if not isinstance(extent, tvm_tirx.ir.IntImm):
                raise ValueError(
                    "sub: dim with multiple layout iters requires concrete iter extents"
                )
            inner *= int(extent)
        if not (
            analyzer.can_prove_equal(start % inner, 0)
            and analyzer.can_prove_equal(length % inner, 0)
        ):
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
    """Indexer returned by ``expr.sub``."""

    def __init__(self, tensor: Var):
        self._tensor = tensor

    def __getitem__(self, indices) -> Var:
        if not isinstance(indices, tuple):
            indices = (indices,)
        buf = self._tensor
        if len(indices) > len(buf.shape):
            raise ValueError(f"sub: {len(indices)} indices for tensor of rank {len(buf.shape)}")
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
                    if not isinstance(extent, tvm_tirx.ir.IntImm):
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
