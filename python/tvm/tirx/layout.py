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
# pylint: disable=super-init-not-called
"""Definition of layout."""

import functools
import operator
import re
from collections.abc import Sequence
from typing import ClassVar, Optional, Union

import tvm_ffi

import tvm
from tvm.runtime import Object
from tvm.tirx.expr import PrimExpr

from . import _ffi_api
from .exec_scope import ExecScope


def _flatten_coord(coord: list[PrimExpr], shape: list[PrimExpr]) -> PrimExpr:
    """Python mirror of ``src/tirx/ir/layout/utils.cc::FlattenCoord``."""

    flat: PrimExpr = 0
    for c, s in zip(coord, shape, strict=False):
        flat = flat * s + c
    return flat


def _split_coord(coord: PrimExpr, extents: list[PrimExpr]) -> list[PrimExpr]:
    """Python mirror of ``src/tirx/ir/layout/utils.cc::SplitCoord``.

    Walks ``extents`` from the innermost (last index, ``%``-ed first) toward
    the outermost (index 0, gets the final remaining ``//``).
    """

    n = len(extents)
    if n == 0:
        return []
    result: list = [None] * n
    remaining = coord
    for i in range(n - 1, -1, -1):
        if i == 0:
            result[0] = remaining
        else:
            result[i] = tvm.tirx.floormod(remaining, extents[i])
            remaining = tvm.tirx.floordiv(remaining, extents[i])
    return result


@tvm_ffi.register_object("tirx.Layout")
class Layout(Object):
    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.Layout)  # pylint: disable=no-member

    def verify_well_formed(self) -> bool:
        """Verify if the layout is well-formed.

        Returns
        -------
        bool
            True if the layout is well-formed, False otherwise
        """
        return _ffi_api.LayoutVerifyWellFormed(self)  # pylint: disable=no-member

    def size(self, axis_name: str | None = None):
        """Get the size of the layout.

        Parameters
        ----------
        axis_name : Optional[str]
            The name of the axis to get the size of. If not provided, the default input size will be returned.
        """  # noqa: E501
        return _ffi_api.LayoutGetSize(self, axis_name)  # pylint: disable=no-member

    def span(self, axis_name: str | None = None):
        """Get the span of the layout.

        Parameters
        ----------
        axis_name : Optional[str]
            The name of the axis to get the span of. If not provided, the default span will be returned.
        """  # noqa: E501
        return _ffi_api.LayoutGetSpan(self, axis_name)  # pylint: disable=no-member

    # Note: no backward-compat alias; `cosize` is removed.

    def apply(
        self, *coord: list[PrimExpr], shape: list[PrimExpr] | None = None
    ) -> dict[str, PrimExpr]:
        """Apply the layout on the input coordinate and get the mapped output.

        Input cases:
        - coord is a single element -> will be treated as a 1D coordinate
        - coord is a list of elements -> will be treated as a multi-dimensional coordinate
        - shape is provided -> turn the coord with shape into a 1D coordinate
        - shape is not provided -> use the default shape

        Returns
        -------
        Dict[str, PrimExpr]
            The mapped output (axis name -> value on the axis)
        """
        if len(coord) == 1:
            # assert shape is None, "shape must be None if coord is not a list or tuple"
            return _ffi_api.LayoutApplyLinear(self, coord[0])  # pylint: disable=no-member
        if shape is None:
            return _ffi_api.LayoutApply(self, coord)  # pylint: disable=no-member
        return _ffi_api.LayoutApplyWithShape(self, coord, shape)  # pylint: disable=no-member

    def apply_to_shape(self, coord: list[PrimExpr], input_shape: list[PrimExpr]) -> list[PrimExpr]:
        """Compute the per-shard value that each shard would take if ``coord``
        were interpreted against ``input_shape``.

        Tries ``self.group(input_shape)`` first. On success, each group owns
        exactly one ``input_shape`` entry, so ``coord[d]`` can be split
        *within* that group's shard extents (bounds stay local to one input
        dim — simpler analyzer simplification, no cross-dim complications).

        Falls back to ``FlattenCoord(coord, input_shape)`` + ``SplitCoord``
        on ``self``'s raw shard shape when the group call fails (e.g. when
        ``input_shape`` does not align with the layout's factor boundaries).

        Returns a list of length ``len(self.shard)``; each entry is the value
        that shard would iterate.
        """

        try:
            grouped, seps = self.group(list(input_shape))
        except Exception:
            flat = _flatten_coord(coord, input_shape)
            return _split_coord(flat, [sh.extent for sh in self.shard])

        results: list = [None] * len(grouped.shard)
        for d in range(len(input_shape)):
            start = seps[d]
            end = seps[d + 1]
            extents = [grouped.shard[i].extent for i in range(start, end)]
            part = _split_coord(coord[d], extents)
            for i, c in zip(range(start, end), part, strict=False):
                results[i] = c
        return results

    def canonicalize(self) -> "Layout":
        """Canonicalize the layout by simplifying and fusing iterators where possible.

        Returns
        -------
        Layout
            The canonicalized layout
        """
        return _ffi_api.LayoutCanonicalize(self)  # pylint: disable=no-member

    def tile(
        self, outer: "TileLayout", outer_shape: list[PrimExpr], inner_shape: list[PrimExpr]
    ) -> Union["TileLayout", "ComposeLayout"]:
        """Tile the current layout with an outer layout.

        Parameters
        ----------
        outer : TileLayout
            The outer layout to tile with
        outer_shape : List[PrimExpr]
            The shape of the outer layout
        inner_shape : List[PrimExpr]
            The shape of the inner layout

        Returns
        -------
        Union[TileLayout, ComposeLayout]
            The resulting tiled layout
        """
        return _ffi_api.LayoutTile(  # pylint: disable=no-member
            self, outer, outer_shape, inner_shape
        )

    def direct_sum(
        self, left: "TileLayout", left_shape: list[PrimExpr], right_shape: list[PrimExpr]
    ) -> Union["TileLayout", "ComposeLayout"]:
        """Direct-sum on the tiling domain (unscaled composition): A + B.

        This layout is treated as the right addend B grouped by `right_shape`.
        The `left` layout is treated as A grouped by `left_shape`.
        The resulting layout is evaluated over the interleaved domain S_A ⊗ S_B,
        without span scaling (unlike tiling).
        """
        return _ffi_api.LayoutDirectSum(  # pylint: disable=no-member
            self, left, left_shape, right_shape
        )

    def is_tile_inner(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: list[PrimExpr],
        inner_shape: list[PrimExpr],
    ) -> Optional["TileLayout"]:
        """Check if a layout is the inner layout of a tiled layout.

        Parameters
        ----------
        tile_layout : Union[TileLayout, ComposeLayout]
            The tiled layout to check
        tiled_shape : List[PrimExpr]
            The shape of the tiled layout
        inner_shape : List[PrimExpr]
            The shape of the inner layout

        Returns
        -------
        Optional[TileLayout]
            The outer layout if it is the inner layout of the tiled layout, None otherwise
        """
        return _ffi_api.LayoutIsTileInner(  # pylint: disable=no-member
            self, tile_layout, tiled_shape, inner_shape
        )

    def is_tile_outer(
        self,
        tile_layout: Union["TileLayout", "ComposeLayout"],
        tiled_shape: list[PrimExpr],
        outer_shape: list[PrimExpr],
    ) -> Optional["Layout"]:
        """Check if a layout is the outer layout of a tiled layout.

        Parameters
        ----------
        tile_layout : Union[TileLayout, ComposeLayout]
            The tiled layout to check
        tiled_shape : List[PrimExpr]
            The shape of the tiled layout
        outer_shape : List[PrimExpr]
            The shape of the outer layout

        Returns
        -------
        Optional[Layout]
            The inner layout if it is the outer layout of the tiled layout, None otherwise
        """
        return _ffi_api.LayoutIsTileOuter(  # pylint: disable=no-member
            self, tile_layout, tiled_shape, outer_shape
        )

    def is_direct_sum_right(
        self,
        sum_layout: Union["TileLayout", "ComposeLayout"],
        interleaved_shape: list[PrimExpr],
        right_shape: list[PrimExpr],
    ) -> Optional["TileLayout"]:
        """Check if this layout is the right addend B in a direct-sum A + B.

        Returns the left addend A if recognized, otherwise None.
        """
        return _ffi_api.LayoutIsDirectSumRight(  # pylint: disable=no-member
            self, sum_layout, interleaved_shape, right_shape
        )

    def is_direct_sum_left(
        self,
        sum_layout: Union["TileLayout", "ComposeLayout"],
        interleaved_shape: list[PrimExpr],
        left_shape: list[PrimExpr],
    ) -> Optional["Layout"]:
        """Check if this layout is the left addend A in a direct-sum A + B.

        Returns the right addend B if recognized, otherwise None.
        """
        return _ffi_api.LayoutIsDirectSumLeft(  # pylint: disable=no-member
            self, sum_layout, interleaved_shape, left_shape
        )

    def slice(
        self, shape: list[PrimExpr], region: list[tuple[PrimExpr, PrimExpr]]
    ) -> Optional["Layout"]:
        """Slice the layout with a given shape and region.

        Parameters
        ----------
        shape : List[PrimExpr]
            The shape of the layout
        region : List[Tuple[PrimExpr, PrimExpr], tvm.ir.Range]
            The region to slice, each element is (begin, end)

        Returns
        -------
        Optional[Layout]
            The sliced layout, or None if slicing is not possible
        """
        assert len(shape) == len(region), "shape and region must have the same length"

        region_list = []
        for range_i in region:
            if isinstance(range_i, tvm.ir.Range):
                region_list.append(range_i)
            else:
                region_list.append(tvm.ir.Range(range_i[0], range_i[1]))
        return _ffi_api.LayoutSlice(self, shape, region_list)  # pylint: disable=no-member

    def tile_to(self, to_shape: list[PrimExpr], current_shape: list[PrimExpr]) -> "Layout":
        """Tile the current layout to the given shape.

        Parameters
        ----------
        to_shape : List[PrimExpr]
            The shape to tile to
        current_shape : List[PrimExpr]
            The current shape of the layout
        """

        tile_shape = [to_shape[i] // current_shape[i] for i in range(len(to_shape))]
        return self.tile(TileLayout(S[tuple(tile_shape)]), tile_shape, current_shape)

    @staticmethod
    def _get_default_strides(data: list[int | PrimExpr], stride: int = 1) -> tuple:
        assert isinstance(data, list | tuple), "data must be a tuple"
        # Promote ``stride`` to the dtype of the shape extents so the resulting
        # strides match what te-create_prim_func / C++ ``GetDefaultStrides``
        # produce for int64-shaped buffers (otherwise the last stride stays a
        # Python ``int`` -> int32 IntImm and breaks structural-equal).
        for t in data:
            if isinstance(t, PrimExpr) and t.dtype != "int32":
                from .expr import IntImm  # pylint: disable=import-outside-toplevel

                stride = IntImm(t.dtype, stride)
                break
        res = list()
        for t in reversed(data):
            assert isinstance(t, int | PrimExpr), f"data must be int or PrimExpr, but got {t}"
            res.append(stride)
            stride *= t
        return list(reversed(res))

    def is_swizzle(self) -> bool:
        """Check if the layout is swizzle."""
        return isinstance(self, SwizzleLayout)

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return False

    def is_trainium(self) -> bool:
        """Check if the layout is trainium layout."""
        if not isinstance(self, TileLayout):
            return False
        return _ffi_api.TileLayoutIsTrainium(self)  # pylint: disable=no-member

    def storage(self) -> "Layout":
        if isinstance(self, TileLayout):
            # Filter out shard with thread axis
            shard = [iter for iter in self.shard if not iter.axis.is_thread()]
            replicate = [iter for iter in self.replica if not iter.axis.is_thread()]
            exclude = {axis: offset for axis, offset in self.offset.items() if not axis.is_thread()}
            return TileLayout.from_iters(shard, replicate, exclude)  # pylint: disable=no-member

        elif isinstance(self, SwizzleLayout):
            return self
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(self.swizzle.storage(), self.tile_layout.storage())
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")

    def unpack(self, num: int) -> "Layout":
        """Unpack the layout, where a single element in the layout is unpacked into num contiguous elements.

        Parameters
        ----------
        num : int
            The number of elements to unpack into

        Returns
        -------
        Layout
            The unpacked layout
        """  # noqa: E501
        if isinstance(self, TileLayout):
            shard = [Iter(iter.extent, iter.stride * num, iter.axis) for iter in self.shard]
            shard.append(Iter(num, 1, Axis.get("m")))
            return TileLayout.from_iters(shard, self.replica, self.offset)
        elif isinstance(self, SwizzleLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            return SwizzleLayout(
                self.per_element + (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.swizzle_inner,
            )
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(self.swizzle.unpack(num), self.tile_layout.unpack(num))
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")

    def pack(self, num: int) -> "Layout":
        """Pack the layout, where num contiguous elements in the layout are packed into a single element.

        Parameters
        ----------
        num : int
            The number of elements to pack into

        Returns
        -------
        Layout
            The packed layout
        """  # noqa: E501
        if isinstance(self, TileLayout):
            inner_iter = self.shard[-1]
            assert (
                inner_iter.stride == 1
                and inner_iter.extent % num == 0
                and inner_iter.axis.is_memory()
            ), f"Layout {self} can not be packed into {num} elements"
            shard = [Iter(iter.extent, iter.stride // num, iter.axis) for iter in self.shard[:-1]]
            shard.append(Iter(inner_iter.extent // num, 1, inner_iter.axis))
            return TileLayout.from_iters(shard, self.replica, self.offset)
        elif isinstance(self, SwizzleLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            assert self.per_element >= num.bit_length() - 1, (
                "per_element must be greater than or equal to num.bit_length() - 1"
            )
            return SwizzleLayout(
                self.per_element - (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.swizzle_inner,
            )
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(self.swizzle.pack(num), self.tile_layout.pack(num))
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")


# Set of axis names registered on the C++ side. Used for lazy resolution of
# both module-level (`from tvm.tirx.layout import laneid`) and class-attribute
# (`Axis.laneid`) accesses. The actual FFI call to look up each axis is
# deferred until first access — keeps `import tvm.tirx.layout` runtime-safe
# (compiler-side FFI need not be present, matching apache's discipline).
_AXIS_NAMES = (
    "pid",
    "bx",
    "by",
    "bz",
    "cbx",
    "cby",
    "cbz",
    "tx",
    "warpid",
    "laneid",
    "wgid",
    "tid_in_wg",
    "wid_in_wg",
    "m",
    "P",
    "F",
    "Bank",
    "TCol",
    "TLane",
)


class _AxisMeta(type(Object)):
    """Metaclass: lazy resolve `Axis.<name>` for registered axes."""

    def __getattr__(cls, name):
        if name in _AXIS_NAMES:
            return cls.get(name)
        raise AttributeError(f"type object 'Axis' has no attribute {name!r}")


@tvm_ffi.register_object("tirx.Axis")
class Axis(Object, metaclass=_AxisMeta):
    """Layout axis wrapper."""

    # ---- forbid direct construction ----
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Cannot create Axis directly; use Axis.get()")

    @staticmethod
    def _register_axis(name: str) -> "Axis":
        return _ffi_api.AxisGet(name)  # pylint: disable=no-member

    # Singleton cache, populated lazily as names are accessed.
    reg_dict: ClassVar[dict[str, "Axis"]] = {}

    @staticmethod
    def get(name: str) -> "Axis":
        """Get or create an axis by name. Unknown names are auto-registered."""
        if name not in Axis.reg_dict:
            Axis.reg_dict[name] = Axis._register_axis(name)
        return Axis.reg_dict[name]

    def is_thread(self) -> bool:
        """Check if the axis is a thread axis."""
        return _ffi_api.AxisIsThreadAxis(self)  # pylint: disable=no-member

    def is_memory(self) -> bool:
        """Check if the axis is a memory axis."""
        return _ffi_api.AxisIsMemoryAxis(self)  # pylint: disable=no-member

    def get_scope(self) -> ExecScope | None:
        """Get the scope of the axis."""
        return _ffi_api.AxisGetScope(self)  # pylint: disable=no-member

    def get_subscope(self) -> ExecScope | None:
        """Get the subscope of the axis."""
        return _ffi_api.AxisGetSubscope(self)  # pylint: disable=no-member

    # Enable syntax like `4 @ Axis.laneid` to attach an axis to a stride/term.
    # This mirrors libraries that overload the matrix multiply operator for DSLs.
    def __rmatmul__(self, other: PrimExpr):  # type: ignore[override]
        # Represent a single value bound to an axis.
        return _OnAxis(other, self)


# ------------------------------------------------------------------
# 2)  Lazy module-level axis lookup
# ------------------------------------------------------------------
# PEP 562 module-level __getattr__ for `from tvm.tirx.layout import laneid`.
# The FFI call to look up each axis is deferred until first access; bare
# `import tvm.tirx.layout` performs zero compiler-side FFI calls.
def __getattr__(name):
    if name in _AXIS_NAMES:
        return Axis.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    __all__  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    __all__ = []  # type: ignore[var-annotated]
__all__ += list(_AXIS_NAMES)
__all__ += ["R", "S"]


def wg_local_layout(cols, rows=128):
    """Return a warpgroup-local register layout.

    The logical ``(rows, cols)`` tile is distributed on ``tid_in_wg`` along rows,
    so each thread owns one row and contiguous ``cols`` local elements.
    """
    return TileLayout(S[(rows, cols) : (1 @ Axis.tid_in_wg, 1)])


# ------------------------------------------------------------------
# Helper types to support `PrimExpr @ Axis` and `sum` for offsets
# ------------------------------------------------------------------
class _OnAxis:
    """Represents a single value attached to an axis, created via `value @ Axis.X`.

    Used in two places:
    - As stride spec in `TileLayout(..., shard=(extents, [value @ Axis.X]))`
    - As terms to build an offset expression like `1 @ Axis.laneid + 512`
    """

    def __init__(self, value: PrimExpr, axis: Axis):
        self.value = value
        self.axis = axis

    # Arithmetic to build offset sums
    def __add__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        base = _OffsetExpr({self.axis: self.value})
        return base + other

    def __radd__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        return self.__add__(other)


class _OffsetExpr:
    """Sum of axis-bound terms forming an offset specification.

    Internally stored as a dict {Axis: PrimExpr}. When a plain PrimExpr is
    provided (without axis), it is treated as `Axis.m` by convention.
    """

    def __init__(self, terms: dict[Axis, PrimExpr] | None = None):
        self.terms: dict[Axis, PrimExpr] = dict(terms or {})

    def _add_term(self, axis: Axis, value: PrimExpr):
        if axis in self.terms:
            # Merge if both exist; rely on tvm arith for symbolic add
            self.terms[axis] = self.terms[axis] + value  # type: ignore[operator]
        else:
            self.terms[axis] = value

    def __add__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        res = _OffsetExpr(dict(self.terms))
        if isinstance(other, _OffsetExpr):
            for ax, v in other.terms.items():
                res._add_term(ax, v)
        elif isinstance(other, _OnAxis):
            res._add_term(other.axis, other.value)
        else:  # PrimExpr-like -> default to Axis.m
            res._add_term(Axis.get("m"), other)  # type: ignore[arg-type]
        return res

    def __radd__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        return self.__add__(other)


_OffsetExprLike = _OffsetExpr | _OnAxis | PrimExpr | int


# ------------------------------------------------------------------
# Composable layout specs: S[shape:stride] + R[shape:stride] + offset
# ------------------------------------------------------------------
class _LayoutSpec:
    """Composable layout specification built via ``S[shape:stride] + R[shape:stride] + offset``.

    Instances are created by the module-level ``S`` and ``R`` builders and
    combined with ``+``.  Pass the result directly to :class:`TileLayout`.
    """

    __slots__ = ("offset", "replica", "shard")

    def __init__(self, shard=None, replica=None, offset=None):
        self.shard = shard  # (shape_tuple, stride_tuple) or (shape_tuple, None)
        self.replica = replica  # (shape_tuple, stride_tuple) or None
        self.offset = offset  # _OffsetExprLike or None

    def __add__(self, other):
        if isinstance(other, _LayoutSpec):
            return _LayoutSpec(
                shard=self.shard or other.shard,
                replica=other.replica if other.replica else self.replica,
                offset=_merge_offset(self.offset, other.offset),
            )
        if isinstance(other, _OnAxis | _OffsetExpr | int):
            return _LayoutSpec(
                shard=self.shard, replica=self.replica, offset=_merge_offset(self.offset, other)
            )
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, _OnAxis | _OffsetExpr | int):
            return _LayoutSpec(
                shard=self.shard, replica=self.replica, offset=_merge_offset(other, self.offset)
            )
        return NotImplemented


def _merge_offset(a: "_OffsetExprLike | None", b: "_OffsetExprLike | None"):
    """Combine two offsets that arrive at a `_LayoutSpec` via successive `+`.

    `_LayoutSpec.__add__` used to overwrite `self.offset` with the new term,
    which made `S[..] + 1 @ laneid + 2 @ warpid` silently drop the first
    axis. Always merge through `_OffsetExpr.__add__` so each axis term is
    accumulated correctly.
    """
    if a is None:
        return b
    if b is None:
        return a
    return _to_offset_expr(a) + _to_offset_expr(b)


class _SpecBuilder:
    """Builder for ``S[shape : stride]`` and ``R[shape : stride]`` syntax.

    - 1-D: ``S[8 : 4@laneid]``
    - N-D: ``S[(8, 4, 2) : (4@laneid, 1@laneid, 1)]``
    - Extents only: ``S[8, 4, 2]``
    """

    __slots__ = ("_kind",)

    def __init__(self, kind: str):
        self._kind = kind  # "shard" or "replica"

    @staticmethod
    def _to_tuple(x):
        if isinstance(x, tuple):
            return x
        if isinstance(x, list):
            return tuple(x)
        return (x,)

    def __getitem__(self, key):
        if isinstance(key, slice):
            pair = (self._to_tuple(key.start), self._to_tuple(key.stop))
        elif isinstance(key, tuple | list):
            pair = (tuple(key), None)  # extents only
        else:
            pair = ((key,), None)  # single extent

        if self._kind == "shard":
            return _LayoutSpec(shard=pair)
        return _LayoutSpec(replica=pair)


S = _SpecBuilder("shard")
R = _SpecBuilder("replica")


def _to_offset_expr(x: _OffsetExprLike) -> _OffsetExpr:
    if isinstance(x, _OffsetExpr):
        return x
    if isinstance(x, _OnAxis):
        return _OffsetExpr({x.axis: x.value})
    # Fallback: treat plain PrimExpr/int as Axis.m
    return _OffsetExpr({Axis.get("m"): x})  # type: ignore[arg-type]


@tvm_ffi.register_object("tirx.Iter")
class Iter(Object):
    """A memory layout that tiles data across devices."""

    extent: PrimExpr
    stride: PrimExpr
    axis: Axis

    def __init__(self, extent: PrimExpr, stride: PrimExpr, axis: Axis | str):
        if isinstance(axis, str):
            axis = Axis.get(axis)
        self.__init_handle_by_constructor__(
            _ffi_api.Iter,
            extent,
            stride,
            axis,  # pylint: disable=no-member
        )


def _spec_to_iters(pair) -> list:
    """Convert a ``(shape, stride)`` pair from :class:`_LayoutSpec` to ``List[Iter]``."""
    if pair is None:
        return []
    shape, strides = pair
    if strides is None:
        strides = Layout._get_default_strides(shape, 1)
    result = []
    for e, s in zip(shape, strides):
        if isinstance(s, _OnAxis):
            result.append(Iter(e, s.value, s.axis))
        elif isinstance(s, str):
            result.append(Iter(e, 1, s))
        elif isinstance(s, tuple):
            result.append(Iter(e, s[0], s[1]))
        else:
            result.append(Iter(e, s, "m"))
    return result


@tvm_ffi.register_object("tirx.TileLayout")
class TileLayout(Layout):
    """A memory layout that tiles data across devices."""

    shard: list[Iter]
    replicate: list[Iter]
    exclude: list[tuple[Axis, PrimExpr]]

    def __init__(self, spec: "_LayoutSpec"):
        shard_iters = _spec_to_iters(spec.shard)
        replica_iters = _spec_to_iters(spec.replica)
        offset_dict = {}
        if spec.offset is not None:
            off_expr = _to_offset_expr(spec.offset)
            offset_dict = dict(off_expr.terms)
        self.__init_handle_by_constructor__(
            _ffi_api.TileLayout,  # pylint: disable=no-member
            shard_iters,
            replica_iters,
            offset_dict,
        )

    @staticmethod
    def from_iters(
        shard: "Sequence[Iter]" = (),
        replica: "Sequence[Iter]" = (),
        offset: dict[Axis | str, PrimExpr] | None = None,
    ) -> "TileLayout":
        """Construct a TileLayout from pre-built Iter objects."""
        if offset:
            offset = {Axis.get(k) if isinstance(k, str) else k: v for k, v in offset.items()}
        return _ffi_api.TileLayout(shard, replica, offset or {})  # pylint: disable=no-member

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return _ffi_api.TileLayoutIsTrivial(self)  # pylint: disable=no-member

    def group(self, shape: list[PrimExpr]) -> tuple["Layout", list[int]]:
        """Group the current layout by the given shape.

        Parameters
        ----------
        shape : List[PrimExpr]
            The shape to group by

        Returns
        -------
        Tuple[Layout, List[int]]
            The grouped layout and the separators
        """
        return _ffi_api.TileLayoutGroup(self, shape)  # pylint: disable=no-member

    def get_scope(self) -> tuple[ExecScope, ExecScope] | None:
        """Get the scope pair of the layout."""
        return _ffi_api.TileLayoutGetScope(self)  # pylint: disable=no-member

    @classmethod
    def trainium(
        cls, annotation: str, shape: tuple[PrimExpr], is_psum: bool = False
    ) -> "TileLayout":
        """Create a TileLayout from an annotation string and a shape."""
        analyzer = tvm.arith.Analyzer()
        assert re.fullmatch(r"[PF]*", annotation), (
            f"annotation {annotation} must be a string of 'P' and 'F'"
        )
        assert len(annotation) == len(shape), (
            f"annotation {annotation} and shape {shape} must have the same length"
        )
        num_p_dim = annotation.count("P")
        if num_p_dim == 1:
            p_idx = annotation.index("P")
            p_dim = shape[p_idx]
            assert analyzer.can_prove(p_dim <= 128 or p_dim % 128 == 0), (
                f"There is only 1 P in the annotation. Partition size {p_dim} must be less than or equal to 128 or a multiple of 128"  # noqa: E501
            )
            if analyzer.can_prove(p_dim > 128):
                # split out the P dimension and put the higher part on the free dimension with largest stride  # noqa: E501
                annotation = "F" + annotation
                shape = (p_dim // 128, *shape[:p_idx], 128, *shape[p_idx + 1 :])
        elif num_p_dim > 1:
            p_dim_prod = functools.reduce(
                operator.mul, [s for s, c in zip(shape, annotation) if c == "P"]
            )
            assert analyzer.can_prove(p_dim_prod <= 128), (
                f"There are {num_p_dim} Ps in the annotation. Partition size {p_dim_prod} must be less than or equal to 128"  # noqa: E501
            )

        f_shape = [s for i, (s, c) in enumerate(zip(shape, annotation)) if c == "F"]
        p_shape = [s for i, (s, c) in enumerate(zip(shape, annotation)) if c == "P"]
        f_strides = Layout._get_default_strides(f_shape, 1)
        p_strides = Layout._get_default_strides(p_shape, 1)
        f_tile_layout = TileLayout(S[tuple(f_shape) : tuple(s @ Axis.F for s in f_strides)])
        p_tile_layout = TileLayout(S[tuple(p_shape) : tuple(s @ Axis.P for s in p_strides)])
        result = []
        f_index = p_index = 0

        for char in annotation:
            if char == "F":
                result.append(f_tile_layout.shard[f_index])
                f_index += 1
            else:  # char == 'P'
                result.append(p_tile_layout.shard[p_index])
                p_index += 1
        if num_p_dim == 1 and analyzer.can_prove(p_dim > 128):
            # put higher part of P to where it belongs
            higher_P = result[0]
            result = result[1:]
            result = [*result[:p_idx], higher_P, *result[p_idx:]]

        res = TileLayout.from_iters(result, [], dict())  # pylint: disable=no-member
        if is_psum:
            res = res.to_psum()
        return res

    kPSUMMaxElemPerBank = 512
    kPSUMBankNum = 8

    def to_psum(self) -> "TileLayout":
        """Convert the layout to a psum layout."""
        analyzer = tvm.arith.Analyzer()
        shard = []
        for i in self.shard:
            if i.axis.name == "F":
                if analyzer.can_prove(i.stride % self.kPSUMMaxElemPerBank == 0):
                    stride = analyzer.simplify(i.stride // self.kPSUMMaxElemPerBank)
                    shard.append(Iter(i.extent, stride, Axis.get("Bank")))
                elif analyzer.can_prove(self.kPSUMMaxElemPerBank % i.stride == 0):
                    c = analyzer.simplify(self.kPSUMMaxElemPerBank // i.stride)
                    if analyzer.can_prove(i.extent < c):
                        shard.append(i)
                    elif analyzer.can_prove(i.extent % c == 0):
                        shard.append(Iter(analyzer.simplify(i.extent // c), 1, Axis.get("Bank")))
                        shard.append(Iter(c, i.stride, Axis.get("F")))
                    else:
                        assert False, f"layout {self} can not be converted to psum layout"
                else:
                    assert False, f"layout {self} can not be converted to psum layout"
            else:
                shard.append(i)
        return TileLayout.from_iters(shard, [], dict())  # pylint: disable=no-member

    def permute_dims(self, perm: list[int]) -> "TileLayout":
        """Permute the dimensions of the layout."""
        assert len(perm) == len(self.shard), (
            "perm must have the same length as the number of dimensions in the layout"
        )
        new_shard = []
        for i in perm:
            new_shard.append(self.shard[i])
        return TileLayout.from_iters(new_shard, self.replica, self.offset)

    def permute_by_groups(self, seps: list[int], perm: list[int]) -> "TileLayout":
        """Permute groups of shard iters defined by ``seps``.

        ``seps`` follows the convention of :meth:`group`'s second return value:
        ``seps[0] == 0`` and group ``i`` covers shard indices
        ``[seps[i], seps[i + 1])``. The number of groups is ``len(seps) - 1``.

        Parameters
        ----------
        seps : list[int]
            Group boundary positions in the shard list.
        perm : list[int]
            Permutation of ``range(len(seps) - 1)`` selecting the new group order.
        """
        n_groups = len(seps) - 1
        assert sorted(perm) == list(range(n_groups)), f"invalid perm {perm}"
        flat = [k for g in perm for k in range(seps[g], seps[g + 1])]
        return self.permute_dims(flat)


@tvm_ffi.register_object("tirx.SwizzleLayout")
class SwizzleLayout(Layout):
    """A memory layout that swizzles elements to improve memory access patterns."""

    per_element: int
    swizzle_len: int
    atom_len: int
    swizzle_inner: bool

    def __init__(
        self, per_element: int, swizzle_len: int, atom_len: int, swizzle_inner: bool = True
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.SwizzleLayout,  # pylint: disable=no-member
            per_element,
            swizzle_len,
            atom_len,
            swizzle_inner,
        )


@tvm_ffi.register_object("tirx.ComposeLayout")
class ComposeLayout(Layout):
    """A memory layout that composes 2 layouts."""

    def __init__(self, layout_A: "SwizzleLayout", layout_B: "TileLayout"):
        self.__init_handle_by_constructor__(
            _ffi_api.ComposeLayout,  # pylint: disable=no-member
            layout_A,
            layout_B,
        )
