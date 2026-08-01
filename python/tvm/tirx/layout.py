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
from tvm.tirx.expr import Expr

from . import _ffi_api
from .exec_scope import ExecScope


def _flatten_coord(coord: list[Expr], shape: list[Expr]) -> Expr:
    """Python mirror of ``src/tirx/ir/layout/utils.cc::FlattenCoord``."""

    flat: Expr = 0
    for c, s in zip(coord, shape, strict=False):
        flat = flat * s + c
    return flat


def _split_coord(coord: Expr, extents: list[Expr]) -> list[Expr]:
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

    def apply(self, *coord: list[Expr], shape: list[Expr] | None = None) -> dict[str, Expr]:
        """Apply the layout on the input coordinate and get the mapped output.

        Input cases:
        - coord is a single element -> will be treated as a 1D coordinate
        - coord is a list of elements -> will be treated as a multi-dimensional coordinate
        - shape is provided -> turn the coord with shape into a 1D coordinate
        - shape is not provided -> use the default shape

        Returns
        -------
        Dict[str, Expr]
            The mapped output (axis name -> value on the axis)
        """
        if len(coord) == 1:
            # assert shape is None, "shape must be None if coord is not a list or tuple"
            return _ffi_api.LayoutApplyLinear(self, coord[0])  # pylint: disable=no-member
        if shape is None:
            return _ffi_api.LayoutApply(self, coord)  # pylint: disable=no-member
        return _ffi_api.LayoutApplyWithShape(self, coord, shape)  # pylint: disable=no-member

    def apply_to_shape(self, coord: list[Expr], input_shape: list[Expr]) -> list[Expr]:
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
        self, outer: "TileLayout", outer_shape: list[Expr], inner_shape: list[Expr]
    ) -> Union["TileLayout", "ComposeLayout"]:
        """Tile the current layout with an outer layout.

        Parameters
        ----------
        outer : TileLayout
            The outer layout to tile with
        outer_shape : List[Expr]
            The shape of the outer layout
        inner_shape : List[Expr]
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
        self, left: "TileLayout", left_shape: list[Expr], right_shape: list[Expr]
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
        tiled_shape: list[Expr],
        inner_shape: list[Expr],
    ) -> Optional["TileLayout"]:
        """Check if a layout is the inner layout of a tiled layout.

        Parameters
        ----------
        tile_layout : Union[TileLayout, ComposeLayout]
            The tiled layout to check
        tiled_shape : List[Expr]
            The shape of the tiled layout
        inner_shape : List[Expr]
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
        tiled_shape: list[Expr],
        outer_shape: list[Expr],
    ) -> Optional["Layout"]:
        """Check if a layout is the outer layout of a tiled layout.

        Parameters
        ----------
        tile_layout : Union[TileLayout, ComposeLayout]
            The tiled layout to check
        tiled_shape : List[Expr]
            The shape of the tiled layout
        outer_shape : List[Expr]
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
        interleaved_shape: list[Expr],
        right_shape: list[Expr],
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
        interleaved_shape: list[Expr],
        left_shape: list[Expr],
    ) -> Optional["Layout"]:
        """Check if this layout is the left addend A in a direct-sum A + B.

        Returns the right addend B if recognized, otherwise None.
        """
        return _ffi_api.LayoutIsDirectSumLeft(  # pylint: disable=no-member
            self, sum_layout, interleaved_shape, left_shape
        )

    def slice(self, shape: list[Expr], region: list[tuple[Expr, Expr]]) -> Optional["Layout"]:
        """Slice the layout with a given shape and region.

        Parameters
        ----------
        shape : List[Expr]
            The shape of the layout
        region : List[Tuple[Expr, Expr], tvm.ir.Range]
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

    def tile_to(self, to_shape: list[Expr], current_shape: list[Expr]) -> "Layout":
        """Tile the current layout to the given shape.

        Parameters
        ----------
        to_shape : List[Expr]
            The shape to tile to
        current_shape : List[Expr]
            The current shape of the layout
        """

        tile_shape = [to_shape[i] // current_shape[i] for i in range(len(to_shape))]
        return self.tile(TileLayout(S[tuple(tile_shape)]), tile_shape, current_shape)

    @staticmethod
    def _get_default_strides(data: list[int | Expr], stride: int = 1) -> tuple:
        assert isinstance(data, list | tuple), "data must be a tuple"
        # Promote ``stride`` to the dtype of the shape extents so the resulting
        # strides match what te-create_prim_func / C++ ``GetDefaultStrides``
        # produce for int64-shaped buffers (otherwise the last stride stays a
        # Python ``int`` -> int32 IntImm and breaks structural-equal).
        for t in data:
            if tvm.ir.is_prim_expr(t) and t.ty.dtype != "int32":
                from .expr import IntImm  # pylint: disable=import-outside-toplevel

                stride = IntImm(t.ty, stride)
                break
        res = list()
        for t in reversed(data):
            assert isinstance(t, int) or tvm.ir.is_prim_expr(t), (
                f"data must be int or Expr, but got {t}"
            )
            res.append(stride)
            stride *= t
        return list(reversed(res))

    def is_swizzle(self) -> bool:
        """Check if the layout is a bare swizzle (ComposeLayout over a trivial tile)."""
        return isinstance(self, ComposeLayout) and self.tile_layout.is_trivial()

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

        elif isinstance(self, ComposeLayout):
            return ComposeLayout(
                self.per_element,
                self.swizzle_len,
                self.atom_len,
                self.tile_layout.storage(),
                self.swizzle_inner,
            )
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
        elif isinstance(self, ComposeLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            return ComposeLayout(
                self.per_element + (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.tile_layout.unpack(num),
                self.swizzle_inner,
            )
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")

    def broadcast(self, num: int, position: int = -1, axis: '"Axis" | str' = "m") -> "Layout":
        """Insert a stride-0 broadcast dim of extent ``num`` at ``position``.

        ``position`` follows Python list-insert semantics (negative indices
        count from the end; ``-1`` appends after the last shard dim).  The
        new dim has stride 0 — accessing along it doesn't move the byte
        offset, so the same physical element is "seen" ``num`` times.

        Useful for layouts where a consumer reads the same SMEM datum
        multiple times (e.g. ``sf_reuse`` over MMA-K steps).
        """
        if isinstance(self, TileLayout):
            if isinstance(axis, str):
                axis = Axis.get(axis)
            new_iter = Iter(num, 0, axis)
            shard = list(self.shard)
            insert_at = position if position >= 0 else len(shard) + 1 + position
            shard.insert(insert_at, new_iter)
            return TileLayout.from_iters(shard, self.replica, self.offset)
        elif isinstance(self, ComposeLayout):
            return ComposeLayout(
                self.per_element,
                self.swizzle_len,
                self.atom_len,
                self.tile_layout.broadcast(num, position, axis),
                self.swizzle_inner,
            )
        else:
            raise ValueError(f"broadcast not supported for {type(self)}")

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
        elif isinstance(self, ComposeLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            assert self.per_element >= num.bit_length() - 1, (
                "per_element must be greater than or equal to num.bit_length() - 1"
            )
            return ComposeLayout(
                self.per_element - (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.tile_layout.pack(num),
                self.swizzle_inner,
            )
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")


# Set of axis names registered on the C++ side. Used for lazy resolution of
# both module-level (`from tvm.tirx.layout import laneid`) and class-attribute
# (`Axis.laneid`) accesses. The actual FFI call to look up each axis is
# deferred until first access — keeps `import tvm.tirx.layout` runtime-safe
# (compiler-side FFI need not be present, matching apache's discipline).
_AXIS_NAMES = (
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
    def __rmatmul__(self, other: Expr):  # type: ignore[override]
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
__all__ += [
    "tcgen05_atom_layout",
    "tmem_datapath_layout",
    "tmem_mma_operand_layout",
    "wg_local_layout",
]


# ============================================================================
# TMEM datapath layouts (PTX ISA §9.7.16.10.5)
# ============================================================================
#
# ``tcgen05.mma`` writes its output matrix C into TMEM using one of several
# **datapath layouts** depending on the MMA's M dimension and ``.ws`` mode.
# Each layout determines *which* physical TMEM lanes (rows) the matrix
# occupies; the leak in the original ``_default_tmem_layout`` was that it
# always used the identity ``(rows, cols) : (1@TLane, 1@TCol)`` mapping,
# which is correct only for Layout D (M=128 full datapath). For Layout F
# (M=64 non-``.ws``) the MMA writes scattered lanes
# ``{0..15, 32..47, 64..79, 96..111}`` — half of each warp's 32-lane
# partition — and the readback path (``.16x*b`` M=64 atom) has the matching
# scatter built into the PTX. To keep the buffer's logical row indexing in
# sync with the physical scatter, the buffer's TileLayout must encode the
# scatter directly.
#
# We surface this via the factory below. Callers pass the datapath letter
# (``"D"`` / ``"F"``) and the logical ``(rows, cols)``; the factory returns
# the appropriate TileLayout. ``tmem_pool.alloc(..., layout=...)`` plumbs
# this into the buffer's layout so the dispatch can structurally verify
# atom ↔ datapath compatibility instead of silently accepting mismatches.
#
# Supported today:
#   - ``"D"``: M=128, ``.cta_group::1``, full datapath. Identity row→lane.
#   - ``"F"``: M=64, non-``.ws``, half datapath (4x1 lane utilization).
#     Logical row r → physical lane
#     (r // 16) * 32 + sub_slab * 16 + (r % 16).
#   - ``"B"``: per-CTA M=64, ``.cta_group::2``, Dense A ("2x2" datapath).
#     PTX names the CTA-pair shape M=128; each CTA owns a logical ``(64, N)``
#     accumulator. Its N columns split into two N/2 halves across physical
#     lanes 0..63 and 64..127:
#       (r, c) → (TLane=r + 64 * (c // (N/2)), TCol=c % (N/2)).
#
# Layouts A / C / E / G are reserved for future expansion.


_TMEM_DATAPATH_ROWS = {"A": 128, "B": 64, "C": 64, "D": 128, "E": 64, "F": 64, "G": 32}


def tmem_datapath_layout(datapath: str, rows: int, cols: int, sub_slab: int = 0) -> "TileLayout":
    """Return the ``TileLayout`` for a tcgen05 MMA datapath.

    See PTX ISA §9.7.16.10.5 for the datapath enumeration. The returned
    layout is shape-compatible with a buffer of ``(rows, cols)`` and
    encodes the logical-row → physical-TMEM-lane mapping that the
    corresponding MMA writes to (and that the matching ``.16x*b`` /
    ``.32x32b`` atom expects to read).

    Parameters
    ----------
    datapath : str
        PTX data path layout letter (all cta_group::2 layouts are the
        per-CTA view of the M-rows this CTA of the pair owns):

        - ``"A"``: M=256, ``.cta_group::2`` — identity over 128 rows.
        - ``"B"``: M=128, ``.cta_group::2``, dense A — 64 rows, column
          halves folded across the two 64-lane halves.
        - ``"C"``: M=128, ``.cta_group::2``, sparse A — 64 rows scattered
          16-per-warp, half datapath (same organization as F).
        - ``"D"``: M=128, ``.cta_group::1`` — identity, full datapath.
        - ``"E"``: M=64, ``.ws`` — 64 rows, column halves folded (same
          organization as B).
        - ``"F"``: M=64, non-``.ws`` — 64 rows scattered 16-per-warp,
          half datapath.
        - ``"G"``: M=32, ``.ws`` — 32 rows, column quarters folded across
          the four 32-lane warp slabs.
    rows : int
        Logical row count of the TMEM buffer. Must match the datapath's M
        dimension: 128 for A/D, 64 for B/C/E/F, 32 for G.
    cols : int
        Logical column count. Datapath B requires an even count because its
        columns split into two equal lane halves.
    sub_slab : int
        For Layout F, select the lower (``0``) or upper (``1``) 16-lane
        half of each warp's 32-lane TMEM partition. The upper half is useful
        as a 64-row read/write view of the high half-slab of a Layout D
        accumulator. Layouts D and B already span both halves and therefore
        only accept ``0``.

    Returns
    -------
    TileLayout
        Buffer-shape-compatible layout for ``(rows, cols)``.
    """
    if datapath not in _TMEM_DATAPATH_ROWS:
        raise ValueError(
            f"tmem_datapath_layout: unknown datapath {datapath!r}; "
            f"supported: {sorted(_TMEM_DATAPATH_ROWS)}"
        )
    expected = _TMEM_DATAPATH_ROWS[datapath]
    if rows != expected:
        raise ValueError(
            f"tmem_datapath_layout: datapath={datapath!r} expects rows={expected}, got {rows}"
        )
    if sub_slab not in (0, 1):
        raise ValueError(f"tmem_datapath_layout: sub_slab must be 0 or 1, got {sub_slab}")
    tlane = Axis.get("TLane")
    tcol = Axis.get("TCol")
    if datapath in ("A", "D"):
        # Identity row→lane (r → lane r). D is the full cta_group::1 datapath;
        # A is the per-CTA view of M=256/cta_group::2 (this CTA's 128 M-rows).
        # Both already span both 16-lane sub-slabs of every warp partition.
        if sub_slab != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath={datapath!r} (M=128) already spans both "
                "sub-slabs; sub_slab must be 0"
            )
        return TileLayout(S[(rows, cols) : (1 @ tlane, 1 @ tcol)])
    if datapath == "G":
        # Layout G (M=32, .ws): quarter datapath — column space splits into 4
        # quarters, lane = row + 32*q, physical col = col % (cols/4).
        if sub_slab != 0:
            raise ValueError(
                "tmem_datapath_layout: datapath='G' folds column quarters across "
                "full 32-lane slabs; sub_slab must be 0"
            )
        if cols % 4 != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath='G' requires cols % 4 == 0, got {cols}"
            )
        return TileLayout(S[(rows, 4, cols // 4) : (1 @ tlane, 32 @ tlane, 1 @ tcol)])
    if datapath in ("B", "E"):
        # Layouts B (M=128 cta_group::2 dense) and E (M=64 .ws): half datapath,
        # lane = row + 64 * (col >= cols/2), physical col = col % (cols/2).
        if sub_slab != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath={datapath!r} folds column halves across "
                "full 64-lane halves; sub_slab must be 0"
            )
        if cols % 2 != 0:
            raise ValueError(
                f"tmem_datapath_layout: datapath={datapath!r} expects even cols, got {cols}"
            )
        return TileLayout(S[(rows, 2, cols // 2) : (1 @ tlane, 64 @ tlane, 1 @ tcol)])
    # Layouts C (M=128 cta_group::2 sparse) and F (M=64 non-.ws): rows-per-CTA
    # scattered, half datapath — logical row wid*16+intra → physical lane
    # wid*32 + sub_slab*16 + intra.
    # ``TileLayout`` decomposes a scalar row index via ``SplitCoord``
    # (src/tirx/ir/layout/utils.cc), which uses row-major ordering: with
    # shape ``(s0, s1)`` the FIRST iter receives ``coord // s1`` (the high
    # bits) and the SECOND receives ``coord % s1`` (the low bits). So we
    # pin the warp selector to iter 0 (extent 4, TLane stride 32) and the
    # within-slab lane to iter 1 (extent 16, TLane stride 1), then shift the
    # TMEM lane offset by 16 for the upper sub-slab.
    spec = S[(4, 16, cols) : (32 @ tlane, 1 @ tlane, 1 @ tcol)]
    if sub_slab:
        spec = spec + (16 @ tlane)
    return TileLayout(spec)


def _mma_datapath_letter(M, cta_group, ws=False, sparse=False):
    """Map (instruction ``M``, ``cta_group``, ``ws``, ``sparse``) to the PTX
    tcgen05 datapath letter. Raises for families rejected in the v1 semantic
    allocator API (sparse Layout C, M=32 Layout G). See
    ``localdoc/claude_plan.txt`` S3a."""
    if sparse:
        raise ValueError(
            f"alloc_tcgen05_mma_AB: sparse A (Layout C) not supported in v1 "
            f"(M={M}, cta_group={cta_group})"
        )
    if ws and not (cta_group == 1 and M == 64):
        raise ValueError(
            f"alloc_tcgen05_mma_AB: ws=True only valid for cta_group=1, M=64 (Layout E); "
            f"got M={M}, cta_group={cta_group}"
        )
    if cta_group == 1:
        if M == 128:
            return "D"
        if M == 64:
            return "E" if ws else "F"
    elif cta_group == 2:
        if M == 256:
            return "A"
        if M == 128:
            return "B"
    raise ValueError(
        f"alloc_tcgen05_mma_AB: no supported datapath for M={M}, cta_group={cta_group}, "
        f"ws={ws} (supported: cta1 M in {{64,128}}, cta2 M in {{128,256}})"
    )


def tmem_mma_operand_layout(
    operand, shape, dtype, *, M, cta_group, ws=False, sparse=False, group=None
):
    """Resolve the TMEM ``TileLayout`` for a tcgen05 MMA operand from operand
    role + instruction params.

    ``operand``: ``"A"`` (A-in-TMEM) or ``"D"`` (accumulator / C).
    ``M`` is the PTX ``tcgen05.mma`` **instruction** M (256/128/64), NOT the
    per-CTA buffer rows; ``per_cta_rows = M // cta_group``. Physical 32-bit
    column count is left to the pool's ``_resolve_cols`` (layout-aware).

    Reproduces the hand-written layouts the three FlashMLA kernels use; the
    supported/reject domain is spec'd in ``localdoc/claude_plan.txt`` S3a.
    """
    tlane = Axis.get("TLane")
    tcol = Axis.get("TCol")
    datapath = _mma_datapath_letter(M, cta_group, ws, sparse)
    per_cta_rows = M // cta_group
    ext = [int(s) for s in shape]

    def _chk_rows(rows):
        if rows != per_cta_rows:
            raise ValueError(
                f"alloc_mma_{operand}: M={M}, cta_group={cta_group} expects "
                f"per-CTA rows {per_cta_rows}, got {rows} (shape {ext})"
            )

    if operand == "D":
        if group is not None:
            # flat buffer (m, N) + explicit grouping -> (m, s, 2, n) layout
            # (codex plan option b): keeps kernel O buffer 2D, layout 4-axis.
            if len(ext) != 2:
                raise ValueError(f"alloc_tcgen05_mma_D group= requires 2D (m,N), got {ext}")
            m, N = ext
            gs = [int(g) for g in group]
            if len(gs) != 3 or gs[1] != 2:
                raise ValueError(f"alloc_tcgen05_mma_D group must be (s,2,n), got {group}")
            s2, _, n2 = gs
            if s2 * 2 * n2 != N:
                raise ValueError(f"alloc_tcgen05_mma_D group {group} product != N={N}")
            if datapath not in ("B", "E"):
                raise ValueError(f"alloc_tcgen05_mma_D group= only for Layout B/E, got {datapath}")
            _chk_rows(m)
            return TileLayout(
                S[(m, s2, 2, n2) : (1 @ tlane, n2 @ tcol, 64 @ tlane, 1 @ tcol)]
            ).canonicalize()
        # Grouped grammar (C3): (m,N) | (m,2,N/2) | (m,s,2,n); plus batched
        # C[2,m,N/2] normalizing to packed. bank axis is fixed by position.
        if len(ext) == 2:
            m, N = ext
            _chk_rows(m)
            layout = tmem_datapath_layout(datapath, m, N)
        elif len(ext) == 3 and ext[0] == 2 and datapath in ("B", "E"):
            _, m, Nh = ext  # batched C[2, m, N/2]
            _chk_rows(m)
            layout = TileLayout(S[(2, m, Nh) : (64 @ tlane, 1 @ tlane, 1 @ tcol)])
        elif len(ext) == 3 and ext[1] == 2 and datapath in ("B", "E"):
            m, _, Nh = ext  # explicit (m, 2, N/2)
            _chk_rows(m)
            layout = TileLayout(S[(m, 2, Nh) : (1 @ tlane, 64 @ tlane, 1 @ tcol)])
        elif len(ext) == 4 and ext[2] == 2 and datapath in ("B", "E"):
            m, s, _, n = ext  # column-segmented (m, s, 2, n): flatten N=s*2*n
            _chk_rows(m)
            layout = TileLayout(S[(m, s, 2, n) : (1 @ tlane, n @ tcol, 64 @ tlane, 1 @ tcol)])
        else:
            raise ValueError(
                f"alloc_tcgen05_mma_D: unsupported D shape {ext} for datapath {datapath}"
            )
    elif operand == "A":
        if datapath in ("A", "D"):
            if len(ext) != 2:
                raise ValueError(
                    f"alloc_tcgen05_mma_A: datapath {datapath} identity needs 2D (m,K), got {ext}"
                )
            m, K = ext
            _chk_rows(m)
            layout = tmem_datapath_layout(datapath, m, K)  # identity
        elif datapath == "B":
            if len(ext) == 2:  # replica A
                m, K = ext
                _chk_rows(m)
                layout = TileLayout(S[(m, K) : (1 @ tlane, 1 @ tcol)] + R[2 : 64 @ tlane])
            elif len(ext) == 3 and ext[0] == 2:  # bank-batched A
                _, m, K = ext
                _chk_rows(m)
                layout = TileLayout(S[(2, m, K) : (64 @ tlane, 1 @ tlane, 1 @ tcol)])
            else:
                raise ValueError(
                    f"alloc_tcgen05_mma_A: datapath B needs (64,K) replica or (2,64,K) "
                    f"bank-batched, got {ext}"
                )
        elif datapath == "E":  # M=64 .ws: bank-batched only, flat rejected
            if len(ext) == 3 and ext[0] == 2:
                _, m, K = ext
                _chk_rows(m)
                layout = TileLayout(S[(2, m, K) : (64 @ tlane, 1 @ tlane, 1 @ tcol)])
            else:
                raise ValueError(
                    f"alloc_tcgen05_mma_A: datapath E (M=64 .ws) requires bank-batched "
                    f"(2,64,K); flat {ext} rejected (hidden second-half read)"
                )
        else:  # F (M=64 non-.ws): A occupies lanes 0..63 identically
            if len(ext) != 2:
                raise ValueError(f"alloc_tcgen05_mma_A: datapath F needs 2D (m,K), got {ext}")
            m, K = ext
            _chk_rows(m)
            layout = TileLayout(S[(m, K) : (1 @ tlane, 1 @ tcol)])
    else:
        raise ValueError(f"tmem_mma_operand_layout: operand must be 'A' or 'D', got {operand!r}")
    return layout.canonicalize()


def wg_local_layout(cols, rows=128):
    """Return a warpgroup-local register layout.

    The logical ``(rows, cols)`` tile is distributed on ``tid_in_wg`` along rows,
    so each thread owns one row and contiguous ``cols`` local elements.
    """
    return TileLayout(S[(rows, cols) : (1 @ Axis.tid_in_wg, 1)])


# Allowed (.shape, .num) combinations for tcgen05.ld/st atoms.
# Source: PTX ISA Table 49 (tcgen05-num-shapes-ld).
_TCGEN05_ATOM_REPS = {
    "32x32b": (1, 2, 4, 8, 16, 32, 64, 128),
    "16x64b": (1, 2, 4, 8, 16, 32, 64, 128),
    "16x128b": (1, 2, 4, 8, 16, 32, 64),
    "16x256b": (1, 2, 4, 8, 16, 32),
}


# Per-warp fp32-column factor for each instr_shape. For .16x*b atoms the
# warpgroup fragment is 64 rows x (factor * rep) fp32 cols; for .32x32b the
# fragment is 128 rows x (factor * rep) fp32 cols with factor=1.
_TCGEN05_COL_FACTOR_FP32 = {"32x32b": 1, "16x64b": 2, "16x128b": 4, "16x256b": 8}

# Allowed fragment row counts per warpgroup for each instr_shape. ``.32x32b``
# normally covers M=128, but a 64-row shape denotes the Layout B readback image:
# the logical (64, N) tile is physically a (128, N/2) ``.32x32b`` register
# file. ``.16x*b`` natively covers M=64 (one 16-row slab per warp, using lanes
# 0..15 of each warp's 32-lane TMEM partition) and can be extended to M=128 by
# issuing the atom twice with row offsets 0 and 16. The M=128 variant doubles
# per-thread registers and treats the extra slab as the highest m-bit.
_TCGEN05_FRAG_ROWS = {
    "32x32b": (64, 128),
    "16x64b": (64, 128),
    "16x128b": (64, 128),
    "16x256b": (64, 128),
}


def tcgen05_atom_layout(instr_shape: str, tensor_shape: tuple[int, int], dtype) -> "TileLayout":
    """Register-side ``TileLayout`` for ``tcgen05.ld``/``tcgen05.st`` atoms.

    Describes the per-warpgroup register tile that ``Tx.copy_async`` produces
    when reading a TMEM fragment via ``tcgen05.{ld,st}.<instr_shape>.xN``.
    ``rep`` (the ``.xN`` qualifier) is inferred from ``tensor_shape``.

    Fragment row count is determined by ``instr_shape``: ``.32x32b`` normally
    covers an M=128 fragment (128 rows per warpgroup), and
    ``.16x{64,128,256}b`` covers an M=64 fragment (64 rows per warpgroup).
    ``("32x32b", (64, N))`` is the fp32 Layout B readback image for a
    ``.cta_group::2`` M=64 accumulator.

    TMEM is kept **dense** for 16-bit dtypes: two 16-bit elements per 32-bit
    TMEM cell (matching the existing ``.32x32b`` convention). The PTX op is
    issued with the plain ``.b32`` form (no ``.pack::16b`` qualifier), and
    the returned layout describes the per-thread register file with two
    packed 16-bit elements per 32-bit register.

    Parameters
    ----------
    instr_shape : str
        The PTX atom's ``.shape`` qualifier. One of ``"32x32b"``, ``"16x64b"``,
        ``"16x128b"``, ``"16x256b"``.
    tensor_shape : tuple[int, int]
        The logical fragment shape in **element units**. Must be
        ``(frag_rows, K)`` where ``frag_rows`` is ``128`` for ``.32x32b`` and
        ``64`` for the other shapes. The fp32 Layout B image is the exception:
        it uses ``("32x32b", (64, N))``. The column extent is divisible by the
        per-warp column factor for the chosen instr_shape and dtype::

            K must be a power-of-two multiple of (factor_fp32 * elem_per_32b)

        where ``factor_fp32`` is ``1`` / ``2`` / ``4`` / ``8`` for ``.32x32b`` /
        ``.16x64b`` / ``.16x128b`` / ``.16x256b``, and ``elem_per_32b`` is
        ``1`` for fp32 and ``2`` for fp16/bf16. The inferred rep must be in PTX
        Table 49's supported set for the chosen instr_shape.
    dtype : str | tvm.DataType
        Element dtype. ``"float32"``, ``"float16"``, or ``"bfloat16"``.

    Returns
    -------
    TileLayout
        A ``tensor_shape``-shaped tile layout. The factory builds it as a
        sequence of fine-grained iters describing the per-(lane, register)
        destination position.

    Examples
    --------
    ``tcgen05_atom_layout("16x64b", (64, 64), "float32")`` → ``.16x64b.x32`` (rep=32, fp32).

    ``tcgen05_atom_layout("16x128b", (64, 256), "float16")`` → ``.16x128b.x32`` (rep=32,
    fp16; two fp16 elements packed per 32-bit register and per 32-bit TMEM cell).
    """
    if instr_shape not in _TCGEN05_ATOM_REPS:
        raise ValueError(
            f"tcgen05_atom_layout instr_shape must be one of "
            f"{list(_TCGEN05_ATOM_REPS)}, got {instr_shape!r}"
        )
    bits = tvm.runtime.DataType(dtype).bits
    if bits not in (16, 32):
        raise ValueError(
            f"tcgen05_atom_layout dtype must be a 32-bit or 16-bit type, got {dtype} ({bits} bits)"
        )
    if len(tensor_shape) != 2:
        raise ValueError(
            f"tcgen05_atom_layout tensor_shape must be 2-D (rows, cols), got {tensor_shape!r}"
        )
    rows, cols = tensor_shape
    allowed_rows = _TCGEN05_FRAG_ROWS[instr_shape]
    if rows not in allowed_rows:
        raise ValueError(
            f"tcgen05_atom_layout {instr_shape!r} expects rows ∈ {allowed_rows}, got {rows}"
        )

    if instr_shape == "32x32b" and rows == 64:
        # Layout B's logical (64, N) tile physically occupies all 128 lanes and
        # N/2 tcols. A .32x32b thread therefore owns its physical lane and N/2
        # fp32 registers. Re-label that (128, N/2) register file as (64, N).
        if bits != 32:
            raise ValueError(
                "tcgen05_atom_layout: 32x32b with 64 rows is the datapath B "
                f"readback image and is fp32-only, got {dtype} ({bits} bits)"
            )
        if cols % 2 != 0:
            raise ValueError(
                f"tcgen05_atom_layout: 32x32b (64, N) datapath B image expects even N, got {cols}"
            )
        n_half = cols // 2
        if n_half not in _TCGEN05_ATOM_REPS["32x32b"]:
            raise ValueError(
                f"tcgen05_atom_layout: 32x32b (64, N) datapath B image needs N/2={n_half} "
                f"(from N={cols}) in the PTX Table 49 set {_TCGEN05_ATOM_REPS['32x32b']}"
            )
        return TileLayout.from_iters(
            [
                Iter(64, 1, Axis.tid_in_wg),
                Iter(2, 64, Axis.tid_in_wg),
                Iter(n_half, 1, "m"),
            ],
            [],
            {},
        )

    elem_per_32b = 32 // bits
    col_factor_elem = _TCGEN05_COL_FACTOR_FP32[instr_shape] * elem_per_32b
    if cols % col_factor_elem != 0:
        raise ValueError(
            f"tcgen05_atom_layout cols={cols} not divisible by the per-rep column "
            f"factor {col_factor_elem} for instr_shape={instr_shape!r} dtype={dtype}; "
            f"valid cols are k * {col_factor_elem} for k in "
            f"{_TCGEN05_ATOM_REPS[instr_shape]}"
        )
    rep = cols // col_factor_elem
    if rep not in _TCGEN05_ATOM_REPS[instr_shape]:
        raise ValueError(
            f"tcgen05_atom_layout inferred rep={rep} (from cols={cols}) is not in "
            f"the PTX Table 49 supported set for {instr_shape}: "
            f"{_TCGEN05_ATOM_REPS[instr_shape]}"
        )

    laneid = Axis.laneid
    wid = Axis.wid_in_wg
    N = rep
    shape = instr_shape
    # All m-strides below are written in fp32-reg units; we multiply by
    # elem_per_32b at the end and prepend a C_pack iter for the 16-bit case
    # (each fp32 reg packs ``elem_per_32b`` elements at adjacent col positions).

    if shape == "32x32b":
        # M=128 fragment, simple thread-rows layout:
        #   (rows=128, cols=K) : (1@tid_in_wg, 1)
        # Each of 128 warpgroup threads owns one row; cols are contiguous in
        # the per-thread storage. For 16-bit dtypes the K cols are packed two
        # per 32-bit register (handled by the per-thread storage element count
        # naturally — m-stride 1 in element units).
        iters = [
            Iter(rows, 1, Axis.tid_in_wg),
            Iter(cols, 1, "m"),
        ]
        return TileLayout.from_iters(iters, [], {})

    # Iter lists are written high-to-low: ``TileLayout`` decomposes a flat
    # coordinate via ``SplitCoord`` (src/tirx/ir/layout/utils.cc) using
    # row-major ordering, where the FIRST iter receives the *high* bits and
    # the LAST iter receives the *low* bits. So R_w (highest-stride row
    # contribution) comes first in row_iters_fp32 and R_t1/t2 (lowest)
    # comes last; same for col.
    if shape == "16x64b":
        # Per-warp tile (fp32 view): (16 rows, 2N cols). Per-lane regs = N.
        # Lane (t0, t1, t2): t0 = laneid & 1, t1 = (laneid >> 1) & 1, t2 = laneid >> 2.
        #   Row = t2 + 8*t0 + 16*wid_in_wg
        #   Col (fp32) = t1 + 2*r,   r ∈ [0, N)
        row_iters_fp32 = [
            (4, 1, wid),  # R_w:  wid_in_wg       → R bits 4..5
            (2, 1, laneid),  # R_t0: laneid bit 0    → R bit 3
            (8, 4, laneid),  # R_t2: laneid bits 2..4 → R bits 0..2
        ]
        col_iters_fp32 = [
            (N, 1, "m"),  # C_r:  register slot   → C bits 1..
            (2, 2, laneid),  # C_t1: laneid bit 1    → C bit 0
        ]
        m_used_M64 = N
    elif shape == "16x128b":
        # Per-warp tile (fp32 view): (16 rows, 4N cols). Per-lane regs = 2N.
        # Lane (t0, t1): t0 = laneid & 3, t1 = laneid >> 2.
        # Reg r = ra + 2*rb, ra ∈ {0,1}, rb ∈ [0, N).
        #   Row = t1 + 8*ra + 16*wid_in_wg
        #   Col (fp32) = t0 + 4*rb
        row_iters_fp32 = [
            (4, 1, wid),  # R_w
            (2, 1, "m"),  # R_ra: reg bit 0        → R bit 3
            (8, 4, laneid),  # R_t1: laneid bits 2..4 → R bits 0..2
        ]
        col_iters_fp32 = [
            (N, 2, "m"),  # C_rb: reg bits 1..     → C bits 2..
            (4, 1, laneid),  # C_t0: laneid bits 0..1 → C bits 0..1
        ]
        m_used_M64 = 2 * N
    else:  # 16x256b
        # Per-warp tile (fp32 view): (16 rows, 8N cols). Per-lane regs = 4N.
        # Lane (t0, t1) as for 16x128b. Reg r = v0p + 2*va + 4*vb.
        #   Row = t1 + 8*va + 16*wid_in_wg
        #   Col (fp32) = v0p + 2*t0 + 8*vb
        row_iters_fp32 = [
            (4, 1, wid),  # R_w
            (2, 2, "m"),  # R_va: reg bit 1 → R bit 3
            (8, 4, laneid),  # R_t1
        ]
        col_iters_fp32 = [
            (N, 4, "m"),  # C_vb:  reg bits 2.. → C bits 3..
            (4, 1, laneid),  # C_t0
            (2, 1, "m"),  # C_v0p: reg bit 0  → C bit 0
        ]
        m_used_M64 = 4 * N

    if rows == 128:
        # M=128 covers both 16-row half-slabs of each warp's 32-lane TMEM
        # partition (the M=64 atom covers only lanes 0..15; the high half
        # 16..31 needs a second PTX issue with row offset 16). We surface
        # the combined fragment as a single (128, K) tile by inserting a
        # v_slab iter right *after* R_w (i.e. as the next-highest row bit).
        # v_slab claims one m-bit at the next free offset
        # (stride = m_used_M64) so reg indices [0, m_used_M64) hold the
        # low slab and [m_used_M64, 2*m_used_M64) hold the high slab — the
        # split the dispatch uses when emitting the two PTX calls. The
        # inserted iter also doubles wid_in_wg's row stride from 16 to 32,
        # so the four warps now tile rows 0..31 / 32..63 / 64..95 / 96..127.
        new_row_iters = []
        for ext, stride, axis in row_iters_fp32:
            new_row_iters.append((ext, stride, axis))
            if axis is wid:
                new_row_iters.append((2, m_used_M64, "m"))
        row_iters_fp32 = new_row_iters

    def _scale(iters):
        out = []
        for ext, stride, axis in iters:
            if axis == "m":
                out.append((ext, stride * elem_per_32b, axis))
            else:
                out.append((ext, stride, axis))
        return out

    row_iters = _scale(row_iters_fp32)
    col_iters = _scale(col_iters_fp32)

    # For the 16-bit packed variant each fp32 register holds two adjacent
    # column elements (low / high halves). Add a C_pack iter of extent
    # ``elem_per_32b`` and m-stride 1 at the *low* end of the col axis —
    # i.e. as the LAST col iter under SplitCoord's high-to-low ordering.
    if elem_per_32b > 1:
        col_iters.append((elem_per_32b, 1, "m"))

    iters = [Iter(ext, stride, axis) for ext, stride, axis in row_iters + col_iters]
    return TileLayout.from_iters(iters, [], {})


# ------------------------------------------------------------------
# Helper types to support `Expr @ Axis` and `sum` for offsets
# ------------------------------------------------------------------
class _OnAxis:
    """Represents a single value attached to an axis, created via `value @ Axis.X`.

    Used in two places:
    - As stride spec in `TileLayout(..., shard=(extents, [value @ Axis.X]))`
    - As terms to build an offset expression like `1 @ Axis.laneid + 512`
    """

    def __init__(self, value: Expr, axis: Axis):
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

    Internally stored as a dict {Axis: Expr}. When a plain Expr is
    provided (without axis), it is treated as `Axis.m` by convention.
    """

    def __init__(self, terms: dict[Axis, Expr] | None = None):
        self.terms: dict[Axis, Expr] = dict(terms or {})

    def _add_term(self, axis: Axis, value: Expr):
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
        else:  # Expr-like -> default to Axis.m
            res._add_term(Axis.get("m"), other)  # type: ignore[arg-type]
        return res

    def __radd__(self, other: "_OffsetExprLike") -> "_OffsetExpr":
        return self.__add__(other)


_OffsetExprLike = _OffsetExpr | _OnAxis | Expr | int


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
        if isinstance(other, _OnAxis | _OffsetExpr | Expr | int):
            return _LayoutSpec(
                shard=self.shard, replica=self.replica, offset=_merge_offset(self.offset, other)
            )
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, _OnAxis | _OffsetExpr | Expr | int):
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
    # Fallback: treat plain Expr/int as Axis.m
    return _OffsetExpr({Axis.get("m"): x})  # type: ignore[arg-type]


@tvm_ffi.register_object("tirx.Iter")
class Iter(Object):
    """A memory layout that tiles data across devices."""

    extent: Expr
    stride: Expr
    axis: Axis

    def __init__(self, extent: Expr, stride: Expr, axis: Axis | str):
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
    exclude: list[tuple[Axis, Expr]]

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
        offset: dict[Axis | str, Expr] | None = None,
    ) -> "TileLayout":
        """Construct a TileLayout from pre-built Iter objects."""
        if offset:
            offset = {Axis.get(k) if isinstance(k, str) else k: v for k, v in offset.items()}
        return _ffi_api.TileLayout(shard, replica, offset or {})  # pylint: disable=no-member

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return _ffi_api.TileLayoutIsTrivial(self)  # pylint: disable=no-member

    def group(self, shape: list[Expr]) -> tuple["Layout", list[int]]:
        """Group the current layout by the given shape.

        Parameters
        ----------
        shape : List[Expr]
            The shape to group by

        Returns
        -------
        Tuple[Layout, List[int]]
            The grouped layout and the separators
        """
        return _ffi_api.TileLayoutGroup(self, shape)  # pylint: disable=no-member

    def group_many(self, shapes: Sequence[Sequence[Expr]]) -> tuple["TileLayout", list[list[int]]]:
        """Group the layout by the minimal common refinement of several shapes.

        Repeated cumulative product boundaries are retained, so an extent-one
        dimension in any input shape becomes a real unit iterator in the
        refined layout.  This operation only splits existing shard iterators;
        it does not canonicalize or reorder the layout.

        Parameters
        ----------
        shapes : Sequence[Sequence[Expr]]
            Logical shapes with provably equal total products.

        Returns
        -------
        Tuple[TileLayout, List[List[int]]]
            The commonly refined layout and one separator list per input shape.
        """
        return _ffi_api.TileLayoutGroupMany(  # pylint: disable=no-member
            self, [list(shape) for shape in shapes]
        )

    def get_scope(self) -> tuple[ExecScope, ExecScope] | None:
        """Get the scope pair of the layout."""
        return _ffi_api.TileLayoutGetScope(self)  # pylint: disable=no-member

    @classmethod
    def trainium(cls, annotation: str, shape: tuple[Expr], is_psum: bool = False) -> "TileLayout":
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


@tvm_ffi.register_object("tirx.ComposeLayout")
class ComposeLayout(Layout):
    """A memory layout that swizzles a tile layout.

    ``per_element`` / ``swizzle_len`` / ``atom_len`` / ``swizzle_inner`` carry the
    swizzle (formerly the standalone ``SwizzleLayout``); ``tile_layout`` is the
    tiled memory map the swizzle is applied to. A bare swizzle is a
    ``ComposeLayout`` over a trivial identity tile.
    """

    per_element: int
    swizzle_len: int
    atom_len: int
    swizzle_inner: bool
    tile_layout: "TileLayout"

    def __init__(
        self,
        per_element: int,
        swizzle_len: int,
        atom_len: int,
        tile_layout: "TileLayout",
        swizzle_inner: bool = True,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.ComposeLayout,  # pylint: disable=no-member
            per_element,
            swizzle_len,
            atom_len,
            tile_layout,
            swizzle_inner,
        )
