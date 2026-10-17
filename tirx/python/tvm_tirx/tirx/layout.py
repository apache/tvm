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

from collections.abc import Sequence
from typing import Union

import tvm_ffi
from tvm_ffi.dataclasses import Enum

import tvm_tirx
from tvm_tirx.tirx.expr import Expr

from . import _ffi_api
from .thread_scope import ThreadScope


@tvm_ffi.register_object("tirx.Layout")
class Layout(tvm_ffi.Object):
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
        if shape is not None:
            return _ffi_api.LayoutApplyWithShape(self, coord, shape)  # pylint: disable=no-member
        if len(coord) == 1:
            return _ffi_api.LayoutApplyLinear(self, coord[0])  # pylint: disable=no-member
        return _ffi_api.LayoutApply(self, coord)  # pylint: disable=no-member

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
        # strides match what te-create_function / C++ ``GetDefaultStrides``
        # produce for int64-shaped tensors (otherwise the last stride stays a
        # Python ``int`` -> int32 IntImm and breaks structural-equal).
        for t in data:
            if tvm_tirx.ir.is_prim_expr(t) and t.ty.dtype != "int32":
                from tvm_tirx.ir import IntImm  # pylint: disable=import-outside-toplevel

                stride = IntImm(t.ty, stride)
                break
        res = list()
        for t in reversed(data):
            assert isinstance(t, int) or tvm_tirx.ir.is_prim_expr(t), (
                f"data must be int or Expr, but got {t}"
            )
            res.append(stride)
            stride *= t
        return list(reversed(res))

    def is_trivial(self) -> bool:
        """Check if the layout is trivial."""
        return False

    def unpack(self, num: int, axis: str = "m") -> "Layout":
        """Unpack the layout, where a single element in the layout is unpacked into num contiguous elements.

        Parameters
        ----------
        num : int
            The number of elements to unpack into
        axis : str
            Storage coordinate whose element size changes. Ownership coordinates
            are preserved. Tensor-memory dtype views use ``TCol``.

        Returns
        -------
        Layout
            The unpacked layout
        """  # noqa: E501
        if isinstance(self, TileLayout):

            def scale(iter):
                return Iter(
                    iter.extent,
                    iter.stride * num if iter.axis.name == axis else iter.stride,
                    iter.axis,
                )

            shard = [scale(iter) for iter in self.shard]
            shard.append(Iter(num, 1, Axis.get(axis)))
            offset = {a: v * num if a.name == axis else v for a, v in self.offset.items()}
            return TileLayout.from_iters(
                shard,
                [scale(i) for i in self.replica],
                offset,
                self.compact,
                self.memory_origin * num if axis == "m" else self.memory_origin,
            )
        elif isinstance(self, ComposeLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            return ComposeLayout(
                self.per_element + (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.tile_layout.unpack(num, axis),
                self.swizzle_inner,
                self.memory_origin * num if axis == "m" else self.memory_origin,
            )
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")

    def pack(self, num: int, axis: str = "m") -> "Layout":
        """Pack the layout, where num contiguous elements in the layout are packed into a single element.

        Parameters
        ----------
        num : int
            The number of elements to pack into
        axis : str
            Storage coordinate whose element size changes. Ownership coordinates
            are preserved. Tensor-memory dtype views use ``TCol``.

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
                and inner_iter.axis.name == axis
            ), f"Layout {self} can not be packed into {num} elements"
            analyzer = tvm_tirx.sym.Analyzer()

            def scale(iter):
                stride = iter.stride
                if iter.axis.name == axis:
                    if not analyzer.can_prove(stride % num == 0):
                        raise ValueError(
                            f"Layout storage stride {stride} is not divisible by {num}"
                        )
                    stride = stride // num
                return Iter(iter.extent, stride, iter.axis)

            offset = {}
            for a, value in self.offset.items():
                if a.name == axis:
                    if not analyzer.can_prove(value % num == 0):
                        raise ValueError(f"Layout storage offset {value} is not divisible by {num}")
                    value = value // num
                offset[a] = value
            shard = [scale(iter) for iter in self.shard[:-1]]
            shard.append(Iter(inner_iter.extent // num, 1, inner_iter.axis))
            if axis == "m" and analyzer.can_prove(self.memory_origin % num != 0):
                raise ValueError(f"Layout memory origin is not divisible by {num}")
            return TileLayout.from_iters(
                shard,
                [scale(i) for i in self.replica],
                offset,
                self.compact,
                self.memory_origin // num if axis == "m" else self.memory_origin,
            )
        elif isinstance(self, ComposeLayout):
            assert num & (num - 1) == 0, "num must be a power of 2"
            assert self.per_element >= num.bit_length() - 1, (
                "per_element must be greater than or equal to num.bit_length() - 1"
            )
            if axis == "m":
                from tvm_tirx.sym import Analyzer

                if Analyzer().can_prove(self.memory_origin % num != 0):
                    raise ValueError(f"Layout memory origin is not divisible by {num}")
            return ComposeLayout(
                self.per_element - (num.bit_length() - 1),
                self.swizzle_len,
                self.atom_len,
                self.tile_layout.pack(num, axis),
                self.swizzle_inner,
                self.memory_origin // num if axis == "m" else self.memory_origin,
            )
        else:
            raise ValueError(f"Unsupported layout type: {type(self)}")


# Result namespaces are available once this class has been created.


# Set of axis names registered on the C++ side. Used for lazy resolution of
# both module-level (`from tvm.tirx.layout import m`) and class-attribute
# (`Axis.m`) accesses. The actual FFI call to look up each axis is
# deferred until first access — keeps `import tvm.tirx.layout` runtime-safe
# (compiler-side FFI need not be present, matching apache's discipline).
_AXIS_NAMES = ("m",)


class _AxisMeta(type(Enum)):
    """Metaclass: lazy resolve `Axis.<name>` for registered axes."""

    def __getattr__(cls, name):
        if name in _AXIS_NAMES:
            return cls.get(name)
        raise AttributeError(f"type object 'Axis' has no attribute {name!r}")


class Axis(Enum, metaclass=_AxisMeta, type_key="tirx.Axis", init=False):
    """Layout axis wrapper."""

    @property
    def name(self) -> str:
        """Return the canonical axis name."""
        return self._str_index

    @classmethod
    def get(cls, name: str) -> "Axis":
        """Get or create the axis singleton named `name`.

        Unknown names are registered without thread or memory attributes.
        """
        return _ffi_api.AxisGet(name)  # pylint: disable=no-member

    def is_thread(self) -> bool:
        """Check if the axis is a thread axis."""
        return _ffi_api.AxisIsThreadAxis(self)  # pylint: disable=no-member

    def is_memory(self) -> bool:
        """Check if the axis is a memory axis."""
        return _ffi_api.AxisIsMemoryAxis(self)  # pylint: disable=no-member

    @property
    def backend(self) -> str | None:
        """Backend owning this axis, or None for a generic axis."""
        return _ffi_api.AxisGetBackend(self)

    def get_scope(self) -> ThreadScope | None:
        """Get the scope of the axis."""
        return _ffi_api.AxisGetScope(self)  # pylint: disable=no-member

    def get_subscope(self) -> ThreadScope | None:
        """Get the subscope of the axis."""
        return _ffi_api.AxisGetSubscope(self)  # pylint: disable=no-member

    # Enable syntax like `4 @ Axis.m` to attach an axis to a stride/term.
    # This mirrors libraries that overload the matrix multiply operator for DSLs.
    def __rmatmul__(self, other: Expr):  # type: ignore[override]
        # Represent a single value bound to an axis.
        return _OnAxis(other, self)


# ------------------------------------------------------------------
# 2)  Lazy module-level axis lookup
# ------------------------------------------------------------------
# PEP 562 module-level __getattr__ for `from tvm.tirx.layout import m`.
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


# ------------------------------------------------------------------
# Helper types to support `Expr @ Axis` and `sum` for offsets
# ------------------------------------------------------------------
class _OnAxis:
    """Represents a single value attached to an axis, created via `value @ Axis.X`.

    Used in two places:
    - As stride spec in `TileLayout(..., shard=(extents, [value @ Axis.X]))`
    - As terms to build an offset expression like `1 @ Axis.m + 512`
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
            # Merge if both exist; rely on tvm sym for symbolic add
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
class Iter(tvm_ffi.Object):
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

    def __init__(self, spec: "_LayoutSpec", *, memory_origin: Expr | None = None):
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
            False,
            memory_origin,
        )

    @staticmethod
    def from_iters(
        shard: "Sequence[Iter]" = (),
        replica: "Sequence[Iter]" = (),
        offset: dict[Axis | str, Expr] | None = None,
        compact: bool = False,
        memory_origin: Expr | None = None,
    ) -> "TileLayout":
        """Construct a TileLayout from pre-built Iter objects."""
        if offset:
            offset = {Axis.get(k) if isinstance(k, str) else k: v for k, v in offset.items()}
        return _ffi_api.TileLayout(shard, replica, offset or {}, compact, memory_origin)  # pylint: disable=no-member

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

    def get_scope(self) -> tuple[ThreadScope, ThreadScope] | None:
        """Get the scope pair of the layout."""
        return _ffi_api.TileLayoutGetScope(self)  # pylint: disable=no-member

    def permute_dims(self, perm: list[int]) -> "TileLayout":
        """Permute the dimensions of the layout."""
        assert len(perm) == len(self.shard), (
            "perm must have the same length as the number of dimensions in the layout"
        )
        new_shard = []
        for i in perm:
            new_shard.append(self.shard[i])
        return TileLayout.from_iters(
            new_shard, self.replica, self.offset, self.compact, self.memory_origin
        )

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


# Result namespaces are available once this class has been created.


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
    memory_origin: Expr

    def __init__(
        self,
        per_element: int,
        swizzle_len: int,
        atom_len: int,
        tile_layout: "TileLayout",
        swizzle_inner: bool = True,
        memory_origin: Expr | None = None,
    ):
        self.__init_handle_by_constructor__(
            _ffi_api.ComposeLayout,  # pylint: disable=no-member
            per_element,
            swizzle_len,
            atom_len,
            tile_layout,
            swizzle_inner,
            memory_origin,
        )


# Result namespaces are available once this class has been created.
