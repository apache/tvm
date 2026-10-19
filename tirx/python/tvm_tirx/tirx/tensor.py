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
"""Tensor types, declarations, and type-owned expression methods."""

from collections.abc import Callable
from numbers import Integral
from typing import ClassVar

import tvm_ffi

import tvm_tirx
from tvm_tirx.ir import Call, PrimType, PtrType, Type, Var
from tvm_tirx.ir.location import UNKNOWN_LOC, Location
from tvm_tirx.runtime import MemoryScope

from . import _ffi_api, _tensor_view

_SCOPE_GLOBAL = MemoryScope("global")

_REARRANGE_PATTERN_UNSET = object()


def _tensor_type_field(name):
    def getter(ty, expr):
        _check_tensor_property_receiver(ty, expr)
        return getattr(ty, name)

    return getter


@tvm_ffi.register_object("tirx.TensorType")
class TensorType(Type):
    """The shape, element type, and storage contract of a TIRx tensor expression.

    Tensor variables expose the type's properties and operations, such as
    ``tensor.shape`` and ``tensor.view(...)``. Strides and element offsets are
    described by ``layout``.

    Parameters
    ----------
    shape : sequence[Expr] or Expr
        Logical dimension extents; a single extent denotes a one-dimensional tensor.
    dtype : PrimType or str
        Primitive element type.
    scope : MemoryScope or str, optional
        Storage scope, defaulting to global memory.
    align : int, optional
        Data-pointer alignment in bytes; zero selects the runtime default.
    layout : Layout or str, optional
        Explicit storage mapping. ``"default"`` creates a contiguous row-major
        layout; None leaves the layout unspecified.
    data : Expr, optional
        Keyword-only pointer expression whose PtrType supplies the storage scope.
        The pointer itself is not stored in TensorType.
    loc : Location, optional
        The source location of the type.
    data_alignment : int, optional
        Keyword-only spelling of ``align``; both must agree when explicitly set.
    """

    dtype: PrimType
    scope: MemoryScope
    shape: list
    data_alignment: int

    layout: object | None

    def __init__(
        self,
        shape,
        dtype,
        scope=_SCOPE_GLOBAL,
        align=0,
        layout=None,
        *,
        data=None,
        loc: Location = UNKNOWN_LOC,
        data_alignment=None,
    ):
        """Construct a tensor access contract.

        Strides and element offsets are described by ``layout``.
        ``align`` is an alias for ``data_alignment``.
        """
        from tvm_tirx.ir.type import _normalize_type
        from tvm_tirx.script.ir_builder.base import MISSING, at

        from .layout import Layout, S, TileLayout

        shape = (
            (shape,) if tvm_tirx.ir.is_prim_expr(shape) or isinstance(shape, Integral) else shape
        )
        shape = tuple(shape)
        dtype = _normalize_type(dtype)
        if not isinstance(dtype, PrimType):
            raise TypeError("Tensor dtype must be a primitive type")
        if data_alignment is None:
            data_alignment = align
        elif align not in (0, data_alignment):
            raise ValueError("align and data_alignment must agree")
        if data is not None:
            if not isinstance(data, tvm_tirx.ir.Expr) or not isinstance(data.ty, PtrType):
                raise TypeError("Tensor data must be an Expr with PtrType")
            if not isinstance(data.ty.element_type, PrimType):
                raise TypeError("Tensor data must point to a primitive type")
            scope = data.ty.scope
        if layout is MISSING:
            layout = None
        if layout == "default":
            default = TileLayout(S[shape])
            layout = default
        elif layout is not None and not isinstance(layout, Layout):
            raise TypeError("layout must be a Layout, None, or 'default'")
        self.__init_handle_by_constructor__(
            _ffi_api.TensorType,
            shape,
            dtype,
            scope,
            data_alignment,
            layout,
            loc if isinstance(loc, Location) else UNKNOWN_LOC,
        )
        at(loc, self)

    __expr_methods__ = (
        "is_scalar",
        "ptr_to",
        "view",
        "permute",
        "rearrange",
    )

    __expr_properties__: ClassVar[dict[str, Callable]] = {
        "scope": lambda ty, expr: lambda: ty._scope(expr),
        "shape": _tensor_type_field("shape"),
        "data_alignment": _tensor_type_field("data_alignment"),
        "layout": _tensor_type_field("layout"),
        "data": lambda ty, expr: ty._data(expr),
        "dtype": lambda ty, expr: ty._dtype(expr),
        "sub": lambda ty, expr: ty._sub(expr),
    }

    def has_default_layout(self, analyzer=None) -> bool:
        """Whether the layout matches implicit row-major storage with zero origin.

        Equivalent split tiles are accepted when their extent covers this shape.
        An optional analyzer supplies existing bindings and constraints; otherwise
        symbolic analysis is initialized only when structural equality is insufficient.
        """
        return _ffi_api.TensorTypeHasDefaultLayout(self, analyzer)

    def effective_layout(self):
        """Return the explicit layout or infer a compact global layout.

        Inference uses the full tensor shape before a region is sliced and
        leaves this type unchanged. Other storage scopes and noncompact global
        tensors require an explicit layout.
        """
        if self.layout is not None:
            return self.layout
        if self.scope != _SCOPE_GLOBAL:
            raise ValueError(f"An explicit layout is required for {self.scope} tensors")

        from .layout import S, TileLayout

        return TileLayout(S[tuple(self.shape)])

    def _scope(self, expr):
        """Return the storage scope associated with this tensor.
        Returns
        -------
        scope : tvm_tirx.runtime.MemoryScope
            The storage scope associated with this tensor.
        """
        _check_tensor_receiver(self, expr)
        return self.scope

    def is_scalar(self, expr, alloc_or_decl=True):
        """Check if the tensor is a scalar.

        Parameters
        ----------
        alloc_or_decl : bool, optional
            Whether to consider alloc_scalar and decl_scalar as scalar. True for alloc_scalar,
            False for decl_scalar.

        Returns
        -------
            bool: True if the tensor is a scalar, False otherwise.
        """
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorIsScalar(expr, alloc_or_decl)

    def ptr_to(self, expr, indices):
        """Get the pointer to the tensor at the given indices (logical indices).

        Note that the tensorload inside requires LowerTIPp pass to apply the layout to get the physical indices.
        """  # noqa: E501
        _check_tensor_receiver(self, expr)
        assert len(indices) == len(self.shape), (
            f"The number of indices {indices} does not match the shape of the tensor {self.shape}"
        )
        return tvm_tirx.tirx.address_of(expr[tuple(indices)])

    def view(self, expr, *args, **kwargs) -> "Var":
        """Creates a new view of the tensor. (used by parser)

        Supported signatures are ``view(*shape, layout=None)``, where shape can contain
        ``-1`` to indicate that the dimension size is auto-inferred, and
        ``view(dtype: Union[str, tvm_ffi.dtype])``.

        Returns
        -------
        view : Var
            The corresponding view tensor.
        """
        _check_tensor_receiver(self, expr)

        return _tensor_view.view(expr, *args, **kwargs)

    def permute(self, expr, *dims) -> "Var":
        """Permute the dimensions of the tensor.

        Parameters
        ----------
        dims : tuple of int
            The permutation of dimensions.

        Returns
        -------
        permuted : Var
            The tensor with permuted dimensions.
        """
        _check_tensor_receiver(self, expr)
        return _tensor_view.permute(expr, *dims)

    def rearrange(self, expr, pattern: str = _REARRANGE_PATTERN_UNSET, /, **sizes) -> "Var":
        """einops-style relayout in one line: ``buf.rearrange("b (2 r) -> 2 b r")``.

        A pure reshape+permute+reshape over the SAME physical bytes, spelled as
        an einops pattern. Lowers to ``view`` (split lhs groups) → ``permute``
        (reorder to rhs atom order) → ``view`` (merge rhs groups), so it inherits
        whatever the underlying axis machinery does: a plain (unswizzled) tensor
        collapses to a flat layout, a swizzled tensor keeps its swizzle,
        and a tmem tensor retains its backing address. It therefore does
        NOT flatten a swizzle atom — the same pattern on a swizzled SMEM tensor
        vs an unswizzled TMEM tensor legitimately yields different physical
        layouts (that is the point: rearrange acts on the operand, not a string).

        ``pattern`` is ``"lhs -> rhs"``; each side is space-separated axis names,
        with ``(a b)`` grouping a product axis. Every lhs group's product must
        equal that input dim; at most one axis per group may be unknown (inferred
        from the dim), the rest supplied via ``**sizes``. Cannot express a
        replica (``R[...]``), a stride-fiction/padded view, or a reshape crossing
        a swizzle-atom boundary — keep those as explicit ``view(layout=...)``.
        """
        _check_tensor_receiver(self, expr)
        if pattern is _REARRANGE_PATTERN_UNSET:
            if "pattern" not in sizes:
                raise TypeError("Var.rearrange() missing required argument: 'pattern'")
            pattern = sizes.pop("pattern")
        return _tensor_view.rearrange(expr, pattern, **sizes)

    def _data(self, expr):
        _check_tensor_property_receiver(self, expr)
        return tensor_data_ptr(expr)

    def _dtype(self, expr):
        _check_tensor_property_receiver(self, expr)
        return self.dtype.dtype

    def _sub(self, expr):
        _check_tensor_property_receiver(self, expr)
        return _tensor_view.sub(expr)


def is_tensor_var(value) -> bool:
    """Return whether ``value`` is an ordinary Var carrying TensorType.

    Unlike ``isinstance(value, Var)``, this predicate distinguishes tensor
    variables from scalar or pointer variables.
    """

    return isinstance(value, tvm_tirx.ir.Var) and isinstance(value.ty, TensorType)


def _check_tensor_receiver(ty, expr):
    if not is_tensor_var(expr):
        raise TypeError("Tensor methods expect a Var with TensorType")
    if not ty.same_as(expr.ty):
        raise TypeError("Tensor method type must be the operand's type")


def _check_tensor_property_receiver(ty, expr):
    if not is_tensor_var(expr):
        raise AttributeError("Tensor properties are only available on a Var with TensorType")
    _check_tensor_receiver(ty, expr)


def tensor_data_ptr(tensor, *, ty=None, loc: Location = UNKNOWN_LOC):
    """Project a tensor variable's physical pointer.

    The result type is inferred from its element type and storage scope.
    ``ty`` may supply an explicit result type; ``loc`` records the source location.
    """
    if not is_tensor_var(tensor):
        raise TypeError("tensor_data_ptr expects a Var with TensorType")
    return Call("tirx.tensor_data_ptr", [tensor], loc=loc, ty=ty)
