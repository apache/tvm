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
    """The structural type carried by an ordinary TIRx tensor variable.

    Declared expression methods receive the tensor variable as ``expr`` after
    the type receiver. Access through ``expr.method`` binds both receivers.
    Read-only expression properties receive the same type and operand through
    the getters in ``__expr_properties__``.
    """

    dtype: PrimType
    scope: MemoryScope
    shape: list
    strides: list
    elem_offset: tvm_tirx.ir.Expr
    data_alignment: int
    offset_factor: int
    layout: object | None

    def __init__(
        self,
        shape,
        dtype,
        scope=_SCOPE_GLOBAL,
        strides=None,
        elem_offset=None,
        align=0,
        offset_factor=0,
        layout=None,
        *,
        data=None,
        byte_offset=None,
        loc: Location = UNKNOWN_LOC,
        data_alignment=None,
    ):
        """Construct a tensor access contract.

        An omitted element offset is zero in the configured index dtype,
        independent of shape and offset_factor. Integer offsets use that dtype;
        explicitly typed expressions retain theirs. Pass a symbolic offset explicitly
        when the contract permits a variable starting position. ``align`` remains a
        compatibility spelling for ``data_alignment``.
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
        if elem_offset is not None and byte_offset is not None:
            raise ValueError("elem_offset and byte_offset cannot be set at the same time")
        if byte_offset is not None:
            elem_offset = byte_offset * 8 // tvm_ffi.dtype(dtype.dtype).bits
        if data_alignment is None:
            data_alignment = align
        elif align not in (0, data_alignment):
            raise ValueError("align and data_alignment must agree")
        if data is not None:
            if not isinstance(data, tvm_tirx.ir.Expr) or not isinstance(data.ty, PtrType):
                raise TypeError("Tensor data must be an Expr with PtrType")
            scope = data.ty.scope
        if layout is MISSING:
            layout = None
        if layout == "default":
            layout = TileLayout(S[shape])
        elif layout is not None and not isinstance(layout, Layout):
            raise TypeError("layout must be a Layout, None, or 'default'")
        self.__init_handle_by_constructor__(
            _ffi_api.TensorType,
            shape,
            dtype,
            scope,
            () if strides is None else strides,
            elem_offset,
            data_alignment,
            offset_factor,
            layout,
            UNKNOWN_LOC,
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
        "strides": _tensor_type_field("strides"),
        "elem_offset": _tensor_type_field("elem_offset"),
        "data_alignment": _tensor_type_field("data_alignment"),
        "offset_factor": _tensor_type_field("offset_factor"),
        "layout": _tensor_type_field("layout"),
        "data": lambda ty, expr: ty._data(expr),
        "dtype": lambda ty, expr: ty._dtype(expr),
        "byte_offset": lambda ty, expr: ty._byte_offset(expr),
        "sub": lambda ty, expr: ty._sub(expr),
    }

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

        from tvm_tirx.sym import Analyzer

        from .layout import S, TileLayout

        analyzer = Analyzer()
        expected = 1
        if self.strides:
            for extent, stride in reversed(list(zip(self.shape, self.strides))):
                if not analyzer.can_prove_equal(extent, 1) and not analyzer.can_prove_equal(
                    stride, expected
                ):
                    raise ValueError("An explicit layout is required for noncompact global tensors")
                expected = analyzer.simplify(expected * extent)
        return TileLayout(S[tuple(self.shape)])

    def _scope(self, expr):
        """Return the storage scope associated with this tensor.
        Returns
        -------
        scope : tvm_tirx.runtime.MemoryScope
            The storage scope associated with this tensor.
        """
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorStorageScope(expr)  # type: ignore

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

    def _byte_offset(self, expr):
        _check_tensor_property_receiver(self, expr)
        return self.elem_offset * tvm_ffi.dtype(self.dtype).bits // 8

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


def decl_tensor(
    shape,
    dtype=None,
    name="tensor",
    data=None,
    strides=None,
    elem_offset=None,
    scope=_SCOPE_GLOBAL,
    data_alignment=-1,
    offset_factor=0,
    loc: Location = UNKNOWN_LOC,
    layout=None,
):
    # pylint: disable=import-outside-toplevel
    from .layout import S, TileLayout

    shape = (shape,) if tvm_tirx.ir.is_prim_expr(shape) or isinstance(shape, Integral) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    if layout == "default":
        layout = TileLayout(S[tuple(shape)]) if shape else None

    if data is not None:
        if not isinstance(data, tvm_tirx.ir.Expr) or not isinstance(data.ty, PtrType):
            raise TypeError("Tensor data must be an Expr with PtrType")
        if not isinstance(data.ty.element_type, PrimType):
            raise TypeError("Tensor data must point to a primitive type")
        scope = data.ty.scope
    tensor_type = _ffi_api.TensorType(  # type: ignore
        shape,
        dtype,
        scope,
        strides,
        elem_offset,
        data_alignment,
        offset_factor,
        layout,
        loc,
    )
    return Var(name, tensor_type, loc=loc)


def tensor_data_ptr(tensor, *, ty=None, loc: Location = UNKNOWN_LOC):
    """Project a tensor variable's physical pointer.

    The result type is inferred from its element type and storage scope.
    ``ty`` may supply an explicit result type; ``loc`` records the source location.
    """
    if not is_tensor_var(tensor):
        raise TypeError("tensor_data_ptr expects a Var with TensorType")
    return Call("tirx.tensor_data_ptr", [tensor], loc=loc, ty=ty)
