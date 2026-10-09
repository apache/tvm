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
from enum import IntEnum
from numbers import Integral
from typing import ClassVar

import tvm_ffi

import tvm
from tvm.ir import Call, PointerType, PrimType, Type, Var
from tvm.runtime import convert

from . import _buffer_view, _ffi_api

_REARRANGE_PATTERN_UNSET = object()


def _tensor_type_field(name):
    def getter(ty, expr):
        _check_tensor_property_receiver(ty, expr)
        return getattr(ty, name)

    return getter


@tvm_ffi.register_object("tirx.TensorType")
class TensorType(Type):
    """The structural type carried by an ordinary TIRx tensor variable."""

    dtype: PrimType
    storage_scope: str
    shape: list
    strides: list
    elem_offset: tvm.ir.Expr
    data_alignment: int
    offset_factor: int
    layout: object | None
    allocated_addr: list

    __expr_methods__ = (
        "access_ptr",
        "vload",
        "vstore",
        "scope",
        "get_flattened_buffer",
        "with_allocated_addr",
        "with_dtype",
        "offset_of",
        "is_scalar",
        "ptr_to",
        "view",
        "local",
        "permute",
        "rearrange",
        "tile",
        "chunk",
    )

    def access_ptr(
        self, expr, access_mask, ptr_type="handle", content_lanes=1, offset=0, extent=None
    ):
        """Get an access pointer to the head of buffer.

        This is the recommended method to get buffer data
        pointers when interacting with external functions.

        Parameters
        ----------
        access_mask : int
            The access pattern MASK. Indicate whether the
            access will read or write to the data content.

        ptr_type : str or tvm.ir.Type, optional
            The data type of the result pointer. Do not specify
            unless we want to cast pointer to specific type.

        content_lanes: int, optional
            The number of lanes for the data type. This value
            is greater than one for vector types.

        offset: Expr, optional
            The offset of pointer. We can use it to offset by
            the number of elements from the address of ptr.

        extent: Expr, optional
            The extent of pointer.

        Examples
        --------
        .. code-block:: python

          # Get access ptr for read
          buffer.access_ptr("r")
          # Get access ptr for read/write with bitmask
          buffer.access_ptr(BufferAccessKind.READ | BufferAccessKind.WRITE)
          # Get access ptr for read/write with str flag
          buffer.access_ptr("rw")
          # Get access ptr for read with offset
          buffer.access_ptr("r", offset = 100)
          # Get access ptr for read with extent
          buffer.access_ptr("r", extent = 100)
        """
        _check_tensor_receiver(self, expr)
        if isinstance(access_mask, str):
            mask = 0
            for value in access_mask:
                if value == "r":
                    mask = mask | BufferAccessKind.READ
                elif value == "w":
                    mask = mask | BufferAccessKind.WRITE
                else:
                    raise ValueError(f"Unknown access_mask {access_mask}")
            access_mask = mask
        if isinstance(ptr_type, str):
            ptr_type = (
                PointerType(PrimType("void"))
                if ptr_type == "handle"
                else PointerType(PrimType(ptr_type))
            )
        elif isinstance(ptr_type, PrimType):
            ptr_type = PointerType(ptr_type)
        offset = convert(offset)
        extent = convert(extent)
        return _ffi_api.TensorAccessPtr(
            expr,
            access_mask,
            ptr_type,
            content_lanes,
            offset,
            extent,  # type: ignore
        )

    def vload(self, expr, begin, dtype=None):
        """Generate an Expr that loads dtype from begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Var.dtype

        dtype : str
            The data type to be loaded,
            can be vector type which have lanes that is multiple of Var.dtype

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        _check_tensor_receiver(self, expr)
        begin = (begin,) if isinstance(begin, int) or tvm.ir.is_prim_expr(begin) else begin
        dtype = dtype if dtype else self.dtype
        return _ffi_api.TensorVLoad(expr, begin, dtype)  # type: ignore

    def vstore(self, expr, begin, value):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Var.dtype

        value : Expr
            The value to be stored.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        _check_tensor_receiver(self, expr)
        begin = (begin,) if isinstance(begin, int) or tvm.ir.is_prim_expr(begin) else begin
        return _ffi_api.TensorVStore(expr, begin, value)  # type: ignore

    def scope(self, expr):
        """Return the storage scope associated with this buffer.
        Returns
        -------
        scope : str
            The storage scope associated with this buffer.
        """
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorStorageScope(expr)  # type: ignore

    def get_flattened_buffer(self, expr):
        """Generate a Var that is a flattened version of this buffer.

        Returns
        -------
        flattened : Var
            The corresponding flat buffer.
        """
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorGetFlattenedTensor(expr)  # type: ignore

    def with_allocated_addr(self, expr, allocated_addr):
        """Return a new buffer with the allocated address."""
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorWithAllocatedAddr(expr, allocated_addr)  # type: ignore

    def with_dtype(self, expr, dtype):
        """Return a new buffer with the dtype."""
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorWithDtype(expr, dtype)  # type: ignore

    def offset_of(self, expr, indices):
        """Determine the offset of the provided indices in the flattened buffer.

        Parameters
        ----------
        indices : Union[Expr, List[Expr]]

            The indices of the element in the original buffer.

        Returns
        -------
        flattened_indices: List[Expr]

            The offset indices of the element in the flattened buffer.
        """
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorOffsetOf(expr, indices)  # type: ignore

    def is_scalar(self, expr, alloc_or_decl=True):
        """Check if the buffer is a scalar.

        Parameters
        ----------
        alloc_or_decl : bool, optional
            Whether to consider alloc_scalar and decl_scalar as scalar. True for alloc_scalar,
            False for decl_scalar.

        Returns
        -------
            bool: True if the buffer is a scalar, False otherwise.
        """
        _check_tensor_receiver(self, expr)
        return _ffi_api.TensorIsScalar(expr, alloc_or_decl)

    def ptr_to(self, expr, indices):
        """Get the pointer to the buffer at the given indices (logical indices).

        Note that the bufferload inside requires LowerTIPp pass to apply the layout to get the physical indices.
        """  # noqa: E501
        _check_tensor_receiver(self, expr)
        assert len(indices) == len(self.shape), (
            f"The number of indices {indices} does not match the shape of the buffer {self.shape}"
        )
        return tvm.tirx.address_of(expr[tuple(indices)])

    def view(self, expr, *args, **kwargs) -> "Var":
        """Creates a new view of the buffer. (used by parser)

        Supported signatures are ``view(*shape, layout=None)``, where shape can contain
        ``-1`` to indicate that the dimension size is auto-inferred, and
        ``view(dtype: Union[str, tvm.DataType])``.

        Returns
        -------
        view : Var
            The corresponding view buffer.
        """
        _check_tensor_receiver(self, expr)

        return _buffer_view.view(expr, *args, **kwargs)

    def local(self, expr, *shape, layout=None) -> "Var":
        """Create a thread-local view of this buffer.

        By default, both the inferred and explicit-shape forms address the
        raw physical storage span.  ``local()[k]`` is the k-th physical
        storage element, including any gaps or layout offset, while
        ``local(d0, d1, ...)`` is a row-major reshape of that same span.
        Pass ``layout=`` to request a mediated view explicitly.  This is an
        escape hatch whose shape is interpreted by the supplied layout.  When
        that shape is explicit, the parent buffer does not need a layout.

        When called with no shape arguments, auto-infers a 1D shape from
        the span of the parent layout's non-thread component (i.e.
        ``expr.layout.storage().span()``).  The explicit-``layout=`` form
        instead infers the parent layout's ``storage().size()`` for
        compatibility.  Either inference requires the parent buffer to have a
        layout.

        Parameters
        ----------
        shape : tuple of Expr
            The shape of the local view for indexing.  Without ``layout=``,
            its product must equal the per-thread physical storage span.
            With an explicit layout, the shape is not constrained by the raw
            span.  If omitted, a matching 1D shape is computed automatically.

        layout : optional
            Override layout. If None, the default (identity) layout is used.

        Returns
        -------
        local : Var
            The corresponding local buffer.
        """
        _check_tensor_receiver(self, expr)
        return _buffer_view.local(expr, *shape, layout=layout)

    def permute(self, expr, *dims) -> "Var":
        """Permute the dimensions of the buffer.

        Parameters
        ----------
        dims : tuple of int
            The permutation of dimensions.

        Returns
        -------
        permuted : Var
            The buffer with permuted dimensions.
        """
        _check_tensor_receiver(self, expr)
        return _buffer_view.permute(expr, *dims)

    def rearrange(self, expr, pattern: str = _REARRANGE_PATTERN_UNSET, /, **sizes) -> "Var":
        """einops-style relayout in one line: ``buf.rearrange("b (2 r) -> 2 b r")``.

        A pure reshape+permute+reshape over the SAME physical bytes, spelled as
        an einops pattern. Lowers to ``view`` (split lhs groups) → ``permute``
        (reorder to rhs atom order) → ``view`` (merge rhs groups), so it inherits
        whatever the underlying axis machinery does: a plain (unswizzled) buffer
        collapses to a flat layout, a swizzled buffer keeps its swizzle,
        and a tmem buffer carries ``allocated_addr`` through. It therefore does
        NOT flatten a swizzle atom — the same pattern on a swizzled SMEM buffer
        vs an unswizzled TMEM buffer legitimately yields different physical
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
        return _buffer_view.rearrange(expr, pattern, **sizes)

    def tile(self, expr, *specs) -> "_buffer_view.TileIndexer":
        """Chunk a dim: split it into factors, pick a chunk, keep the rest.

        Rank-preserving — the picked dim's remaining factors merge back into
        that one dim, and every other dim is untouched, so N dims in gives N
        dims out. Chunk multiple dims by chaining (dims never shift):
        ``buf.tile(0, (nx, -1))[cx, :].tile(1, (-1, ny))[:, cy]``.

        Call as ``tile(dim, factors)`` for one dim, or pass several
        ``(dim, factors)`` specs as sugar for a chain. ``factors`` is the
        tuple the dim splits into (row-major, like :meth:`unflatten`; one
        ``-1`` inferred). The indexer takes one entry per factor: an ``int`` /
        ``Expr`` **picks** it (fixing the chunk, dropping the axis, folding
        its offset) and ``:`` **keeps** it. At least one factor per dim must be
        picked — a pure keep-everything split is :meth:`unflatten`, not a
        chunk::

            # 64 rows split into (stripe, warp, row) = (-1, WARPS, 4); this
            # warp's 16 interleaved rows (stripe x row merged):
            buf.tile(1, (-1, WARPS, 4))[:, warp, :]

            tile(d, (n, -1))[c, :]   # contiguous block c
            tile(d, (-1, n))[:, c]   # round-robin chunk c

        A picked index may be a dynamic Expr (e.g. a warp id); picking
        several factors of one dim is allowed.
        """
        _check_tensor_receiver(self, expr)
        return _buffer_view.tile(expr, *specs)

    def chunk(self, expr, spec) -> "_buffer_view.ChunkIndexer":
        """Split dims into equal contiguous chunks and pick a chunk per dim —
        **rank-preserving**. Index the result with ``[picks]``.

        ``spec`` is a per-dim tuple (length = rank). Each entry is ``None``
        (leave the dim) or a positive int ``n`` (split that dim, extent ``E``
        with ``E % n == 0``, into ``n`` equal chunks of ``E // n``). Then
        ``chunk(spec)[picks]`` takes one entry per dim: a chunked dim's pick is
        the chunk index (int / Expr) and **narrows that dim** to the chunk's
        ``[c*E//n : (c+1)*E//n)`` range — the dim is kept at ``E // n``, no
        dimension is added; an unchunked dim's pick is a normal index (``:`` /
        int / slice). The result is the *same TensorRegion* as the hand-written
        slice — one line instead of the ``c*k : (c+1)*k`` arithmetic::

            X[.., c * k : (c + 1) * k, ..]        # before (k = E // n)
            X.chunk((None, .., n, ..))[.., c, ..]  # after (k inferred)
        """
        _check_tensor_receiver(self, expr)
        return _buffer_view.chunk(expr, spec)

    def _data(self, expr):
        _check_tensor_property_receiver(self, expr)
        return tensor_data_ptr(expr)

    def _dtype(self, expr):
        _check_tensor_property_receiver(self, expr)
        return self.dtype.dtype

    def _byte_offset(self, expr):
        _check_tensor_property_receiver(self, expr)
        return self.elem_offset * tvm.DataType(self.dtype).bits // 8

    def _sub(self, expr):
        _check_tensor_property_receiver(self, expr)
        return _buffer_view.sub(expr)

    __expr_properties__: ClassVar[dict[str, Callable]] = {
        "shape": _tensor_type_field("shape"),
        "strides": _tensor_type_field("strides"),
        "elem_offset": _tensor_type_field("elem_offset"),
        "data_alignment": _tensor_type_field("data_alignment"),
        "offset_factor": _tensor_type_field("offset_factor"),
        "layout": _tensor_type_field("layout"),
        "allocated_addr": _tensor_type_field("allocated_addr"),
        "data": _data,
        "dtype": _dtype,
        "byte_offset": _byte_offset,
        "sub": _sub,
    }


def is_tensor_var(value) -> bool:
    """Return whether ``value`` is an ordinary Var carrying TensorType.

    Unlike ``isinstance(value, Var)``, this predicate distinguishes tensor
    variables from scalar or pointer variables.
    """

    return isinstance(value, tvm.ir.Var) and isinstance(value.ty, TensorType)


class BufferAccessKind(IntEnum):
    """Buffer access modes accepted by :func:`buffer_access_ptr`."""

    READ = 1
    WRITE = 2


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
    name="buffer",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    data_alignment=-1,
    offset_factor=0,
    span=None,
    layout="default",
):
    # pylint: disable=import-outside-toplevel
    from .expr import Var
    from .layout import S, TileLayout

    shape = (shape,) if tvm.ir.is_prim_expr(shape) or isinstance(shape, Integral) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    if layout == "default":
        layout = TileLayout(S[tuple(shape)]) if shape else None

    if offset_factor != 0 and elem_offset is None:
        shape_ty = shape[0].ty if shape and tvm.ir.is_prim_expr(shape[0]) else "int32"
        elem_offset = Var(f"{name}_elem_offset", shape_ty)
    storage_scope = scope
    if data is not None:
        if not isinstance(data, tvm.ir.Expr) or not isinstance(data.ty, PointerType):
            raise TypeError("Tensor data must be an Expr with PointerType")
        if not isinstance(data.ty.element_type, PrimType):
            raise TypeError("Tensor data must point to a primitive type")
        storage_scope = data.ty.storage_scope
    buffer_type = _ffi_api.TensorType(  # type: ignore
        storage_scope,
        dtype,
        shape,
        strides,
        elem_offset,
        data_alignment,
        offset_factor,
        layout,
        (),
        span,
    )
    return _ffi_api.TensorVar(name, buffer_type, span)  # type: ignore


def tensor_data_ptr(tensor, *, ty=None, span=None):
    """Project a tensor variable's physical pointer.

    The result type is inferred from its element type and storage scope.
    ``ty`` may supply an explicit result type; ``span`` records the source location.
    """
    if not is_tensor_var(tensor):
        raise TypeError("tensor_data_ptr expects a Var with TensorType")
    return Call("tirx.tensor_data_ptr", [tensor], ty=ty, span=span)


def buffer_data_pointer_type(buffer):
    """Return the pointer type produced by :func:`tensor_data_ptr`."""

    if not is_tensor_var(buffer):
        raise TypeError("buffer_data_pointer_type expects a Var with TensorType")
    return _ffi_api.TensorDataPointerType(buffer)
