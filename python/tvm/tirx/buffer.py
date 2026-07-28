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
"""Abstraction for array data structures."""

import functools
from enum import IntEnum
from numbers import Integral

import tvm_ffi

import tvm
from tvm.ir import PointerType, PrimType, Range, Type
from tvm.runtime import Object, convert

from . import _ffi_api


@tvm_ffi.register_object("tirx.BufferType")
class BufferType(Type):
    """The structural type carried by an ordinary buffer variable."""

    dtype: PrimType
    storage_scope: str
    shape: list
    strides: list
    elem_offset: tvm.ir.Expr
    data_alignment: int
    offset_factor: int
    layout: object | None
    allocated_addr: list


def is_buffer_var(value) -> bool:
    """Return whether ``value`` is an ordinary Var carrying BufferType.

    Use this predicate instead of ``isinstance(value, Buffer)``.  ``Buffer`` is
    a source-compatibility alias for :class:`tvm.ir.Var` and therefore does not
    discriminate buffer variables from scalar or pointer variables.
    """

    return isinstance(value, tvm.ir.Var) and isinstance(value.ty, BufferType)


class BufferAccessKind(IntEnum):
    """Buffer access modes accepted by :func:`buffer_access_ptr`."""

    READ = 1
    WRITE = 2


class _BufferMethods:
    """Symbolic data buffer in TVM.

    Buffer provide a way to represent data layout
    specialization of data structure in TVM.

    Do not construct directly, use :py:func:`~decl_buffer` instead.
    See the documentation of :py:func:`decl_buffer` for more details.

    See Also
    --------
    decl_buffer : Declare a buffer
    """

    def access_ptr(self, access_mask, ptr_type="handle", content_lanes=1, offset=0, extent=None):
        """Get an access pointer to the head of buffer.

        This is the recommended method to get buffer data
        ptress when interacting with external functions.

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
        return _ffi_api.BufferAccessPtr(
            self,
            access_mask,
            ptr_type,
            content_lanes,
            offset,
            extent,  # type: ignore
        )

    def vload(self, begin, dtype=None, predicate=None):
        """Generate an Expr that loads dtype from begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        dtype : str
            The data type to be loaded,
            can be vector type which have lanes that is multiple of Buffer.dtype

        predicate : Optional[Expr]
            A vector mask of boolean values indicating which lanes of a vector are to be
            loaded. The number lanes of the mask must be equal to the number of lanes being loaded.

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        begin = (begin,) if isinstance(begin, int) or tvm.ir.is_prim_expr(begin) else begin
        dtype = dtype if dtype else self.ty.dtype
        return _ffi_api.BufferVLoad(self, begin, dtype, predicate)  # type: ignore

    def vstore(self, begin, value, predicate=None):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        value : Expr
            The value to be stored.

        predicate : Optional[Expr]
            A vector mask of boolean values indicating which lanes of a vector are to be
            stored. The number lanes of the mask must be equal to the number of lanes in
            value.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        begin = (begin,) if isinstance(begin, int) or tvm.ir.is_prim_expr(begin) else begin
        return _ffi_api.BufferVStore(self, begin, value, predicate)  # type: ignore

    def scope(self):
        """Return the storage scope associated with this buffer.
        Returns
        -------
        scope : str
            The storage scope associated with this buffer.
        """
        return _ffi_api.BufferStorageScope(self)  # type: ignore

    def get_flattened_buffer(self):
        """Generate a Buffer that is a flattened version of this buffer.

        Returns
        -------
        flattened : Buffer
            The corresponding flat buffer.
        """
        return _ffi_api.BufferGetFlattenedBuffer(self)  # type: ignore

    def with_allocated_addr(self, allocated_addr):
        """Return a new buffer with the allocated address."""
        return _ffi_api.BufferWithAllocatedAddr(self, allocated_addr)  # type: ignore

    def with_dtype(self, dtype):
        """Return a new buffer with the dtype."""
        return _ffi_api.BufferWithDtype(self, dtype)  # type: ignore

    def offset_of(self, indices):
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
        return _ffi_api.BufferOffsetOf(self, indices)  # type: ignore

    @property
    def byte_offset(self):
        """Get the byte offset of the buffer."""
        return self.ty.elem_offset * tvm.DataType(self.ty.dtype).bits // 8

    def elem_offset_of(self, indices, inner=True):
        """Get the element offset of the buffer at the given indices.
        Note that indices subject to buffer's layout mapping.

        Parameters
        ----------
        indices : Union[Expr, List[Expr]]
            The indices of the element in the original buffer.

        inner : bool, optional
            If False, the offset is relative to the original buffer.
            Default is True.

        Returns
        -------
        offset: Expr
            The element offset of the buffer at the given indices.
        """
        if inner:
            return _ffi_api.BufferOffsetOfp(self, indices)
        return self.ty.elem_offset + _ffi_api.BufferOffsetOfp(self, indices)

    def byte_offset_of(self, indices, inner=True):
        """Get the byte offset of the buffer at the given indices.
        Note that indices subject to buffer's layout mapping.

        Parameters
        ----------
        indices : Union[Expr, List[Expr]]
            The indices of the element in the original buffer.

        inner : bool, optional
            If False, the offset is relative to the original buffer.
            Default is True.

        Returns
        -------
        offset: Expr
            The byte offset of the buffer at the given indices.
        """
        return self.elem_offset_of(indices, inner) * tvm.DataType(self.ty.dtype).bits // 8

    def is_scalar(self, alloc_or_decl=True):
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
        return _ffi_api.BufferIsScalar(self, alloc_or_decl)

    def ptr_to(self, indices):
        """Get the pointer to the buffer at the given indices (logical indices).

        Note that the bufferload inside requires LowerTIPp pass to apply the layout to get the physical indices.
        """  # noqa: E501
        assert len(indices) == len(self.ty.shape), (
            f"The number of indices {indices} does not match "
            f"the shape of the buffer {self.ty.shape}"
        )
        return tvm.tirx.address_of(self[tuple(indices)])

    def view(self, *args, **kwargs) -> "Buffer":
        """Creates a new view of the buffer. (used by parser)

        Supported signatures are ``view(*shape, layout=None)``, where shape can contain
        ``-1`` to indicate that the dimension size is auto-inferred, and
        ``view(dtype: Union[str, tvm.DataType])``.

        Returns
        -------
        view : DeclBufferFrame
            The corresponding view buffer.
        """

        def _infer_shape(shape):
            shape = list(shape)
            if -1 in shape and shape.count(-1) == 1:
                size = functools.reduce(lambda x, y: x * y, self.ty.shape)
                n_size = functools.reduce(lambda x, y: x * y, [s for s in shape if s != -1], 1)
                shape[shape.index(-1)] = size // n_size
            else:
                # Only validate the shape product when both old and new shapes
                # are fully concrete: a Expr `==` returns an `EQ` node, not
                # a Python bool, and `assert <Expr>` raises (no __bool__).
                if all(isinstance(s, int) for s in shape) and all(
                    isinstance(s, int) for s in self.ty.shape
                ):
                    assert functools.reduce(lambda x, y: x * y, shape) == functools.reduce(
                        lambda x, y: x * y, self.ty.shape
                    ), (
                        "The shape of the buffer "
                        + str(self.ty.shape)
                        + " and the new shape "
                        + str(shape)
                        + " are not compatible"
                    )
            return shape

        if len(args) == 1 and isinstance(args[0], str | tvm.DataType) and not kwargs:
            cast_dtype = tvm.DataType(args[0])
            cur_dtype = tvm.DataType(self.ty.dtype)
            if cast_dtype.bits > cur_dtype.bits:
                # cast up
                assert cast_dtype.bits % cur_dtype.bits == 0
                ratio = cast_dtype.bits // cur_dtype.bits
                layout = self.ty.layout.pack(ratio)
                shape = [s for s in self.ty.shape[:-1]] + [self.ty.shape[-1] // ratio]
                new_elem_offset = self.ty.elem_offset // ratio
            else:
                # cast down
                assert cur_dtype.bits % cast_dtype.bits == 0
                ratio = cur_dtype.bits // cast_dtype.bits
                layout = self.ty.layout.unpack(ratio)
                shape = [s for s in self.ty.shape[:-1]] + [self.ty.shape[-1] * ratio]
                new_elem_offset = self.ty.elem_offset * ratio
            return tvm.tirx.script.builder.decl_buffer(
                shape,
                cast_dtype,
                buffer_data(self),
                self.ty.strides,
                new_elem_offset,
                None,
                self.ty.storage_scope,
                self.ty.data_alignment,
                self.ty.offset_factor,
                layout,
            )
        else:
            # --- Signature 1: view(*shape, **opts) ---
            # Check if all positional args are integers/PrimExprs with dtype int32 or int64 (the shape)  # noqa: E501
            shape = args
            assert all(
                isinstance(arg, int)
                or (tvm.ir.is_prim_expr(arg) and arg.ty.dtype in ["int32", "int64"])
                for arg in shape
            ), "shape must be a list of integers or PrimExprs with dtype int32 or int64"
            # Safely get optional keyword arguments
            layout = kwargs.get("layout", None)
            # Assert there are no other kwargs
            assert set(kwargs.keys()).issubset({"layout"}), (
                f"Unsupported kwargs for view: {set(kwargs.keys()) - {'layout'}}"
            )

            if layout is None:
                shape = _infer_shape(shape)

            return tvm.tirx.script.builder.decl_buffer(
                shape,
                self.ty.dtype,
                buffer_data(self),
                self.ty.strides,
                self.ty.elem_offset,
                None,
                self.ty.storage_scope,
                self.ty.data_alignment,
                self.ty.offset_factor,
                self.ty.layout if layout is None else layout,
            )

    def local(self, *shape, layout=None) -> "Buffer":
        """Create a thread-local view of this buffer.

        When called with no shape arguments, auto-infers a 1D shape from
        the layout's non-thread component (i.e. ``layout.storage().shard``).

        Parameters
        ----------
        shape : tuple of Expr
            The shape of the local view for indexing. If omitted, a 1D
            shape is computed automatically.

        layout : optional
            Override layout. If None, uses the storage layout
            (parent layout with thread axes removed).

        Returns
        -------
        local : DeclBufferFrame
            The corresponding local buffer.
        """
        if not shape:
            local_layout = self.ty.layout.storage()
            total = functools.reduce(
                lambda x, y: x * y, [it.extent for it in local_layout.shard], 1
            )
            shape = (total,)
        return tvm.tirx.script.builder.decl_buffer(
            shape,
            self.ty.dtype,
            buffer_data(self),
            self.ty.strides,
            self.ty.elem_offset,
            None,
            self.ty.storage_scope,
            self.ty.data_alignment,
            self.ty.offset_factor,
            self.ty.layout.storage() if layout is None else layout,
        )

    def permute(self, *dims) -> "Buffer":
        """Permute the dimensions of the buffer.

        Parameters
        ----------
        dims : tuple of int
            The permutation of dimensions.

        Returns
        -------
        permuted : DeclBufferFrame
            The buffer with permuted dimensions.
        """
        new_shape = [self.ty.shape[d] for d in dims]
        # Permute *logical* dims, not the layout's fine-grained shard iters: a
        # tcgen05/atom layout maps several shard iters to each logical axis, so
        # group by the current shape first and permute whole groups. ``group``
        # returns a regrouped layout (degenerate extent-1 iters folded away)
        # plus seps over *that* layout — permute the regrouped one, not
        # ``self.layout``. For a simple layout (one shard iter per axis) this
        # reduces to ``permute_dims(dims)``.
        grouped, seps = self.ty.layout.group(list(self.ty.shape))
        new_layout = grouped.permute_by_groups(seps, list(dims))
        return tvm.tirx.script.builder.decl_buffer(
            new_shape,
            self.ty.dtype,
            buffer_data(self),
            self.ty.strides,
            self.ty.elem_offset,
            None,
            self.ty.storage_scope,
            self.ty.data_alignment,
            self.ty.offset_factor,
            new_layout,
        )

    def __getitem__(self, indices):
        if not is_buffer_var(self):
            return _ORIGINAL_VAR_GETITEM(self, indices)

        from ..arith import Analyzer  # pylint: disable=import-outside-toplevel
        from .expr import BufferLoad, Ramp  # pylint: disable=import-outside-toplevel
        from .stmt import BufferRegion  # pylint: disable=import-outside-toplevel

        if not isinstance(indices, tuple | list):
            indices = [indices]
        has_slice = any(isinstance(i, slice) for i in indices)
        has_step = any(
            isinstance(i, slice) and (i.step is not None and i.step != 1) for i in indices
        )
        has_implicit_slice = len(indices) < len(self.ty.shape)
        analyzer = Analyzer()
        if (has_slice and not has_step) or has_implicit_slice:
            region = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.ty.shape[i] if index.stop is None else index.stop
                    region.append(Range.from_min_extent(start, analyzer.simplify(stop - start)))
                else:
                    region.append(
                        Range.from_min_extent(
                            index,
                            tvm.tirx.expr.IntImm(index.ty, 1) if tvm.ir.is_prim_expr(index) else 1,
                        )
                    )
            if has_implicit_slice:
                for i in range(len(indices), len(self.ty.shape)):
                    region.append(Range.from_min_extent(0, self.ty.shape[i]))
            return BufferRegion(self, region)
        else:
            expr_indices = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.ty.shape[i] if index.stop is None else index.stop
                    step = 1 if index.step is None else index.step
                    # We should ensure the dtype of start is the same with that of step.
                    if tvm.ir.is_prim_expr(start) and isinstance(step, int):
                        step = tvm.tirx.expr.IntImm(start.ty, step)
                    lanes = analyzer.simplify((stop - start + step - 1) // step)
                    if lanes == 1:
                        expr_indices.append(start)
                    else:
                        expr_indices.append(Ramp(start, step, int(lanes)))
                else:
                    expr_indices.append(index)
            return BufferLoad(self, expr_indices)


def decl_buffer(
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
        if not isinstance(data, tvm.ir.Var) or not isinstance(data.ty, PointerType):
            raise TypeError("Buffer data must be a Var with PointerType")
        if not isinstance(data.ty.element_type, PrimType):
            raise TypeError("Buffer data must point to a primitive type")
        storage_scope = data.ty.storage_scope
    buffer_type = _ffi_api.BufferType(  # type: ignore
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
    return _ffi_api.BufferVar(name, buffer_type, span)  # type: ignore


def buffer_data(buffer):
    """Project the physical pointer associated with a buffer variable."""

    if not is_buffer_var(buffer):
        raise TypeError("buffer_data expects a Var with BufferType")
    return _ffi_api.BufferData(buffer)


def buffer_data_pointer_type(buffer):
    """Return the pointer type produced by :func:`buffer_data`."""

    if not is_buffer_var(buffer):
        raise TypeError("buffer_data_pointer_type expects a Var with BufferType")
    return _ffi_api.BufferDataPointerType(buffer)


# Buffer values intentionally retain runtime type key ``ir.Var``.  Importing
# ``tvm.tirx`` therefore augments ``tvm.ir.Var`` process-wide with the legacy
# buffer operation and metadata surface.  Non-buffer Vars reject the metadata
# properties with AttributeError, preserving correct ``hasattr`` behavior.
_ORIGINAL_VAR_GETITEM = tvm.ir.Var.__getitem__
for _name, _value in _BufferMethods.__dict__.items():
    if _name.startswith("__") and _name != "__getitem__":
        continue
    if callable(_value) or isinstance(_value, property):
        setattr(tvm.ir.Var, _name, _value)


def _buffer_type_field(name):
    def getter(value):
        if not is_buffer_var(value):
            raise AttributeError(f"{name} is only available on a Var with BufferType")
        return getattr(value.ty, name)

    return property(getter)


# Preserve Buffer's public metadata surface while keeping BufferType as the
# single source of truth.
for _name in (
    "shape",
    "strides",
    "elem_offset",
    "data_alignment",
    "offset_factor",
    "layout",
    "allocated_addr",
):
    setattr(tvm.ir.Var, _name, _buffer_type_field(_name))


def _buffer_dtype_property(value):
    if not is_buffer_var(value):
        raise AttributeError("dtype is only available on a Var with BufferType")
    # Preserve the pre-migration Python Buffer surface.  BufferType stores a
    # PrimType, while Python callers historically receive its runtime DataType.
    return value.ty.dtype.dtype


tvm.ir.Var.dtype = property(_buffer_dtype_property)


# Keep the established ``A.data`` TVMScript surface as syntax sugar.  Compiler
# and builder code calls ``buffer_data(A)`` directly.
def _buffer_data_property(value):
    if not is_buffer_var(value):
        raise AttributeError("data is only available on a Var with BufferType")
    return buffer_data(value)


tvm.ir.Var.data = property(_buffer_data_property)

# Source compatibility for annotations and imports only.  There is no
# ``tirx.Buffer`` runtime object; constructors return ``tvm.ir.Var``.  In
# particular, ``isinstance(value, Buffer)`` matches every Var.  Runtime checks
# must use ``is_buffer_var(value)``.
Buffer = tvm.ir.Var


@tvm_ffi.register_object("tirx.DataProducer")
class DataProducer(Object):
    pass
