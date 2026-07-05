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
import re
from numbers import Integral

import tvm_ffi

import tvm
from tvm.ir import PointerType, PrimExpr, PrimType, Range
from tvm.runtime import Object, Scriptable, convert

from . import _ffi_api


@tvm_ffi.register_object("tirx.Buffer")
class Buffer(Object, Scriptable):
    """Symbolic data buffer in TVM.

    Buffer provide a way to represent data layout
    specialization of data structure in TVM.

    Do not construct directly, use :py:func:`~decl_buffer` instead.
    See the documentation of :py:func:`decl_buffer` for more details.

    See Also
    --------
    decl_buffer : Declare a buffer
    """

    READ = 1
    WRITE = 2

    def access_ptr(self, access_mask, ptr_type="handle", content_lanes=1, offset=0, extent=None):
        """Get an access pointer to the head of buffer.

        This is the recommended method to get buffer data
        ptress when interacting with external functions.

        Parameters
        ----------
        access_mask : int
            The access pattern MASK. Indicate whether the
            access will read or write to the data content.

        ptr_type : str, optional
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
          buffer.access_ptr(Buffer.READ | Buffer.WRITE)
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
                    mask = mask | Buffer.READ
                elif value == "w":
                    mask = mask | Buffer.WRITE
                else:
                    raise ValueError(f"Unknown access_mask {access_mask}")
            access_mask = mask
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

        predicate : Optional[PrimExpr]
            A vector mask of boolean values indicating which lanes of a vector are to be
            loaded. The number lanes of the mask must be equal to the number of lanes being loaded.

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        begin = (begin,) if isinstance(begin, int | PrimExpr) else begin
        dtype = dtype if dtype else self.dtype
        return _ffi_api.BufferVLoad(self, begin, dtype, predicate)  # type: ignore

    def vstore(self, begin, value, predicate=None):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        value : Expr
            The value to be stored.

        predicate : Optional[PrimExpr]
            A vector mask of boolean values indicating which lanes of a vector are to be
            stored. The number lanes of the mask must be equal to the number of lanes in
            value.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        begin = (begin,) if isinstance(begin, int | PrimExpr) else begin
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

    def with_data(self, data):
        """Return a new buffer with the data."""
        return _ffi_api.BufferWithData(self, data)  # type: ignore

    def offset_of(self, indices):
        """Determine the offset of the provided indices in the flattened buffer.

        Parameters
        ----------
        indices : Union[PrimExpr, List[PrimExpr]]

            The indices of the element in the original buffer.

        Returns
        -------
        flattened_indices: List[PrimExpr]

            The offset indices of the element in the flattened buffer.
        """
        return _ffi_api.BufferOffsetOf(self, indices)  # type: ignore

    @property
    def byte_offset(self):
        """Get the byte offset of the buffer."""
        return self.elem_offset * tvm.DataType(self.dtype).bits // 8

    def elem_offset_of(self, indices, inner=True):
        """Get the element offset of the buffer at the given indices.
        Note that indices subject to buffer's layout mapping.

        Parameters
        ----------
        indices : Union[PrimExpr, List[PrimExpr]]
            The indices of the element in the original buffer.

        inner : bool, optional
            If False, the offset is relative to the original buffer.
            Default is True.

        Returns
        -------
        offset: PrimExpr
            The element offset of the buffer at the given indices.
        """
        if inner:
            return _ffi_api.BufferOffsetOfp(self, indices)
        return self.elem_offset + _ffi_api.BufferOffsetOfp(self, indices)

    def byte_offset_of(self, indices, inner=True):
        """Get the byte offset of the buffer at the given indices.
        Note that indices subject to buffer's layout mapping.

        Parameters
        ----------
        indices : Union[PrimExpr, List[PrimExpr]]
            The indices of the element in the original buffer.

        inner : bool, optional
            If False, the offset is relative to the original buffer.
            Default is True.

        Returns
        -------
        offset: PrimExpr
            The byte offset of the buffer at the given indices.
        """
        return self.elem_offset_of(indices, inner) * tvm.DataType(self.dtype).bits // 8

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
        assert len(indices) == len(self.shape), (
            f"The number of indices {indices} does not match the shape of the buffer {self.shape}"
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
                size = functools.reduce(lambda x, y: x * y, self.shape)
                n_size = functools.reduce(lambda x, y: x * y, [s for s in shape if s != -1], 1)
                shape[shape.index(-1)] = size // n_size
            else:
                # Only validate the shape product when both old and new shapes
                # are fully concrete: a PrimExpr `==` returns an `EQ` node, not
                # a Python bool, and `assert <PrimExpr>` raises (no __bool__).
                if all(isinstance(s, int) for s in shape) and all(
                    isinstance(s, int) for s in self.shape
                ):
                    assert functools.reduce(lambda x, y: x * y, shape) == functools.reduce(
                        lambda x, y: x * y, self.shape
                    ), (
                        "The shape of the buffer "
                        + str(self.shape)
                        + " and the new shape "
                        + str(shape)
                        + " are not compatible"
                    )
            return shape

        if len(args) == 1 and isinstance(args[0], str | tvm.DataType) and not kwargs:
            cast_dtype = tvm.DataType(args[0])
            cur_dtype = tvm.DataType(self.dtype)
            if cast_dtype.bits > cur_dtype.bits:
                # cast up
                assert cast_dtype.bits % cur_dtype.bits == 0
                ratio = cast_dtype.bits // cur_dtype.bits
                layout = self.layout.pack(ratio)
                shape = [s for s in self.shape[:-1]] + [self.shape[-1] // ratio]
                new_elem_offset = self.elem_offset // ratio
            else:
                # cast down
                assert cur_dtype.bits % cast_dtype.bits == 0
                ratio = cur_dtype.bits // cast_dtype.bits
                layout = self.layout.unpack(ratio)
                shape = [s for s in self.shape[:-1]] + [self.shape[-1] * ratio]
                new_elem_offset = self.elem_offset * ratio
            return tvm.tirx.script.builder.decl_buffer(
                shape,
                cast_dtype,
                self.data,
                self.strides,
                new_elem_offset,
                None,
                self.scope(),
                self.data_alignment,
                self.offset_factor,
                "",
                self.axis_separators,
                layout,
            )
        else:
            # --- Signature 1: view(*shape, **opts) ---
            # Check if all positional args are integers/PrimExprs with dtype int32 or int64 (the shape)  # noqa: E501
            shape = args
            assert all(
                isinstance(arg, int)
                or (isinstance(arg, PrimExpr) and arg.ty.dtype in ["int32", "int64"])
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
                self.dtype,
                self.data,
                self.strides,
                self.elem_offset,
                None,
                self.scope(),
                self.data_alignment,
                self.offset_factor,
                "",
                self.axis_separators,
                self.layout if layout is None else layout,
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
            local_layout = self.layout.storage()
            total = functools.reduce(
                lambda x, y: x * y, [it.extent for it in local_layout.shard], 1
            )
            shape = (total,)
        return tvm.tirx.script.builder.decl_buffer(
            shape,
            self.dtype,
            self.data,
            self.strides,
            self.elem_offset,
            None,
            self.scope(),
            self.data_alignment,
            self.offset_factor,
            "",
            self.axis_separators,
            self.layout.storage() if layout is None else layout,
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
        new_shape = [self.shape[d] for d in dims]
        # Permute *logical* dims, not the layout's fine-grained shard iters: a
        # tcgen05/atom layout maps several shard iters to each logical axis, so
        # group by the current shape first and permute whole groups. ``group``
        # returns a regrouped layout (degenerate extent-1 iters folded away)
        # plus seps over *that* layout — permute the regrouped one, not
        # ``self.layout``. For a simple layout (one shard iter per axis) this
        # reduces to ``permute_dims(dims)``.
        layout = self.layout
        swizzle = None
        if isinstance(layout, tvm.tirx.layout.ComposeLayout):
            # The swizzle permutes the flat offset, so it commutes with a
            # permutation of the logical dims: permute the inner tile layout
            # and re-compose.
            swizzle = layout.swizzle
            layout = layout.tile_layout
        grouped, seps = layout.group(list(self.shape))
        new_layout = grouped.permute_by_groups(seps, list(dims))
        if swizzle is not None:
            new_layout = tvm.tirx.layout.ComposeLayout(swizzle, new_layout)
        return tvm.tirx.script.builder.decl_buffer(
            new_shape,
            self.dtype,
            self.data,
            self.strides,
            self.elem_offset,
            None,
            self.scope(),
            self.data_alignment,
            self.offset_factor,
            "",
            self.axis_separators,
            new_layout,
        )

    # ------------------------------------------------------------------
    # Dimension-surgery views (torch-aligned): unflatten / flatten /
    # select / narrow, the numpy-style ``sub`` indexer, and einops-style
    # ``rearrange``. All return derived views sharing this buffer's data;
    # the physical placement (layout iters + swizzle) is carried, never
    # restated, and data is never moved: an operation either yields a
    # valid view or raises.
    # ------------------------------------------------------------------

    def _normalized_dim(self, dim, name):
        ndim = len(self.shape)
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise ValueError(f"{name}: dim {dim} out of range for buffer of rank {ndim}")
        return dim

    @staticmethod
    def _concrete_int(value):
        """Return ``value`` as a python int when it is statically known."""
        if isinstance(value, bool):
            return None
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, tvm.tirx.IntImm):
            return int(value)
        return None

    def unflatten(self, dim, sizes) -> "Buffer":
        """Split ``dim`` into the given factor sizes (row-major).

        Torch-aligned: ``x.unflatten(1, (4, 4, 4))``; one size may be ``-1``
        and is inferred from the dim's extent. Pure view: the layout is
        carried and regrouped against the new shape. The split must be exact:
        when the extent and factors are statically known, a non-bijective
        factorization is rejected (symbolic values are the caller's
        responsibility).
        """
        dim = self._normalized_dim(dim, "unflatten")
        sizes = list(sizes)
        extent = self._concrete_int(self.shape[dim])
        negatives = [i for i, s in enumerate(sizes) if isinstance(s, int) and s == -1]
        if len(negatives) > 1:
            raise ValueError("unflatten: at most one -1 is allowed in sizes")
        for i, size in enumerate(sizes):
            if negatives and i == negatives[0]:
                continue
            size_c = self._concrete_int(size)
            if size_c is not None and size_c <= 0:
                raise ValueError(
                    f"unflatten: sizes must be positive (or a single -1); got {size_c}"
                )
        if negatives:
            known = functools.reduce(
                lambda a, b: a * b,
                [s for i, s in enumerate(sizes) if i != negatives[0]],
                1,
            )
            known_c = self._concrete_int(known)
            if extent is not None and known_c is not None:
                if known_c <= 0 or extent % known_c != 0:
                    raise ValueError(
                        f"unflatten: dim {dim} extent {extent} is not divisible "
                        f"by the known factors (product {known_c})"
                    )
                sizes[negatives[0]] = extent // known_c
            else:
                raw_extent = self.shape[dim]
                raw_extent = (
                    int(raw_extent) if isinstance(raw_extent, tvm.tirx.IntImm) else raw_extent
                )
                sizes[negatives[0]] = raw_extent // known
        else:
            product = functools.reduce(lambda a, b: a * b, sizes, 1)
            product_c = self._concrete_int(product)
            if extent is not None and product_c is not None and product_c != extent:
                raise ValueError(
                    f"unflatten: sizes {sizes} multiply to {product_c}, "
                    f"but dim {dim} has extent {extent}"
                )
        new_shape = list(self.shape[:dim]) + sizes + list(self.shape[dim + 1 :])
        return self.view(*new_shape)

    def flatten(self, start_dim=0, end_dim=-1) -> "Buffer":
        """Merge dims ``start_dim..end_dim`` into one (torch-aligned).

        Unlike torch this never copies: the merge is expressed in the layout
        (a logical axis may carry several strided iters), so it either yields
        a view or the downstream consumer rejects the layout loudly.
        """
        start = self._normalized_dim(start_dim, "flatten")
        end = self._normalized_dim(end_dim, "flatten")
        if end < start:
            raise ValueError(f"flatten: end_dim {end} < start_dim {start}")
        merged = functools.reduce(lambda a, b: a * b, list(self.shape[start : end + 1]), 1)
        new_shape = [*self.shape[:start], merged, *self.shape[end + 1 :]]
        return self.view(*new_shape)

    def _surgery_parts(self):
        """Split ``self.layout`` into (inner tile layout, optional swizzle)."""
        layout = self.layout
        swizzle = None
        if isinstance(layout, tvm.tirx.layout.ComposeLayout):
            swizzle = layout.swizzle
            layout = layout.tile_layout
        return layout, swizzle

    @staticmethod
    def _rewrap_swizzle(layout, swizzle):
        if swizzle is not None:
            return tvm.tirx.layout.ComposeLayout(swizzle, layout)
        return layout

    @staticmethod
    def _dim_group_offset(group, index):
        """Offset of ``index`` along a dim made of the given layout iters.

        The dim's iters are outer→inner in row-major coordinate order:
        decompose ``index`` mixed-radix over their extents and weight each
        coordinate by its iter's stride. A single-iter dim reduces to
        ``index * stride``. ``index`` may be a PrimExpr; multi-iter dims
        need concrete inner extents.
        """
        if len(group) == 1:
            return index * group[0].stride
        extents = []
        for it in group[1:]:
            extent = it.extent
            if not isinstance(extent, tvm.tirx.IntImm):
                raise ValueError(
                    "select/narrow: dim with multiple layout iters requires concrete iter extents"
                )
            extents.append(int(extent))
        offset = None
        remaining = index
        for pos, it in enumerate(group):
            inner = functools.reduce(lambda a, b: a * b, extents[pos:], 1)
            if inner == 1:
                coord = remaining
            elif isinstance(remaining, Integral):
                coord = remaining // inner
                remaining = remaining % inner
            else:
                coord = tvm.tirx.floordiv(remaining, inner)
                remaining = tvm.tirx.floormod(remaining, inner)
            term = coord * it.stride
            offset = term if offset is None else offset + term
        return offset

    def _swizzle_offset_commutes(self, swizzle, extra_offset):
        """Whether ``extra_offset`` commutes with the swizzle permutation.

        Folding an offset into ``elem_offset`` moves it *outside* the layout,
        so the address becomes ``offset + swizzle(rest)``. The swizzle XORs
        bits within a period of ``2^(per_element + atom_len + swizzle_len)``
        elements, so this equals the true ``swizzle(offset + rest)`` only
        when the offset is a multiple of that period.
        """
        if swizzle is None or extra_offset is None:
            return True
        sw_len = int(swizzle.swizzle_len)
        if sw_len == 0:
            return True  # identity permutation, everything commutes
        period = 1 << (int(swizzle.per_element) + int(swizzle.atom_len) + sw_len)
        offset_c = self._concrete_int(extra_offset)
        if offset_c is not None:
            return offset_c % period == 0
        from ..arith import Analyzer  # pylint: disable=import-outside-toplevel

        return Analyzer().can_prove_equal(tvm.tirx.floormod(extra_offset, period), 0)

    def _rebuild_view(self, new_shape, new_shard, grouped, swizzle, extra_offset):
        """Rebuild a derived view; ``extra_offset`` goes into ``elem_offset``
        when it commutes with the swizzle (provable period multiple),
        otherwise it stays inside the tile layout's offset so the swizzle
        keeps applying to it and the view addresses the same bytes."""
        offset_map = dict(grouped.offset.items())
        if extra_offset is not None and not self._swizzle_offset_commutes(swizzle, extra_offset):
            m_axis = tvm.tirx.layout.Axis.get("m")
            prev = offset_map.get(m_axis)
            offset_map[m_axis] = extra_offset if prev is None else prev + extra_offset
            extra_offset = None
        new_layout = tvm.tirx.layout.TileLayout.from_iters(
            new_shard, list(grouped.replica), offset_map
        )
        new_layout = self._rewrap_swizzle(new_layout, swizzle)
        elem_offset = self.elem_offset
        if extra_offset is not None:
            elem_offset = elem_offset + extra_offset
        return tvm.tirx.script.builder.decl_buffer(
            new_shape,
            self.dtype,
            self.data,
            self.strides,
            elem_offset,
            None,
            self.scope(),
            self.data_alignment,
            self.offset_factor,
            "",
            self.axis_separators,
            new_layout,
        )

    def select(self, dim, index) -> "Buffer":
        """Return a view with ``dim`` removed, fixed at ``index``.

        Torch-aligned (``Tensor.select``); ``index`` may be a dynamic
        PrimExpr, folded into the view's ``elem_offset`` through the dim's
        layout iters. Statically known indices are bounds-checked; dynamic
        indices are the caller's responsibility. On a swizzled layout the
        offset folds into ``elem_offset`` only when it provably commutes
        with the swizzle (a swizzle-period multiple); otherwise it stays
        inside the derived layout's offset, where the swizzle applies to it.
        """
        dim = self._normalized_dim(dim, "select")
        index_c = self._concrete_int(index)
        extent_c = self._concrete_int(self.shape[dim])
        if index_c is not None:
            if index_c < 0 or (extent_c is not None and index_c >= extent_c):
                raise ValueError(
                    f"select: index {index_c} out of range for dim {dim} "
                    f"of extent {extent_c if extent_c is not None else self.shape[dim]}"
                )
        layout, swizzle = self._surgery_parts()
        grouped, seps = layout.group(list(self.shape))
        iters = list(grouped.shard)
        lo, hi = seps[dim], seps[dim + 1]
        offset = self._dim_group_offset(iters[lo:hi], index)
        new_shape = list(self.shape[:dim]) + list(self.shape[dim + 1 :])
        new_shard = iters[:lo] + iters[hi:]
        return self._rebuild_view(new_shape, new_shard, grouped, swizzle, offset)

    def narrow(self, dim, start, length) -> "Buffer":
        """Return a view of ``dim`` narrowed to ``[start, start + length)``.

        Torch-aligned (``Tensor.narrow``); ``start`` may be a dynamic
        PrimExpr when the dim maps to a single layout iter. For dims made of
        several iters, ``start`` and ``length`` must be concrete multiples of
        the inner iter block so the range stays a contiguous iter prefix.
        Statically known bounds are checked (``start >= 0``, ``length >= 1``,
        ``start + length <= extent``); dynamic values are the caller's
        responsibility. On a swizzled layout the offset folds into
        ``elem_offset`` only when it provably commutes with the swizzle
        (a swizzle-period multiple); otherwise it stays inside the derived
        layout's offset, where the swizzle applies to it.
        """
        dim = self._normalized_dim(dim, "narrow")
        start_c = self._concrete_int(start)
        length_c = self._concrete_int(length)
        extent_c = self._concrete_int(self.shape[dim])
        if start_c is not None and start_c < 0:
            raise ValueError(f"narrow: start {start_c} must be non-negative")
        if length_c is not None and length_c < 1:
            raise ValueError(f"narrow: length {length_c} must be positive")
        if (
            start_c is not None
            and length_c is not None
            and extent_c is not None
            and start_c + length_c > extent_c
        ):
            raise ValueError(
                f"narrow: range [{start_c}, {start_c + length_c}) exceeds "
                f"dim {dim} extent {extent_c}"
            )
        layout, swizzle = self._surgery_parts()
        grouped, seps = layout.group(list(self.shape))
        iters = list(grouped.shard)
        lo, hi = seps[dim], seps[dim + 1]
        group = iters[lo:hi]
        Iter = tvm.tirx.layout.Iter
        if len(group) == 1:
            it = group[0]
            offset = start * it.stride
            new_group = [Iter(length, it.stride, it.axis)]
        else:
            inner = 1
            for it in group[1:]:
                extent = it.extent
                if not isinstance(extent, tvm.tirx.IntImm):
                    raise ValueError(
                        "narrow: dim with multiple layout iters requires concrete iter extents"
                    )
                inner *= int(extent)
            if not (isinstance(start, Integral) and isinstance(length, Integral)):
                raise ValueError(
                    f"narrow: dim {dim} is made of {len(group)} layout iters; "
                    f"start/length must be concrete multiples of {inner}"
                )
            if start % inner != 0 or length % inner != 0:
                raise ValueError(
                    f"narrow: dim {dim} start/length must be multiples of the "
                    f"inner iter block {inner}; got start={start} length={length}"
                )
            outer = group[0]
            offset = (start // inner) * outer.stride
            new_group = [Iter(length // inner, outer.stride, outer.axis), *group[1:]]
        new_shape = list(self.shape)
        new_shape[dim] = length
        new_shard = iters[:lo] + new_group + iters[hi:]
        return self._rebuild_view(new_shape, new_shard, grouped, swizzle, offset)

    @property
    def sub(self) -> "_SubIndexer":
        """Numpy-style view indexer: ``buf.sub[2, 4:8, ::4]``.

        Unlike plain ``buf[...]`` (BufferLoad for scalar indices, extent-1
        BufferRegion dims for tile-primitive operands), ``sub`` follows numpy
        basic-indexing semantics as a *view constructor*: an integer index
        removes the dim (``select``), ``a:b`` narrows it, and ``a::s`` takes
        every s-th element (requires the extent divisible by ``s`` and
        ``a < s``). Trailing dims are kept whole.
        """
        return _SubIndexer(self)

    def rearrange(self, pattern, **sizes) -> "Buffer":
        """Einops-style dimension rearrangement, e.g.
        ``buf.rearrange("b (s w r) c -> b w (s r) c", w=4, r=4)``.

        Pure bijective reshuffling (split / permute / merge) compiled onto
        :meth:`unflatten`, :meth:`permute` and :meth:`flatten`; indexing is
        out of scope (use :meth:`select` / ``sub``). Composite-axis sizes
        come from keyword arguments; at most one factor per composite may be
        omitted and is inferred from the dim extent.
        """
        return _rearrange(self, pattern, **sizes)

    def __getitem__(self, indices):
        from ..arith import Analyzer  # pylint: disable=import-outside-toplevel
        from .expr import BufferLoad, Ramp  # pylint: disable=import-outside-toplevel
        from .stmt import BufferRegion  # pylint: disable=import-outside-toplevel

        if not isinstance(indices, tuple | list):
            indices = [indices]
        has_slice = any(isinstance(i, slice) for i in indices)
        has_step = any(
            isinstance(i, slice) and (i.step is not None and i.step != 1) for i in indices
        )
        has_implicit_slice = len(indices) < len(self.shape)
        analyzer = Analyzer()
        if (has_slice and not has_step) or has_implicit_slice:
            region = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.shape[i] if index.stop is None else index.stop
                    region.append(Range.from_min_extent(start, analyzer.simplify(stop - start)))
                else:
                    region.append(
                        Range.from_min_extent(
                            index,
                            tvm.tirx.expr.IntImm(index.ty, 1) if isinstance(index, PrimExpr) else 1,
                        )
                    )
            if has_implicit_slice:
                for i in range(len(indices), len(self.shape)):
                    region.append(Range.from_min_extent(0, self.shape[i]))
            return BufferRegion(self, region)
        else:
            expr_indices = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.shape[i] if index.stop is None else index.stop
                    step = 1 if index.step is None else index.step
                    # We should ensure the dtype of start is the same with that of step.
                    if isinstance(start, tvm.tirx.expr.PrimExpr) and isinstance(step, int):
                        step = tvm.tirx.expr.IntImm(start.ty, step)
                    lanes = analyzer.simplify((stop - start + step - 1) // step)
                    if lanes == 1:
                        expr_indices.append(start)
                    else:
                        expr_indices.append(Ramp(start, step, int(lanes)))
                else:
                    expr_indices.append(index)
            return BufferLoad(self, expr_indices)


_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _rearrange_parse_side(text: str, pattern: str) -> list[list[str]]:
    """Parse one side of a pattern into a list of axis groups.

    Each top-level entry corresponds to one buffer dim: either a single axis
    name or a parenthesized composite of axis names.
    """
    groups: list[list[str]] = []
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    current: list[str] | None = None
    for token in tokens:
        if token == "(":
            if current is not None:
                raise ValueError(f"rearrange: nested '(' in pattern {pattern!r}")
            current = []
        elif token == ")":
            if current is None:
                raise ValueError(f"rearrange: unmatched ')' in pattern {pattern!r}")
            if not current:
                raise ValueError(f"rearrange: empty '()' group in pattern {pattern!r}")
            groups.append(current)
            current = None
        else:
            if not _NAME_RE.match(token):
                raise ValueError(f"rearrange: bad axis name {token!r} in pattern {pattern!r}")
            if current is not None:
                current.append(token)
            else:
                groups.append([token])
    if current is not None:
        raise ValueError(f"rearrange: unmatched '(' in pattern {pattern!r}")
    if not groups:
        raise ValueError(f"rearrange: empty side in pattern {pattern!r}")
    return groups


def _rearrange_flat_names(groups: list[list[str]], pattern: str) -> list[str]:
    names = [name for group in groups for name in group]
    if len(set(names)) != len(names):
        raise ValueError(f"rearrange: duplicate axis name in pattern {pattern!r}")
    return names


def _rearrange(buffer, pattern, **sizes):
    """Apply an einops-style rearrangement pattern to ``buffer`` as a view."""
    if "->" not in pattern:
        raise ValueError(f"rearrange: pattern {pattern!r} must contain '->'")
    lhs_text, rhs_text = pattern.split("->", 1)
    lhs = _rearrange_parse_side(lhs_text, pattern)
    rhs = _rearrange_parse_side(rhs_text, pattern)
    lhs_names = _rearrange_flat_names(lhs, pattern)
    rhs_names = _rearrange_flat_names(rhs, pattern)
    if set(lhs_names) != set(rhs_names):
        missing = set(lhs_names) ^ set(rhs_names)
        raise ValueError(
            f"rearrange: axes {sorted(missing)} appear on only one side of {pattern!r}"
        )
    unknown = set(sizes) - set(lhs_names)
    if unknown:
        raise ValueError(f"rearrange: sizes for unknown axes {sorted(unknown)} in {pattern!r}")
    if len(lhs) != len(buffer.shape):
        raise ValueError(
            f"rearrange: pattern {pattern!r} has {len(lhs)} input dims, "
            f"buffer has rank {len(buffer.shape)}"
        )

    # Resolve every axis size. Non-composite axes take the dim extent;
    # composite factors come from ``sizes`` with at most one inferred.
    axis_size: dict[str, object] = dict(sizes)
    for dim, group in enumerate(lhs):
        extent = buffer.shape[dim]
        extent = int(extent) if isinstance(extent, tvm.tirx.IntImm) else extent
        if len(group) == 1:
            name = group[0]
            if name not in axis_size:
                axis_size[name] = extent
            else:
                given_c = Buffer._concrete_int(axis_size[name])
                extent_c = Buffer._concrete_int(extent)
                if given_c is not None and extent_c is not None and given_c != extent_c:
                    raise ValueError(
                        f"rearrange: size {name}={given_c} does not match dim {dim} "
                        f"extent {extent_c} in {pattern!r}"
                    )
            continue
        unknown_factors = [name for name in group if name not in axis_size]
        if len(unknown_factors) > 1:
            raise ValueError(
                f"rearrange: composite {'(' + ' '.join(group) + ')'} in {pattern!r} "
                f"has multiple unknown factors {unknown_factors}; pass their sizes"
            )
        if unknown_factors:
            known = 1
            for name in group:
                if name != unknown_factors[0]:
                    known = known * axis_size[name]
            known_c = Buffer._concrete_int(known)
            extent_c = Buffer._concrete_int(extent)
            if (
                extent_c is not None
                and known_c is not None
                and (known_c <= 0 or extent_c % known_c != 0)
            ):
                raise ValueError(
                    f"rearrange: composite {'(' + ' '.join(group) + ')'} in "
                    f"{pattern!r} does not factor dim extent {extent_c} "
                    f"(known factors multiply to {known_c})"
                )
            axis_size[unknown_factors[0]] = extent // known

    # 1. Split composites, right-to-left so dim indices stay valid.
    result = buffer
    for dim in range(len(lhs) - 1, -1, -1):
        group = lhs[dim]
        if len(group) > 1:
            result = result.unflatten(dim, tuple(axis_size[name] for name in group))

    # 2. Permute flat axes into the output order.
    perm = [lhs_names.index(name) for name in rhs_names]
    if perm != list(range(len(perm))):
        result = result.permute(*perm)

    # 3. Merge output composites, right-to-left over flat positions.
    position = len(rhs_names)
    for group in reversed(rhs):
        position -= len(group)
        if len(group) > 1:
            result = result.flatten(position, position + len(group) - 1)
    return result


class _SubIndexer:
    """Indexer object returned by :attr:`Buffer.sub`.

    Translates numpy basic indexing into a chain of
    :meth:`Buffer.select` / :meth:`Buffer.narrow` /
    :meth:`Buffer.unflatten` view operations.
    """

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
                    buf = buf.narrow(dim, start, stop - start)
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
                    # a::s over N = unflatten into (N//s, s) and fix the
                    # remainder coordinate at a (requires 0 <= a < s).
                    start_c = Buffer._concrete_int(start)
                    if start_c is not None and not 0 <= start_c < step:
                        raise ValueError(
                            f"sub: stepped-slice start {start_c} must be in [0, {step})"
                        )
                    buf = buf.unflatten(dim, (extent // step, step)).select(dim + 1, start)
                    dim += 1
            else:
                buf = buf.select(dim, item)
        return buf


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
    buffer_type="",
    axis_separators=None,
    span=None,
    layout="default",
):
    # pylint: disable=import-outside-toplevel
    from .expr import Var
    from .layout import S, TileLayout

    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    if axis_separators is None:
        axis_separators = []

    if layout == "default":
        layout = TileLayout(S[tuple(shape)]) if shape else None

    if offset_factor != 0 and elem_offset is None:
        shape_ty = shape[0].ty if shape and isinstance(shape[0], PrimExpr) else "int32"
        elem_offset = Var(f"{name}_elem_offset", shape_ty)
    if data is None:
        # Bool is represented as uint1 in the IR, but stored as int8
        storage_type = dtype if isinstance(dtype, PrimType) else PrimType(dtype)
        storage_type = PrimType("int8") if storage_type.dtype == "bool" else storage_type
        data = Var(name, PointerType(storage_type, scope), span)
    return _ffi_api.Buffer(  # type: ignore
        data,
        dtype,
        shape,
        strides,
        elem_offset,
        name,
        data_alignment,
        offset_factor,
        buffer_type,
        axis_separators,
        span,
        layout,
    )


@tvm_ffi.register_object("tirx.DataProducer")
class DataProducer(Object):
    pass
