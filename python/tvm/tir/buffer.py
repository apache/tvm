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
from numbers import Integral

import tvm._ffi
from tvm._ffi.base import string_types
from tvm.ir import PointerType, PrimExpr, PrimType, Range
from tvm.runtime import Object, Scriptable, convert

from . import _ffi_api


@tvm._ffi.register_object("tir.Buffer")
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
        if isinstance(access_mask, string_types):
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
            self, access_mask, ptr_type, content_lanes, offset, extent  # type: ignore
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
        begin = (begin,) if isinstance(begin, (int, PrimExpr)) else begin
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
        begin = (begin,) if isinstance(begin, (int, PrimExpr)) else begin
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

    def __getitem__(self, indices):
        from ..arith import Analyzer  # pylint: disable=import-outside-toplevel
        from .expr import BufferLoad, Ramp, const  # pylint: disable=import-outside-toplevel
        from .stmt import BufferRegion  # pylint: disable=import-outside-toplevel

        if not isinstance(indices, (tuple, list)):
            indices = [indices]
        has_slice = any(isinstance(i, slice) for i in indices)
        has_step = any(isinstance(i, slice) and i.step is not None for i in indices)
        analyzer = Analyzer()
        if has_slice and not has_step:
            region = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.shape[i] if index.stop is None else index.stop
                    region.append(Range.from_min_extent(start, analyzer.simplify(stop - start)))
                else:
                    region.append(
                        Range.from_min_extent(
                            index, const(1, index.dtype) if isinstance(index, PrimExpr) else 1
                        )
                    )
            return BufferRegion(self, region)
        else:
            expr_indices = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.shape[i] if index.stop is None else index.stop
                    step = 1 if index.step is None else index.step
                    # We should ensure the dtype of start is the same with that of step.
                    if isinstance(start, tvm.tir.expr.PrimExpr) and isinstance(step, int):
                        step = tvm.tir.expr.IntImm(start.dtype, step)
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
    buffer_type="",
    axis_separators=None,
    span=None,
):
    """Declare a new symbolic buffer.

    Normally buffer is created automatically during lower and build.
    This is only needed if user want to specify their own buffer layout.

    See the note below for detailed discussion on usage of buffer.

    Parameters
    ----------
    shape : tuple of Expr
        The shape of the buffer.

    dtype : str, optional
        The data type of the buffer.

    name : str, optional
        The name of the buffer.

    data : tir.Var, optional
        The data pointer in the buffer.

    strides: array of Expr
        The stride of the buffer.

    elem_offset: Expr, optional
        The beginning offset of the array to data.
        In terms of number of elements of dtype.

    scope: str, optional
        The storage scope of the buffer, if not global.
        If scope equals empty string, it means it is global memory.

    data_alignment: int, optional
        The alignment of data pointer in bytes.
        If -1 is passed, the alignment will be set to TVM's internal default.

    offset_factor: int, optional
        The factor of elem_offset field, when set,
        elem_offset is required to be multiple of offset_factor.
        If 0 is pssed, the alignment will be set to 1.
        if non-zero is passed, we will created a Var for elem_offset if elem_offset is not None.

    buffer_type: str, optional, {"", "auto_broadcast"}
        auto_broadcast buffer allows one to implement broadcast computation
        without considering whether dimension size equals to one.
        TVM maps buffer[i][j][k] -> buffer[i][0][k] if dimension j's shape equals 1.

    axis_separators : list of int, optional
        If passed, a list of separators between groups of axes,
        each of which is flattened to an output axis.  For flat
        memory spaces, should either be None, or an empty list.

    span: Optional[Span]
        The location of the decl_buffer creation in the source.

    Returns
    -------
    buffer : tvm.tir.Buffer
        The created buffer

    Example
    -------
    Here's an example of how broadcast buffer can be used to define a symbolic broadcast operation,

    .. code-block:: python

        m0, m1, m2 = te.var("m0"), te.var("m1"), te.var("m2")
        n0, n1, n2 = te.var("n0"), te.var("n1"), te.var("n2")
        o0, o1, o2 = te.var("o0"), te.var("o1"), te.var("o2")
        A = te.placeholder((m0, m1, m2), name='A')
        B = te.placeholder((n0, n1, n2), name='B')
        C = te.compute((o0, o1, o2), lambda i, j, k: A[i, j, k] + B[i, j, k], name='C')
        Ab = tvm.tir.decl_buffer(A.shape, A.dtype, name="Ab", buffer_type="auto_broadcast")
        Bb = tvm.tir.decl_buffer(B.shape, B.dtype, name="Bb", buffer_type="auto_broadcast")
        s = te.create_schedule(C.op)
        fadd = tvm.build(s, [A, B, C], target='llvm', name='bcast_add', binds={A:Ab, B:Bb})
        dev = tvm.cpu(0)
        a = tvm.nd.array(np.random.uniform(size=(2, 4, 3)).astype(A.dtype), dev)
        b = tvm.nd.array(np.random.uniform(size=(2, 1, 3)).astype(B.dtype), dev)
        c = tvm.nd.array(np.zeros((2, 4, 3), dtype=C.dtype), dev)
        fadd(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    Note
    ----
    Buffer data structure reflects the DLTensor structure in dlpack.
    While DLTensor data structure is very general, it is usually helpful
    to create function that only handles specific case of data structure
    and make compiled function benefit from it.

    If user pass strides and elem_offset is passed as None
    when constructing the function, then the function will be specialized
    for the DLTensor that is compact and aligned.
    If user pass a fully generic symbolic array to the strides,
    then the resulting function becomes fully generic.
    """
    # pylint: disable=import-outside-toplevel
    from .expr import Var

    shape = (shape,) if isinstance(shape, (PrimExpr, Integral)) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    if axis_separators is None:
        axis_separators = []

    if offset_factor != 0 and elem_offset is None:
        shape_dtype = shape[0].dtype if shape and hasattr(shape[0], "dtype") else "int32"
        elem_offset = Var(f"{name}_elem_offset", shape_dtype)
    if data is None:
        # Bool is represented as uint1 in the IR, but stored as int8
        storage_type = PrimType(dtype)
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
    )


@tvm._ffi.register_object("tir.DataProducer")
class DataProducer(Object):
    pass
