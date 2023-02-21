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
# pylint: disable=invalid-name
"""Diagonal scatter operator """
from tvm import te, tir
from . import utils


def diagonal_scatter(data, src, offset=0, dim1=0, dim2=1):
    """Diagonal scatter embeds the values of the src tensor into data along its diagonal
    with respect to dim1 and dim2. It can be lateral diagonal instead of main one.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input data to the operator.

    src : tvm.te.Tensor
        The  tensor to embed into data.

    offset : int, optional
        Which diagonal to consider.
        If offset > 0, it is above the main diagonal.
        If offset < 0, it is below the main diagonal.
        Default: 0 (main diagonal).

    dim1 : int, optional
        First dimension with respect to which to take diagonal. Default: 0.

    dim2 : int, optional
        Second dimension with respect to which to take diagonal. Default: 1.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    # Prepare ranges and strides
    if not isinstance(offset, int):
        offset = utils.get_const_int(offset)
    if not isinstance(dim1, int):
        dim1 = utils.get_const_int(dim1)
    if not isinstance(dim2, int):
        dim2 = utils.get_const_int(dim2)

    shape = data.shape
    rank = len(shape)
    assert rank > 1, "Multidimensional input tensor is expected (rank>=2)"
    assert 0 <= dim1 < rank, "First given dimension is out of bounds"
    assert 0 <= dim2 < rank, "Second given dimension is out of bounds"
    # Check some statements for using by gen_ir without check
    assert dim1 != dim2, "Given dimensions should not be the same"
    assert shape[dim1] == shape[dim2], "The slice for diagonal is assumed square"
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1

    # Prepare ranges and strides
    axis_range = shape[dim1]
    cut_data_range = 1
    mid_tail_stride = 1
    stride2 = 1
    for i, value in enumerate(shape, 0):
        if i not in (dim1, dim2):
            cut_data_range *= value
        if i > dim1 and i != dim2:
            mid_tail_stride *= value
        if i > dim2:
            stride2 *= value
    stride1 = axis_range * mid_tail_stride
    istride = axis_range * stride1
    jstride = axis_range * stride2
    mstride = stride1 + stride2
    data_range = cut_data_range * axis_range * axis_range

    src_range = 1
    for i, value in enumerate(src.shape, 0):
        src_range *= value

    base_offset = 0
    if offset >= 0:
        base_offset += offset * stride2
    else:
        base_offset -= offset * stride1

    def gen_diagonal_scatter_2d(data, src, out):
        ib = tir.ir_builder.create()

        data_ptr = ib.buffer_ptr(data)
        src_ptr = ib.buffer_ptr(src)
        out_ptr = ib.buffer_ptr(out)

        # Copy initial input data to output
        with ib.for_range(0, data_range, "i", kind="parallel") as i:
            out_ptr[i] = data_ptr[i]

        with ib.for_range(0, src_range, "j", kind="parallel") as j:
            out_index = j * mstride + base_offset
            out_ptr[out_index] = src_ptr[j]

        return ib.get()

    def gen_ir(data, src, out):
        ib = tir.ir_builder.create()

        data_ptr = ib.buffer_ptr(data)
        src_ptr = ib.buffer_ptr(src)
        out_ptr = ib.buffer_ptr(out)

        # Copy initial input data to output
        with ib.for_range(0, data_range, "i", kind="parallel") as i:
            out_ptr[i] = data_ptr[i]

        with ib.for_range(0, cut_data_range, "fused", kind="parallel") as fused:
            i = fused // mid_tail_stride
            j = fused // stride2
            k = fused % stride2
            out_preindex = base_offset + i * istride + j * jstride + k
            with ib.for_range(0, src_range, "m") as m:
                out_ptr[out_preindex + m * mstride] = src_ptr[m]

        return ib.get()

    gen_diagonal_scatter_ir = None
    if rank == 2:
        gen_diagonal_scatter_ir = gen_diagonal_scatter_2d
    else:
        gen_diagonal_scatter_ir = gen_ir

    out_buf = tir.decl_buffer(data.shape, data.dtype, "out_buf")
    return te.extern(
        [data.shape],
        [data, src],
        lambda ins, outs: gen_diagonal_scatter_ir(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="diagonal_scatter.generic",
        tag="diagonal_scatter.generic",
    )
