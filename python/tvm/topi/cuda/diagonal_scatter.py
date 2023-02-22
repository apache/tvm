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
# pylint: disable=invalid-name, unused-argument
"""Diagonal scatter operator """
import tvm
from tvm import te, tir
from ..utils import ceil_div, get_const_int


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

    def gen_diagonal_scatter_2d(data, src, out, offset, dim1, dim2):
        ib = tir.ir_builder.create()

        data_ptr = ib.buffer_ptr(data)
        src_ptr = ib.buffer_ptr(src)
        out_ptr = ib.buffer_ptr(out)

        # Prepare ranges and strides
        shape = data.shape

        data_stride = shape[1]
        diag_stride = data_stride + 1
        data_range = shape[0] * data_stride

        src_range = src.shape[0]

        base_offset = 0
        with ib.if_scope(offset >= 0):
            base_offset += offset
        with ib.else_scope():
            base_offset -= offset * data_stride

        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        # Copy initial input data to output
        with ib.new_scope():
            num_blocks = ceil_div(data_range, max_threads)
            bx = te.thread_axis("blockIdx.x")
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(bx, "thread_extent", num_blocks)
            ib.scope_attr(tx, "thread_extent", max_threads)

            index = bx * max_threads + tx
            with ib.if_scope(index < data_range):
                out_ptr[index] = data_ptr[index]

        with ib.new_scope():
            num_blocks_1 = ceil_div(src_range, max_threads)
            bx1 = te.thread_axis("blockIdx.x")
            tx1 = te.thread_axis("threadIdx.x")
            ib.scope_attr(bx1, "thread_extent", num_blocks_1)
            ib.scope_attr(tx1, "thread_extent", max_threads)

            ind_fused = bx1 * max_threads + tx1
            with ib.if_scope(ind_fused < src_range):
                out_index = ind_fused * diag_stride + base_offset
                out_ptr[out_index] = src_ptr[ind_fused]

        return ib.get()

    def gen_ir(data, src, out, offset, dim1, dim2):
        ib = tir.ir_builder.create()

        data_ptr = ib.buffer_ptr(data)
        src_ptr = ib.buffer_ptr(src)
        out_ptr = ib.buffer_ptr(out)

        # Prepare ranges and strides
        shape = data.shape

        axis1_range = shape[dim1]
        axis2_range = shape[dim2]
        data_range = 1
        mid_tail_stride = 1
        stride2 = 1
        for i, value in enumerate(shape, 0):
            data_range *= value
            if i > dim1 and i != dim2:
                mid_tail_stride *= value
            if i > dim2:
                stride2 *= value
        stride1 = axis2_range * mid_tail_stride
        istride = axis1_range * stride1
        jstride = axis2_range * stride2
        mstride = stride1 + stride2

        src_shape = src.shape
        src_rank = len(src_shape)
        src_range_wo_diag = 1
        src_diag_len = 1
        for i, value in enumerate(src.shape, 0):
            if i != (src_rank - 1):
                src_range_wo_diag *= value
            else:
                src_diag_len = value

        base_offset = 0
        if offset >= 0:
            base_offset += offset * stride2
        else:
            base_offset -= offset * stride1

        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        # Copy initial input data to output
        with ib.new_scope():
            num_blocks = ceil_div(data_range, max_threads)
            bx = te.thread_axis("blockIdx.x")
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(bx, "thread_extent", num_blocks)
            ib.scope_attr(tx, "thread_extent", max_threads)

            index = bx * max_threads + tx
            with ib.if_scope(index < data_range):
                out_ptr[index] = data_ptr[index]

        with ib.new_scope():
            num_blocks_1 = ceil_div(src_range_wo_diag, max_threads)
            bx1 = te.thread_axis("blockIdx.x")
            tx1 = te.thread_axis("threadIdx.x")
            ib.scope_attr(bx1, "thread_extent", num_blocks_1)
            ib.scope_attr(tx1, "thread_extent", max_threads)

            index1 = bx1 * max_threads + tx1
            with ib.if_scope(index1 < src_range_wo_diag):
                i = index1 // mid_tail_stride
                j = index1 // stride2
                k = index1 % stride2
                out_preindex = base_offset + i * istride + j * jstride + k
                src_preindex = index1 * src_diag_len
                with ib.for_range(0, src_diag_len, "m") as m:
                    out_ptr[out_preindex + m * mstride] = src_ptr[src_preindex + m]

        return ib.get()

    if not isinstance(offset, int):
        offset = get_const_int(offset)
    if not isinstance(dim1, int):
        dim1 = get_const_int(dim1)
    if not isinstance(dim2, int):
        dim2 = get_const_int(dim2)

    shape = data.shape
    rank = len(shape)
    assert rank > 1, "Multidimensional input tensor is expected (rank>=2)"
    assert 0 <= dim1 < rank, "First given dimension is out of bounds"
    assert 0 <= dim2 < rank, "Second given dimension is out of bounds"
    # Check some statements for using by gen_ir without check
    assert dim1 < dim2, "First dimension less than second one is supported only"

    gen_diagonal_scatter_ir = None
    if rank == 2:
        gen_diagonal_scatter_ir = gen_diagonal_scatter_2d
    else:
        gen_diagonal_scatter_ir = gen_ir

    out_buf = tir.decl_buffer(data.shape, data.dtype, "out_buf")
    return te.extern(
        [data.shape],
        [data, src],
        lambda ins, outs: gen_diagonal_scatter_ir(ins[0], ins[1], outs[0], offset, dim1, dim2),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="diagonal_scatter_cuda",
        tag="diagonal_scatter_cuda",
    )
