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
"""Unique operator"""
import tvm
from tvm import te, tir
from ...te import hybrid
from .scan import cumsum
from .sort import sort, argsort
from ..utils import ceil_div


def _get_max_threads(batch_size):
    target = tvm.target.Target.current()
    max_threads = tvm.target.Target.current(allow_none=False).max_num_threads
    if "vulkan" in str(target) and not isinstance(batch_size, tvm.tir.IntImm):
        # SPIR-V does not support dynamic thread group size
        return max_threads
    return tir.min(batch_size, max_threads)


def _calc_adjacent_diff_ir(data, output, binop=tir.Sub):
    """Low level IR to calculate adjacent difference in an 1-D array.

    Parameters
    ----------
    data : Buffer
        Input 1-D Buffer.

    output: Buffer
        A buffer to store adjacent difference, of the same shape as data. The adjacent difference
        is defined as: output[0] = 0, output[i] = binop(data[i], data[i-1])
        where i > 0 and i < len(data).

    binop: function, optional
        A binary associative op to use for calculating adjacent difference. The function takes two
        TIR expressions and produce a new TIR expression. By default it uses tvm.tir.Sub to
        compute the adjacent difference.
    """
    ib = tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    output_ptr = ib.buffer_ptr(output)
    batch_size = data.shape[0]
    max_threads = _get_max_threads(batch_size)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            with ib.if_scope(tid == 0):
                output_ptr[tid] = 0
            with ib.else_scope():
                output_ptr[tid] = tir.Cast(output.dtype, binop(data_ptr[tid], data_ptr[tid - 1]))
    return ib.get()


def _calc_adjacent_diff(data, out_dtype="int32", binop=tir.Sub):
    """Function calculate adjacent difference in an 1-D array.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input 1-D tensor.

    output_dtype : str
        The output tensor data type.

    binop: function, optional
        A binary associative op to use for calculating difference. The function takes two
        TIR expressions and produce a new TIR expression. By default it uses tvm.tir.Sub to
        compute the adjacent difference.

    Returns
    -------
    output : tvm.te.Tensor
        1-D tensor storing the adjacent difference of the input tensor. The adjacent difference
        is defined as: output[0] = 0, output[i] = binop(data[i], data[i-1])
        where i > 0 and i < len(data).
    """
    data_buf = tir.decl_buffer(data.shape, data.dtype, "sorted_data_buf", data_alignment=8)
    output_buf = tir.decl_buffer(data.shape, out_dtype, "output_buf", data_alignment=8)
    return te.extern(
        [data.shape],
        [data],
        lambda ins, outs: _calc_adjacent_diff_ir(ins[0], outs[0], binop=binop),
        dtype=[out_dtype],
        in_buffers=[data_buf],
        out_buffers=[output_buf],
        name="_calc_adjacent_diff",
        tag="_calc_adjacent_diff_gpu",
    )


@hybrid.script
def _calc_num_unique(inc_scan):
    """Helper function to get the number of unique elements fron inc_scan tensor"""
    output = output_tensor((1,), "int32")
    for i in bind("threadIdx.x", 1):
        output[i] = inc_scan[inc_scan.shape[0] - 1] + int32(1)
    return output


def _calc_unique_ir(
    data, argsorted_indices, inc_scan, index_converter, unique_elements, inverse_indices, counts
):
    """Low level IR to calculate unique elements, inverse indices, and counts (optional) of
    unique elements of 1-D array.

    Parameters
    ----------
    data : Buffer
        Input 1-D Buffer.

    argsorted_indices : Buffer
        A buffer that stores the argsorted indices of the input data.

    inc_scan : Buffer
        A buffer that stores the inclusive scan of the binary tir.NE adjacent difference
        of the sorted data.

    index_converter (optional) : Buffer
        An optional index converter that transforms the unique element index
        such that new_idx = index_converter[old_idx].

    unique_elements : Buffer
        A buffer that stores the unique elements.

    inverse_indices : Buffer
        A buffer that stores the index of each input data element in the unique element array.

    counts (optional) : Buffer
        A buffer that stores the count of each unique element.
    """
    ib = tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    unique_elements_ptr = ib.buffer_ptr(unique_elements)
    inverse_indices_ptr = ib.buffer_ptr(inverse_indices)

    index_converter_ptr = None
    if isinstance(index_converter, tir.Buffer):
        index_converter_ptr = ib.buffer_ptr(index_converter)

    if isinstance(counts, tir.Buffer):
        counts_ptr = ib.buffer_ptr(counts)
        # use indices_ptr as a tmp buffer to store tids with inc_scan[tid] != inc_scan[tid-1]
        unique_seq_indices_ptr = ib.buffer_ptr(inverse_indices)

    batch_size = data.shape[0]
    max_threads = _get_max_threads(batch_size)

    # if need to return counts
    if isinstance(counts, tir.Buffer):
        num_unique = inc_scan_ptr[inc_scan.shape[0] - 1] + 1
        num_elements = data.shape[0]
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = ceil_div(batch_size, max_threads)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            tid = bx * max_threads + tx
            with ib.if_scope(tid < batch_size):
                with ib.if_scope(tid == 0):
                    unique_seq_indices_ptr[num_unique - 1] = num_elements
                with ib.else_scope():
                    with ib.if_scope(inc_scan_ptr[tid] != inc_scan_ptr[tid - 1]):
                        unique_seq_indices_ptr[inc_scan_ptr[tid] - 1] = tid
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = ceil_div(batch_size, max_threads)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            tid = bx * max_threads + tx
            with ib.if_scope(tid < num_unique):
                unique_idx = tid if not index_converter_ptr else index_converter_ptr[tid]
                with ib.if_scope(tid == 0):
                    counts_ptr[unique_idx] = unique_seq_indices_ptr[tid]
                with ib.else_scope():
                    counts_ptr[unique_idx] = (
                        unique_seq_indices_ptr[tid] - unique_seq_indices_ptr[tid - 1]
                    )
    # calculate unique elements and inverse indices
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            data_idx = argsorted_indices_ptr[tid]
            unique_idx = (
                inc_scan_ptr[tid]
                if not index_converter_ptr
                else index_converter_ptr[inc_scan_ptr[tid]]
            )
            inverse_indices_ptr[data_idx] = unique_idx
            with ib.if_scope(tid == 0):
                unique_elements_ptr[unique_idx] = data_ptr[data_idx]
            with ib.else_scope():
                with ib.if_scope(inc_scan_ptr[tid] != inc_scan_ptr[tid - 1]):
                    unique_elements_ptr[unique_idx] = data_ptr[data_idx]
    return ib.get()


def _calc_first_occurence_ir(argsorted_indices, inc_scan, first_occurence):
    """Low level IR to calculate the first occurence of each unique element in the input data.

    Parameters
    ----------
    argsorted_indices : Buffer
        A buffer that stores the argsorted indices of the input data.

    inc_scan : Buffer
        A buffer that stores the inclusive scan of the binary tir.NE adjacent difference
        of the sorted data.

    first_occurence : Buffer
        A buffer that stores the first occurence of each unique element in the input data.
    """
    ib = tir.ir_builder.create()
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    first_occurence_ptr = ib.buffer_ptr(first_occurence)
    batch_size = argsorted_indices.shape[0]
    max_threads = _get_max_threads(batch_size)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            first_occurence_ptr[tid] = batch_size
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            with ib.if_scope(tid == 0):
                first_occurence_ptr[inc_scan_ptr[tid]] = argsorted_indices_ptr[tid]
            with ib.else_scope():
                with ib.if_scope(inc_scan_ptr[tid] != inc_scan_ptr[tid - 1]):
                    first_occurence_ptr[inc_scan_ptr[tid]] = argsorted_indices_ptr[tid]
    return ib.get()


def unique(data, is_sorted=True, return_counts=False):
    """
    Find the unique elements of a 1-D tensor. Please note `output` and `counts` are all padded to
    have the same length of `data` and element with index >= num_unique[0] has undefined value.

    Parameters
    ----------
    data : tvm.te.Tensor
        A 1-D tensor of integers.

    sorted : bool
        Whether to sort the unique elements in ascending order before returning as output.

    return_counts : bool
        Whether to return the count of each unique element.

    Returns
    -------
    unique : tvm.te.Tensor
        A 1-D tensor containing the unique elements of the input data tensor. The same size as
        the input data. If there are less unique elements than input data, the end of the tensor
        is padded with zeros.

    indices : tvm.te.Tensor
        A 1-D tensor. The same size as output. For each entry in output, it contains
        the index of its first occurence in the input data. The end of the tensor is padded
        with the length of the input data.

    inverse_indices : tvm.te.Tensor
        A 1-D tensor. For each entry in data, it contains the index of that data element in the
        unique array. (Note that inverse_indices is very similar to indices if output is not
        sorted)

    num_unique : tvm.te.Tensor
        A 1-D tensor with size=1 containing the number of unique elements in the input data tensor.

    counts (optional) : tvm.te.Tensor
        A 1-D tensor containing the count of each unique element in the output.

    Examples
    --------
    .. code-block:: python

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, False)
        output          =  [4, 5, 1, 2, 3, ?, ?, ?]
        indices         =  [0, 1, 2, 3, 4, ?, ?, ?]
        inverse_indices =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique      =  [5]

        [output, indices, num_unique, counts] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, True)
        output          =  [4, 5, 1, 2, 3, ?, ?, ?]
        indices         =  [0, 1, 2, 3, 4, ?, ?, ?]
        inverse_indices =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique      =  [5]
        counts          =  [2, 2, 1, 1, 2, ?, ?, ?]

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], True)
        output          =  [1, 2, 3, 4, 5, ?, ?, ?]
        indices         =  [2, 3, 4, 0, 1, ?, ?, ?]
        inverse_indices =  [3, 4, 0, 1, 2, 2, 3, 4]
        num_unique      =  [5]
    """
    sorted_data = sort(data)
    argsorted_indices = argsort(data, dtype="int32")
    # adjacent difference
    adjacent_diff = _calc_adjacent_diff(sorted_data, out_dtype="int32", binop=tir.NE)
    # inclusive scan
    inc_scan = cumsum(adjacent_diff, dtype="int32", exclusive=0)
    # total number of unique elements
    num_unique_elements = _calc_num_unique(inc_scan)
    # buffers
    data_buf = tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    argsorted_indices_buf = tir.decl_buffer(
        data.shape, "int32", "argsorted_indices_buf", data_alignment=8
    )
    inc_scan_buf = tvm.tir.decl_buffer(data.shape, "int32", "inc_scan_buf", data_alignment=8)
    unique_elements_buf = tir.decl_buffer(
        data.shape, data.dtype, "unique_elements_buf", data_alignment=8
    )
    inverse_indices_buf = tvm.tir.decl_buffer(
        data.shape, "int32", "inverse_indices_buf", data_alignment=8
    )
    # prepare outputs
    if return_counts:
        counts_buf = tir.decl_buffer(data.shape, "int32", "counts_buf", data_alignment=8)
        out_data_shape = [data.shape] * 3
        out_buffers = [unique_elements_buf, inverse_indices_buf, counts_buf]
        out_dtypes = [data.dtype, "int32", "int32"]
    else:
        out_data_shape = [data.shape] * 2
        out_buffers = [unique_elements_buf, inverse_indices_buf]
        out_dtypes = [data.dtype, "int32"]
    # prepare inputs and fcompute
    # calculate first occurence
    first_occurence_buf = tir.decl_buffer(
        data.shape, "int32", "first_occurence_buf", data_alignment=8
    )
    first_occurence = te.extern(
        [data.shape],
        [argsorted_indices, inc_scan],
        lambda ins, outs: _calc_first_occurence_ir(ins[0], ins[1], outs[0]),
        dtype=["int32"],
        in_buffers=[argsorted_indices_buf, inc_scan_buf],
        out_buffers=[first_occurence_buf],
        name="_calc_first_occurence",
        tag="_calc_first_occurence_gpu",
    )
    if is_sorted:
        in_data = [data, argsorted_indices, inc_scan]
        in_buffers = [data_buf, argsorted_indices_buf, inc_scan_buf]
        if return_counts:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, None, *outs)
        else:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, None, *outs, None)
        indices = first_occurence
    else:
        # calculate index converter by sorting unique elements by their first occurence
        argsorted_first_occurence = argsort(first_occurence, dtype="int32")
        index_converter = argsort(argsorted_first_occurence, dtype="int32")
        index_converter_buf = tir.decl_buffer(
            data.shape, "int32", "index_converter_buf", data_alignment=8
        )
        in_data = [data, argsorted_indices, inc_scan, index_converter]
        in_buffers = [data_buf, argsorted_indices_buf, inc_scan_buf, index_converter_buf]
        if return_counts:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, *outs)
        else:
            fcompute = lambda ins, outs: _calc_unique_ir(*ins, *outs, None)
        indices = sort(first_occurence)
    outs = te.extern(
        out_data_shape,
        in_data,
        fcompute,
        dtype=out_dtypes,
        in_buffers=in_buffers,
        out_buffers=out_buffers,
        name="_calc_unique",
        tag="_calc_unique_gpu",
    )
    if return_counts:
        return [outs[0], indices, outs[1], num_unique_elements, outs[2]]
    return [outs[0], indices, outs[1], num_unique_elements]
