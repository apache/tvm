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
# pylint: disable=invalid-name, no-else-return
"""Unique operator"""
from tvm import te, tir
import tvm

from ...te import hybrid
from .scan import cumsum
from .sort import sort, argsort
from ..utils import ceil_div


def _calc_adjacent_diff_ir(data, adjacent_diff):
    ib = tvm.tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    adjacent_diff_ptr = ib.buffer_ptr(adjacent_diff)
    batch_size = data.shape[0]
    max_threads = tir.min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
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
                adjacent_diff_ptr[tid] = 0
            with ib.else_scope():
                with ib.if_scope(data_ptr[tid] != data_ptr[tid - 1]):
                    adjacent_diff_ptr[tid] = 1
                with ib.else_scope():
                    adjacent_diff_ptr[tid] = 0
    return ib.get()


@hybrid.script
def _calc_num_unique(data):
    output = output_tensor((1,), "int32")
    for i in bind("threadIdx.x", 1):
        output[i] = data[data.shape[0] - 1] + int32(1)
    return output


def _calc_unique_sorted_ir(data, argsorted_indices, inc_scan, unique_elements, indices):
    ib = tvm.tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    unique_elements_ptr = ib.buffer_ptr(unique_elements)
    indices_ptr = ib.buffer_ptr(indices)

    batch_size = data.shape[0]
    max_threads = tir.min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            indices_ptr[argsorted_indices_ptr[tid]] = inc_scan_ptr[tid]
            with ib.if_scope(tid == 0):
                unique_elements_ptr[inc_scan_ptr[tid]] = data_ptr[argsorted_indices_ptr[tid]]
            with ib.else_scope():
                with ib.if_scope(inc_scan_ptr[tid] != inc_scan_ptr[tid - 1]):
                    unique_elements_ptr[inc_scan_ptr[tid]] = data_ptr[argsorted_indices_ptr[tid]]
    return ib.get()


def _calc_counts_sorted_ir(inc_scan, counts):
    ib = tvm.tir.ir_builder.create()
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    counts_ptr = ib.buffer_ptr(counts)

    batch_size = inc_scan.shape[0]
    max_threads = tir.min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            counts_ptr[tid] = 0
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        atomic_add_return = ib.allocate(counts.dtype, (1,), name="atomic_add_return", scope="local")
        with ib.if_scope(tid < batch_size):
            index = inc_scan_ptr[tid]
            atomic_add_return[0] = tvm.tir.call_intrin(
                counts.dtype,
                "tir.atomic_add",
                tvm.tir.call_intrin("handle", "tir.address_of", counts_ptr[index]),
                1,
            )
    return ib.get()


def _calc_first_occurence_ir(argsorted_indices, inc_scan, first_occurence):
    ib = tvm.tir.ir_builder.create()
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    first_occurence_ptr = ib.buffer_ptr(first_occurence)
    batch_size = argsorted_indices.shape[0]
    max_threads = tir.min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
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


def _calc_unique_unsorted_ir(
    data, argsorted_indices, inc_scan, index_converter, unique_elements, indices
):
    ib = tvm.tir.ir_builder.create()
    data_ptr = ib.buffer_ptr(data)
    argsorted_indices_ptr = ib.buffer_ptr(argsorted_indices)
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    index_converter_ptr = ib.buffer_ptr(index_converter)
    unique_elements_ptr = ib.buffer_ptr(unique_elements)
    indices_ptr = ib.buffer_ptr(indices)

    batch_size = data.shape[0]
    max_threads = tir.min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            indices_ptr[argsorted_indices_ptr[tid]] = index_converter_ptr[inc_scan_ptr[tid]]
            with ib.if_scope(tid == 0):
                unique_elements_ptr[index_converter_ptr[inc_scan_ptr[tid]]] = data_ptr[
                    argsorted_indices_ptr[tid]
                ]
            with ib.else_scope():
                with ib.if_scope(inc_scan_ptr[tid] != inc_scan_ptr[tid - 1]):
                    unique_elements_ptr[index_converter_ptr[inc_scan_ptr[tid]]] = data_ptr[
                        argsorted_indices_ptr[tid]
                    ]
    return ib.get()


def _calc_counts_unsorted_ir(inc_scan, index_converter, counts):
    ib = tvm.tir.ir_builder.create()
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    index_converter_ptr = ib.buffer_ptr(index_converter)
    counts_ptr = ib.buffer_ptr(counts)

    batch_size = inc_scan.shape[0]
    max_threads = tir.min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            counts_ptr[tid] = 0
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        atomic_add_return = ib.allocate(counts.dtype, (1,), name="atomic_add_return", scope="local")
        with ib.if_scope(tid < batch_size):
            index = index_converter_ptr[inc_scan_ptr[tid]]
            atomic_add_return[0] = tvm.tir.call_intrin(
                counts.dtype,
                "tir.atomic_add",
                tvm.tir.call_intrin("handle", "tir.address_of", counts_ptr[index]),
                1,
            )
    return ib.get()


def unique(data, is_sorted=True, return_counts=False):
    """
    Find the unique elements of a tensor
    Parameters
    ----------
    data : relay.Expr
        A 1-D tensor of integers
    sorted : bool
        Whether to sort the unique elements in ascending order before returning as output
    return_counts : bool
        Whether to return the array with count of each unique element
    Returns
    -------
    output : relay.Expr
        A 1-D tensor containing the unique elements of the input data tensor
    indices : relay.Expr
        A 1-D tensor containing the index of each data element in the output tensor
    num_unique : relay.Expr
        A 0-D tensor containing the number of unique elements in the input data tensor
    counts (optional) : relay.Expr
        A 1-D tensor containing the count of each unique element in the output
    Examples
    --------
    .. code-block:: python
        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, False)
        output         =  [4, 5, 1, 2, 3, ?, ?, ?]
        indices        =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique     =  [5]

        [output, indices, num_unique, counts] = unique([4, 5, 1, 2, 3, 3, 4, 5], False, True)
        output         =  [4, 5, 1, 2, 3, ?, ?, ?]
        indices        =  [0, 1, 2, 3, 4, 4, 0, 1]
        num_unique     =  [5]
        counts         =  [2, 2, 1, 1, 2, ?, ?, ?]

        [output, indices, num_unique] = unique([4, 5, 1, 2, 3, 3, 4, 5], True)
        output         =  [1, 2, 3, 4, 5, ?, ?, ?]
        indices        =  [3, 4, 0, 1, 2, 2, 3, 4]
        num_unique     =  [5]
    """

    sorted_data = sort(data)
    argsorted_indices = argsort(data, dtype="int32")
    # calculate adjacent difference
    sorted_data_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "sorted_data_buf", data_alignment=8
    )
    adjacent_diff_buf = tvm.tir.decl_buffer(
        data.shape, "int32", "adjacent_diff_buf", data_alignment=8
    )
    adjacent_diff = te.extern(
        [data.shape],
        [sorted_data],
        lambda ins, outs: _calc_adjacent_diff_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[sorted_data_buf],
        out_buffers=[adjacent_diff_buf],
        name="_calc_adjacent_diff",
        tag="_calc_adjacent_diff_gpu",
    )
    # calculate inclusive scan
    inc_scan = cumsum(adjacent_diff, dtype="int32", exclusive=0)
    # calculate number of unique elements
    num_unique_elements = _calc_num_unique(inc_scan)
    # declare buffers
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    argsorted_indices_buf = tvm.tir.decl_buffer(
        data.shape, "int32", "argsorted_indices_buf", data_alignment=8
    )
    inc_scan_buf = tvm.tir.decl_buffer(data.shape, "int32", "inc_scan_buf", data_alignment=8)
    unique_elements_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "unique_elements_buf", data_alignment=8
    )
    inverse_indices_buf = tvm.tir.decl_buffer(
        data.shape, "int32", "inverse_indices_buf", data_alignment=8
    )
    if is_sorted:
        # calculate unique elements and inverse indices
        unique_elements, inverse_indices = te.extern(
            [data.shape, data.shape],
            [data, argsorted_indices, inc_scan],
            lambda ins, outs: _calc_unique_sorted_ir(*ins, *outs),
            dtype=[data.dtype, "int32"],
            in_buffers=[data_buf, argsorted_indices_buf, inc_scan_buf],
            out_buffers=[unique_elements_buf, inverse_indices_buf],
            name="_calc_unique_sorted",
            tag="_calc_unique_sorted_gpu",
        )
        if not return_counts:
            return [unique_elements, inverse_indices, num_unique_elements]
        else:
            # calculate counts of unique elements
            counts_buf = tvm.tir.decl_buffer(data.shape, "int32", "counts_buf", data_alignment=8)
            counts = te.extern(
                [data.shape],
                [inc_scan],
                lambda ins, outs: _calc_counts_sorted_ir(ins[0], outs[0]),
                dtype=["int32"],
                in_buffers=[inc_scan_buf],
                out_buffers=[counts_buf],
                name="calc_counts_sorted",
                tag="calc_counts_sorted_gpu",
            )
            return [unique_elements, inverse_indices, num_unique_elements, counts]
    else:
        # calculate first occurence
        first_occurence_buf = tvm.tir.decl_buffer(
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
        # calculate index converter by sorting unique elements by their first occurence
        argsorted_first_occurence = argsort(first_occurence, dtype="int32")
        index_converter = argsort(argsorted_first_occurence, dtype="int32")
        # calculate unique elements and inverse indices
        index_converter_buf = tvm.tir.decl_buffer(
            data.shape, "int32", "index_converter_buf", data_alignment=8
        )
        unique_elements, inverse_indices = te.extern(
            [data.shape, data.shape],
            [data, argsorted_indices, inc_scan, index_converter],
            lambda ins, outs: _calc_unique_unsorted_ir(*ins, *outs),
            dtype=[data.dtype, "int32"],
            in_buffers=[data_buf, argsorted_indices_buf, inc_scan_buf, index_converter_buf],
            out_buffers=[unique_elements_buf, inverse_indices_buf],
            name="_calc_unique_unsorted",
            tag="_calc_unique_unsorted_gpu",
        )
        if not return_counts:
            return [unique_elements, inverse_indices, num_unique_elements]
        else:
            # calculate counts of unique elements
            counts_buf = tvm.tir.decl_buffer(data.shape, "int32", "counts_buf", data_alignment=8)
            counts = te.extern(
                [data.shape],
                [inc_scan, index_converter],
                lambda ins, outs: _calc_counts_unsorted_ir(ins[0], ins[1], outs[0]),
                dtype=["int32"],
                in_buffers=[inc_scan_buf, index_converter_buf],
                out_buffers=[counts_buf],
                name="calc_counts_unsorted",
                tag="calc_counts_unsorted_gpu",
            )
            return [unique_elements, inverse_indices, num_unique_elements, counts]
