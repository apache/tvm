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
from ...te import hybrid
from .scan import cumsum
from .sort import sort, argsort
from tvm import te
import tvm
from ..utils import ceil_div
from .nms import atomic_add


@hybrid.script
def _calc_adjacent_diff(data):
    output = output_tensor(data.shape, "int32")
    idx = allocate((1,), "int32", "local")
    i_extent = min(data.shape[0], max_num_threads(False))
    j_extent = ceil_div(data.shape[0], i_extent)
    for i in bind("threadIdx.x", i_extent):
        for j in range(j_extent):
            idx[0] = j * i_extent + i
            if idx[0] == 0:
                output[0] = int32(0)
            elif idx[0] < data.shape[0]:
                output[idx[0]] = int32(1) if data[idx[0]] != data[idx[0] - 1] else int32(0)
    return output


@hybrid.script
def _calc_num_unique(data):
    output = output_tensor((1,), "int32")
    for i in bind("threadIdx.x", 1):
        output[0] = data[data.shape[0] - 1] + int32(1)
    return output


@hybrid.script
def _calc_unique_sorted(data, argsorted_indices, inc_scan):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    idx = allocate((1,), "int32", "local")
    i_extent = min(data.shape[0], max_num_threads(False))
    j_extent = ceil_div(data.shape[0], i_extent)
    for i in bind("threadIdx.x", i_extent):
        for j in range(j_extent):
            idx[0] = j * i_extent + i
            if idx[0] < data.shape[0]:
                indices[argsorted_indices[idx[0]]] = inc_scan[idx[0]]
                if idx[0] == 0 or inc_scan[idx[0]] != inc_scan[idx[0] - 1]:
                    unique_elements[inc_scan[idx[0]]] = data[argsorted_indices[idx[0]]]
    return unique_elements, indices


def _calc_counts_sorted_ir(inc_scan, counts):
    ib = tvm.tir.ir_builder.create()
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    counts_ptr = ib.buffer_ptr(counts)
    batch_size = inc_scan.shape[0]
    max_threads = min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
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


@hybrid.script
def _calc_first_occurence(argsorted_indices, inc_scan):
    first_occurence = output_tensor(argsorted_indices.shape, "int32")
    idx = allocate((1,), "int32", "local")
    i_extent = min(argsorted_indices.shape[0], max_num_threads(False))
    j_extent = ceil_div(argsorted_indices.shape[0], i_extent)
    for i in bind("threadIdx.x", i_extent):
        for j in range(j_extent):
            idx[0] = j * i_extent + i
            if idx[0] < argsorted_indices.shape[0]:
                first_occurence[idx[0]] = argsorted_indices.shape[0]
    for i in bind("threadIdx.x", i_extent):
        for j in range(j_extent):
            idx[0] = j * i_extent + i
            if idx[0] < argsorted_indices.shape[0]:
                if idx[0] == 0 or inc_scan[idx[0]] != inc_scan[idx[0] - 1]:
                    first_occurence[inc_scan[idx[0]]] = argsorted_indices[idx[0]]
    return first_occurence


@hybrid.script
def _calc_unique_unsorted(data, argsorted_indices, inc_scan, index_converter):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    for i in parallel(data.shape[0]):
        new_unique_idx = index_converter[inc_scan[i]]
        new_data_idx = argsorted_indices[i]
        indices[new_data_idx] = new_unique_idx
        if i == 0 or inc_scan[i] != inc_scan[i - 1]:
            unique_elements[new_unique_idx] = data[new_data_idx]
    return unique_elements, indices


@hybrid.script
def _calc_unique_unsorted(data, argsorted_indices, inc_scan, index_converter):
    unique_elements = output_tensor(data.shape, data.dtype)
    indices = output_tensor(data.shape, "int32")
    idx = allocate((1,), "int32", "local")
    i_extent = min(data.shape[0], max_num_threads(False))
    j_extent = ceil_div(data.shape[0], i_extent)
    for i in bind("threadIdx.x", i_extent):
        for j in range(j_extent):
            idx[0] = j * i_extent + i
            if idx[0] < data.shape[0]:
                indices[argsorted_indices[idx[0]]] = index_converter[inc_scan[idx[0]]]
                if idx[0] == 0 or inc_scan[idx[0]] != inc_scan[idx[0] - 1]:
                    unique_elements[index_converter[inc_scan[idx[0]]]] = data[
                        argsorted_indices[idx[0]]
                    ]
    return unique_elements, indices


def _calc_counts_unsorted_ir(inc_scan, index_converter, counts):
    ib = tvm.tir.ir_builder.create()
    inc_scan_ptr = ib.buffer_ptr(inc_scan)
    index_converter_ptr = ib.buffer_ptr(index_converter)
    counts_ptr = ib.buffer_ptr(counts)
    batch_size = inc_scan.shape[0]
    max_threads = min(batch_size, tvm.target.Target.current(allow_none=False).max_num_threads)
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
    adjacent_diff = _calc_adjacent_diff(sorted_data)
    inc_scan = cumsum(adjacent_diff, dtype="int32", exclusive=0)
    num_unique_elements = _calc_num_unique(inc_scan)
    if is_sorted:
        unique_elements, inverse_indices = _calc_unique_sorted(data, argsorted_indices, inc_scan)
        if not return_counts:
            return [unique_elements, inverse_indices, num_unique_elements]
        else:
            inc_scan_buf = tvm.tir.decl_buffer(
                data.shape, "int32", "inc_scan_buf", data_alignment=8
            )
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
        first_occurence = _calc_first_occurence(argsorted_indices, inc_scan)
        argsorted_first_occurence = argsort(first_occurence, dtype="int32")
        index_converter = argsort(argsorted_first_occurence, dtype="int32")
        unique_elements, inverse_indices = _calc_unique_unsorted(
            data, argsorted_indices, inc_scan, index_converter
        )
        if not return_counts:
            return [unique_elements, inverse_indices, num_unique_elements]
        else:
            inc_scan_buf = tvm.tir.decl_buffer(
                data.shape, "int32", "inc_scan_buf", data_alignment=8
            )
            index_converter_buf = tvm.tir.decl_buffer(
                data.shape, "int32", "index_converter_buf", data_alignment=8
            )
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
