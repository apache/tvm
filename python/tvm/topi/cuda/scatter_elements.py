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
"""Scatter operator """
import tvm
from tvm import te, tir
from ..utils import ceil_div, get_const_int
from ..math import cast
from .nms import atomic_add


def gen_scatter_add_1d_atomic(data, indices, updates, out, axis, _):
    """Generate ir for scatter elements for reduction sum for 1d inputs,
    using atomic_add instruction

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices : tir.Tensor
        The index locations to update.

    updates : tir.Tensor
        The values to update.

    out : tir.Tensor
        The output tensor.

    axis : int
        The axis to scatter on

    Returns
    -------
    ret : tir
        The computational ir.
    """
    assert axis == 0
    n = data.shape[0]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    with ib.new_scope():
        nthread_bx = ceil_div(n, nthread_tx)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * nthread_tx + tx
        with ib.if_scope(tid < n):
            out_ptr[tid] = data_ptr[tid]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)

    ni = indices.shape[0]

    atomic_add_return = ib.allocate(updates.dtype, (1,), name="atomic_add_return", scope="local")

    with ib.new_scope():
        nthread_bx = ceil_div(ni, nthread_tx)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * nthread_tx + tx

        with ib.if_scope(tid < ni):
            index = indices_ptr[tid]
            with ib.if_scope(index < 0):
                atomic_add_return[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", out_ptr[index + n]),
                    updates_ptr[tid],
                )
            with ib.else_scope():
                atomic_add_return[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", out_ptr[index]),
                    updates_ptr[tid],
                )

    return ib.get()


def gen_ir(data, indices, updates, out, axis, reduce_func):
    """Generate ir for scatter elements

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices : tir.Tensor
        The index locations to update.

    updates : tir.Tensor
        The values to update.

    out : tir.Tensor
        The output tensor.

    axis : int
        The axis to scatter on

    reduce_func : Any
        The function reduced update and output to output

    Returns
    -------
    ret : tir
        The computational ir.
    """
    ib = tir.ir_builder.create()

    data_ptr = ib.buffer_ptr(data)
    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    out_ptr = ib.buffer_ptr(out)

    # Prepare ranges and strides
    shape = data.shape
    if axis < 0:
        axis = len(shape) + axis
    axis_range = cast(shape[axis], indices.dtype)

    before_axis_range = 1
    after_axis_range = 1
    for i, value in enumerate(shape, 0):
        if i < axis:
            before_axis_range *= value
        elif i > axis:
            after_axis_range *= value
    before_axis_stride = axis_range * after_axis_range
    full_range = before_axis_range * before_axis_stride

    ind_shape = indices.shape
    ind_axis_range = ind_shape[axis]

    ind_before_axis_range = 1
    ind_after_axis_range = 1
    for i, value in enumerate(ind_shape, 0):
        if i < axis:
            ind_before_axis_range *= value
        elif i > axis:
            ind_after_axis_range *= value
    ind_before_axis_stride = ind_axis_range * ind_after_axis_range
    ind_full_range_excl_axis = ind_before_axis_range * ind_after_axis_range

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    # Copy initial input data to output
    with ib.new_scope():
        num_blocks = cast(ceil_div(full_range, max_threads), "int32")
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(bx, "thread_extent", num_blocks)
        ib.scope_attr(tx, "thread_extent", max_threads)

        index = bx * max_threads + tx
        with ib.if_scope(index < full_range):
            out_ptr[index] = data_ptr[index]

    with ib.new_scope():
        num_blocks_2 = ceil_div(ind_full_range_excl_axis, max_threads)
        bx2 = te.thread_axis("blockIdx.x")
        tx2 = te.thread_axis("threadIdx.x")
        ib.scope_attr(bx2, "thread_extent", num_blocks_2)
        ib.scope_attr(tx2, "thread_extent", max_threads)

        ind_fused = bx2 * max_threads + tx2
        with ib.if_scope(ind_fused < ind_full_range_excl_axis):
            i = ind_fused // ind_after_axis_range
            j = ind_fused % ind_after_axis_range
            pre_index1 = i * ind_before_axis_stride + j
            pre_index2 = i * before_axis_stride + j
            with ib.for_range(0, ind_axis_range, "k") as k:
                # Offset along indices or updates
                index1 = pre_index1 + k * ind_after_axis_range
                # Get index and shift to positive side if need
                new_index = indices_ptr[index1]
                shifted_index = new_index + (new_index < 0) * axis_range
                # Offset along data
                index2 = pre_index2 + shifted_index * after_axis_range
                reduce_func(out_ptr, index2, updates_ptr[index1])

    return ib.get()


def scatter_elements(data, indices, updates, axis=0, reduction="update"):
    """Scatter elements from updates to corresponding indices of copied data.

    Data, indices, updates and output have the same shape.
    Indices can not have duplicates (if idx1 != idx2, then indices[idx1] != indices[idx2])
    if reduction == "update".

    .. code-block::

        output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0
        output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1

    where the update function f is determinted by the reduction.
    Five types of the function are supported: "update", "add", "mul", "min" and "max" (see below)

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    updates : tvm.te.Tensor
        The updates to apply at the Indices

    axis : optional, int
        The axis to scatter on. It is zero by default.

    reduction : optional, string
        The update mode for the algorithm, either "update", "add", "mul", "mean", "min" or "max"
        If update, the update values will replace the input data
        If add, the update values will be added to the input data
        If mul, the input data will be multiplied on the update values
        If mean, the input data will be mean between the update values and the input data
        If min, there is choice of minimal between the update values and the input data
        If max, there is choice of maximal between the update values and the input data
        It is "update" by default

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if not isinstance(axis, int):
        axis = get_const_int(axis)

    def update_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = update

    def add_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] += update

    def mul_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] *= update

    def mean_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = (dst_ptr[dst_index] + update) / 2

    def min_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = tir.min(dst_ptr[dst_index], update)

    def max_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = tir.max(dst_ptr[dst_index], update)

    reduce_func = None
    if reduction == "update":
        reduce_func = update_func
    elif reduction == "add":
        reduce_func = add_func
    elif reduction == "mul":
        reduce_func = mul_func
    elif reduction == "mean":
        reduce_func = mean_func
    elif reduction == "min":
        reduce_func = min_func
    elif reduction == "max":
        reduce_func = max_func
    else:
        raise NotImplementedError(
            "scatter_elements reduction not in [update, add, mul, min, max]:", reduction
        )

    shape = data.shape
    rank = len(shape)
    cur_target_kind = str(tvm.target.Target.current(allow_none=False).kind)
    gen_scatter_elements_ir = None
    if (
        reduction == "add"
        and rank == 1
        and cur_target_kind not in ["vulkan", "metal"]
        and updates.dtype in ["int32", "float32"]
    ):
        gen_scatter_elements_ir = gen_scatter_add_1d_atomic
    else:
        gen_scatter_elements_ir = gen_ir

    out_buf = tir.decl_buffer(shape, data.dtype, "out_buf")
    return te.extern(
        [shape],
        [data, indices, updates],
        lambda ins, outs: gen_scatter_elements_ir(
            ins[0], ins[1], ins[2], outs[0], axis, reduce_func
        ),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_elements_cuda",
        tag="scatter_elements_cuda",
    )
