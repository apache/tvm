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
from tvm import te
from tvm import tir
from ..utils import ceil_div, get_const_int
from ..math import cast


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
        The update mode for the algorithm, either "update", "add", "mul", "min" or "max"
        If update, the update values will replace the input data
        If add, the update values will be added to the input data
        If mul, the update values will be multiply to the input data
        If min, there is choice of minimal between the update values and the input data
        If max, there is choice of maximal between the update values and the input data
        It is "update" by default

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if not isinstance(axis, int):
        axis = get_const_int(axis)

    if axis < 0:
        axis = len(shape) + axis

    # Prepare ranges and strides
    shape = data.shape
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
    ind_axis_range = shape[axis]

    ind_before_axis_range = 1
    ind_after_axis_range = 1
    for i, value in enumerate(ind_shape, 0):
        if i < axis:
            ind_before_axis_range *= value
        elif i > axis:
            ind_after_axis_range *= value
    ind_before_axis_stride = ind_axis_range * ind_after_axis_range
    ind_full_range_excl_axis = ind_before_axis_range * ind_after_axis_range

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        ib = tir.ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        updates = ib.buffer_ptr(updates_ptr)
        out = ib.buffer_ptr(out_ptr)

        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        # Copy initial input data to output
        with ib.new_scope():
            num_blocks = ceil_div(full_range, max_threads)
            bx = te.thread_axis("blockIdx.x")
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(bx, "thread_extent", num_blocks)
            ib.scope_attr(tx, "thread_extent", max_threads)

            index = bx * max_threads + tx
            with ib.if_scope(index < full_range):
                out[index] = data[index]

        # TODO (vvchernov): use atomic function for special conditions (see cuda.scatter_nd)
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
                with ib.for_range(0, ind_axis_range, "k") as k:
                    # Offset along indices or updates
                    index1 = i * ind_before_axis_stride + k * ind_after_axis_range + j
                    # TODO(vvchernov): assert for out of bounds, separated check for indices
                    k_new = indices[index1]
                    index_check = tir.LT(k_new, tir.const(0, indices.dtype))
                    k_new += tir.Select(index_check, axis_range, tir.const(0, indices.dtype))
                    # Offset along data
                    index2 = i * before_axis_stride + k_new * after_axis_range + j
                    if reduction == "update":
                        out[index2] = updates[index1]
                    elif reduction == "add":
                        out[index2] += updates[index1]
                    elif reduction == "mul":
                        out[index2] *= updates[index1]
                    elif reduction == "min":
                        tir.min(out[index2], updates[index1])
                    elif reduction == "max":
                        tir.max(out[index2], updates[index1])
                    else:
                        raise NotImplementedError(
                            "scatter_elements reduction not in [update, add, mul, min, max]:",
                            reduction,
                        )

        return ib.get()

    out_buf = tir.decl_buffer(data.shape, data.dtype, "out_buf")
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_elements_cuda",
        tag="scatter_elements_cuda",
    )
