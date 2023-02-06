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
"""ScatterElements operator"""
from tvm import te
from tvm import tir
from . import utils


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
        axis = utils.get_const_int(axis)

    shape = data.shape
    axis_range = int(shape[axis])

    if axis < 0:
        axis = len(shape) + axis

    # Prepare ranges and strides
    before_axis_range = 1
    after_axis_range = 1
    for i, value in enumerate(shape, 0):
        if i < axis:
            before_axis_range *= value
        elif i > axis:
            after_axis_range *= value
    before_axis_stride = axis_range * after_axis_range
    full_range = before_axis_range * before_axis_stride

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        # pylint: disable=invalid-name
        ib = tir.ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        updates = ib.buffer_ptr(updates_ptr)
        out = ib.buffer_ptr(out_ptr)

        # Copy initial input data to output
        with ib.for_range(0, full_range, "i", kind="parallel") as i:
            out[i] = data[i]

        # TODO(vvchernov): find optimal parallel approach
        with ib.for_range(0, before_axis_range, "i", kind="parallel") as i:
            with ib.for_range(0, after_axis_range, "j") as j:
                with ib.for_range(0, axis_range, "k") as k:
                    pre_index = i * before_axis_stride + j
                    index1 = pre_index + k * after_axis_range
                    # TODO(vvchernov): assert for out of bounds, separated check for indices
                    k_new = indices[index1]
                    index_check = tir.LT(k_new, tir.const(0, indices.dtype))
                    k_new += tir.Select(index_check, axis_range, tir.const(0, indices.dtype))
                    index2 = pre_index + k_new * after_axis_range
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
        name="scatter_elements.generic",
        tag="scatter_elements.generic",
    )
