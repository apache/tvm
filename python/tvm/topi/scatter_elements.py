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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-nested-blocks
"""ScatterElements operator"""
from ..te import extern
from ..tir import min, max, decl_buffer, ir_builder


def scatter_elements(data, indices, updates, axis=0, reduction="update"):
    """Scatter elements from updates to corresponding indices of copied data.

    Data, indices, updates and output have the same shape.
    Indices can not have duplicates (if idx1 != idx2, then indices[idx1] != indices[idx2])
    if reduction == "update".

    .. code-block::

        output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
        output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,

    where the update function f is determinted by the reduction.
    Five types of the function are supported: replace, +, *, min or max

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

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        # pylint: disable=invalid-name
        ib = ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        updates = ib.buffer_ptr(updates_ptr)
        out = ib.buffer_ptr(out_ptr)

        # Prepare ranges and strides
        before_axis_range = 1
        for i in data_ptr.shape[:axis]:
            before_axis_range *= i

        axis_range = data_ptr.shape[axis]

        after_axis_range = 1
        for i in data_ptr.shape[axis + 1 :]:
            after_axis_range *= i
        before_axis_stride = axis_range * after_axis_range

        # Copy initial input data to output
        fused_shape = before_axis_range * before_axis_stride

        with ib.for_range(0, fused_shape) as i:
            out[i] = data[i]

        with ib.for_range(0, before_axis_range, kind="parallel") as i:
            with ib.for_range(0, after_axis_range, kind="parallel") as j:
                with ib.for_range(0, axis_range, kind="parallel") as k:
                    pre_index = i * before_axis_stride + j
                    index1 = pre_index + k * after_axis_range
                    index2 = pre_index + indices[index1]
                    if reduction == "update":
                        out[index2] = updates[index1]
                    elif reduction == "add":
                        out[index2] += updates[index1]
                    elif reduction == "mul":
                        out[index2] *= updates[index1]
                    elif reduction == "min":
                        min(out[index2], updates[index1])
                    elif reduction == "max":
                        max(out[index2], updates[index1])
                    else:
                        raise NotImplementedError(
                            "scatter_elements reduction not in [update, add, mul, min, max]:",
                            reduction,
                        )

        return ib.get()

    out_buf = decl_buffer(data.shape, data.dtype, "out_buf")
    return extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_elements.generic",
        tag="scatter_elements.generic",
    )
