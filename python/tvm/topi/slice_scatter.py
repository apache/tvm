# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""SliceScatter operator"""
from tvm import topi
from . import utils


def slice_scatter(input_tensor, src, start, end, step, axis):
    """
    Scatters a slice of src into input along the given axis (SSA form).

    Args:
        input_tensor (te.Tensor): The input tensor to scatter into.
        src (te.Tensor): The source tensor to scatter from.
        start (int): The starting index of the slice.
        end (int): The ending index of the slice.
        step (int): The step size of the slice.
        axis (int): The axis to scatter along.

    Returns:
        list[te.Tensor]: A list containing the output tensor with the slice scattered.
    """

    dim_size_expr = input_tensor.shape[axis]  # Expression for dimension size
    dim_size = utils.get_const_int(dim_size_expr)  # Dimension size (as constant int)

    if start == 0 and end == dim_size and step == 1:
        return topi.identity(src)

    mask = topi.full((dim_size,), "bool", True)
    idx = topi.arange(start=0, stop=dim_size, step=1, dtype="int64")

    if start != 0:
        mask = topi.logical_and(mask, topi.greater_equal(idx, start))

    if end != dim_size:
        mask = topi.logical_and(mask, topi.less(idx, end))

    if step != 1:
        step_mask = topi.equal(topi.floor_mod(idx - start, step), 0)
        mask = topi.logical_and(mask, step_mask)

    mask_shape_base = [1] * len(input_tensor.shape)
    mask_shape_base[axis] = dim_size
    mask_shape = tuple(mask_shape_base)

    mask_reshaped = topi.reshape(mask, mask_shape)

    idx_new_pre = idx - start + (step - 1)
    idx_new_div = topi.floor_divide(idx_new_pre, step)
    idx_new = topi.clip(idx_new_div, 0, dim_size - 1)

    temp = topi.take(src, idx_new, axis=axis)

    mask_shape_expanded_base = list(input_tensor.shape)
    mask_shape_expanded = tuple(mask_shape_expanded_base)

    mask_expanded = topi.broadcast_to(mask_reshaped, mask_shape_expanded)

    output = topi.where(mask_expanded, temp, input_tensor)

    return [output]
