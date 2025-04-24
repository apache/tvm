# Licensed to the Apache Software Foundation (ASF) under one
# or more contrir_builderutor license agreements.  See the NOTICE file
# distrir_builderuted with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distrir_builderuted under the License is distrir_builderuted on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""IndexPut operator"""
from tvm import te
from tvm import tir
from . import utils


def index_put(data, indices, values, accumulate=False):
    """Put values into an array according to indices.

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array to be modified.

    indices : Tuple[tvm.te.Tensor]
        Tuple of 1D index tensors (one for each dimension) specifying positions.

    values : tvm.te.Tensor
        The values to place at the specified indices.

    accumulate : bool, optional
        Whether to accumulate (add) values rather than replace.
        If True, performs tensor[indices] += values
        If False, performs tensor[indices] = values
        Default is False.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    if not isinstance(indices, (list, tuple)):
        indices = [indices]

    # Check indices match data dimensions
    if len(indices) != len(data.shape):
        raise ValueError(
            f"Number of index tensors ({len(indices)}) must match "
            f"data dimensions ({len(data.shape)})"
        )

    # Prepare ranges and strides
    shape = data.shape
    full_range = 1
    for dim in shape:
        full_range *= dim

    # Check all indices have same length
    index_len = len(indices[0])
    for idx in indices[1:]:
        if not utils.equal_const_int(len(idx), index_len):
            raise ValueError("All index tensors must have same length")

    def gen_ir(data_ptr, index_ptrs, values_ptr, out_ptr, reduce_func):
        ir_builder = tir.ir_builder.create()

        data = ir_builder.buffer_ptr(data_ptr)
        indices = [ir_builder.buffer_ptr(idx) for idx in index_ptrs]
        values = ir_builder.buffer_ptr(values_ptr)
        out = ir_builder.buffer_ptr(out_ptr)

        with ir_builder.for_range(0, full_range, "i", kind="parallel") as i:
            out[i] = data[i]

        with ir_builder.for_range(0, index_len, "k", kind="parallel") as k:
            # Calculate multi-dimensional index
            flat_index = 0
            stride = 1
            for dim in range(len(shape) - 1, -1, -1):
                # Get index and shift to positive if needed
                idx_val = indices[dim][k]
                shifted_idx = idx_val + (idx_val < 0) * shape[dim]
                flat_index += shifted_idx * stride
                stride *= shape[dim]

            reduce_func(out, flat_index, values[k])

        return ir_builder.get()

    def update_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = update

    def add_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] += update

    reduce_func = add_func if accumulate else update_func

    # Prepare input buffers
    in_buffers = [data]
    in_buffers.extend(indices)
    in_buffers.append(values)

    out_buf = tir.decl_buffer(data.shape, data.dtype, "out_buf")
    return te.extern(
        [data.shape],
        in_buffers,
        lambda ins, outs: gen_ir(ins[0], ins[1:-1], ins[-1], outs[0], reduce_func),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="index_put.generic",
        tag="index_put.generic",
    )
