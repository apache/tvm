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
"""searchsorted operator"""
from . import utils
from . import te
from ..tir import ir_builder
from .math import cast


def binary_search(
    ib, sequence_offset, search_range, index, sorted_sequence, values, out_indices, side, out_dtype
):
    """Common IR generator for CPU and GPU searchsorted."""
    lo = ib.allocate(out_dtype, (1,), name="lo", scope="local")
    hi = ib.allocate(out_dtype, (1,), name="hi", scope="local")

    v = values[index]
    lo[0] = cast(0, out_dtype)
    hi[0] = cast(search_range, out_dtype)

    # Reference: pytorch/aten/src/ATen/native/cuda/Bucketization.cu
    def condition(current_val, target_val):
        if side == "left":
            return current_val < target_val
        return current_val <= target_val

    with ib.while_loop(lo[0] < hi[0]):
        mid = lo[0] + (hi[0] - lo[0] >> 1)
        with ib.if_scope(condition(sorted_sequence[sequence_offset + mid], v)):
            lo[0] = mid + 1
        with ib.else_scope():
            hi[0] = mid

    out_indices[index] = lo[0]


def searchsorted(sorted_sequence, values, side="left", out_dtype="int64"):
    """Find indices where elements should be inserted to maintain order.
       If `sorted_sequence` is N-dimensional, the innermost dimension of
       `values` are searched in the corresponding dimension of `sorted_sequence`.

    Parameters
    ----------
    sorted_sequence : te.Tensor
        N-D or 1-D Tensor, containing monotonically increasing sequence
        on the innermost dimension.

    values : te.Tensor
        N-D Tensor containing the search values. When `sorted_sequence` is 1-D,
        the shape of `values` can be arbitrary. Otherwise, ranks of `sorted_sequence`
        and `values` must be the same, and outer N-1 axes must have the same size.

    side : string, optional
        It can be `left` or `right`. If `left`, gets the lower bound index for each value
        in `values` on the corresponding innermost dimension of the `sorted_sequence`.
        If `right`, gets the upper bound index instead.

    dtype : string, optional
        The data type of the output indices.

    Returns
    -------
    indices : te.Tensor
        Tensor with same shape as values, representing the indices of
        elements of `values` if they are inserted in `sorted_sequence`.
    """

    def ir(sorted_sequence, values, indices):
        ib = ir_builder.create()
        sorted_sequence_shape = sorted_sequence.shape
        values_shape = values.shape
        num_search = utils.prod(values_shape)
        search_range = sorted_sequence_shape[-1]

        sorted_sequence = ib.buffer_ptr(sorted_sequence)
        values = ib.buffer_ptr(values)
        indices = ib.buffer_ptr(indices)

        with ib.for_range(0, num_search, name="i", kind="parallel") as i:
            if len(sorted_sequence_shape) == 1:
                sequence_offset = 0
            else:
                sequence_id = i // values_shape[-1]
                sequence_offset = sequence_id * search_range

            binary_search(
                ib,
                sequence_offset,
                search_range,
                i,
                sorted_sequence,
                values,
                indices,
                side,
                out_dtype,
            )

        return ib.get()

    return te.extern(
        values.shape,
        [sorted_sequence, values],
        lambda ins, outs: ir(ins[0], ins[1], outs[0]),
        name="searchsorted",
        dtype=out_dtype,
    )
