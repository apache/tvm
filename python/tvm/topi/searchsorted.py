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
"""searchsorted operator"""

import tvm
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T

from . import te, utils
from .math import cast


def binary_search(sequence_offset, search_range, sorted_sequence, value, right, out_dtype):
    """Common IR generator for binary search used by CPU and GPU backends.

    Must be called within an active IRBuilder context.

    `sorted_sequence` is a N-D Buffer whose innermost dimension we want to search for `value`,
    and `search_range` is the size of the innermost dimension. `sequence_offset` is
    a 1-D linearlized offset specifying which of innermost sequences to search.

    So the search for `value` is performed over
    `sorted_sequence[sequence_offset:(sequence_offset + search_range)]`.
    Note that we index N-D Buffer by 1-D linearlized indices.

    """
    with T.frame_scope(
        [
            T.allocate([1], out_dtype, scope="local"),
            T.allocate([1], out_dtype, scope="local"),
        ]
    ) as (lo_ptr, hi_ptr):
        lo = T.buffer_proxy(tvm.tir.decl_buffer([1], out_dtype, "lo", data=lo_ptr, scope="local"))
        hi = T.buffer_proxy(tvm.tir.decl_buffer([1], out_dtype, "hi", data=hi_ptr, scope="local"))

        lo[0] = cast(0, out_dtype)
        hi[0] = cast(search_range, out_dtype)

        # Reference: pytorch/aten/src/ATen/native/cuda/Bucketization.cu
        def condition(current_val, target_val):
            if right:
                return current_val <= target_val
            return current_val < target_val

        with T.While(lo[0] < hi[0]):
            mid = lo[0] + (hi[0] - lo[0] >> 1)
            with T.If(condition(sorted_sequence[sequence_offset + mid], value)):
                with T.Then():
                    lo[0] = mid + 1
                with T.Else():
                    hi[0] = mid

    return lo[0]


def searchsorted(sorted_sequence, values, right=False, out_dtype="int64"):
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

    right : bool, optional
        Controls which index is returned if a value lands exactly on one of sorted values. If
        False, the index of the first suitable location found is given. If true, return the
        last such index. If there is no suitable index, return either 0 or N (where N is the
        size of the innermost dimension).

    dtype : string, optional
        The data type of the output indices.

    Returns
    -------
    indices : te.Tensor
        Tensor with same shape as values, representing the indices of
        elements of `values` if they are inserted in `sorted_sequence`.
    """

    def ir(sorted_sequence, values, indices):
        with IRBuilder() as ib:
            sorted_sequence_shape = sorted_sequence.shape
            values_shape = values.shape
            num_search = utils.prod(values_shape)
            search_range = sorted_sequence_shape[-1]

            sorted_sequence = T.buffer_proxy(sorted_sequence)
            values = T.buffer_proxy(values)
            indices = T.buffer_proxy(indices)

            with T.parallel(0, num_search) as i:
                if len(sorted_sequence_shape) == 1:
                    sequence_offset = 0
                else:
                    sequence_id = i // values_shape[-1]
                    sequence_offset = sequence_id * search_range

                indices[i] = binary_search(
                    sequence_offset,
                    search_range,
                    sorted_sequence,
                    values[i],
                    right,
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
