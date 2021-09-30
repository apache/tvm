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
from . import utils
from . import te
from ..tir import ir_builder
from .math import cast


def searchsorted(sorted_sequence, values, side="left", out_dtype="int64"):
    def binary_search(ib, sequence_offset, search_range, sorted_sequence, i, values, out_indices):
        lo = ib.allocate(out_dtype, (1,), name="lo", scope="local")
        hi = ib.allocate(out_dtype, (1,), name="hi", scope="local")

        v = values[i]
        lo[0] = cast(0, out_dtype)
        hi[0] = cast(search_range, out_dtype)

        with ib.while_loop(lo[0] < hi[0]):
            mid = lo[0] + (hi[0] - lo[0] >> 1)
            with ib.if_scope(sorted_sequence[sequence_offset + mid] < v):
                lo[0] = mid + 1
            with ib.else_scope():
                hi[0] = mid

        out_indices[i] = lo[0]

    def ir(sorted_sequence, values, indices):
        ib = ir_builder.create()
        sorted_sequence_shape = sorted_sequence.shape
        values_shape = values.shape
        num_search = utils.prod(values_shape)
        num_sequence = utils.prod(sorted_sequence_shape[:-1])
        search_range = sorted_sequence_shape[-1]

        sorted_sequence = ib.buffer_ptr(sorted_sequence)
        values = ib.buffer_ptr(values)
        indices = ib.buffer_ptr(indices)

        with ib.for_range(0, num_search, name="i", kind="parallel") as i:
            sequence_id = i // values_shape[-1]
            sequence_offset = sequence_id * search_range
            binary_search(ib, sequence_offset, search_range, sorted_sequence, i, values, indices)

        return ib.get()

    return te.extern(
        values.shape,
        [sorted_sequence, values],
        lambda ins, outs: ir(ins[0], ins[1], outs[0]),
        name="searchsorted_ir",
        dtype=out_dtype,
    )
