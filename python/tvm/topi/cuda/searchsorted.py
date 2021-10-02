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
"""searchsorted operator for GPU"""
import tvm
from tvm import te
from .. import utils
from ..searchsorted import binary_search


def searchsorted(sorted_sequence, values, side="left", out_dtype="int64"):
    """TODO"""

    def ir(sorted_sequence, values, indices):
        ib = tvm.tir.ir_builder.create()
        sorted_sequence_shape = sorted_sequence.shape
        values_shape = values.shape
        num_search = utils.prod(values_shape)
        search_range = sorted_sequence_shape[-1]

        sorted_sequence = ib.buffer_ptr(sorted_sequence)
        values = ib.buffer_ptr(values)
        indices = ib.buffer_ptr(indices)

        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(
            bx, "thread_extent", tvm.tir.indexdiv(num_search + max_threads - 1, max_threads)
        )
        ib.scope_attr(tx, "thread_extent", max_threads)
        tid = bx * max_threads + tx

        with ib.if_scope(tid < num_search):
            if len(sorted_sequence_shape) == 1:
                sequence_offset = 0
            else:
                sequence_id = tid // values_shape[-1]
                sequence_offset = sequence_id * search_range

            binary_search(
                ib,
                sequence_offset,
                search_range,
                sorted_sequence,
                tid,
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
