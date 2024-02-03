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
# pylint: disable=missing-docstring
"""A RMS norm schedule rule for GPU operators."""

import tvm
from tvm import tir
from tvm.tir import Block, BufferStore
from tvm.tir.expr import Cast, BufferLoad, Call
from tvm.target import Target

from ..base import ScheduleRule


def identify_cast_or_load_block(block: Block) -> bool:
    if len(block.reads) != 1 or len(block.writes) != 1:
        return False

    if not isinstance(block.body, BufferStore):
        return False
    store = block.body

    # check types
    if isinstance(store.value, BufferLoad):
        load = store.value
    elif isinstance(store.value, Cast):
        load = store.value.value
        if not isinstance(load, BufferLoad):
            return False
    else:
        return False

    # check indices
    if len(load.indices) != len(store.indices):
        return False

    for lhs, rhs in zip(load.indices, store.indices):
        if not lhs.same_as(rhs):
            return False

    return True


def identify_rsqrt_block(block: Block) -> bool:
    if len(block.reads) != 1 or len(block.writes) != 1:
        return False

    if not isinstance(block.body, BufferStore):
        return False
    store = block.body

    if not isinstance(store.value, Call):
        return False
    call = store.value
    op = call.op

    return op == tvm.ir.op.Op.get("tir.rsqrt")


class RMSNorm(ScheduleRule):
    """A rule for RMS norm."""

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> tir.Schedule:
        if target.kind.name == "cuda":
            num_tx = 512
        else:
            num_tx = 64

        sch = tir.Schedule(func)
        root = sch.get_block(name="root", func_name="main")

        blocks = sch.get_child_blocks(root)

        if not any([identify_rsqrt_block(sch.get(block)) for block in blocks]):
            return None

        read = sch.cache_read(block=blocks[0], read_buffer_index=0, storage_scope="local")
        write = sch.cache_write(block=blocks[-1], write_buffer_index=0, storage_scope="local")

        for block in blocks:
            if identify_cast_or_load_block(sch.get(block)):
                sch.compute_inline(block)

        blocks = sch.get_child_blocks(root)

        read, sqr, redsum, rsqrt, norm, write = blocks

        if not identify_rsqrt_block(sch.get(rsqrt)):
            return None

        for name in [read, sqr, redsum, rsqrt, norm, write]:
            loops = sch.get_loops(name)
            sch.fuse(*loops[:-1])

        block_loop, loops = sch.get_loops(block=read)
        thread_loop, _, _ = sch.split(
            loop=loops, factors=[num_tx, None, 8], preserve_unit_iters=True
        )
        sch.bind(block_loop, thread_axis="blockIdx.x")
        sch.bind(thread_loop, thread_axis="threadIdx.x")
        sch.vectorize(sch.get_loops(block=read)[-1])
        sch.reverse_compute_at(block=sqr, loop=thread_loop)
        sch.reverse_compute_at(block=redsum, loop=thread_loop)

        sch.reverse_compute_at(block=rsqrt, loop=block_loop, index=-1)
        sch.reverse_compute_at(block=norm, loop=block_loop, index=-1)
        block_loop, loops = sch.get_loops(block=norm)
        thread_loop, _, _ = sch.split(
            loop=loops, factors=[num_tx, None, 8], preserve_unit_iters=True
        )
        sch.bind(thread_loop, thread_axis="threadIdx.x")

        sch.reverse_compute_at(block=write, loop=thread_loop, index=-1)
        sch.vectorize(sch.get_loops(block=write)[-1])

        sch.set_scope(block=sqr, buffer_index=0, storage_scope="local")
        sch.set_scope(block=redsum, buffer_index=0, storage_scope="local")
        sch.set_scope(block=rsqrt, buffer_index=0, storage_scope="shared")
        sch.set_scope(block=norm, buffer_index=0, storage_scope="local")

        return sch
