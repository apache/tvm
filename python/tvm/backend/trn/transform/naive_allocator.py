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

import functools

import tvm_ffi

from tvm.tirx import AllocBuffer, IntImm
from tvm.tirx.buffer import Buffer
from tvm.tirx.transform.function_pass import prim_func_pass


def is_const_shape(buffer: Buffer) -> bool:
    for i in buffer.ty.shape:
        if not isinstance(i, IntImm):
            return False
    return True


def get_buffer_size(buffer: Buffer) -> int:
    if buffer.scope() == "trn.sbuf":
        if buffer.ty.layout is None:
            # the first dimension is partition size
            num_elem = functools.reduce(lambda x, y: x * y, buffer.ty.shape[1:])
        else:
            par_size = buffer.ty.layout.size("P")
            num_elem = functools.reduce(lambda x, y: x * y, buffer.ty.shape) // par_size
    elif buffer.scope().startswith("shared"):
        num_elem = functools.reduce(lambda x, y: x * y, buffer.ty.shape)
    else:
        return None
    if not is_const_shape(buffer):
        raise ValueError(
            f"Buffer {buffer.name} has non-constant shape. Do not know how to allocate it."
        )
    return int(num_elem * buffer.ty.dtype.dtype.itemsize)


def _get_alloc_pool_start(stmt) -> int:
    alloc_pool_start = 0

    def collect_alloc_buffer(op: AllocBuffer):
        nonlocal alloc_pool_start
        buffer = op.buffer
        if len(buffer.ty.allocated_addr) == 0:
            return
        buffer_size = get_buffer_size(buffer)
        if buffer_size is None:
            return
        alloc_pool_start = max(alloc_pool_start, buffer.ty.allocated_addr[-1] + buffer_size)

    tvm_ffi.structural_walk(stmt, (AllocBuffer, collect_alloc_buffer), order="post")
    return alloc_pool_start


def _allocate_missing_buffers(stmt, alloc_pool_start: int):
    alloc_offset = alloc_pool_start
    buffer_map = {}

    def allocate_buffer(op: AllocBuffer):
        nonlocal alloc_offset
        buffer = op.buffer
        buffer_size = get_buffer_size(buffer)
        if len(buffer.ty.allocated_addr) == 0 and buffer_size is not None:
            new_buffer = buffer.with_allocated_addr([alloc_offset])
            buffer_map[buffer] = new_buffer
            alloc_offset += buffer_size
            return AllocBuffer(new_buffer, op.annotations, op.span)
        return op

    def replace_buffer(op):
        return buffer_map.get(op, op)

    return tvm_ffi.structural_map(
        stmt,
        [(AllocBuffer, allocate_buffer), (Buffer, replace_buffer)],
        order="pre",
    )


@prim_func_pass(opt_level=0, name="TrnNaiveAllocator")
class TrnNaiveAllocator:
    def transform_function(self, func, mod, ctx):
        alloc_pool_start = _get_alloc_pool_start(func.body)
        new_body = _allocate_missing_buffers(func.body, alloc_pool_start)
        return func.with_body(new_body)
