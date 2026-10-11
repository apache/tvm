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

from tvm.ir import Bind, Call, Op, Tuple, Var
from tvm.tirx import IntImm
from tvm.tirx.transform.function_pass import function_pass


def is_const_shape(shape) -> bool:
    for i in shape:
        if not isinstance(i, IntImm):
            return False
    return True


def get_tensor_size(tensor: Var, shape, dtype, scope: str) -> int:
    if scope == "trn.sbuf":
        if tensor.ty.layout is None:
            # the first dimension is partition size
            num_elem = functools.reduce(lambda x, y: x * y, shape[1:])
        else:
            par_size = tensor.ty.layout.size("P")
            num_elem = functools.reduce(lambda x, y: x * y, shape) // par_size
    elif scope.startswith("shared"):
        num_elem = functools.reduce(lambda x, y: x * y, shape)
    else:
        return None
    if not is_const_shape(shape):
        raise ValueError(
            f"Var {tensor.name} has non-constant shape. Do not know how to allocate it."
        )
    return int(num_elem * dtype.itemsize)


def _get_alloc_pool_start(stmt) -> int:
    alloc_pool_start = 0

    def collect_alloc_tensor(op: Bind):
        nonlocal alloc_pool_start
        if not isinstance(op.value, Call) or op.value.op != Op.get("tirx.alloc_tensor"):
            return
        tensor = op.var
        allocated_addr = op.value.args[3].fields if len(op.value.args) == 4 else []
        if len(allocated_addr) == 0:
            return
        shape = op.value.args[0].fields
        dtype = op.value.args[1].value
        scope = op.value.args[2].value
        tensor_size = get_tensor_size(tensor, shape, dtype, scope)
        if tensor_size is None:
            return
        alloc_pool_start = max(alloc_pool_start, allocated_addr[-1] + tensor_size)

    tvm_ffi.structural_walk(stmt, (Bind, collect_alloc_tensor), order="post")
    return alloc_pool_start


def _allocate_missing_tensors(stmt, alloc_pool_start: int):
    alloc_offset = alloc_pool_start

    def allocate_tensor(op: Bind):
        nonlocal alloc_offset
        if not isinstance(op.value, Call) or op.value.op != Op.get("tirx.alloc_tensor"):
            return op
        tensor = op.var
        shape = op.value.args[0].fields
        dtype = op.value.args[1].value
        scope = op.value.args[2].value
        tensor_size = get_tensor_size(tensor, shape, dtype, scope)
        allocated_addr = op.value.args[3].fields if len(op.value.args) == 4 else []
        if len(allocated_addr) == 0 and tensor_size is not None:
            args = list(op.value.args[:3])
            args.append(Tuple([IntImm("int32", int(alloc_offset))]))
            alloc_offset += tensor_size
            return Bind(
                tensor,
                Call(
                    op.value.op,
                    args,
                    attrs=op.value.attrs,
                    ty_args=op.value.ty_args,
                    loc=op.value.loc,
                    ty=op.value.ty,
                ),
                op.loc,
            )
        return op

    return tvm_ffi.structural_map(
        stmt,
        (Bind, allocate_tensor),
        order="pre",
    )


@function_pass(opt_level=0, name="TrnNaiveAllocator")
class TrnNaiveAllocator:
    def transform_function(self, func, mod, ctx):
        alloc_pool_start = _get_alloc_pool_start(func.body)
        new_body = _allocate_missing_tensors(func.body, alloc_pool_start)
        return func.with_body(new_body)
