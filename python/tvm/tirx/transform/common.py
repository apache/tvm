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


import tvm_ffi

from tvm.ir import Call, Op, is_prim_expr
from tvm.tirx import Evaluate, Expr, Stmt, TilePrimitiveCall, Var, decl_buffer
from tvm.tirx.buffer import Buffer, is_buffer_var
from tvm.tirx.layout import Iter, TileLayout


class BufferReplacer:
    """
    Replace buffer with another buffer.
    Buffer values are ordinary Vars, so the same mapping also rewrites
    ``buffer_data`` projections.
    """

    def __init__(
        self, buffer_map: dict[Buffer, Buffer] | None = None, var_map: dict[Var, Var] | None = None
    ):
        super().__init__()
        self.buffer_map = buffer_map if buffer_map is not None else {}
        self.var_map = var_map if var_map is not None else {}
        for old_buffer, new_buffer in self.buffer_map.items():
            self.var_map[old_buffer] = new_buffer

    def __call__(self, node):
        def replace_var(op: Var):
            if is_buffer_var(op):
                return self._mutate_buffer(op)
            return self.var_map.get(op, op)

        def replace_op_call(op: TilePrimitiveCall):
            new_workspace = {key: self._mutate_buffer(value) for key, value in op.workspace.items()}
            new_config = {
                key: self._replace_expr(value) if is_prim_expr(value) else value
                for key, value in op.config.items()
            }
            return TilePrimitiveCall(
                *op.args,
                op=op.op,
                workspace=new_workspace,
                config=new_config,
                dispatch=op.dispatch,
                scope=op.scope,
            )

        return tvm_ffi.structural_map(
            node,
            [(TilePrimitiveCall, replace_op_call), (Var, replace_var)],
            order="post",
        )

    def _replace_expr(self, expr: Expr):
        return tvm_ffi.structural_map(
            expr,
            (Var, lambda var: self.var_map.get(var, var)),
            order="post",
        )

    def _mutate_buffer(self, buffer: Buffer):
        if buffer in self.buffer_map:
            return self.buffer_map[buffer]

        new_shape = [self._replace_expr(expr) for expr in buffer.ty.shape]
        new_strides = [self._replace_expr(expr) for expr in buffer.ty.strides]
        new_elem_offset = (
            self._replace_expr(buffer.ty.elem_offset) if buffer.ty.elem_offset is not None else None
        )
        if isinstance(buffer.ty.layout, TileLayout):
            new_shard = [
                Iter(self._replace_expr(it.extent), self._replace_expr(it.stride), it.axis)
                for it in buffer.ty.layout.shard
            ]
            new_replicate = [
                Iter(self._replace_expr(it.extent), self._replace_expr(it.stride), it.axis)
                for it in buffer.ty.layout.replica
            ]
            new_layout = TileLayout.from_iters(
                new_shard,
                new_replicate,
                offset=buffer.ty.layout.offset,
            )
        else:
            new_layout = buffer.ty.layout
        new_allocated_addr = [self._replace_expr(expr) for expr in buffer.ty.allocated_addr]

        unchanged = (
            all(old is new for old, new in zip(buffer.ty.shape, new_shape))
            and all(old is new for old, new in zip(buffer.ty.strides, new_strides))
            and buffer.ty.elem_offset is new_elem_offset
            and buffer.ty.layout is new_layout
            and all(old is new for old, new in zip(buffer.ty.allocated_addr, new_allocated_addr))
        )
        if unchanged:
            return buffer

        new_buffer = decl_buffer(
            new_shape,
            buffer.ty.dtype,
            buffer.name,
            None,
            new_strides,
            new_elem_offset,
            buffer.scope(),
            buffer.ty.data_alignment,
            buffer.ty.offset_factor,
            layout=new_layout,
        )
        if new_allocated_addr:
            new_buffer = new_buffer.with_allocated_addr(new_allocated_addr)
        self.buffer_map[buffer] = new_buffer
        self.var_map[buffer] = new_buffer
        return new_buffer


def seek_kernel_replace_point(stmt: Stmt, body: Stmt) -> Stmt:
    """Replace the kernel replacement point in ``stmt`` with ``body``."""

    def replace_evaluate(op: Evaluate):
        value = op.value
        if isinstance(value, Call) and value.op.same_as(Op.get("tirx.tvm_kernel_replace_point")):
            return body
        return op

    return tvm_ffi.structural_map(stmt, (Evaluate, replace_evaluate), order="post")
