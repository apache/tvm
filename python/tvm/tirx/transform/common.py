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


from tvm.ir import Call, Op, TensorLoad, is_prim_expr
from tvm.tirx import (
    AllocBuffer,
    BufferLoad,
    BufferRegion,
    BufferStore,
    DeclBuffer,
    Evaluate,
    Expr,
    Stmt,
    TilePrimitiveCall,
    Var,
    decl_buffer,
)
from tvm.tirx.buffer import Buffer
from tvm.tirx.layout import Iter, TileLayout
from tvm.tirx.stmt_functor import StmtExprMutator, StmtMutator


class BufferReplacer(StmtExprMutator):
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
        self.buffer_attr_var_mutated = False
        for old_buffer, new_buffer in self.buffer_map.items():
            self.var_map[old_buffer] = new_buffer

    def mutate_buffer(self, buffer: Buffer):
        if buffer in self.buffer_map:
            return self.buffer_map[buffer]

        # Track mutations for this specific buffer only.  Without this reset,
        # unrelated buffers can be spuriously cloned and introduce alias buffers.
        prev_mutated = self.buffer_attr_var_mutated
        self.buffer_attr_var_mutated = False
        new_shape = [self.visit_expr(expr) for expr in buffer.ty.shape]
        new_strides = [self.visit_expr(expr) for expr in buffer.ty.strides]
        new_elem_offset = (
            self.visit_expr(buffer.ty.elem_offset) if buffer.ty.elem_offset is not None else None
        )
        if isinstance(buffer.ty.layout, TileLayout):
            new_shard = []
            new_replicate = []
            for iter in buffer.ty.layout.shard:
                new_iter = Iter(
                    self.visit_expr(iter.extent), self.visit_expr(iter.stride), iter.axis
                )
                new_shard.append(new_iter)
            for iter in buffer.ty.layout.replica:
                new_iter = Iter(
                    self.visit_expr(iter.extent), self.visit_expr(iter.stride), iter.axis
                )
                new_replicate.append(new_iter)
            new_layout = TileLayout.from_iters(
                new_shard, new_replicate, offset=buffer.ty.layout.offset
            )
        else:
            new_layout = buffer.ty.layout
        new_allocated_addr = [self.visit_expr(expr) for expr in buffer.ty.allocated_addr]
        buffer_attr_mutated = self.buffer_attr_var_mutated
        self.buffer_attr_var_mutated = prev_mutated or buffer_attr_mutated
        if not buffer_attr_mutated:
            return None
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

    def visit_var_(self, op: Var):
        if op in self.var_map:
            self.buffer_attr_var_mutated = True
            return self.var_map[op]
        return super().visit_var_(op)

    def visit_buffer_load_(self, op: TensorLoad):
        new_buffer = self.mutate_buffer(op.source)
        op = super().visit_buffer_load_(op)
        if new_buffer is not None:
            return BufferLoad(new_buffer, op.indices)
        return op

    def visit_buffer_store_(self, op: BufferStore):
        new_buffer = self.mutate_buffer(op.buffer)
        op = super().visit_buffer_store_(op)
        if new_buffer is not None:
            return BufferStore(new_buffer, op.value, op.indices)
        return op

    def visit_buffer_region_(self, op: BufferRegion):
        new_buffer = self.mutate_buffer(op.buffer)
        op = super().visit_buffer_region_(op)
        if new_buffer is not None:
            return BufferRegion(new_buffer, op.region)
        return op

    def visit_decl_buffer_(self, op: DeclBuffer):
        new_buffer = self.mutate_buffer(op.buffer)
        data = self.visit_expr(op.data)
        if new_buffer is not None or data is not op.data:
            return DeclBuffer(
                new_buffer if new_buffer is not None else op.buffer,
                data=data,
                span=op.span,
            )
        return op

    def visit_array_prim_expr_(self, op: list[Expr]):
        return [self.visit_expr(expr) for expr in op]

    def visit_alloc_buffer_(self, op: AllocBuffer):
        op = super().visit_alloc_buffer_(op)
        if op.buffer in self.buffer_map:
            return AllocBuffer(self.buffer_map[op.buffer], op.annotations, op.span)
        return op

    def visit_op_call_(self, op):
        op = super().visit_op_call_(op)
        new_workspace = {}
        for key, value in op.workspace.items():
            new_buffer = self.mutate_buffer(value)
            if new_buffer is not None:
                new_workspace[key] = new_buffer
            else:
                new_workspace[key] = value
        new_config = {}
        for key, value in op.config.items():
            if is_prim_expr(value):
                new_config[key] = self.visit_expr(value)
            else:
                new_config[key] = value
        args = list()
        for arg in op.args:
            args.append(arg)
        return TilePrimitiveCall(
            *args,
            op=op.op,
            workspace=new_workspace,
            config=new_config,
            dispatch=op.dispatch,
            scope=op.scope,
        )


class KernelReplacePointSearcher(StmtMutator):
    def __init__(self, body: Stmt):
        super().__init__()
        self.body = body

    def visit_evaluate_(self, op: Evaluate):
        value = op.value
        if isinstance(value, Call) and value.op.same_as(Op.get("tirx.tvm_kernel_replace_point")):
            return self.body
        return super().visit_evaluate_(op)


def seek_kernel_replace_point(stmt: Stmt, body: Stmt) -> Stmt:
    """replace kernel replace point in stmt with body"""
    return KernelReplacePointSearcher(body)(stmt)
