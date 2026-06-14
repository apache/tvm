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


from tvm.ir import Range
from tvm.target import Target
from tvm.tirx.buffer import Buffer
from tvm.tirx.operator.tile_primitive.dispatch_context import DispatchContext
from tvm.tirx.stmt import (
    AllocBuffer,
    AttrStmt,
    For,
    SeqStmt,
    Stmt,
    TilePrimitiveCall,
)
from tvm.tirx.stmt_functor import StmtMutator, StmtVisitor
from tvm.tirx.transform.common import seek_kernel_replace_point
from tvm.tirx.transform.function_pass import prim_func_pass


class PrivateAllocCollector(StmtVisitor):
    def __init__(self, target: Target):
        super().__init__()
        self.target = target
        self.launch_params = {}
        self.var_range_map = {}
        self.buffer_dict = {}
        self.private_buf_refs = {}

    def visit_attr_(self, op: AttrStmt):
        if op.attr_key == "thread_extent":
            self.launch_params[op.node.thread_tag] = op.value
        super().visit_attr_(op)

    def visit_for_(self, op: For):
        self.var_range_map[op.loop_var] = Range.from_min_extent(op.min, op.extent)
        super().visit_for_(op)

    def visit_op_call_(self, op: TilePrimitiveCall):
        # Scope is a per-call field on the node; read it directly.
        exec_scope = op.scope
        scope_kind = op.scope.name
        sctx = DispatchContext(
            target=self.target,
            exec_scope=exec_scope,
            launch_params=self.launch_params,
            var_range_map=self.var_range_map,
            alloc_only=True,
            scope_kind=scope_kind,
        )
        op = TilePrimitiveCall.downcast(op)
        private_buf_refs = op.get_private_buffers(self.buffer_dict, sctx)
        self.private_buf_refs[op] = private_buf_refs


class PrivateAllocMutator(StmtMutator):
    def __init__(
        self,
        alloc_buffers: list[Buffer],
        init_stmts: list[Stmt],
        added_workspace: dict[TilePrimitiveCall, dict[str, Buffer]],
    ):
        super().__init__()
        self.alloc_buffers = alloc_buffers
        self.init_stmts = init_stmts
        self.added_workspace = added_workspace
        self.is_outer_block = True

    def visit_attr_(self, op: AttrStmt):
        # AttrStmt(kDeviceEntry) marks the device-region root: inject the
        # collected init stmts + alloc_buffers into its body.
        if op.attr_key == "tirx.device_entry":
            is_outer_block = self.is_outer_block
            self.is_outer_block = False
            op = super().visit_attr_(op)
            if is_outer_block:
                body = op.body
                for stmt in self.init_stmts:
                    body = seek_kernel_replace_point(stmt, body)
                for buffer in reversed(self.alloc_buffers):
                    body = SeqStmt([AllocBuffer(buffer), body])
                return AttrStmt(op.node, op.attr_key, op.value, body)
            return op
        return super().visit_attr_(op)

    def visit_op_call_(self, op):
        if op not in self.added_workspace:
            return op
        new_workspace = dict(op.workspace)
        new_workspace.update(self.added_workspace[op])
        op = TilePrimitiveCall.downcast(op).with_workspace(new_workspace)
        return op


def private_alloc(stmt: Stmt, target: Target) -> Stmt:
    collector = PrivateAllocCollector(target)
    collector(stmt)

    alloc_buffers = [buffer for buffer, _ in collector.buffer_dict.values()]
    init_stmts = [stmt for _, stmt in collector.buffer_dict.values() if stmt is not None]
    added_workspace = {
        op: {
            name: collector.buffer_dict[ref][0]
            for name, ref in collector.private_buf_refs[op].items()
        }
        for op in collector.private_buf_refs
    }

    mutator = PrivateAllocMutator(alloc_buffers, init_stmts, added_workspace)
    return mutator(stmt)


@prim_func_pass(opt_level=0, name="TrnPrivateBufferAlloc")
class TrnPrivateBufferAlloc:
    """Generate private buffer allocations for each TilePrimitiveCall"""

    def transform_function(self, func, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None:
            target = Target.current(allow_none=False)
        new_body = private_alloc(func.body, target)
        new_func = func.with_body(new_body)
        return new_func
