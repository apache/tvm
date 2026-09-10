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

from tvm.ir import Range
from tvm.target import Target
from tvm.tirx.buffer import Buffer
from tvm.tirx.stmt import (
    AllocBuffer,
    AttrStmt,
    For,
    SeqStmt,
    Stmt,
)
from tvm.tirx.tile_primitive import DispatchContext, TilePrimitiveCall
from tvm.tirx.transform.common import seek_kernel_replace_point
from tvm.tirx.transform.function_pass import prim_func_pass


def _collect_private_allocations(stmt: Stmt, target: Target):
    launch_params = {}
    var_range_map = {}
    buffer_dict = {}
    private_buf_refs = {}

    def visit_attr(op: AttrStmt):
        if op.attr_key == "thread_extent":
            launch_params[op.node.thread_tag] = op.value

    def visit_for(op: For):
        var_range_map[op.loop_var] = Range.from_min_extent(op.min, op.extent)

    def visit_op_call(op: TilePrimitiveCall):
        # Scope is a per-call field on the node; read it directly.
        exec_scope = op.scope
        scope_kind = op.scope.name
        sctx = DispatchContext(
            target=target,
            exec_scope=exec_scope,
            launch_params=launch_params,
            var_range_map=var_range_map,
            alloc_only=True,
            scope_kind=scope_kind,
        )
        op = TilePrimitiveCall.downcast(op)
        private_buf_refs[op] = op.get_private_buffers(buffer_dict, sctx)

    tvm_ffi.structural_walk(
        stmt,
        [(AttrStmt, visit_attr), (For, visit_for), (TilePrimitiveCall, visit_op_call)],
        order="pre",
    )
    return buffer_dict, private_buf_refs


def _inject_private_allocations(
    stmt: Stmt,
    alloc_buffers: list[Buffer],
    init_stmts: list[Stmt],
    added_workspace: dict[TilePrimitiveCall, dict[str, Buffer]],
) -> Stmt:
    is_outer_block = True

    def visit_attr(op: AttrStmt):
        nonlocal is_outer_block
        # AttrStmt(kDeviceEntry) marks the device-region root: inject the
        # collected init stmts + alloc_buffers into its body.
        if op.attr_key == "tirx.device_entry":
            is_outer = is_outer_block
            is_outer_block = False
            if is_outer:
                body = op.body
                for init_stmt in init_stmts:
                    body = seek_kernel_replace_point(init_stmt, body)
                for buffer in reversed(alloc_buffers):
                    body = SeqStmt([AllocBuffer(buffer), body])
                return AttrStmt(op.node, op.attr_key, op.value, body)
        return op

    def visit_op_call(op: TilePrimitiveCall):
        if op not in added_workspace:
            return op
        new_workspace = dict(op.workspace)
        new_workspace.update(added_workspace[op])
        return TilePrimitiveCall.downcast(op).with_workspace(new_workspace)

    return tvm_ffi.structural_map(
        stmt,
        [(AttrStmt, visit_attr), (TilePrimitiveCall, visit_op_call)],
        order="pre",
    )


def private_alloc(stmt: Stmt, target: Target) -> Stmt:
    buffer_dict, private_buf_refs = _collect_private_allocations(stmt, target)

    alloc_buffers = [buffer for buffer, _ in buffer_dict.values()]
    init_stmts = [stmt for _, stmt in buffer_dict.values() if stmt is not None]
    added_workspace = {
        op: {name: buffer_dict[ref][0] for name, ref in private_buf_refs[op].items()}
        for op in private_buf_refs
    }

    return _inject_private_allocations(stmt, alloc_buffers, init_stmts, added_workspace)


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
