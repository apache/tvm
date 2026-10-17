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

from tvm_tirx.ir import (
    Bind,
    Call,
    For,
    Op,
    Range,
    RegionStmt,
    SeqStmt,
    Stmt,
    Var,
)
from tvm_tirx.target import Target
from tvm_tirx.tirx.tile.dispatch import DispatchContext
from tvm_tirx.tirx.tile.instruction import SPECS, TensorCall
from tvm_tirx.tirx.transform.common import seek_kernel_replace_point
from tvm_tirx.tirx.transform.function_pass import function_pass


def _collect_private_allocations(stmt: Stmt, target: Target):
    launch_params = {}
    var_range_map = {}
    tensor_dict = {}
    private_buf_refs = {}

    def visit_region(op: RegionStmt):
        if op.op.name != "tirx.launch_thread":
            return None
        tag = op.args[0].value
        if tag.startswith("vthread"):
            return None
        previous_launch_params = dict(launch_params)
        launch_params[tag] = (op.body_params[0], op.args[1])
        visit(op.body)
        launch_params.clear()
        launch_params.update(previous_launch_params)
        return tvm_ffi.WalkResult.SKIP

    def visit_for(op: For):
        var_range_map[op.var] = Range(op.begin, op.end)

    def visit_op_call(op: Call):
        if getattr(op.op, "name", "") not in SPECS:
            return
        call = op
        op = TensorCall.decode(call)
        thread_scope = op.scope
        scope_kind = op.scope.local_name
        sctx = DispatchContext(
            target=target,
            thread_scope=thread_scope,
            launch_params=launch_params,
            var_range_map=var_range_map,
            alloc_only=True,
            scope_kind=scope_kind,
        )
        private_buf_refs[call] = op.get_private_tensors(tensor_dict, sctx)

    def visit(node):
        tvm_ffi.structural_walk(
            node,
            [(RegionStmt, visit_region), (For, visit_for), (Call, visit_op_call)],
            order="pre",
        )

    visit(stmt)
    return tensor_dict, private_buf_refs


def _inject_private_allocations(
    stmt: Stmt,
    alloc_tensors: list[Bind],
    init_stmts: list[Stmt],
    added_workspace: dict[Call, dict[str, Var]],
) -> Stmt:
    is_outer_block = True

    def visit_region(op: RegionStmt):
        nonlocal is_outer_block
        # The device-entry region marks the root: inject the
        # collected init stmts + alloc_tensors into its body.
        if op.op.same_as(Op.get("tirx.device_entry")):
            is_outer = is_outer_block
            is_outer_block = False
            if is_outer:
                body = op.body
                for init_stmt in init_stmts:
                    body = seek_kernel_replace_point(init_stmt, body)
                for allocation in reversed(alloc_tensors):
                    body = SeqStmt([allocation, body])
                return RegionStmt(
                    op.op, op.args, op.body_params, op.attrs, body, op.result_vars, op.loc
                )
        return op

    def visit_op_call(op: Call):
        if op not in added_workspace:
            return op
        new_workspace = dict(TensorCall.decode(op).workspaces)
        new_workspace.update(added_workspace[op])
        return TensorCall.decode(op).with_workspaces(new_workspace)

    return tvm_ffi.structural_map(
        stmt,
        [(RegionStmt, visit_region), (Call, visit_op_call)],
        order="pre",
    )


def private_alloc(stmt: Stmt, target: Target) -> Stmt:
    tensor_dict, private_buf_refs = _collect_private_allocations(stmt, target)

    alloc_tensors = [allocation for allocation, _ in tensor_dict.values()]
    init_stmts = [stmt for _, stmt in tensor_dict.values() if stmt is not None]
    added_workspace = {
        op: {name: tensor_dict[ref][0].var for name, ref in private_buf_refs[op].items()}
        for op in private_buf_refs
    }

    return _inject_private_allocations(stmt, alloc_tensors, init_stmts, added_workspace)


@function_pass(opt_level=0, name="TrnPrivateTensorAlloc")
class TrnPrivateTensorAlloc:
    """Allocate and initialize instruction-private Trainium workspaces.

    Insert workspaces at the device-entry root and attach them to tile calls.
    Run before tile dispatch with a Trainium target. Calls needing no
    workspace are unchanged.

    Examples
    --------
    .. code:: python

        import tvm_tirx as tvm
        from tvm_tirx.script import tirx as T
        from tvm_tirx.backend.trn.axis import P, F
        from tvm_tirx.backend.trn.transform import TrnPrivateTensorAlloc

        tile = T.Tensor((128, 512), "float32", scope="trn.sbuf",
                        layout=T.TileLayout(T.S[(128, 512):(1 @ P, 1 @ F)]))

        @T.ir_module
        class Before:
            @T.function
            def main():
                T.device_entry()
                A = T.alloc_tensor(ty_args=[tile])
                C = T.alloc_tensor(ty_args=[tile])
                T.trn.tile.activation(C, A, bias=T.float32(1), opcode="exp")

        @T.ir_module
        class Expected:
            @T.function
            def main():
                T.device_entry()
                const_bias = T.alloc_tensor(
                    ty_args=[T.Tensor((128, 512), "float32", scope="trn.sbuf")])
                with T.nki.tensorized_instruction():
                    for p in T.serial(128, annotations={"nki_dim": "P"}):
                        for f in T.serial(512, annotations={"nki_dim": "F"}):
                            T.nki.memset(const_bias[p, f], T.float32(1))
                A = T.alloc_tensor(ty_args=[tile])
                C = T.alloc_tensor(ty_args=[tile])
                T.trn.tile.activation(C, A, bias=T.float32(1), const_bias=const_bias, opcode="exp")

        with tvm.target.Target("aws/trn1/trn1.2xlarge"):
            after = TrnPrivateTensorAlloc()(Before)
        tvm.ir.assert_structural_equal(after, Expected)
    """

    def transform_function(self, func, mod, ctx):
        target = func.attrs.get("target", None)
        if target is None:
            target = Target.current(allow_none=False)
        new_body = private_alloc(func.body, target)
        new_func = func.with_body(new_body)
        return new_func
