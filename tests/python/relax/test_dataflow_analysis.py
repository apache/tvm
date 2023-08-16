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
from typing import List, Optional
import tvm
from tvm import relax
from tvm.relax.analysis.dataflow_analysis import ControlFlowGraph, BasicBlock, ExtractCFG
from tvm.script import ir as I, relax as R
import tvm.testing


def assert_pred_succ_lists(graph: ControlFlowGraph, expected_preds: List[List[int]]) -> None:
    assert tuple([tuple(preds) for preds in graph.preds]) == tuple(
        [tuple(exp_preds) for exp_preds in expected_preds]
    )

    expected_succs = [[] for preds in expected_preds]
    # we can automatically invert the predecessor list
    # this also guarantees consistency
    for i, pred_list in enumerate(expected_preds):
        for pred in pred_list:
            expected_succs[pred].append(i)

    assert tuple([tuple(succs) for succs in graph.succs]) == tuple(
        [tuple(exp_succs) for exp_succs in expected_succs]
    )


# common pattern in normalization that we can check for:
# if condition:
#    ...
#    z = value1
# else:
#    ...
#    z = value2
#
# results in:
#
# VarBinding(
#     z,
#     If(
#         condition,
#         SeqExpr([..., BindingBlock([..., VarBinding(new_var1, value1)])], body=new_var1),
#         SeqExpr([..., BindingBlock([..., VarBinding(new_var2, value2)])], body=new_var2)
#     )
# )
# This function can be used for checking the SeqExprs inside the branches
def assert_ret_is_final_binding_in_seq(block: BasicBlock, check_op: Optional[str] = None):
    seq_body = block.seq.body
    final_binding = block.seq.blocks[-1].bindings[-1]
    assert seq_body == final_binding.var
    assert block.ret == seq_body
    if check_op is not None:
        assert isinstance(final_binding.value, relax.Call)
        assert final_binding.value.op.name == check_op


# ensure that the exprs in each list match each other and that they do not match those in the other lists
def assert_distinct(*groups: List[relax.Expr]):
    for i, group in enumerate(groups):
        if len(group) == 0:
            continue
        for item in group[1:]:
            assert item == group[0]
        for other_group in groups[i + 1 :]:
            for item in other_group:
                assert group[0] != item


def test_trivial_CFG():
    @I.ir_module
    class TrivialFunc:
        @R.function
        def main() -> R.Tensor((), "int32"):
            return R.const(1, dtype="int32")

    graph = ExtractCFG(TrivialFunc["main"])
    assert len(graph.blocks) == 1
    assert_pred_succ_lists(graph, [[]])
    assert graph.blocks[0].ret == TrivialFunc["main"].body.body
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    assert graph.blocks[0].end_block_idx == 0


def test_sequence_of_bindings():
    @I.ir_module
    class FuncWithBindings:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.add(y, x)
            q = R.multiply(z, x)
            return q

    graph = ExtractCFG(FuncWithBindings["main"])
    assert len(graph.blocks) == 1
    assert_pred_succ_lists(graph, [[]])
    assert graph.blocks[0].ret == FuncWithBindings["main"].body.body
    assert graph.blocks[0].args == FuncWithBindings["main"].params
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    assert graph.blocks[0].end_block_idx == 1


def test_dataflow_block():
    @I.ir_module
    class FuncWithDataflow:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            y = R.add(x, x)
            z = R.add(y, y)
            with R.dataflow():
                q = R.multiply(z, z)
                r = R.add(q, q)
                R.output(r)
            s = R.add(r, r)
            with R.dataflow():
                t = R.multiply(s, s)
                u = R.add(t, t)
                R.output(u)
            return u

    graph = ExtractCFG(FuncWithDataflow["main"])
    assert len(graph.blocks) == 1
    assert_pred_succ_lists(graph, [[]])
    assert graph.blocks[0].ret == FuncWithDataflow["main"].body.body
    assert graph.blocks[0].args == FuncWithDataflow["main"].params
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    # there are four binding blocks but they form one basic block
    assert graph.blocks[0].end_block_idx == 4


def test_simple_branch():
    @I.ir_module
    class SimpleBranch:
        @R.function
        def main(cond: R.Tensor((), "bool")) -> R.Tensor((), "int32"):
            if cond:
                x = R.const(1, dtype="int32")
                y = R.add(x, x)
                z = R.multiply(y, y)
            else:
                x = R.const(2, dtype="int32")
                y = R.add(x, x)
                z = R.multiply(y, y)
            return z

    # basic blocks:
    # 1. the starting block (no bindings) whose return is the branch condition
    # 2. the true branch body (return: R.multiply(y, y))
    # 3. the false branch body (return: R.multiply(y, y))
    # 4. the merge block (no bindings, argument is z) whose return is z
    graph = ExtractCFG(SimpleBranch["main"])
    assert len(graph.blocks) == 4
    assert_pred_succ_lists(graph, [[], [0], [0], [1, 2]])

    assert graph.blocks[0].args == SimpleBranch["main"].params
    assert graph.blocks[0].ret == SimpleBranch["main"].params[0]
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    assert graph.blocks[0].end_block_idx == 0
    assert graph.blocks[0].end_binding_idx == 0

    assert len(graph.blocks[1].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[1], "relax.multiply")
    assert graph.blocks[1].start_block_idx == 0
    assert graph.blocks[1].start_binding_idx == 0
    assert graph.blocks[1].end_block_idx == 1

    assert len(graph.blocks[2].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[2], "relax.multiply")
    assert graph.blocks[2].start_block_idx == 0
    assert graph.blocks[2].start_binding_idx == 0
    assert graph.blocks[2].end_block_idx == 1

    assert len(graph.blocks[3].args) == 1
    assert graph.blocks[3].args[0].name_hint == "z"
    assert graph.blocks[3].ret == SimpleBranch["main"].body.body
    # the if was the last binding in the block, so we're past the end
    assert graph.blocks[3].start_block_idx == 1
    assert graph.blocks[3].end_block_idx == 1

    assert_distinct(
        [graph.blocks[0].seq, graph.blocks[3].seq], [graph.blocks[1]], [graph.blocks[2]]
    )


def test_bindings_after_branch():
    @I.ir_module
    class BranchAndBind:
        @R.function
        def main(cond: R.Tensor((), "bool")) -> R.Tensor((), "int32"):
            x = R.const(1, dtype="int32")
            y = R.add(x, x)
            if cond:
                z = R.multiply(y, y)
            else:
                z = R.add(y, y)
            q = R.add(z, z)
            return q

    graph = ExtractCFG(BranchAndBind["main"])
    assert len(graph.blocks) == 4
    assert_pred_succ_lists(graph, [[], [0], [0], [1, 2]])

    # same as above example, except there are bindings preceding the if (included in block 0)
    # and after the if (included in block 3)

    assert graph.blocks[0].args == BranchAndBind["main"].params
    assert graph.blocks[0].ret == BranchAndBind["main"].params[0]
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    assert graph.blocks[0].end_block_idx == 0
    assert graph.blocks[0].end_binding_idx == 2

    assert len(graph.blocks[1].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[1], "relax.multiply")
    assert graph.blocks[1].start_block_idx == 0
    assert graph.blocks[1].start_binding_idx == 0
    assert graph.blocks[1].end_block_idx == 1

    assert len(graph.blocks[2].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[2], "relax.add")
    assert graph.blocks[2].start_block_idx == 0
    assert graph.blocks[2].start_binding_idx == 0
    assert graph.blocks[2].end_block_idx == 1

    assert len(graph.blocks[3].args) == 1
    assert graph.blocks[3].args[0].name_hint == "z"
    assert graph.blocks[3].ret.name_hint == "q"
    assert graph.blocks[3].start_block_idx == 0
    assert graph.blocks[3].start_binding_idx == 3
    assert graph.blocks[3].end_block_idx == 1
    assert graph.blocks[3].end_binding_idx == 0

    assert_distinct(
        [graph.blocks[0].seq, graph.blocks[3].seq], [graph.blocks[1]], [graph.blocks[2]]
    )


def test_branch_with_multiple_blocks():
    @I.ir_module
    class LongBranches:
        @R.function
        def main(cond: R.Tensor((), "bool")) -> R.Tensor((), "int32"):
            if cond:
                x = R.const(1, dtype="int32")
                y = R.add(x, x)
                with R.dataflow():
                    z = R.multiply(y, y)
                    w = R.add(z, z)
                    v = R.multiply(w, w)
                    R.output(v)
                q = R.add(v, v)
                r = R.multiply(q, q)
            else:
                x = R.const(2, dtype="int32")
                y = R.multiply(x, x)
                with R.dataflow():
                    z = R.add(y, y)
                    w = R.multiply(z, z)
                    v = R.add(w, w)
                    R.output(v)
                q = R.multiply(v, v)
                r = R.add(q, q)
            return r

    graph = ExtractCFG(LongBranches["main"])
    # empty entry block, one block for each branch, and an empty exit block
    assert len(graph.blocks) == 4
    assert_pred_succ_lists(graph, [[], [0], [0], [1, 2]])

    assert graph.blocks[0].args == LongBranches["main"].params
    assert graph.blocks[0].ret == LongBranches["main"].params[0]
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    assert graph.blocks[0].end_block_idx == 0
    assert graph.blocks[0].end_binding_idx == 0

    # there are 3 binding blocks included in each branch
    assert len(graph.blocks[1].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[1], "relax.multiply")
    assert graph.blocks[1].start_block_idx == 0
    assert graph.blocks[1].start_binding_idx == 0
    assert graph.blocks[1].end_block_idx == 3

    assert len(graph.blocks[2].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[2], "relax.add")
    assert graph.blocks[2].start_block_idx == 0
    assert graph.blocks[2].start_binding_idx == 0
    assert graph.blocks[2].end_block_idx == 3

    assert len(graph.blocks[3].args) == 1
    assert graph.blocks[3].args[0].name_hint == "r"
    assert graph.blocks[3].ret.name_hint == "r"
    assert graph.blocks[3].start_block_idx == 1
    assert graph.blocks[3].end_block_idx == 1

    assert_distinct(
        [graph.blocks[0].seq, graph.blocks[3].seq], [graph.blocks[1]], [graph.blocks[2]]
    )


def test_nested_branches():
    @I.ir_module
    class NestedBranches:
        @R.function
        def main(x: R.Tensor((), "int32")) -> R.Tensor((), "int32"):
            cond1 = R.const(True, dtype="bool")
            if cond1:
                cond2 = R.const(False, dtype="bool")
                if cond2:
                    y = R.add(x, x)
                else:
                    y = R.multiply(x, x)
                z = R.add(y, y)
            else:
                cond3 = R.const(True, dtype="bool")
                if cond3:
                    y = R.multiply(x, x)
                else:
                    y = R.add(x, x)
                z = R.multiply(y, y)
            return z

    graph = ExtractCFG(NestedBranches["main"])
    # basic blocks: entry block to func, entry block to true branch, true branch in true branch,
    #   false branch in true branch, merge block in true branch,
    #   entry to false branch, true branch in false branch, false branch in false branch,
    #   merge block in false branch, merge block in outer function
    assert len(graph.blocks) == 10
    assert_pred_succ_lists(
        graph,
        [
            [],  # function entry
            [0],  # true branch entry
            [1],  # true branch's true branch
            [1],  # true branch's false branch
            [2, 3],  # true branch's exit
            [0],  # false branch entry
            [5],  # false branch's true branch
            [5],  # false branch's false branch
            [6, 7],  # false branch exit
            [4, 8],  # function exit
        ],
    )

    assert graph.blocks[0].args == NestedBranches["main"].params
    assert graph.blocks[0].ret.name_hint == "cond1"
    assert graph.blocks[0].start_block_idx == 0
    assert graph.blocks[0].start_binding_idx == 0
    assert graph.blocks[0].end_block_idx == 0
    assert graph.blocks[0].end_binding_idx == 1

    assert len(graph.blocks[1].args) == 0
    assert graph.blocks[1].ret.name_hint == "cond2"
    assert graph.blocks[1].start_block_idx == 0
    assert graph.blocks[1].start_binding_idx == 0
    assert graph.blocks[1].end_block_idx == 0
    assert graph.blocks[1].end_binding_idx == 1

    assert len(graph.blocks[2].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[2], "relax.add")
    assert graph.blocks[2].start_block_idx == 0
    assert graph.blocks[2].start_binding_idx == 0
    assert graph.blocks[2].end_block_idx == 1

    assert len(graph.blocks[3].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[3], "relax.multiply")
    assert graph.blocks[3].start_block_idx == 0
    assert graph.blocks[3].start_binding_idx == 0
    assert graph.blocks[3].end_block_idx == 1

    assert len(graph.blocks[4].args) == 1
    assert graph.blocks[4].args[0].name_hint == "y"
    assert_ret_is_final_binding_in_seq(graph.blocks[4], "relax.add")
    assert graph.blocks[4].start_block_idx == 0
    assert graph.blocks[4].start_binding_idx == 2
    assert graph.blocks[4].end_block_idx == 1

    assert len(graph.blocks[5].args) == 0
    assert graph.blocks[5].ret.name_hint == "cond3"
    assert graph.blocks[5].start_block_idx == 0
    assert graph.blocks[5].start_binding_idx == 0
    assert graph.blocks[5].end_block_idx == 0
    assert graph.blocks[5].end_binding_idx == 1

    assert len(graph.blocks[6].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[6], "relax.multiply")
    assert graph.blocks[6].start_block_idx == 0
    assert graph.blocks[6].start_binding_idx == 0
    assert graph.blocks[6].end_block_idx == 1

    assert len(graph.blocks[7].args) == 0
    assert_ret_is_final_binding_in_seq(graph.blocks[7], "relax.add")
    assert graph.blocks[7].start_block_idx == 0
    assert graph.blocks[7].start_binding_idx == 0
    assert graph.blocks[7].end_block_idx == 1

    assert len(graph.blocks[8].args) == 1
    assert graph.blocks[8].args[0].name_hint == "y"
    assert_ret_is_final_binding_in_seq(graph.blocks[8], "relax.multiply")
    assert graph.blocks[8].start_block_idx == 0
    assert graph.blocks[8].start_binding_idx == 2
    assert graph.blocks[8].end_block_idx == 1

    assert len(graph.blocks[9].args) == 1
    assert graph.blocks[9].args[0].name_hint == "z"
    assert graph.blocks[9].ret.name_hint == "z"
    assert graph.blocks[9].start_block_idx == 1
    assert graph.blocks[9].end_block_idx == 1


if __name__ == "__main__":
    tvm.testing.main()
