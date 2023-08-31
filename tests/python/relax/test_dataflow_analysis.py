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
from typing import Any, Callable, List, Optional
import tvm
from tvm import relax
from tvm.relax.analysis.dataflow_analysis import (
    ControlFlowGraph,
    ExtractCFG,
    DataflowAnalysis,
    BindingNodeKind,
)
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


def assert_binding_fields(
    graph: ControlFlowGraph,
    idx: int,
    block_idx: int,
    binding_idx: int,
    kind: BindingNodeKind = BindingNodeKind.kBinding,
    args: Optional[List[relax.Var]] = None,
):
    binding = graph.bindings[idx]
    assert binding.block_idx == block_idx
    assert binding.binding_idx == binding_idx
    assert binding.kind == kind.value
    if args is not None:
        assert len(binding.args) == len(args)
        for i in range(len(args)):
            assert binding.args[i] == args[i]


# assert that the SeqExprs for each bindings match within groups and do not match other groups
def assert_distinct_seqs(cfg: ControlFlowGraph, *groups: List[int]):
    for i, group in enumerate(groups):
        if len(group) == 0:
            continue
        for idx in group[1:]:
            assert cfg.bindings[idx].seq == cfg.bindings[group[0]].seq
        for other_group in groups[i + 1 :]:
            for idx in other_group:
                assert cfg.bindings[group[0]].seq != cfg.bindings[idx].seq


def test_trivial_CFG():
    @I.ir_module
    class TrivialFunc:
        @R.function
        def main() -> R.Tensor((), "int32"):
            return R.const(1, dtype="int32")

    graph = ExtractCFG(TrivialFunc["main"])
    assert len(graph.bindings) == 1
    assert_pred_succ_lists(graph, [[]])
    assert_binding_fields(graph, 0, 0, 0, kind=BindingNodeKind.kSeqBody)


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
    assert len(graph.bindings) == 4
    assert_pred_succ_lists(graph, [[], [0], [1], [2]])
    assert_binding_fields(graph, 0, 0, 0, args=[FuncWithBindings["main"].params[0]])
    assert_binding_fields(graph, 1, 0, 1)
    assert_binding_fields(graph, 2, 0, 2)
    assert_binding_fields(graph, 3, 1, 0, kind=BindingNodeKind.kSeqBody)


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
    assert len(graph.bindings) == 8
    assert_pred_succ_lists(graph, [[], [0], [1], [2], [3], [4], [5], [6]])
    assert_binding_fields(graph, 0, 0, 0, args=FuncWithDataflow["main"].params)
    assert_binding_fields(graph, 1, 0, 1)
    assert_binding_fields(graph, 2, 1, 0)
    assert_binding_fields(graph, 3, 1, 1)
    assert_binding_fields(graph, 4, 2, 0)
    assert_binding_fields(graph, 5, 3, 0)
    assert_binding_fields(graph, 6, 3, 1)
    assert_binding_fields(graph, 7, 4, 0, kind=BindingNodeKind.kSeqBody)


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

    graph = ExtractCFG(SimpleBranch["main"])

    # cond binding + 3 bindings in true branch + true branch end
    #   + 3 bindings in false branch + false branch end + merge + seq body
    assert len(graph.bindings) == 11
    assert_pred_succ_lists(graph, [[], [0], [1], [2], [3], [0], [5], [6], [7], [4, 8], [9]])

    assert_binding_fields(
        graph, 0, 0, 0, kind=BindingNodeKind.kIfCond, args=SimpleBranch["main"].params
    )
    assert_binding_fields(graph, 1, 0, 0)
    assert_binding_fields(graph, 2, 0, 1)
    assert_binding_fields(graph, 3, 0, 2)
    assert_binding_fields(graph, 4, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 5, 0, 0)
    assert_binding_fields(graph, 6, 0, 1)
    assert_binding_fields(graph, 7, 0, 2)
    assert_binding_fields(graph, 8, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 9, 0, 0, kind=BindingNodeKind.kIfMerge)
    assert_binding_fields(graph, 10, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_distinct_seqs(graph, [0, 9], [1, 4], [5, 8])


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
    assert len(graph.bindings) == 10
    assert_pred_succ_lists(graph, [[], [0], [1], [2], [3], [2], [5], [4, 6], [7], [8]])
    assert_binding_fields(graph, 0, 0, 0, args=BranchAndBind["main"].params)
    assert_binding_fields(graph, 1, 0, 1)
    assert_binding_fields(graph, 2, 0, 2, kind=BindingNodeKind.kIfCond)
    assert_binding_fields(graph, 3, 0, 0)
    assert_binding_fields(graph, 4, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 5, 0, 0)
    assert_binding_fields(graph, 6, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 7, 0, 2, kind=BindingNodeKind.kIfMerge)
    assert_binding_fields(graph, 8, 0, 3)
    assert_binding_fields(graph, 9, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_distinct_seqs(graph, [0, 2, 7, 9], [3, 4], [5, 6])


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
    assert len(graph.bindings) == 19
    assert_pred_succ_lists(
        graph,
        [
            [],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [0],
            [9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [8, 16],
            [17],
        ],
    )

    assert_binding_fields(
        graph, 0, 0, 0, kind=BindingNodeKind.kIfCond, args=LongBranches["main"].params
    )
    assert_binding_fields(graph, 1, 0, 0)
    assert_binding_fields(graph, 2, 0, 1)
    assert_binding_fields(graph, 3, 1, 0)
    assert_binding_fields(graph, 4, 1, 1)
    assert_binding_fields(graph, 5, 1, 2)
    assert_binding_fields(graph, 6, 2, 0)
    assert_binding_fields(graph, 7, 2, 1)
    assert_binding_fields(graph, 8, 3, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 9, 0, 0)
    assert_binding_fields(graph, 10, 0, 1)
    assert_binding_fields(graph, 11, 1, 0)
    assert_binding_fields(graph, 12, 1, 1)
    assert_binding_fields(graph, 13, 1, 2)
    assert_binding_fields(graph, 14, 2, 0)
    assert_binding_fields(graph, 15, 2, 1)
    assert_binding_fields(graph, 16, 3, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 17, 0, 0, kind=BindingNodeKind.kIfMerge)
    assert_binding_fields(graph, 18, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_distinct_seqs(graph, [0, 17, 18], [1, 8], [9, 16])


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
    assert len(graph.bindings) == 22
    assert_pred_succ_lists(
        graph,
        [
            [],  # first binding
            [0],  # branch cond
            [1],  # first binding in true branch
            [2],  # mested if condition
            [3],  # binding inside nested true branch
            [4],  # end of nested true branch
            [3],  # binding inside nested false branch
            [6],  # end of nested false branch
            [5, 7],  # merge for nested if
            [8],  # binding after nested if
            [9],  # end of outer true branch
            [1],  # first binding in false branch
            [11],  # nested if condition,
            [12],  # binding inside nested true branch
            [13],  # end of nested true branch
            [12],  # binding inside nested false branch
            [15],  # end of nested false branch
            [14, 16],  # merge after nested if
            [17],  # binding after nested if
            [18],  # end of outer false branch
            [10, 19],  # merge after outer if
            [20],  # end of body
        ],
    )

    assert_binding_fields(graph, 0, 0, 0, args=NestedBranches["main"].params)
    assert_binding_fields(graph, 1, 0, 1, kind=BindingNodeKind.kIfCond)
    assert_binding_fields(graph, 2, 0, 0)
    assert_binding_fields(graph, 3, 0, 1, kind=BindingNodeKind.kIfCond)
    assert_binding_fields(graph, 4, 0, 0)
    assert_binding_fields(graph, 5, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 6, 0, 0)
    assert_binding_fields(graph, 7, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 8, 0, 1, kind=BindingNodeKind.kIfMerge)
    assert_binding_fields(graph, 9, 0, 2)
    assert_binding_fields(graph, 10, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 11, 0, 0)
    assert_binding_fields(graph, 12, 0, 1, kind=BindingNodeKind.kIfCond)
    assert_binding_fields(graph, 13, 0, 0)
    assert_binding_fields(graph, 14, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 15, 0, 0)
    assert_binding_fields(graph, 16, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 17, 0, 1, kind=BindingNodeKind.kIfMerge)
    assert_binding_fields(graph, 18, 0, 2)
    assert_binding_fields(graph, 19, 1, 0, kind=BindingNodeKind.kSeqBody)
    assert_binding_fields(graph, 20, 0, 1, kind=BindingNodeKind.kIfMerge)
    assert_binding_fields(graph, 21, 1, 0, kind=BindingNodeKind.kSeqBody)

    assert_distinct_seqs(
        graph,
        [0, 1, 20, 21],
        [2, 3, 8, 9, 10],
        [4, 5],
        [6, 7],
        [11, 12, 17, 18, 19],
        [13, 14],
        [15, 16],
    )


def test_simple_analysis():
    @I.ir_module
    class TrivialFunc:
        @R.function
        def main() -> R.Tensor((), "int32"):
            return R.const(1, dtype="int32")

    # only one binding to consider here
    init = {"a": 1}

    def transfer_func(_, domain):
        # the input domain will be converted into an immutable TVM Map,
        # so we have to create a new domain
        new_domain = {}
        for k, v in domain.items():
            new_domain[k] = v
        new_domain["b"] = 2
        return new_domain

    # there will not be a merge here
    merge_func = lambda domain1, _: domain1

    def check_expected_maps(in_map, out_map):
        # we expect the in map to be the init value and the out map to have the key b
        assert len(in_map[0]) == 1
        assert in_map[0]["a"] == 1
        assert len(out_map[0]) == 2
        assert out_map[0]["a"] == 1
        assert out_map[0]["b"] == 2

    cfg = ExtractCFG(TrivialFunc["main"])
    in_map, out_map = DataflowAnalysis(cfg, init, transfer_func, merge_func, forward=True)
    check_expected_maps(in_map, out_map)
    # backward will just flip in and out
    in_map, out_map = DataflowAnalysis(cfg, init, transfer_func, merge_func, forward=False)
    check_expected_maps(out_map, in_map)


def test_simple_analysis_with_merge():
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

    init = {"a": 1}

    def transfer_func(_, domain):
        new_domain = {}
        for k, v in domain.items():
            new_domain[k] = v + 1
        return new_domain

    def merge_func(domain1, domain2):
        new_domain = {}
        for k, v in domain1.items():
            new_domain[k] = v
        for k, v in domain2.items():
            if k not in new_domain or (k in new_domain and new_domain[k] < v):
                new_domain[k] = v
        if "merge" not in new_domain:
            new_domain["merge"] = 1
        return new_domain

    cfg = ExtractCFG(SimpleBranch["main"])
    in_map, out_map = DataflowAnalysis(cfg, init, transfer_func, merge_func, forward=True)
    # start and true branch
    for i in range(5):
        assert in_map[i]["a"] == i + 1
        assert out_map[i]["a"] == i + 2
    # false branch
    for i in range(5, 9):
        assert in_map[i]["a"] == i - 3
        assert out_map[i]["a"] == i - 2
    # index 9 is the merge
    assert in_map[9]["a"] == 6
    assert in_map[9]["merge"] == 1
    assert out_map[9]["a"] == 7
    assert out_map[9]["merge"] == 2
    # index 10 is the last
    assert in_map[10]["a"] == 7
    assert in_map[10]["merge"] == 2
    assert out_map[10]["a"] == 8
    assert out_map[10]["merge"] == 3

    in_map, out_map = DataflowAnalysis(cfg, init, transfer_func, merge_func, forward=False)
    # backward direction: start with index 10
    # end of seq through false branch
    for i in range(6):
        assert out_map[10 - i]["a"] == i + 1
        assert in_map[10 - i]["a"] == i + 2
    # true branch
    for i in range(4):
        assert out_map[4 - i]["a"] == i + 3
        assert in_map[4 - i]["a"] == i + 4
    # the if condition is the merge
    assert out_map[0]["a"] == 7
    assert out_map[0]["merge"] == 1
    assert in_map[0]["a"] == 8
    assert in_map[0]["merge"] == 2


if __name__ == "__main__":
    tvm.testing.main()
