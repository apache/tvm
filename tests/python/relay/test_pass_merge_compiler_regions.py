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
"""Unit tests for merge compiler regions."""
import tvm
from tvm import relay
import tvm.relay.transform as transform
from tvm.relay.op.annotation import compiler_begin, compiler_end
from tvm.relay.testing import run_opt_pass


def test_diamond_graph_fanouts():
    """
    This tests that the data dependencies present in a diamond-shaped
    graph are correctly resolved by the merging pass.

    O = supported by target
    X = not supported by target

       O         O
      / \\      /               \\
     O   X --> O    +       +    X
     \\ /             \\ /
       O                O

    Note that we can't just merge the three supported operators together,
    otherwise both subgraphs would depend on the other.
    """

    def diamond_graph_fanouts():
        data = relay.var("data", shape=(10, 10))
        cb_1 = compiler_begin(data, "test")
        O_1 = relay.abs(cb_1)
        ce_1 = compiler_end(O_1, "test")
        ce_2 = compiler_end(O_1, "test")
        cb_2 = compiler_begin(ce_1, "test")
        cb_3 = compiler_begin(ce_2, "default")
        O_2 = relay.nn.relu(cb_2)
        ce_3 = compiler_end(O_2, "test")

        X = relay.tanh(cb_3)
        ce_4 = compiler_end(X, "default")

        cb_4 = compiler_begin(ce_3, "test")
        cb_5 = compiler_begin(ce_4, "test")
        O_3 = relay.add(cb_4, cb_5)
        ce_5 = compiler_end(O_3, "test")

        diamond = relay.Function([data], ce_5)
        return diamond

    def expected():
        data = relay.var("data", shape=(10, 10))
        cb_1 = compiler_begin(data, "test")
        O_1 = relay.abs(cb_1)
        ce_2 = compiler_end(O_1, "test")
        O_2 = relay.nn.relu(O_1)
        ce_3 = compiler_end(O_2, "test")

        cb_3 = compiler_begin(ce_2, "default")
        X = relay.tanh(cb_3)
        ce_4 = compiler_end(X, "default")

        cb_4 = compiler_begin(ce_3, "test")
        cb_5 = compiler_begin(ce_4, "test")
        O_3 = relay.add(cb_4, cb_5)
        ce_5 = compiler_end(O_3, "test")

        func = relay.Function([data], ce_5)
        return func

    result = run_opt_pass(diamond_graph_fanouts(), relay.transform.MergeCompilerRegions())
    golden = run_opt_pass(expected(), relay.transform.InferType())
    tvm.ir.assert_structural_equal(result, golden)


def test_example_graph():
    """This tests the merging algorithm on the example used in the RFC.

    See the RFC here: https://discuss.tvm.apache.org/t/relay-improved-graph-partitioning-algorithm/5830
    Blue nodes are adds (target: test), red nodes are subtracts (target: default).
    """

    def annotated():
        in_1 = relay.var("in_1", shape=(10, 10), dtype="float32")
        in_2 = relay.var("in_2", shape=(10, 10), dtype="float32")
        in_3 = relay.var("in_3", shape=(10, 10), dtype="float32")
        in_4 = relay.var("in_4", shape=(10, 10), dtype="float32")
        in_5 = relay.var("in_5", shape=(10, 10), dtype="float32")
        in_6 = relay.var("in_6", shape=(10, 10), dtype="float32")
        in_7 = relay.var("in_7", shape=(10, 10), dtype="float32")
        in_8 = relay.var("in_8", shape=(10, 10), dtype="float32")
        in_9 = relay.var("in_9", shape=(10, 10), dtype="float32")
        in_10 = relay.var("in_10", shape=(10, 10), dtype="float32")

        begin0 = compiler_begin(in_1, "test")
        begin1 = compiler_begin(in_2, "test")
        begin2 = compiler_begin(in_3, "test")
        begin3 = compiler_begin(in_4, "test")
        node0 = relay.add(begin0, begin1)
        node1 = relay.add(begin2, begin3)
        end0 = compiler_end(node0, "test")
        end1 = compiler_end(node1, "test")
        begin4 = compiler_begin(end0, "test")
        begin5 = compiler_begin(end1, "test")
        node2 = relay.add(begin4, begin5)
        end2 = compiler_end(node2, "test")

        dbegin0 = compiler_begin(in_5, "default")
        dbegin1 = compiler_begin(in_6, "default")
        node3 = relay.subtract(dbegin0, dbegin1)
        dbegin2 = compiler_begin(in_7, "default")
        dend1 = compiler_end(node3, "default")
        dbegin3 = compiler_begin(dend1, "default")
        node4 = relay.subtract(dbegin2, dbegin3)
        dend2 = compiler_end(node4, "default")

        begin6 = compiler_begin(end2, "test")
        begin7 = compiler_begin(dend2, "test")
        node5 = relay.add(begin6, begin7)
        end3 = compiler_end(node5, "test")
        end4 = compiler_end(node5, "test")
        dbegin4 = compiler_begin(in_8, "default")
        dbegin5 = compiler_begin(end3, "default")
        node6 = relay.subtract(dbegin4, dbegin5)
        begin8 = compiler_begin(in_9, "test")
        begin9 = compiler_begin(end4, "test")
        node7 = relay.add(begin8, begin9)
        end5 = compiler_end(node7, "test")

        dend3 = compiler_end(node6, "default")
        begin10 = compiler_begin(dend3, "test")
        begin11 = compiler_begin(end5, "test")
        node8 = relay.add(begin10, begin11)
        end6 = compiler_end(node8, "test")
        begin12 = compiler_begin(in_10, "test")
        begin13 = compiler_begin(end6, "test")
        node9 = relay.add(begin12, begin13)
        end7 = compiler_end(node9, "test")

        f = relay.Function([in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10], end7)
        mod = tvm.IRModule.from_expr(f)
        return mod

    def expected():
        in_1 = relay.var("in_1", shape=(10, 10), dtype="float32")
        in_2 = relay.var("in_2", shape=(10, 10), dtype="float32")
        in_3 = relay.var("in_3", shape=(10, 10), dtype="float32")
        in_4 = relay.var("in_4", shape=(10, 10), dtype="float32")
        in_5 = relay.var("in_5", shape=(10, 10), dtype="float32")
        in_6 = relay.var("in_6", shape=(10, 10), dtype="float32")
        in_7 = relay.var("in_7", shape=(10, 10), dtype="float32")
        in_8 = relay.var("in_8", shape=(10, 10), dtype="float32")
        in_9 = relay.var("in_9", shape=(10, 10), dtype="float32")
        in_10 = relay.var("in_10", shape=(10, 10), dtype="float32")

        begin0 = compiler_begin(in_1, "test")
        begin1 = compiler_begin(in_2, "test")
        begin2 = compiler_begin(in_3, "test")
        begin3 = compiler_begin(in_4, "test")
        node0 = relay.add(begin0, begin1)
        node1 = relay.add(begin2, begin3)
        node2 = relay.add(node0, node1)

        dbegin0 = compiler_begin(in_5, "default")
        dbegin1 = compiler_begin(in_6, "default")
        dbegin2 = compiler_begin(in_7, "default")
        node3 = relay.subtract(dbegin0, dbegin1)
        node4 = relay.subtract(dbegin2, node3)
        dend0 = compiler_end(node4, "default")

        begin4 = compiler_begin(dend0, "test")
        begin5 = compiler_begin(in_9, "test")
        node5 = relay.add(node2, begin4)
        end1 = compiler_end(node5, "test")

        dbegin4 = compiler_begin(end1, "default")
        dbegin5 = compiler_begin(in_8, "default")
        node6 = relay.subtract(dbegin5, dbegin4)
        dend1 = compiler_end(node6, "default")

        node7 = relay.add(begin5, node5)
        end2 = compiler_end(node7, "test")
        begin6 = compiler_begin(end2, "test")
        begin7 = compiler_begin(dend1, "test")

        node8 = relay.add(begin7, begin6)

        begin8 = compiler_begin(in_10, "test")
        node9 = relay.add(begin8, node8)
        end3 = compiler_end(node9, "test")

        f = relay.Function([in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10], end3)
        mod = tvm.IRModule.from_expr(f)
        return mod

    mod = annotated()
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    ref_mod = expected()
    ref_mod = relay.transform.InferType()(ref_mod)
    tvm.ir.assert_structural_equal(mod, ref_mod)


def test_if_else():
    """
    This tests that the restriction regions propagate successful in
    if_else control flow.

    O = supported by target
    X = not supported by target


           O1 - - - |      O1 --|
            |       |               |
            X       |               X
            |       |                              |
    If cond ? O1: X | -->       +       +  If cond ? O1: X  +
            |       |                                           |
           O2 <- - -|                                          O2 <-|


    Avoid O1 merge to O2.
    """

    target = "test_if_else_merge"

    @tvm.ir.register_op_attr("sigmoid", "target." + target)
    def sigmoid(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("erf", "target." + target)
    def erf(expr):  # pylint: disable=unused-variable
        return True

    @tvm.ir.register_op_attr("add", "target." + target)
    def add(expr):  # pylint: disable=unused-variable
        return True

    """Test that If-else nodes merges regions correctly."""

    def get_mod():
        data = relay.var("data", shape=(1, 32))
        add0 = relay.add(data, data)
        sub0 = relay.subtract(add0, data)
        eq = relay.equal(relay.sum(add0), relay.sum(sub0))

        true_branch = relay.sigmoid(add0)
        false_branch = relay.sigmoid(sub0)
        ife = relay.If(eq, true_branch, false_branch)
        erf = relay.erf(ife)
        out = relay.add(add0, erf)
        func = relay.Function([data], out)
        mod = tvm.IRModule.from_expr(func)

        return mod

    for annotate_non_call_ops in [True, False]:
        result = transform.AnnotateTarget(target, annotate_non_call_ops)(get_mod())
        merge = transform.MergeCompilerRegions()(result)
        # Ensure partition finished without segment fault.
        partition = transform.PartitionGraph()(merge)


if __name__ == "__main__":
    test_diamond_graph_fanouts()
    test_example_graph()
    test_if_else()
