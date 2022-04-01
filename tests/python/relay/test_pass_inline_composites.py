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
# pylint: disable=invalid-name, missing-docstring, too-many-statements
"""Unit tests for inline composites."""
import pytest
import tvm
from tvm import relay, tir
from tvm.relay.dataflow_pattern import TupleGetItemPattern, is_op, wildcard
from tvm.relay.testing import run_opt_pass

"""
The inline composite pass is designed to inline multiple kernel generated through 
the merge composite composite pass. The underlying idea is to inline N kernels 
produced from merge composite based on a given set of pattern into a single IR module.
Also, clears Composite and PartionedFromPatterns that infer with certain BYOC implementations

For example suppose we have the graph:

        a  b                   
        \ /              
        add     
         |            
       relu                            

Merge composite will wrap each standalone op to it's own function, while setting Composite and
PartitionedFromPattern attrs. 
       
Relay IR after merge composite pass when registering each op as a standalone pattern: 
fn (%a: Tensor[(10, 10), float32], %b: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
  %0 = fn (%FunctionVar_0_01: Tensor[(10, 10), float32], %FunctionVar_0_1: Tensor[(10, 10), float32], PartitionedFromPattern="add_", Composite="add") -> Tensor[(10, 10), float32] {
    add(%FunctionVar_0_01, %FunctionVar_0_1) /* ty=Tensor[(10, 10), float32] */
  };
  %1 = %0(%a, %b) /* ty=Tensor[(10, 10), float32] */;
  %2 = fn (%FunctionVar_0_0: Tensor[(10, 10), float32], PartitionedFromPattern="nn.relu_", Composite="nn.relu") -> Tensor[(10, 10), float32] {
    nn.relu(%FunctionVar_0_0) /* ty=Tensor[(10, 10), float32] */
  };
  %2(%1) /* ty=Tensor[(10, 10), float32] */
}

Relay IR after inline composites pass:
fn (%a: Tensor[(10, 10), float32], %b: Tensor[(10, 10), float32]) -> Tensor[(10, 10), float32] {
  %0 = add(%a, %b) /* ty=Tensor[(10, 10), float32] */;
  nn.relu(%0) /* ty=Tensor[(10, 10), float32] */
}

One convenient use of this pass is to use Pattern-based operator support to move away
from the original operator predicates, and inline them into a single primitive function to offload it 
to an external BYOC backend, such as TensorRT.
"""


def make_add_relu_pattern():
    r"""Create a pattern to match the following graph.

     add
      |
    relu
    """
    add_node = wildcard() + wildcard()
    r = is_op("nn.relu")(add_node)
    return r


def make_relu_pattern():
    r"""Create a pattern to match the following graph
     a
     |
    relu
     |
    """
    pattern = is_op("nn.relu")(wildcard())
    return pattern


def make_add_pattern():
    r"""Create a pattern to match the following graph
    a  b
    \  /
    add
     |
    """
    pattern = is_op("add")(wildcard(), wildcard())
    return pattern


def check_success_composite_pass(func):
    return func.body.op.attrs["Composite"] is not None


def check_result(pattern_table, expected_graph, import_prelude=False):
    """Utility function to check inline composites results."""
    result = run_opt_pass(
        expected_graph, relay.transform.MergeComposite(pattern_table), import_prelude=import_prelude
    )
    assert check_success_composite_pass(
        result
    ), "Merge Composite pass didn't produced partioned from Pattern"
    result = run_opt_pass(
        expected_graph, relay.transform.InlineComposites(target=""), import_prelude=import_prelude
    )
    assert not relay.analysis.free_vars(result), "Found free vars in the result graph: {0}".format(
        str(result)
    )
    expected = run_opt_pass(expected_graph, relay.transform.InferType())
    assert tvm.ir.structural_equal(
        result, expected, map_free_vars=True
    ), "Graph mismatch: output vs. expected\n{0}\n=====\n{1}".format(str(result), str(expected))


def test_single_op_registry():
    r"""Test inline composite pass is correctly inline the post-merge composite graph.

    We could expect the patterns `make_add_pattern` and `make_relu_pattern` to be inlined
    into a single func instead of an single func per registered pattern.

    """
    pattern_table = [("add", make_add_pattern()), ("nn.relu", make_relu_pattern())]

    def expected():
        in_1 = relay.var("in_1", shape=(10, 10))
        in_2 = relay.var("in_2", shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        relu_node = relay.nn.relu(add_node)
        add_relu = relay.Function([in_1, in_2], relu_node)
        return add_relu

    check_result(pattern_table, expected())


def test_mix_fused_and_single_op():
    r"""Test inline composite pass is correctly inline the merge composite result"""
    pattern_table = [("add_relu", make_add_relu_pattern()), ("nn.relu", make_relu_pattern())]

    def expected():
        a = relay.var("a", shape=(10, 10))
        b = relay.var("b", shape=(10, 10))

        # add_relu function
        in_1 = relay.var("in_1", shape=(10, 10))
        in_2 = relay.var("in_2", shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        relu_node = relay.nn.relu(add_node)
        relu_nd = relay.nn.relu(relu_node)
        add_relu = relay.Function([in_1, in_2], relu_nd)
        return add_relu

    check_result(pattern_table, expected())


if __name__ == "__main__":
    pytest.main()
