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

import tvm
from tvm import relay
from tvm.contrib.relay_viz import node_edge_gen
from tvm.contrib.relay_viz.node_edge_gen import DefaultNodeEdgeGenerator

# the testing focus on that DefaultNodeEdgeGenerator can
# parse Relay IR properly.


def test_var():
    ne_gen = DefaultNodeEdgeGenerator()
    shape = (10, 10)
    input_var = relay.var("input", shape=shape)
    node, edges = ne_gen.get_node_edges(input_var, {}, {input_var: 1})
    assert node.identity == 1, "node_id should be 1."
    assert "input" in node.detail, "detail should have name_hint."
    assert str(shape) in node.detail, "detail should have shape."
    assert len(edges) == 0, "relay.var doesn't cause any edge."


def test_function():
    ne_gen = DefaultNodeEdgeGenerator()
    input_var = relay.var("input")
    bias_var = relay.var("bias")
    add_bias = relay.add(input_var, bias_var)
    func = relay.Function([input_var, bias_var], add_bias)
    node, edges = ne_gen.get_node_edges(func, {}, {func: 99, add_bias: 199})
    assert node.identity == 99, "node_id should be 99."
    assert edges[0].start == 199, "edge.start should be node 199."
    assert edges[0].end == 99, "edge.end should be node 99."


def test_call():
    ne_gen = DefaultNodeEdgeGenerator()
    input_var = relay.var("input")
    bias_var = relay.var("bias")
    add_bias = relay.add(input_var, bias_var)
    node, edges = ne_gen.get_node_edges(add_bias, {}, {add_bias: 1, input_var: 0, bias_var: 2})
    assert "add" in node.type_str, "node_type shuold contain op_name."
    assert len(edges) == 2, "the length of edges should be 2, from two var to relay.add."


def test_tuple():
    ne_gen = DefaultNodeEdgeGenerator()
    elemt0_var = relay.var("elemt0")
    elemt1_var = relay.var("elemt1")
    tup = relay.Tuple([elemt0_var, elemt1_var])
    node, edges = ne_gen.get_node_edges(tup, {}, {tup: 123, elemt0_var: 0, elemt1_var: 1})
    assert node.identity == 123, "node_id should be 123."
    assert len(edges) == 2, "the length of edges should be 2, from two relay.var to tuple."
    assert edges[0].start == 0 and edges[0].end == 123, "edges[0] should be 0 -> 123."
    assert edges[1].start == 1 and edges[1].end == 123, "edges[1] should be 1 -> 123."


def test_constant():
    ne_gen = DefaultNodeEdgeGenerator()
    arr = tvm.nd.array(10)
    const = relay.Constant(arr)
    node, edges = ne_gen.get_node_edges(const, {}, {const: 999})
    assert node.identity == 999, "node_id should be 999."
    assert len(edges) == 0, "constant should not cause edges."

    arr = tvm.nd.array([[10, 11]])
    const = relay.Constant(arr)
    node, edges = ne_gen.get_node_edges(const, {}, {const: 111})
    assert str(const.data.shape) in node.detail, "node_detail should contain shape."


if __name__ == "__main__":
    test_var()
    test_function()
    test_call()
    test_tuple()
    test_constant()
