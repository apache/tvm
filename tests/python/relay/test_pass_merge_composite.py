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
"""Unit tests for merge composite."""
from tvm import relay
from tvm.relay.testing import run_opt_pass


def make_add_sub_mul_pattern():
    """Create a pattern to match the following graph.

        add  sub
         \   /
          \ /
          mul
    """
    x = relay.var('x')
    y = relay.var('y')
    add_node = relay.add(x, y)
    sub_node = relay.subtract(x, y)
    mul_node = relay.multiply(add_node, sub_node)
    return mul_node


def make_add_relu_pattern():
    """Create a pattern to match the following graph.

        add
         |
       ReLu
    """
    x = relay.var('x')
    y = relay.var('y')
    add_node = relay.add(x, y)
    r = relay.nn.relu(add_node)
    return r


def test_simple_merge():
    """Test composite function is correctly produced from simple graph.

    We could expect the pattern `make_add_relu_pattern` to be merged
    into a single op `add_relu`.

        a  b
        \ /               a  b
        add    ====>      \ /
         |             add_relu
       ReLu

    """
    pattern_table = {
        "add_sub_mul": make_add_relu_pattern()
    }

    def before():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        add_node = relay.add(a, b)
        r = relay.nn.relu(add_node)
        return relay.Function([a, b], r)

    def expected():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))

        # add_relu function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        relu_node = relay.nn.relu(add_node)
        add_relu = relay.Function([in_1, in_2], relu_node)

        # merged function
        r = relay.Call(add_relu, [a, b])
        return relay.Function([a, b], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert relay.analysis.alpha_equal(result, expected)


def test_branch_merge():
    """Test composite function is correctly produced from branching graph.

    We would expect the pattern `make_add_sub_mul_pattern` to be merged
    into a single op `add_sub_mul`.

       a  b  a  b
        \/    \/
        add  sub                       a  b
         \   /                          \/
          \ /                      add_sub_mul
          mul                     c     |
          /  \                     \    |
       c /  c |       ====>        add_sub_mul
       \/   \/                          |
       add  sub                         |
        \   /                         ReLu
         \ /
         mul
          |
          |
        ReLu
    """

    pattern_table = {
        "add_sub_mul": make_add_sub_mul_pattern()
    }

    def before():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        c = relay.var('c', shape=(10, 10))
        add_node = relay.add(a, b)
        sub_node = relay.subtract(a, b)
        mul_node = relay.multiply(add_node, sub_node)
        add_node_2 = relay.add(c, mul_node)
        sub_node_2 = relay.subtract(c, mul_node)
        mul_node_2 = relay.multiply(add_node_2, sub_node_2)
        r = relay.nn.relu(mul_node_2)
        return relay.Function([a, b, c], r)

    def expected():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        c = relay.var('c', shape=(10, 10))

        # add_sub_mul function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        sub_node = relay.subtract(in_1, in_2)
        mul_node = relay.multiply(add_node, sub_node)
        add_sub_mul = relay.Function([in_1, in_2], mul_node)

        # merged function
        add_sub_mul_1 = relay.Call(add_sub_mul, [a, b])
        add_sub_mul_2 = relay.Call(add_sub_mul, [c, add_sub_mul_1])
        r = relay.nn.relu(add_sub_mul_2)
        return relay.Function([a, b, c], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert relay.analysis.alpha_equal(result, expected)
