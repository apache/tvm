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
import tvm
from tvm import relay
from tvm import tir
from tvm.relay.testing import run_opt_pass

"""
The merge composite pass is designed to merge multiple relay operators, that
match a given pattern, and combine them into a single relay function.

For example suppose we have the graph:

    conv2d
      |       (merge composite pass)
   bias_add            ====>           conv2d_bias_relu
      |            (our target)
     relu

Our Relay IR before the pass:
    fn (%data: Tensor[(1, 512, 28, 28), float32], %kernel: Tensor[(256, 512, 1, 1), float32],
            %bias: Tensor[(256), float32]) -> Tensor[(1, 256, 28, 28), float32] {
        %0 = nn.conv2d(%data, %kernel, kernel_size=[1, 1])
            /* ty=Tensor[(1, 256, 28, 28), float32] */;
        %1 = nn.bias_add(%0, %bias) /* ty=Tensor[(1, 256, 28, 28), float32] */;
        nn.relu(%1) /* ty=Tensor[(1, 256, 28, 28), float32] */
    }

Our Relay IR after the pass:
    fn (%data: Tensor[(1, 512, 28, 28), float32], %kernel: Tensor[(256, 512, 1, 1), float32],
            %bias: Tensor[(256), float32]) -> Tensor[(1, 256, 28, 28), float32] {
      %2 = fn (%x: Tensor[(1, 512, 28, 28), float32], %y: Tensor[(256, 512, 1, 1), float32],
            %z: Tensor[(256), float32], Primitive=1, Composite="conv2d_bias_relu") ->
            Tensor[(1, 256, 28, 28), float32] {
        %0 = nn.conv2d(%x, %y, kernel_size=[1, 1]) /* ty=Tensor[(1, 256, 28, 28), float32] */;
        %1 = nn.bias_add(%0, %z) /* ty=Tensor[(1, 256, 28, 28), float32] */;
        nn.relu(%1) /* ty=Tensor[(1, 256, 28, 28), float32] */
      };
      %2(%data, %kernel, %bias) /* ty=Tensor[(1, 256, 28, 28), float32] */
    }

As you can see in the second relay example, the pattern we specified has been wrapped
in a function. The function is then called, producing the same result as the first relay
example.

One convenient use for this pass is to offload multiple operators to a single external
codegen function.
"""


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
       relu
    """
    x = relay.var('x')
    y = relay.var('y')
    add_node = relay.add(x, y)
    r = relay.nn.relu(add_node)
    return r


def make_conv_bias_relu_pattern():
    """Create a pattern to match the following graph.

       conv2d
         |
      bias_add
         |
       relu
    """
    x = relay.var('x')
    y = relay.var('y')
    z = relay.var('z')
    conv_node = relay.nn.conv2d(x, y)
    bias_node = relay.nn.bias_add(conv_node, z)
    r = relay.nn.relu(bias_node)
    return r


def make_add_add_add_pattern():
    """Create a pattern to match the following graph.
       Useful for testing re-using a call node.

        x    y
      /  \  /
      |  add
       \  |  \
         add |
          | /
         add
    """
    x = relay.var('x')
    y = relay.var('y')
    add_node = relay.add(x, y)
    add_node_1 = relay.add(x, add_node)
    r = relay.add(add_node_1, add_node)
    return r

def make_bn_relu_pattern():
    """Create a pattern to match the following graph.

     batch_norm
         |
    TupleGetItem(0)
         |
       relu
    """
    x = relay.var('x')
    gamma = relay.var("gamma")
    beta = relay.var("beta")
    moving_mean = relay.var("moving_mean")
    moving_var = relay.var("moving_var")
    bn_node = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
    tuple_get_item_node = bn_node[0]
    r = relay.nn.relu(tuple_get_item_node)
    return r


def test_simple_merge():
    """Test composite function is correctly produced from simple graph.

    We could expect the pattern `make_add_relu_pattern` to be merged
    into a single op `add_relu`.

        a  b
        \ /               a  b
        add    ====>      \ /
         |             add_relu
       relu

    """
    pattern_table = [
        ("add_relu", make_add_relu_pattern())
    ]

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
        add_relu = add_relu.with_attr("Composite", "add_relu")

        # merged function
        r = relay.Call(add_relu, [a, b])
        return relay.Function([a, b], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


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
        \   /                         relu
         \ /
         mul
          |
          |
        relu
    """

    pattern_table = [
        ("add_sub_mul", make_add_sub_mul_pattern())
    ]

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
        add_sub_mul = add_sub_mul.with_attr("Composite", "add_sub_mul")

        # add_sub_mul1 function
        in_3 = relay.var('in_3', shape=(10, 10))
        in_4 = relay.var('in_4', shape=(10, 10))
        add_node_1 = relay.add(in_3, in_4)
        sub_node_1 = relay.subtract(in_3, in_4)
        mul_node_1 = relay.multiply(add_node_1, sub_node_1)
        add_sub_mul_1 = relay.Function([in_3, in_4], mul_node_1)
        add_sub_mul_1 = add_sub_mul_1.with_attr("Composite", "add_sub_mul")

        # merged function
        m_add_sub_mul_1 = relay.Call(add_sub_mul, [a, b])
        m_add_sub_mul_2 = relay.Call(add_sub_mul_1, [c, m_add_sub_mul_1])
        r = relay.nn.relu(m_add_sub_mul_2)
        return relay.Function([a, b, c], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_reuse_call_merge():
    """Test composite function is correctly produced from simple graph
       which re-uses call nodes.

    We could expect the pattern `make_add_add_add` to be merged
    into a single op `add_add_add`.

        x     y
         \   / \
          sub  |           x     y
        /  |  /             \   / |
        | add      ====>     sub  |
         \ |  \               |  /
          add |           add_add_add
           | /
          add

    """
    pattern_table = [
        ("add_add_add", make_add_add_add_pattern())
    ]

    def before():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))
        sub_node = relay.subtract(a, b)

        # pattern
        add_node = relay.add(sub_node, b)
        add_node_1 = relay.add(sub_node, add_node)
        r = relay.add(add_node_1, add_node)

        return relay.Function([a, b], r)

    def expected():
        a = relay.var('a', shape=(10, 10))
        b = relay.var('b', shape=(10, 10))

        # add_relu_add function
        in_1 = relay.var('in_1', shape=(10, 10))
        in_2 = relay.var('in_2', shape=(10, 10))
        add_node = relay.add(in_1, in_2)
        add_node_1 = relay.add(in_1, add_node)
        add_node_2 = relay.add(add_node_1, add_node)
        add_add_add = relay.Function([in_1, in_2], add_node_2)
        add_add_add = add_add_add.with_attr("Composite", "add_add_add")

        # merged function
        sub_node = relay.subtract(a, b)
        call = relay.Call(add_add_add, [sub_node, b])
        return relay.Function([a, b], call)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_multiple_patterns():
    """Test different patterns are merged correctly in the graph.

    We would expect the pattern `make_conv_bias_relu_pattern` to be merged
    into a single op `conv_bias_relu`. We would also expect `make_add_relu_pattern`
    to be merged into a single op `add_relu`.

        data   kernel
          \      /
           \    /
           conv2d                   data   kernel   bias
             |                         \      |      /
             |   bias                 conv2d_bias_relu
             |   /                            |
          bias_add        ====>               |    a
             |                                |   /
           relu  a                        add_relu
             \  /                             |
             add                              |  b
              |                               | /
            relu  b                          mul
              |  /
             mul
    """
    pattern_table = [
        ("conv2d_bias_relu", make_conv_bias_relu_pattern()),
        ("add_relu", make_add_relu_pattern())
    ]

    def before():
        data = relay.var('data', shape=(1, 512, 28, 28))
        kernel = relay.var('kernel', shape=(256, 512, 1, 1))
        bias = relay.var('bias', shape=(256,))
        a = relay.var('a', shape=(1, 256, 28, 28))
        b = relay.var('b', shape=(1, 256, 28, 28))

        conv_node = relay.nn.conv2d(data,
                                    kernel,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    strides=(1, 1))

        bias_node = relay.nn.bias_add(conv_node, bias)
        relu_node = relay.nn.relu(bias_node)
        add_node = relay.add(relu_node, a)
        relu_node_2 = relay.nn.relu(add_node)
        r = relay.multiply(relu_node_2, b)
        return relay.Function([data, kernel, bias, a, b], r)

    def expected():
        data = relay.var('data', shape=(1, 512, 28, 28))
        kernel = relay.var('kernel', shape=(256, 512, 1, 1))
        bias = relay.var('bias', shape=(256,))
        a = relay.var('a', shape=(1, 256, 28, 28))
        b = relay.var('b', shape=(1, 256, 28, 28))

        # conv_bias_relu function
        in_1 = relay.var('in_1', shape=(1, 512, 28, 28))
        in_2 = relay.var('in_2', shape=(256, 512, 1, 1))
        in_3 = relay.var('in_3', shape=(256,))

        conv_node = relay.nn.conv2d(in_1,
                                    in_2,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    strides=(1, 1))

        bias_node = relay.nn.bias_add(conv_node, in_3)
        r = relay.nn.relu(bias_node)
        conv_bias_add_relu = relay.Function([in_1, in_2, in_3], r)
        conv_bias_add_relu = conv_bias_add_relu.with_attr("Composite",
                                                          "conv2d_bias_relu")

        # add_relu function
        in_4 = relay.var('in_4', shape=(1, 256, 28, 28))
        in_5 = relay.var('in_5', shape=(1, 256, 28, 28))
        add_node = relay.add(in_4, in_5)
        r = relay.nn.relu(add_node)
        add_relu = relay.Function([in_4, in_5], r)
        add_relu = add_relu.with_attr("Composite", "add_relu")

        # merged function
        conv_bias_add_relu_1 = relay.Call(conv_bias_add_relu, [data, kernel, bias])
        add_relu_1 = relay.Call(add_relu, [conv_bias_add_relu_1, a])
        r = relay.multiply(add_relu_1, b)
        return relay.Function([data, kernel, bias, a, b], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_merge_order():
    """Test that patterns are merged in the order they exist in the pattern table.

    There can be cases where one pattern is a subgraph of another, in which case
    it is not clear which match should take priority. The priority should come
    from the order in which the patterns are declared in the pattern table. The
    first patterns will be merged with highest priority and the last with lowest.

    A:       B:       C:
    add      add      abs
     |        |        |
    abs      abs      relu
     |
    relu

    """

    def pattern_A():
        x = relay.var('x')
        y = relay.var('y')
        out = relay.add(x, y)
        out = relay.abs(out)
        out = relay.nn.relu(out)
        return out

    def pattern_B():
        x = relay.var('x')
        y = relay.var('y')
        out = relay.add(x, y)
        out = relay.abs(out)
        return out

    def pattern_C():
        x = relay.var('x')
        out = relay.abs(x)
        out = relay.nn.relu(x)
        return out

    def before():
        input_1 = relay.var('input_1', shape=(10, 10))
        input_2 = relay.var('input_2', shape=(10, 10))
        out = relay.add(input_1, input_2)
        out = relay.abs(out)
        out = relay.nn.relu(out)
        return relay.Function([input_1, input_2], out)

    def after_A_priority(composite_name):
        input_1 = relay.var('input_1', shape=(10, 10))
        input_2 = relay.var('input_2', shape=(10, 10))
        x = relay.var('x')
        y = relay.var('y')
        out = relay.add(x, y)
        out = relay.abs(out)
        out = relay.nn.relu(out)
        merged_func = relay.Function([x, y], out)
        merged_func = merged_func.with_attr('Composite', composite_name)
        ret = relay.Call(merged_func, [input_1, input_2])
        return relay.Function([input_1, input_2], ret)

    # check A highest priority
    pattern_table = [
        ("A", pattern_A()),
        ("B", pattern_B()),
        ("C", pattern_C()),
    ]
    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(after_A_priority("A"), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)

    # check B highest priority
    pattern_table = [
        ("B", pattern_A()),
        ("C", pattern_B()),
        ("A", pattern_C()),
    ]
    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(after_A_priority("B"), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)

    # check C highest priority
    pattern_table = [
        ("C", pattern_A()),
        ("A", pattern_B()),
        ("B", pattern_C()),
    ]
    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(after_A_priority("C"), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_parallel_merge():
    """Tests that parallel patterns relying on the same inputs are correctly merged.

    The test graph is difficult to draw out as ascii art. It is essentially two parallel
    add-sub-mul units which both consume input_1 and input_2 with their results being multiplied
    to give the output. We expect both parallel branches should get merged and both should still
    consume the same input variables, input_1 and input_2."""

    def before():
        input_1 = relay.var('input_1', shape=(10, 10))
        input_2 = relay.var('input_2', shape=(10, 10))
        branch_1_add = relay.add(input_1, input_2)
        branch_1_sub = relay.subtract(input_1, input_2)
        branch_1 = relay.multiply(branch_1_add, branch_1_sub)
        branch_2_add = relay.add(input_1, input_2)
        branch_2_sub = relay.subtract(input_1, input_2)
        branch_2 = relay.multiply(branch_2_add, branch_2_sub)
        out = relay.multiply(branch_1, branch_2)
        return relay.Function([input_1, input_2], out)

    def after():
        input_1 = relay.var('input_1', shape=(10, 10))
        input_2 = relay.var('input_2', shape=(10, 10))
        x = relay.var('x')
        y = relay.var('y')
        branch_1 = relay.multiply(relay.add(x, y), relay.subtract(x, y))
        func_1 = relay.Function([x, y], branch_1)
        func_1 = func_1.with_attr('Composite', "add_sub_mul")
        call_1 = relay.Call(func_1, [input_1, input_2])
        x1 = relay.var('x1')
        y1 = relay.var('y1')
        branch_2 = relay.multiply(relay.add(x1, y1), relay.subtract(x1, y1))
        func_2 = relay.Function([x1, y1], branch_2)
        func_2 = func_2.with_attr('Composite', "add_sub_mul")
        call_2 = relay.Call(func_2, [input_1, input_2])
        out = relay.multiply(call_1, call_2)
        return relay.Function([input_1, input_2], out)

    pattern_table = [
        ("add_sub_mul", make_add_sub_mul_pattern())
    ]
    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(after(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_multiple_input_subgraphs():
    """Test the case when multiple input subgraphs feed into another subgraph.

     (1)    (2)    (3)    (4)
    add    add    add    add
     |      |      |      |
    relu   relu   relu   relu
     \      /      \      /
      \   /         \   /
       add           sub
        \            /
          \        /
            \    /
              mul

    ----> When 1=3 and 2=4 (Case 'A')

    add_relu  add_relu
       \         /
        \      /
       add_sub_mul

    ----> When 1!=3 and 2!=4 (Case 'B')

    add_relu  add_relu  add_relu  add_relu
       \       /           \       /
         \   /               \   /
          add                 sub
           \                  /
            --------     -----
                   \    /
                    mul

    The difference in behaviour comes from the fact that add_sub_mul expects that the
    inputs to add and sub are identical (the same two relay expressions). So when you
    have 4 independent inputs, the pattern should not be merged.
    """

    def before():
        before_funcs = {}
        inputs = [relay.var('input_' + str(i), shape=(10, 10)) for i in range(8)]
        add_relu_1 = relay.add(inputs[0], inputs[1])
        add_relu_1 = relay.nn.relu(add_relu_1)
        add_relu_2 = relay.add(inputs[2], inputs[3])
        add_relu_2 = relay.nn.relu(add_relu_2)
        add_relu_3 = relay.add(inputs[4], inputs[5])
        add_relu_3 = relay.nn.relu(add_relu_3)
        add_relu_4 = relay.add(inputs[6], inputs[7])
        add_relu_4 = relay.nn.relu(add_relu_4)
        add = relay.add(add_relu_1, add_relu_2)
        sub = relay.subtract(add_relu_3, add_relu_4)
        out = relay.multiply(add, sub)
        before_funcs['B'] = relay.Function(inputs, out)
        sub = relay.subtract(add_relu_1, add_relu_2)
        out = relay.multiply(add, sub)
        before_funcs['A'] = relay.Function(inputs[:4], out)
        return before_funcs

    def after_A():
        inputs = [relay.var('input_' + str(i), shape=(10, 10)) for i in range(4)]
        x = relay.var('x')
        y = relay.var('y')
        add_relu_1 = relay.add(x, y)
        add_relu_1 = relay.nn.relu(add_relu_1)
        add_relu_1 = relay.Function([x, y], add_relu_1)
        add_relu_1 = add_relu_1.with_attr('Composite', 'add_relu')
        add_relu_call_1 = relay.Call(add_relu_1, [inputs[0], inputs[1]])
        x1 = relay.var('x1')
        y1 = relay.var('y1')
        add_relu_2 = relay.add(x1, y1)
        add_relu_2 = relay.nn.relu(add_relu_2)
        add_relu_2 = relay.Function([x1, y1], add_relu_2)
        add_relu_2 = add_relu_2.with_attr('Composite', 'add_relu')
        add_relu_call_2 = relay.Call(add_relu_2, [inputs[2], inputs[3]])
        x2 = relay.var('x2')
        y2 = relay.var('y2')
        add = relay.add(x2, y2)
        sub = relay.subtract(x2, y2)
        add_sub_mul = relay.multiply(add, sub)
        add_sub_mul = relay.Function([x2, y2], add_sub_mul)
        add_sub_mul = add_sub_mul.with_attr('Composite', 'add_sub_mul')
        add_sub_mul_call = relay.Call(add_sub_mul, [add_relu_call_1, add_relu_call_2])
        return relay.Function(inputs, add_sub_mul_call)

    def after_B():
        inputs = [relay.var('input_' + str(i), shape=(10, 10)) for i in range(8)]
        add_relu_calls = []
        for i in range(4):
            x = relay.var('x' + str(i))
            y = relay.var('x' + str(i))
            add_relu = relay.add(x, y)
            add_relu = relay.nn.relu(add_relu)
            add_relu = relay.Function([x, y], add_relu)
            add_relu = add_relu.with_attr('Composite', 'add_relu')
            add_relu_call = relay.Call(add_relu, [inputs[i*2], inputs[i*2+1]])
            add_relu_calls.append(add_relu_call)

        add = relay.add(add_relu_calls[0], add_relu_calls[1])
        sub = relay.subtract(add_relu_calls[2], add_relu_calls[3])
        out = relay.multiply(add, sub)
        return relay.Function(inputs, out)

    pattern_table = [
        ("add_sub_mul", make_add_sub_mul_pattern()),
        ("add_relu", make_add_relu_pattern())
    ]
    # check case 'A'
    result = run_opt_pass(before()['A'], relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(after_A(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)

    # check case 'B'
    result = run_opt_pass(before()['B'], relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(after_B(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_tuple_get_item_merge():
    """Test composite function can be merged from pattern containing TupleGetItem nodes."""
    pattern_table = [
        ("bn_relu", make_bn_relu_pattern())
    ]

    def before():
        x = relay.var('x', shape=(1, 8))
        gamma = relay.var("gamma", shape=(8,))
        beta = relay.var("beta", shape=(8,))
        moving_mean = relay.var("moving_mean", shape=(8,))
        moving_var = relay.var("moving_var", shape=(8,))
        bn_node = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
        tuple_get_item_node = bn_node[0]
        r = relay.nn.relu(tuple_get_item_node)
        return relay.Function([x, gamma, beta, moving_mean, moving_var], r)

    def expected():
        x = relay.var('x', shape=(1, 8))
        beta = relay.var("beta", shape=(8,))
        gamma = relay.var("gamma", shape=(8,))
        moving_mean = relay.var("moving_mean", shape=(8,))
        moving_var = relay.var("moving_var", shape=(8,))

        # bn_relu function
        in_1 = relay.var('x1', shape=(1, 8))
        in_2 = relay.var('gamma1', shape=(8,))
        in_3 = relay.var('beta1', shape=(8,))
        in_4 = relay.var('moving_mean1', shape=(8,))
        in_5 = relay.var('moving_var1', shape=(8,))
        bn_node = relay.nn.batch_norm(in_1, in_2, in_3, in_4, in_5)
        tuple_get_item_node = bn_node[0]
        relu_node = relay.nn.relu(tuple_get_item_node)
        bn_relu = relay.Function([in_1, in_2, in_3, in_4, in_5], relu_node)
        bn_relu = bn_relu.with_attr("Composite", "bn_relu")

        # merged function
        r = relay.Call(bn_relu, [x, gamma, beta, moving_mean, moving_var])
        return relay.Function([x, gamma, beta, moving_mean, moving_var], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)


def test_pattern_with_check():
    def before():
        x = relay.var('x', shape=(1, 10, 10, 10))
        w = relay.var('w', shape=(10, 10, 3, 3))
        b = relay.var('b', shape=(8,))
        conv = relay.nn.conv2d(x,
                               w,
                               kernel_size=(3, 3),
                               kernel_layout="OIHW",
                               data_layout="NHWC")
        bias = relay.nn.bias_add(conv, b)
        relu = relay.nn.relu(bias)
        return relay.Function([x, w, b], relu)

    def _check_true(extract):
        conv = extract.args[0].args[0]
        return conv.attrs.data_layout == "NHWC"

    def _check_false(extract):
        conv = extract.args[0].args[0]
        return conv.attrs.data_layout == "NCHW"

    pattern_table_true = [
        ("conv_bias_relu", make_conv_bias_relu_pattern(), _check_true)
    ]
    pattern_table_false = [
        ("conv_bias_relu", make_conv_bias_relu_pattern(), _check_false)
    ]

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table_false))
    expected = run_opt_pass(before(), relay.transform.InferType())
    assert tvm.ir.structural_equal(result, expected, map_free_vars=True)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table_true))
    assert result.body.op.attrs["Composite"] == "conv_bias_relu"


if __name__ == "__main__":
    test_simple_merge()
    test_branch_merge()
    test_multiple_patterns()
    test_merge_order()
    test_parallel_merge()
    test_multiple_input_subgraphs()
    test_reuse_call_merge()
    test_tuple_get_item_merge()
    test_pattern_with_check()
