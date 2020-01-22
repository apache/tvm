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
    pattern_table = {
        "add_relu": make_add_relu_pattern()
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
    assert not relay.analysis.free_vars(result)
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
        \   /                         relu
         \ /
         mul
          |
          |
        relu
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
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert relay.analysis.alpha_equal(result, expected)


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
    pattern_table = {
        "conv2d_bias_relu": make_conv_bias_relu_pattern(),
        "add_relu": make_add_relu_pattern()
    }

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
        return relay.Function([data, kernel, bias], r)

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

        # add_relu function
        in_4 = relay.var('in_4', shape=(1, 256, 28, 28))
        in_5 = relay.var('in_5', shape=(1, 256, 28, 28))
        add_node = relay.add(in_4, in_5)
        r = relay.nn.relu(add_node)
        add_relu = relay.Function([in_4, in_5], r)

        # merged function
        conv_bias_add_relu_1 = relay.Call(conv_bias_add_relu, [data, kernel, bias])
        add_relu_1 = relay.Call(add_relu, [conv_bias_add_relu_1, a])
        r = relay.multiply(add_relu_1, b)
        return relay.Function([data, kernel, bias, a, b], r)

    result = run_opt_pass(before(), relay.transform.MergeComposite(pattern_table))
    assert not relay.analysis.free_vars(result)
    expected = run_opt_pass(expected(), relay.transform.InferType())
    assert relay.analysis.alpha_equal(result, expected)


if __name__ == "__main__":
    test_simple_merge()
    test_branch_merge()
    test_multiple_patterns()
