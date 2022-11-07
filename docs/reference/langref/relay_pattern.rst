..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.


=========================
Pattern Matching in Relay
=========================

There are many places in TVM where we identify pure data-flow sub-graphs of the Relay program and attempt to transform them in some way example passes include fusion, quantization, external code generation, and device specific optimizations such as bitpacking, and layer slicing used by VTA.

Many of these passes today require a lots of boring boilerplate code in order to implement as well as requiring users to think in terms of visitors and AST matching. Many of these transformations can easily be described in terms of graph rewrites. In order to build a rewriter or other advanced machinery we first need a language of patterns to describe what we can match.

Such a language is not just useful for building a rewriter but also providing extension points for existing passes. For example the fusion pass could be parameterized by a set of fusion patterns which describes the capability of your hardware, and the quantization pass could take a set of patterns which describe which operators can be quantized on a given platform.

In the backend world, we could use the same machinery to build a higher level API using bring your own code generation. This API takes set of patterns describing your hardware capabilities and an external compiler, providing a relatively smooth heterogeneous experience out of the box.

Pattern Examples
================

There are quite a few properties of operators that are worth matching. Below we examine how to match tree properties, and expand on some use cases that are not fully explored in the prototype. This section
demonstrates how to write patterns. It is recommended to check `tests/python/relay/test_dataflow_pattern.py`_
for more use cases.

.. _tests/python/relay/test_dataflow_pattern.py: https://github.com/apache/tvm/blob/main/tests/python/relay/test_dataflow_pattern.py

.. note::

    If you cannot find the corresponding pattern node to match the Relay node you want,
    you are welcome to raise an issue or submit a PR to add it.

Matching One of Two Ops
***********************

The first example is a simple case where we want to match one operator with a single input OR
another operator with a single input:

.. code-block:: python

    def test_match_op_or():
        is_add_or_sub = is_op('add') | is_op('subtract')
        assert is_add_or_sub.match(relay.op.op.get("add"))
        assert is_add_or_sub.match(relay.op.op.get("subtract"))


Matching an Op with Attributes
******************************

The next example is a dense operation with any operator that is marked element-wise:

.. code-block:: python

    def test_no_match_attr():
        op = is_op('nn.dense').has_attr({"TOpPattern": K_ELEMWISE})
        op_pat = op(wildcard(), wildcard())
        x = relay.var('x')
        y = relay.var('y')
        assert not op_pat.match(relay.op.nn.dense(x, y))

Here is another example to match an op with a specific attribute:

.. code-block:: python

    def test_match_data_layout():
        is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard()).has_attr({"data_layout": "NHWC"})
        x = relay.var('x')
        y = relay.var('y')
        assert not is_conv2d.match(relay.op.nn.conv2d(x, y))

Or a convolution with a specific kernel size:

.. code-block:: python

    def test_match_kernel_size():
        is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"kernel_size": [3, 3]})
        x = relay.var('x')
        y = relay.var('y')
        assert is_conv2d.match(relay.op.nn.conv2d(x, y, kernel_size=[3, 3]))



Matching an Optional Op
***********************

The next example is matching a pattern with one optional operator. In this pattern,
we can match the graph of conv2d+bias_add+relu or the graph of conv2d+bias_add.

.. code-block:: python

    def test_match_optional():
        conv_node = is_op('nn.conv2d')(wildcard(), wildcard())
        bias_node = is_op('nn.bias_add')(conv_node, wildcard())
        pat = bias_node.optional(lambda x: is_op('nn.relu')(x))

        x = relay.var('x')
        y = relay.var('y')
        z = relay.var('z')
        conv2d = relay.op.nn.conv2d(x, y)
        bias = relay.op.nn.bias_add(conv2d, z)
        assert pat.match(bias)
        relu = relay.op.nn.relu(bias)
        assert pat.match(relu)


Matching Types
**************

In addition to matching ops with attributes, we can also make a pattern to match their types, in interms of the shape and data type. Here are some examples:

.. code-block:: python

    def test_match_type():
        # Match any op with float32
        pat1 = has_dtype('float32')
        x = relay.var('x', shape=(10, 10), dtype='float32')
        assert pat1.match(x)

        # Match any op with shape (10, 10)
        pat2 = has_shape((10, 10))
        x = relay.var('x', shape=(10, 10), dtype='float32')
        assert pat2.match(x)

        # Match conv2d+relu with a certain shape
        conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
        pat3 = is_op('nn.relu')(conv2d).has_shape((1, 32, 28, 28))

        x = relay.var('x', shape=(1, 3, 28, 28), dtype='float32')
        w = relay.var('w', shape=(32, 3, 3, 3), dtype='float32')
        conv2d = relay.nn.conv2d(x, w, strides=(1, 1), padding=(1, 1))
        relu = relay.nn.relu(conv2d)
        assert pat3.match(relu)


Matching Non-Call Nodes
***********************

Sometimes we may also want to match a pattern that includes Tuple or TupleGetItem nodes.
Since there are not call nodes, we need to use specific pattern nodes to match them:

.. code-block:: python

    def test_match_tuple():
        x = relay.var('x')
        y = relay.var('y')
        z = relay.var('z')
        tuple_pattern = is_tuple((wildcard(), wildcard(), wildcard()))
        assert tuple_pattern.match(relay.expr.Tuple((x,y,z)))

The next example is matching a pattern of batch_norm -> get(0) -> relu. Note that you can also use `is_tuple_get_item(bn_node)` to match a `TupleGetItem` node with any index.

.. code-block:: python

    def test_match_tuple_get_item():
        bn_node = is_op('nn.batch_norm')(wildcard(), wildcard(), wildcard(), wildcard(), wildcard())
        tuple_get_item_node = is_tuple_get_item(bn_node, 0)
        pat = is_op('nn.relu')(tuple_get_item_node)

        x = relay.var('x', shape=(1, 8))
        gamma = relay.var("gamma", shape=(8,))
        beta = relay.var("beta", shape=(8,))
        moving_mean = relay.var("moving_mean", shape=(8,))
        moving_var = relay.var("moving_var", shape=(8,))
        bn_node = relay.nn.batch_norm(x, gamma, beta, moving_mean, moving_var)
        tuple_get_item_node = bn_node[0]
        out = relay.nn.relu(tuple_get_item_node)
        pat.match(out)

If we have a pattern that crosses a function boundary, we might want to match the Function itself


.. code-block:: python

  def test_match_func():
      x = relay.var("x")
      y = relay.var("y")
      wc1 = wildcard()
      wc2 = wildcard()
      func_pattern = FunctionPattern([wc1, wc2], wc1 + wc2)
      assert func_pattern.match(relay.Function([x, y], x + y))

The next example is matching a constant node regarding its values. This is useful to check
if a specific parameter in a subgraph has been bound or not.

.. code-block:: python

    def test_match_constant():
        conv2d = is_op('nn.conv2d')(wildcard(), is_constant())
        pattern = is_op('nn.bias_add')(conv2d, wildcard())

        x = relay.var('x', shape=(1, 3, 224, 224))
        w = relay.var('w', shape=(3, 3, 3, 3))
        b = relay.var('b', shape=(3, ))
        conv2d = relay.op.nn.conv2d(x, w)
        out = relay.op.nn.bias_add(conv2d, b)
        func = relay.Function([x, w, b], out)
        mod = tvm.IRModule.from_expr(func)

        # Two inputs of the conv2d in the graph are VarNode by default, so no match.
        assert not pattern.match(mod['main'].body)

        # The second input (weight) has been bind with constant values so it is now a constant node.
        mod["main"] = bind_params_by_name(mod["main"],
                                        {'w': tvm.nd.array(np.ones(shape=(3, 3, 3, 3)))})
        assert pattern.match(mod['main'].body)

On the other hand, if you need to match the constant with a specific value, you can directly
use ``is_expr``. This could be useful for algebraic simplify.

.. code-block:: python

    def test_match_plus_zero():
        zero = (is_expr(relay.const(0)) | is_expr(relay.const(0.0)))
        pattern = wildcard() + zero

        x = relay.Var('x')
        y = x + relay.const(0)
        assert pattern.match(y)

The next example is matching function nodes with a specific attribute:

.. code-block:: python

    def test_match_function():
        pattern = wildcard().has_attr({"Composite": "add"})

        x = relay.var('x')
        y = relay.var('y')
        f = relay.Function([x, y], x + y).with_attr("Composite", "add")
        assert pattern.match(f)

A Relay ``If`` expression can be matched if all of its condition, true branch and false branch
are matched:

.. code-block:: python

    def test_match_if():
        x = is_var("x")
        y = is_var("y")
        pat = is_if(is_op("less")(x, y), x, y)

        x = relay.var("x")
        y = relay.var("y")
        cond = x < y

        assert pat.match(relay.expr.If(cond, x, y))


A Relay ``Let`` expression can be matched if all of its variable, value, and body
are matched:

.. code-block:: python

  def test_match_let():
      x = is_var("x")
      y = is_var("y")
      let_var = is_var("let")
      pat = is_let(let_var, is_op("less")(x, y), let_var)

      x = relay.var("x")
      y = relay.var("y")
      lv = relay.var("let")
      cond = x < y
      assert pat.match(relay.expr.Let(lv, cond, lv))

Matching Diamonds and Post-Dominator Graphs
*******************************************

The next example is matching a diamond with two inputs at the top of the diamond::

    def test_match_diamond():
        # Pattern
        is_conv2d = is_op('nn.conv2d')(is_var(), is_var())
        path1 = is_op('nn.relu')(is_conv2d)
        path2 = is_op('nn.leaky_relu')(is_conv2d)
        diamond = is_op('add')(path1, path2)

        # Expr
        inp = relay.var('input')
        weight = relay.var('weight')
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
        out = relu + leaky_relu

        # Check
        assert diamond.match(out)

The final example is matching diamonds with a post-dominator relationship. We embed dominator analysis as type of matching in the pattern language in order to allow for pattern matching with unknown topology. This is important because we want to be able to use the language to describe fuse patterns, like elementwise operations followed by a conv2d::

    def test_match_dom_diamond():
        # Pattern
        is_conv2d = is_op('nn.conv2d')(is_var(), is_var())
        reduction = is_op('add')(wildcard(), wildcard())
        diamond = dominates(is_conv2d, is_elemwise, reduction)

        # Expr
        inp = relay.var('input')
        weight = relay.var('weight')
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
        out = relu + leaky_relu

        # Check
        assert diamond.match(out)


Matching Fuzzy Patterns
=======================

The Dominator analysis above lets one match a subgraph of Relay AST that doesn't correspond to a set of patterns nodes exactly 1-to-1. There are a few other places where we support such "fuzzy" matching.

Tuples, Functions, and Call nodes with any number of inputs can be matched by passing `None` as the argument value, i.e.::

    tuple_pattern = is_tuple(None)
    func_pattern = FunctionPattern(None, wildcard() + wildcard())
    call_pattern = func_pattern(None)

These patterns allow matching more generic classes patterns by constraining the use of the arguments rather than the number of arguments.

Additionally, we support matching Functions with fuzzy bodies, i.e., a function body that is under constrained by the pattern. The pattern `FunctionPattern([is_var(), is_var()], wildcard() + wildcard()])` will match `relay.Function([x, y], x + y)`, but it will also match `relay.Function([x, y], x * x + y)`. In the second case, the pattern doesn't perfectly constrain the body of the function, so the resulting match is fuzzy.


Pattern Language Design
=======================

The pattern language proposed is designed to be a mirror of Relay's IR with additional support for common scenarios. The goal of the pattern language is to provide a regular-expression like capability for matching data-flow graphs and doing rewriting.

The high level design is to introduce a language of patterns for now we propose the language as::

    Pattern ::= expr
            | *
            | pattern(pattern1, ... patternN)
            | has_type(type)
            | has_dtype(type)
            | has_shape(shape)
            | has_attr(attrs)
            | is_var(name)
            | is_constant()
            | is_expr(expr)
            | is_op(op_name)
            | is_tuple()
            | is_tuple_get_item(pattern, index = None)
            | is_if(cond, tru, fls)
            | is_let(var, value, body)
            | pattern1 `|` pattern2
            | dominates(parent_pattern, path_pattern, child_pattern)
            | FunctionPattern(params, body)

The above language then provides a matching interface with both can select sub-graphs as well as verify that the graph does match the pattern.

Expression Pattern
******************

Match a literal expression.

Wildcard
********

Match any expression.

Type Pattern
************

Check that the expression matched by the nested pattern has a particular type.

DType Pattern
*************

Check that the expression matched by the nested pattern has a particular data type.

Shape Pattern
*************

Check that the expression matched by the nested pattern has a particular output shape.

Attribute Pattern
*****************

Check that the operator matched by the pattern has an attribute with a particular value.

Variable Pattern
****************

Check that the expression is a relay Variable, and optional provide a name to match to the Variable name.


Alternate
*********

Either match the first pattern or the second pattern.

Domination
**********

Match child pattern, find a match for the parent pattern, insuring that the child ultimately dominates the parent (i.e., no nodes outside the pattern use outputs of the parent), and that ever node between the child and the pattern matches the path pattern.

Function Pattern
****************

Match a Function with a body and parameters

If Pattern
**********

Match an If with condition, true branch, and false branch

Let Pattern
***********

Match a Let with a variable, value, and body

Applications
============

The pattern language provides not only the pattern matching but also pattern processing.
Here we introduce two pattern processing approaches and provide some examples.

Pattern Rewriting
*****************

If you would like to replace the matched pattern with another subgraph, you can leverage
the ``rewrite`` transformation. Here is an example of rewriting a series of arithmetic operators
with a single batch_norm op. The constructor parameter ``require_type`` indicates whether InferType
is required to be run before the callback.

.. code-block:: python

    class BatchnormCallback(DFPatternCallback):
        # A callback class to rewrite the matched pattern to a batch_norm op.
        def __init__(self, require_type=False):
            super().__init__(require_type)
            self.x = wildcard()
            self.var = wildcard()
            self.mean = wildcard()
            self.beta = wildcard()
            self.gamma = wildcard()
            self.eps = wildcard()

            self.pattern = self.gamma * (self.x - self.mean)/is_op("sqrt")(self.var + self.eps) + self.beta

        def callback(self, pre, post, node_map):
            x = node_map[self.x][0]
            var = node_map[self.var][0]
            mean = node_map[self.mean][0]
            beta = node_map[self.beta][0]
            gamma = node_map[self.gamma][0]
            eps = node_map[self.eps][0]
            return relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = eps.data.numpy().item())[0]

        # A graph of arithmetic operators that are functional equivalent to batch_norm.
        x = relay.var('x')
        var = relay.var('var')
        mean = relay.var('mean')
        beta = relay.var('beta')
        gamma = relay.var('gamma')
        BN = gamma * (x - mean)/relay.op.sqrt(var + relay.const(1e-5)) + beta

        from tvm.relay.dataflow_pattern import rewrite
        out = rewrite(BatchnormCallback(), BN)
        assert tvm.ir.structural_equal(out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = 1e-5)[0])

The function ``def callback(self, pre, post, node_map)`` will be invoked when the rewriter matches
``self.pattern``. ``node_map`` is a dictionary mapping from pattern nodes to matched nodes in the graph.

The callback function will be invoked recursively on the returned pattern until the pattern stops changing. As a result, if ``self.pattern`` matches any part of the graph that the callback returned, the rewriter will run in a loop. If you want to avoid multiple rewrites, you can pass a ``rewrite_once=True`` parameter to the constructor.

Pattern Partitioning
********************

If you would like to perform a more complex processing for matched subgraphs and you are not
satisfied with ``rewrite``, you may consider partitioning the matched subgraphs to a separate
Relay function and perform other processes to the function. Here we use ``pattern.partition``
to create a new Relay function for each matched subgraph. The functionality is similar to
the op fusion pass in TVM:

.. code-block:: python

    # A pattern matching conv2d+relu.
    pattern = is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))

    # A graph.
    x = relay.var('input')
    w = relay.var('weight')
    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)
    print('relu')
    # free_var %x: Tensor[(1, 3, 224, 224), float32]
    # free_var %w: Tensor[(3, 3, 3, 3), float32]
    # %0 = nn.conv2d(%x, %w, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 3, 222, 222), float32] */;
    # free_var %b: Tensor[(3), float32]
    # nn.bias_add(%0, %b) /* ty=Tensor[(1, 3, 222, 222), float32] */

    # After partition.
    print(pattern.partition(relu))
    # free_var %x: Tensor[(1, 3, 224, 224), float32]
    # free_var %w: Tensor[(3, 3, 3, 3), float32]
    # free_var %b: Tensor[(3), float32]
    # %1 = fn (%FunctionVar_0_0, %FunctionVar_0_1,
    #          %FunctionVar_0_2, PartitionedFromPattern="nn.conv2d_nn.bias_add_") {
    #   %0 = nn.conv2d(%FunctionVar_0_0, %FunctionVar_0_1, padding=[0, 0, 0, 0]);
    #   nn.bias_add(%0, %FunctionVar_0_2)
    # };
    # %1(%x, %w, %b)

Note that you can also specify the attributes for the created functions:

.. code-block:: python

    print(pattern.partition(relu, {'Composite': 'one_layer'}))
    # free_var %x: Tensor[(1, 3, 224, 224), float32]
    # free_var %w: Tensor[(3, 3, 3, 3), float32]
    # free_var %b: Tensor[(3), float32]
    # %1 = fn (%FunctionVar_0_0, %FunctionVar_0_1,
    #          %FunctionVar_0_2, Composite="one_layer",
    #                            PartitionedFromPattern="nn.conv2d_nn.bias_add_") {
    #   %0 = nn.conv2d(%FunctionVar_0_0, %FunctionVar_0_1, padding=[0, 0, 0, 0]);
    #   nn.bias_add(%0, %FunctionVar_0_2)
    # };
    # %1(%x, %w, %b)

If you need a customized checking function that cannot be specified using pattern language,
you can specify ``check`` function when partitioning. The following example demonstrates a
case that checks input data layout of a subgraph:

.. code-block:: python

    def check(pre):
        conv = pre.args[0]
        return (conv.attrs.data_layout == "NCHW") and bool(conv.checked_type.shape[0] == 1)

    pattern.partition(relu, check=check)

In this example, we check if the first argument of the matched subgraph (i.e., ``pre.args[0]``)
has data layout "NCHW" and if its batch size is 1. This feature is useful if the conditions
of matching a pattern cannot be verified by analyzing the pattern itself.
