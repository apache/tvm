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

Examples
========

There are quite a few properties that are worth matching of operators below we examine how to match tree properties, and expand on some use cases that are not fully explored in the prototype. The first example is a simple case where we want to match one operator with a single input OR another operator with a single input, see the below diagram for a graphical representation and corresponding code::

    def test_match_op_or():
        is_add_or_sub = is_op('add') | is_op('subtract')
        assert is_add_or_sub.match(relay.op.op.get("add"))
        assert is_add_or_sub.match(relay.op.op.get("subtract"))

The next example is a dense operation with any operator that is marked element-wise::

    def test_no_match_attr():
        op = is_op('nn.dense').has_attr("TOpPattern", K_ELEMWISE)
        op_pat = op(wildcard(), wildcard())
        x = relay.var('x')
        y = relay.var('y')
        assert not op_pat.match(relay.op.nn.dense(x, y))

The next example is matching a diamond with two inputs at the top of the diamond::

    def test_match_diamond():
        # Pattern
        is_conv2d = is_op('nn.conv2d')(is_input(), is_input())
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

The final example  is matching diamonds with a post-dominator relationship. We embed dominator analysis as type of matching in the pattern language in order to allow for pattern matching with unknown topology. This is important because we want to be able to use the language to describe fuse patterns, like elementwise operations followed by a conv2d::

    def test_match_dom_diamond():
        # Pattern
        is_conv2d = is_op('nn.conv2d')(is_input(), is_input())
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

Design
======

The pattern language proposed is designed to be a mirror of Relay's IR with additional support for common scenarios. The goal of the pattern language is to provide a regular-expression like capability for matching data-flow graphs and doing rewriting.

The high level design is to introduce a language of patterns for now we propose the language as::

    Pattern ::= expr
            | *
            | pattern(pattern1, ... patternN)
            | has_type(pattern, type)
            | has_attr(pattern, attr, attr_value)
            | is_input(name)
            | pattern1 `|` pattern2
            | dominates(parent_pattern, path_pattern, child_pattern)

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

Attribute Pattern
*****************

Check that the operator matched by the pattern has an attribute with a particular value.

Input
*****

Check that the expression is an input, i.e has no parents and is a variable.


Alternate
*********

Either match the first pattern or the second pattern.

Domination
**********

Match child pattern, find a match for the parent pattern, insuring that the child ultimately dominates the parrent (i.e., no nodes outside the pattern use outputs of the parent), and that ever node betwen the child and the pattern matches the path pattern.
