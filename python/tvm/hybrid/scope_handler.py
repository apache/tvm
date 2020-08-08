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
"""Hybrid Script Parser Scope Handler Functions

This module provides the functions registered into parser under with_scope or for_scope category.
Scope handler nodes are StmtNodes with body, which are used to handle such scenarios.

.. code-block:: python

    for x in tir.name():
    with tir.name():
    tir.name() # with scope handlers + concise scoping

"""
# pylint: disable=redefined-builtin, unused-argument, invalid-name
import tvm.tir
from .registry import register_with_scope, register_for_scope


# With scope handler
@register_with_scope(concise=False)
def Assert(parser, node, condition, message, body):
    """ With scope handler function assert(condition, message, body) """

    return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), body)


@register_with_scope(concise=False)
def let(parser, node, var, value, body):
    """ With scope handler function let(var, value, body) """

    return tvm.tir.LetStmt(var, value, body)


@register_with_scope(concise=True)
def realize(parser, node, buffer_bounds, body, condition=True):
    """ With scope handler function realize(buffer_bounds, condition, body) """

    buffer, bounds = buffer_bounds
    return tvm.tir.BufferRealize(buffer, bounds, condition, body)


@register_with_scope(concise=True)
def attr(parser, node, attr_node, attr_key, value, body):
    """ With scope handler function attr(attr_node, attr_key, value, body) """

    return tvm.tir.AttrStmt(attr_node, attr_key, tvm.runtime.convert(value), body)


@register_with_scope(concise=True)
def allocate(parser, node, buffer_var, dtype, extents, body, condition=True):
    """ With scope handler function allocate(buffer_var, dtype, extents, condition, body) """

    return tvm.tir.Allocate(buffer_var, dtype, extents, tvm.runtime.convert(condition), body)


# For scope handler
@register_for_scope()
def range(parser, node, begin, end, for_type="serial"):
    """ For scope handler function range(begin, end, annotation)"""
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    loop_var_name = node.target.id
    loop_var = tvm.te.var(loop_var_name, dtype="int32")

    parser.scope_emitter.new_scope()
    parser.scope_emitter.update_symbol(loop_var_name, loop_var)
    parser.scope_emitter.node_stack[-1].extend(reversed(node.body))
    body = parser.get_body()
    parser.scope_emitter.pop_scope()

    for_type_dict = {"serial": 0, "parallel": 1, "vectorized": 2, "unroll": 3}
    if for_type not in for_type_dict:
        parser.report_error("unknown for type " + for_type)
    return tvm.tir.For(loop_var, begin, extent, for_type_dict[for_type], 0, body)
