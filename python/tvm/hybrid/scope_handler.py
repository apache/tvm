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
1. For scope handler
When registering a for scope handler, the first 4 arguments must be parser, node, body, loop_vars
and these arguments will provided by Hybrid Script parser automatically
.. code-block:: python
    for loop_vars in tir.xxx():
2. With scope handler
There are 4 subtypes of with scope handlers, classified by
    1) with or without as
    2) allow concise scoping or not
1) with as    & concise
the first 2 arguments must be parser, node
Need to parse the body manually
Example : tir.alloc_with_scope
.. code-block:: python
    target = tir.xxx()
    with tir.xxx() as target:
2) with as    & not concise
the first 2 arguments must be parser, node
Need to parse the body manually
Example : None atm
.. code-block:: python
    with tir.xxx() as target:
3) without as & concise
the first 3 arguments must be parser, node, body
Hybrid Script parser will parse the body automatically
Example : tir.allocate()/tir.realize()/tir.attr()
.. code-block:: python
    tir.xxx()
    with tir.xxx():
4) without as & not concise
the first 3 arguments must be parser, node, body
Hybrid Script parser will parse the body automatically
Example : tir.assert()/tir.let()
.. code-block:: python
    with tir.xxx():
"""
# pylint: disable=redefined-builtin, unused-argument, invalid-name

from typed_ast import ast3 as ast
import tvm.tir
from .registry import register_with_scope, register_for_scope


# With scope handler
@register_with_scope(concise=True, with_var=True)
def allocate(parser, node, extents, dtype, scope, condition=True):
    """ With scope handler function tir.alloc_with_scope(var, extents, dtype, scope, condition) """
    # defining buffer var and parse the body manually

    buffer_var = tvm.te.var(parser.target[0], "handle")
    # (TODO) Uncomment this line if we have richer type info for buffer var
    # buffer_var = tvm.te.var(parser.target[0], tvm.ir.PointerType(tvm.ir.PrimType(dtype)))
    if isinstance(node, ast.With):
        parser.scope_emitter.new_scope()
        parser.scope_emitter.update_symbol(buffer_var.name, buffer_var)
        parser.scope_emitter.node_stack[-1].extend(reversed(node.body))
        body = parser.get_body()
        parser.scope_emitter.pop_scope()
    else:
        parser.scope_emitter.update_symbol(buffer_var.name, buffer_var)
        body = parser.get_body()
    condition = tvm.runtime.convert(condition)
    scope = tvm.runtime.convert(scope)
    body = tvm.tir.Allocate(buffer_var, dtype, extents, condition, body)
    return tvm.tir.AttrStmt(buffer_var, "storage_scope", scope, body)


@register_with_scope(concise=True)
def launch_thread(parser, node, body, env_var, extent):
    extent = tvm.runtime.convert(extent)
    return tvm.tir.AttrStmt(
        tvm.tir.IterVar(
            None, env_var, getattr(tvm.tir.IterVar, "ThreadIndex"), parser.var_env_dict[env_var]
        ),
        "thread_extent",
        extent,
        body,
    )


@register_with_scope(concise=True)
def realize(parser, node, body, buffer_bounds, scope, condition=True):
    """ With scope handler function tir.realize(buffer_bounds, scope, condition) """
    buffer, bounds = buffer_bounds
    scope = tvm.runtime.convert(scope)
    return tvm.tir.AttrStmt(
        buffer, "realize_scope", scope, tvm.tir.BufferRealize(buffer, bounds, condition, body)
    )


@register_with_scope(concise=True)
def attr(parser, node, body, attr_node, attr_key, value):
    """ With scope handler function tir.attr(attr_node, attr_key, value) """
    attr_node = tvm.runtime.convert(attr_node)
    value = tvm.runtime.convert(value)
    return tvm.tir.AttrStmt(attr_node, attr_key, value, body)


@register_with_scope(concise=False)
def Assert(parser, node, body, condition, message):
    """ With scope handler function tir.Assert(condition, message) """
    return tvm.tir.AssertStmt(condition, tvm.runtime.convert(message), body)


@register_with_scope(concise=False)
def let(parser, node, body, var, value):
    """ With scope handler function tir.let(var, value) """
    return tvm.tir.LetStmt(var, value, body)


# For scope handler
@register_for_scope()
def serial(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.serial(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 0, 0, body)


@register_for_scope()
def parallel(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.parallel(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 1, 0, body)


@register_for_scope()
def vectorized(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.vectorized(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 2, 0, body)


@register_for_scope()
def unroll(parser, node, body, loop_vars, begin, end):
    """ For scope handler function tir.unroll(begin, end)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    return tvm.tir.For(loop_vars[0], begin, extent, 3, 0, body)


@register_for_scope(name="range")
def Range(parser, node, body, loop_vars, begin, end, annotation=None):
    """ For scope handler function range(begin, end, annotation)"""
    if len(loop_vars) != 1:
        parser.report_error("Expect exact 1 loop var")
    ana = tvm.arith.Analyzer()
    extent = end if begin == 0 else ana.simplify(end - begin)
    if annotation is None:
        annotation = []
    else:
        annotation = [
            tvm.tir.Annotation(key, tvm.runtime.convert(val) if isinstance(val, str) else val)
            for key, val in annotation.items()
        ]
    return tvm.tir.Loop(loop_vars[0], begin, extent, annotation, body)


@register_for_scope()
def grid(parser, node, body, loop_vars, *extents):
    """ For scope handler function tir.grid(*extents) """
    if len(loop_vars) != len(extents):
        parser.report_error("Inconsitent number of loop vars and extents")
    for loop_var, extent in zip(reversed(loop_vars), reversed(extents)):
        body = tvm.tir.Loop(loop_var, 0, extent, [], body)
    return body
