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
# pylint: disable=invalid-name
"""
Utilities for building Relay loops.
"""
from .scope_builder import ScopeBuilder
from . import expr as _expr
from . import function as _function


def while_loop(cond, loop_vars, loop_bodies):
    """
    Construct a while loop.

    Parameters
    ----------

    cond: Callable[Tuple[relay.Expr], relay.Expr]
        The condition of the loop.

    loop_vars:  Tuple[relay.Expr]
        The variables being looped over.
        The initial values of the loop, will be used to
        construct the loop variables.

    loop_bodies: Callable[Tuple[relay.Expr], Tuple[relay.Expr]]
        The body of the loop, should be a function which
        given loop variables produces the output result
        also as a tuple

    Returns
    -------
    loop: relay.Expr
        The loop expression.
    """
    sb = ScopeBuilder()
    loop = _expr.Var("while_loop")
    fresh_vars = []

    for i, loop_var in enumerate(loop_vars):
        name = loop_var.name_hint if isinstance(loop_var, _expr.Var) else "arg{}".format(i)
        new_var = _expr.var(name, type_annotation=sb.type_of(loop_var))
        fresh_vars.append(new_var)

    with sb.if_scope(cond(*fresh_vars)):
        sb.ret(loop(*loop_bodies(*fresh_vars)))
    with sb.else_scope():
        sb.ret(_expr.Tuple(fresh_vars))

    func = _function.Function(fresh_vars, sb.get())
    let = _expr.Let(loop, func, loop)
    return let
