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
from . import op as _op
from .prelude import Prelude

def while_loop(cond, loop_vars, loop_bodies, loop_name=None):
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

    if loop_name is None:
        loop_name = 'while_loop'

    loop = _expr.Var(loop_name)
    fresh_vars = []

    for i, loop_var in enumerate(loop_vars):
        name = loop_var.name_hint if isinstance(loop_var, _expr.Var) else "arg{}".format(i)
        new_var = _expr.var(name, type_annotation=sb.type_of(loop_var))
        fresh_vars.append(new_var)

    with sb.if_scope(cond(*fresh_vars)):
        sb.ret(loop(*loop_bodies(*fresh_vars)))
    with sb.else_scope():
        sb.ret(_expr.Tuple(fresh_vars))

    func = _expr.Function(fresh_vars, sb.get())
    let = _expr.Let(loop, func, loop)
    return let

def foreach(tensor, iter, init_states, axis=None, loop_name=None, mod=None):
    assert isinstance(tensor, _expr.Expr), "data to slice must be a tensor"
    assert isinstance(init_states, list), "initial states must be a list"

    Prelude(mod)
    list_var = mod.get_global_type_var('list')
    list_type = mod[list_var]
    nil, cons = list_type.constructors
    reverse = mod.get_global_var('rev')

    sb = ScopeBuilder()
    state_vars = []
    for i, st in enumerate(init_states):
        state_ty = sb.type_of(st)
        state_vars.append(
            _expr.var(f"st{i}", type_annotation=state_ty))

    data_ty = sb.type_of(tensor)
    data_var = _expr.var("data", type_annotation=data_ty)

    if axis is None:
        axis = 0

    tensor_sh = _op.shape_of(data_var)
    end = _op.take(tensor_sh, _expr.const(axis, dtype='int32'), axis=0)

    def _foreach_iter(i, outs, *states):
        slice = _op.take(data_var, i, axis=axis)
        step = iter(slice, *states)
        out = _expr.TupleGetItem(step, 0)
        states = []
        for st_i in range(len(init_states)):
            states.append(_expr.TupleGetItem(step, 1 + st_i))
        outs = _expr.Call(cons, [out, outs])
        return [i + _expr.const(1, dtype="int32"), outs, *states]

    def _foreach_cond(i, outs, *states):
        return _op.less(i, end)

    i = _expr.var('i', shape=(), dtype='int32')
    outs = _expr.Call(nil, [])
    loop = while_loop(_foreach_cond, [i, outs] + state_vars, _foreach_iter)
    backwards_result = loop(_expr.const(0, dtype="int32"), outs, *state_vars)
    outs = _expr.TupleGetItem(backwards_result, 1)
    states = []
    for st_i in range(len(init_states)):
        states.append(_expr.TupleGetItem(backwards_result, 2 + st_i))
    sb.ret(_expr.Tuple([reverse(outs), *states]))
    return _expr.Function([data_var, *state_vars], sb.get())
