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

"""The scope builder interface."""
from __future__ import absolute_import

from . import ty as _ty
from . import expr as _expr
from .._ffi import base as _base


class WithScope(object):
    """A wrapper for builder methods which introduce scoping.

    Parameters
    ----------
    enter_value: object
        The value returned by enter.
    """

    def __init__(self, enter_value, exit_cb):
        self._enter_value = enter_value
        self._exit_cb = exit_cb

    def __enter__(self):
        return self._enter_value

    def __exit__(self, ptype, value, trace):
        if value:
            raise value
        self._exit_cb()


def _make_lets(bindings, ret_value):
    """Make a nested let expressions.

    Parameters
    ----------
    bindings: List[Tuple[tvm.relay.Var,tvm.relay.Expr]]
        The sequence of let bindings

    ret_value: tvm.relay.Expr
        The final value of the expression.

    Returns
    -------
    lets: tvm.relay.Expr
        A nested let expression.
    """
    if ret_value is None:
        raise RuntimeError("ret is not called in this scope")
    if isinstance(ret_value, _expr.If) and ret_value.false_branch is None:
        raise RuntimeError("Creating an If expression without else.")
    let_expr = ret_value
    for var, value in reversed(bindings):
        let_expr = _expr.Let(var, value, let_expr)
    return let_expr


class ScopeBuilder(object):
    """Scope builder class.

    Enables users to build up a nested
    scope(let, if) expression easily.

    Examples
    --------
    .. code-block: python

        sb = relay.ScopeBuilder()
        cond = relay.var("cond", 'bool')
        x = relay.var("x")
        y = relay.var("y")

        with sb.if_scope(cond):
            one = relay.const(1, "float32")
            t1 = sb.let(t1, relay.add(x, one))
            sb.ret(t1)
        with sb.else_scope():
            sb.ret(y)

        print(sb.get().astext())
    """

    def __init__(self):
        self._bindings = [[]]
        self._ret_values = [None]

    def _enter_scope(self):
        self._bindings.append([])
        self._ret_values.append(None)

    def _exit_scope(self):
        bindings = self._bindings.pop()
        ret_value = self._ret_values.pop()
        return bindings, ret_value

    def let(self, var, value):
        """Create a new let binding.

        Parameters
        ----------
        var: Union[Tuple[str, relay.Type], tvm.relay.Var]
            The variable or name of variable.

        value: tvm.relay.Expr
            The value to be bound
        """
        if isinstance(var, (tuple, list)):
            if len(var) > 2:
                raise ValueError("Expect var to be Tuple[str, relay.Type]")
            var = _expr.var(*var)
        elif isinstance(var, _base.string_types):
            var = _expr.var(var)
        self._bindings[-1].append((var, value))
        return var

    def if_scope(self, cond):
        """Create a new if scope.

        Parameters
        ----------
        cond: tvm.relay.expr.Expr
            The condition

        Returns
        -------
        scope: WithScope
            The if scope.

        Note
        ----
        The user must follows with an else scope.
        """
        self._enter_scope()

        def _on_exit():
            bindings, ret_value = self._exit_scope()
            if self._ret_values[-1] is not None:
                raise RuntimeError("result already returned before if scope")
            true_branch = _make_lets(bindings, ret_value)
            self._ret_values[-1] = _expr.If(cond, true_branch, None)

        return WithScope(None, _on_exit)

    def else_scope(self):
        """Create a new else scope.

        Returns
        -------
        scope: WithScope
            The if scope.
        """
        self._enter_scope()

        def _on_exit():
            bindings, ret_value = self._exit_scope()
            partial_if = self._ret_values[-1]
            no_else = not isinstance(partial_if, _expr.If) or partial_if.false_branch is not None
            if no_else:
                raise RuntimeError("else scope must follows")
            false_branch = _make_lets(bindings, ret_value)
            self._ret_values[-1] = _expr.If(partial_if.cond, partial_if.true_branch, false_branch)

        return WithScope(None, _on_exit)

    def type_of(self, expr):
        """
        Compute the type of an expression.

        Parameters
        ----------
        expr: relay.Expr
            The expression to compute the type of.
        """
        if isinstance(expr, _expr.Var):
            return expr.type_annotation

        ity = _ty.IncompleteType()
        var = _expr.var("unify", ity)
        self.let(var, expr)
        return ity

    def ret(self, value):
        """Set the return value of this scope.

        Parameters
        ----------
        value: tvm.relay.expr.Expr
            The return value.
        """
        if self._ret_values[-1] is not None:
            raise RuntimeError("ret value is already set in this scope.")
        self._ret_values[-1] = value

    def get(self):
        """Get the generated result.

        Returns
        -------
        value: tvm.relay.expr.Expr
            The final result of the expression.
        """
        if len(self._bindings) != 1:
            raise RuntimeError("can only call get at the outmost scope")
        return _make_lets(self._bindings[-1], self._ret_values[-1])
