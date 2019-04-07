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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The expression functor of Relay."""

from .expr import Function, Call, Let, Var, GlobalVar
from .expr import If, Tuple, TupleGetItem, Constant
from .expr import RefCreate, RefRead, RefWrite
from .adt import Constructor, Match, Clause
from .op import Op

class ExprFunctor:
    """
    An abstract visitor defined over Expr.

    Defines the default dispatch over expressions, and
    implements memoization.
    """
    def __init__(self):
        self.memo_map = {}

    # pylint: disable=no-else-return
    def visit(self, expr):
        """Apply the visitor to an expression."""
        found = self.memo_map.get(expr)
        if found:
            return found

        if isinstance(expr, Function):
            res = self.visit_function(expr)
        elif isinstance(expr, Call):
            res = self.visit_call(expr)
        elif isinstance(expr, Let):
            res = self.visit_let(expr)
        elif isinstance(expr, Var):
            res = self.visit_var(expr)
        elif isinstance(expr, GlobalVar):
            res = self.visit_global_var(expr)
        elif isinstance(expr, If):
            res = self.visit_if(expr)
        elif isinstance(expr, Tuple):
            res = self.visit_tuple(expr)
        elif isinstance(expr, TupleGetItem):
            res = self.visit_tuple_getitem(expr)
        elif isinstance(expr, Constant):
            res = self.visit_constant(expr)
        elif isinstance(expr, Op):
            res = self.visit_op(expr)
        elif isinstance(expr, RefCreate):
            res = self.visit_ref_create(expr)
        elif isinstance(expr, RefRead):
            res = self.visit_ref_read(expr)
        elif isinstance(expr, RefWrite):
            res = self.visit_ref_write(expr)
        elif isinstance(expr, Constructor):
            res = self.visit_constructor(expr)
        elif isinstance(expr, Match):
            res = self.visit_match(expr)
        else:
            raise Exception("warning unhandled case: {0}".format(type(expr)))

        self.memo_map[expr] = res

        return res

    def visit_function(self, _):
        raise NotImplementedError()

    def visit_let(self, _):
        raise NotImplementedError()

    def visit_call(self, _):
        raise NotImplementedError()

    def visit_var(self, _):
        raise NotImplementedError()

    def visit_type(self, typ):
        return typ

    def visit_if(self, _):
        raise NotImplementedError()

    def visit_tuple(self, _):
        raise NotImplementedError()

    def visit_tuple_getitem(self, _):
        raise NotImplementedError()

    def visit_global_var(self, _):
        raise NotImplementedError()

    def visit_op(self, _):
        raise NotImplementedError()

    def visit_constant(self, _):
        raise NotImplementedError()

    def visit_ref_create(self, _):
        raise NotImplementedError()

    def visit_ref_write(self, _):
        raise NotImplementedError()

    def visit_ref_read(self, _):
        raise NotImplementedError()

    def visit_constructor(self, _):
        raise NotImplementedError()

    def visit_match(self, _):
        raise NotImplementedError()


class ExprMutator(ExprFunctor):
    """
    A functional visitor over Expr.

    The default behavior recursively traverses the AST
    and reconstructs the AST.
    """
    def visit_function(self, fn):
        new_body = self.visit(fn.body)
        return Function(
            list(fn.params),
            new_body,
            fn.ret_type,
            fn.type_params,
            fn.attrs)

    def visit_let(self, let):
        new_var = self.visit(let.var)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        return Let(new_var, new_val, new_body)

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        return Call(new_fn, new_args, call.attrs)

    def visit_var(self, rvar):
        return rvar

    def visit_global_id(self, global_var):
        return global_var

    def visit_if(self, ite):
        return If(
            self.visit(ite.cond),
            self.visit(ite.true_branch),
            self.visit(ite.false_branch))

    def visit_tuple(self, tup):
        return Tuple([self.visit(field) for field in tup.fields])

    def visit_tuple_getitem(self, op):
        tuple_value = self.visit(op.tuple_value)
        if not tuple_value.same_as(op.tuple_value):
            return TupleGetItem(tuple_value, op.index)
        return op

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_constant(self, const):
        return const

    def visit_constructor(self, con):
        return con

    def visit_match(self, m):
        return Match(self.visit(m.data), [Clause(c.lhs, self.visit(c.rhs)) for c in m.pattern])

    def visit_ref_create(self, r):
        return RefCreate(self.visit(r.value))

    def visit_ref_write(self, r):
        return RefWrite(self.visit(r.ref), self.visit(r.value))

    def visit_ref_read(self, r):
        return RefRead(self.visit(r.ref))
