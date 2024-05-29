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
from tvm.ir import Op

from .function import Function, FunctionWithFields
from .expr import Call, Let, Var, GlobalVar
from .expr import If, Tuple, TupleGetItem, Constant
from .expr import RefCreate, RefRead, RefWrite
from .adt import Constructor, Match, Clause


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
        if expr in self.memo_map:
            return self.memo_map[expr]

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
            raise Exception(f"warning unhandled case: {type(expr)}")

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


class ExprVisitor(ExprFunctor):
    """
    A visitor over Expr.

    The default behavior recursively traverses the AST.
    """

    def visit_tuple(self, tup):
        for x in tup.fields:
            self.visit(x)

    def visit_call(self, call):
        self.visit(call.op)
        for a in call.args:
            self.visit(a)

    def visit_var(self, var):
        pass

    def visit_let(self, let):
        self.visit(let.var)
        self.visit(let.value)
        self.visit(let.body)

    def visit_function(self, fn):
        for x in fn.params:
            self.visit(x)
        self.visit(fn.body)

    def visit_if(self, i):
        self.visit(i.cond)
        self.visit(i.true_branch)
        self.visit(i.false_branch)

    def visit_global_var(self, gv):
        pass

    def visit_constructor(self, c):
        pass

    def visit_op(self, op):
        pass

    def visit_constant(self, const):
        pass

    def visit_ref_create(self, r):
        self.visit(r.value)

    def visit_ref_read(self, r):
        self.visit(r.ref)

    def visit_ref_write(self, r):
        self.visit(r.ref)
        self.visit(r.value)

    def visit_tuple_getitem(self, t):
        self.visit(t.tuple_value)

    def visit_match(self, m):
        self.visit(m.data)
        for c in m.clauses:
            self.visit(c.rhs)


class ExprMutator(ExprFunctor):
    """
    A functional visitor over Expr.

    The default behavior recursively traverses the AST
    and reconstructs the AST.
    """

    def visit_function(self, fn):
        new_params = [self.visit(x) for x in fn.params]
        new_body = self.visit(fn.body)
        if new_params == list(fn.params) and new_body == fn.body:
            return fn
        return FunctionWithFields(fn, list(new_params), new_body)

    def visit_let(self, let):
        new_var = self.visit(let.var)
        new_val = self.visit(let.value)
        new_body = self.visit(let.body)
        if new_var == let.var and new_val == let.value and new_body == let.body:
            return let
        return Let(new_var, new_val, new_body)

    def visit_call(self, call):
        new_fn = self.visit(call.op)
        new_args = [self.visit(arg) for arg in call.args]
        if new_fn == call.op and new_args == list(call.args):
            return call
        return Call(new_fn, new_args, call.attrs, call.type_args, call.span)

    def visit_var(self, var):
        return var

    def visit_global_id(self, global_var):
        return global_var

    def visit_if(self, ite):
        new_cond = self.visit(ite.cond)
        new_true_branch = self.visit(ite.true_branch)
        new_false_branch = self.visit(ite.false_branch)
        if (
            new_cond == ite.cond
            and new_true_branch == ite.true_branch
            and new_false_branch == ite.false_branch
        ):
            return ite
        return If(new_cond, new_true_branch, new_false_branch)

    def visit_tuple(self, tup):
        new_fields = [self.visit(field) for field in tup.fields]
        if new_fields == list(tup.fields):
            return tup
        return Tuple(new_fields, tup.span)

    def visit_tuple_getitem(self, op):
        new_tuple_value = self.visit(op.tuple_value)
        if new_tuple_value == op.tuple_value:
            return op
        return TupleGetItem(new_tuple_value, op.index, span=op.span)

    def visit_global_var(self, gvar):
        return gvar

    def visit_op(self, op):
        return op

    def visit_constant(self, const):
        return const

    def visit_constructor(self, con):
        return con

    def visit_match(self, m):
        new_data = self.visit(m.data)
        new_clauses = [Clause(c.lhs, self.visit(c.rhs)) for c in m.clauses]
        if new_data == m.data and all(x.rhs == y.rhs for x, y in zip(new_clauses, m.clauses)):
            return m
        return Match(new_data, new_clauses, complete=m.complete)

    def visit_ref_create(self, r):
        new_value = self.visit(r.value)
        if new_value == r.value:
            return r
        return RefCreate(new_value)

    def visit_ref_write(self, r):
        new_ref = self.visit(r.ref)
        new_value = self.visit(r.value)
        if new_ref == r.ref and new_value == r.value:
            return r
        return RefWrite(new_ref, new_value)

    def visit_ref_read(self, r):
        new_ref = self.visit(r.ref)
        if new_ref == r.ref:
            return r
        return RefRead(new_ref)
