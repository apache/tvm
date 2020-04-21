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
"""Defines a unary natural number (Peano natural number) abstract
data type for Relay and provides some utility functions for it.
Nats are useful for testing purposes, as they make it easy to write
test cases for recursion and pattern matching."""

from tvm.relay.adt import Constructor, TypeData, Clause, Match, PatternConstructor, PatternVar
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay.expr import Var, GlobalVar
from tvm.relay.function import Function
from tvm.relay.ty import GlobalTypeVar, TypeVar, FuncType

def define_nat_adt(prelude):
    """Defines a Peano (unary) natural number ADT.
    Zero is represented by z(). s(n) adds 1 to a nat n.
    Adds the fields nat, z, and s to the preluide, representing
    (respectively) the nat ADT and the z and s constructors.
    """
    prelude.nat = GlobalTypeVar("nat")
    prelude.z = Constructor("z", [], prelude.nat)
    prelude.s = Constructor("s", [prelude.nat()], prelude.nat)
    prelude.mod[prelude.nat] = TypeData(prelude.nat, [], [prelude.z, prelude.s])


def define_nat_double(prelude):
    """Defines a function that doubles a nat. Adds a field called
    'double' to the prelude, giving the GlobalVar pointing to
    the function.
    """
    prelude.double = GlobalVar("double")
    x = Var("x", prelude.nat())
    y = Var("y")
    z_case = Clause(PatternConstructor(prelude.z), prelude.z())
    s_case = Clause(PatternConstructor(prelude.s, [PatternVar(y)]),
                    prelude.s(prelude.s(prelude.double(y))))
    prelude.mod[prelude.double] = Function([x], Match(x, [z_case, s_case]))


def define_nat_add(prelude):
    """Defines a function that adds two nats and adds a field to the
    prelude 'add' giving the GlobalVar pointing to that function.
    """
    prelude.add = GlobalVar("add")
    x = Var("x", prelude.nat())
    y = Var("y", prelude.nat())
    a = Var("a")
    z_case = Clause(PatternConstructor(prelude.z), y)
    s_case = Clause(PatternConstructor(prelude.s, [PatternVar(a)]),
                    prelude.s(prelude.add(a, y)))
    prelude.mod[prelude.add] = Function([x, y], Match(x, [z_case, s_case]))


# versions of prelude functions that use nats instead of scalars

def define_nat_nth(prelude):
    """Defines a function to get the nth eleemnt of a list using
    a nat to index into the list.

    nat_nth(l, n): fun<a>(list[a], nat) -> a
    """
    prelude.nat_nth = GlobalVar("nat_nth")
    a = TypeVar("a")
    x = Var("x", prelude.l(a))
    n = Var("n", prelude.nat())
    y = Var("y")

    z_case = Clause(PatternConstructor(prelude.z), prelude.hd(x))
    s_case = Clause(PatternConstructor(prelude.s, [PatternVar(y)]),
                    prelude.nat_nth(prelude.tl(x), y))

    prelude.mod[prelude.nat_nth] = Function([x, n],
                                            Match(n, [z_case, s_case]),
                                            a, [a])


def define_nat_update(prelude):
    """Defines a function to update the nth element of a list and return the updated list.

    nat_update(l, i, v) : fun<a>(list[a], nat, a) -> list[a]
    """
    prelude.nat_update = GlobalVar("nat_update")
    a = TypeVar("a")
    # pylint: disable=invalid-name
    l = Var("l", prelude.l(a))
    n = Var("n", prelude.nat())
    v = Var("v", a)
    y = Var("y")

    z_case = Clause(PatternConstructor(prelude.z),
                    prelude.cons(v, prelude.tl(l)))
    s_case = Clause(PatternConstructor(prelude.s, [PatternVar(y)]),
                    prelude.cons(
                        prelude.hd(l),
                        prelude.nat_update(prelude.tl(l), y, v)))

    prelude.mod[prelude.nat_update] = Function([l, n, v],
                                               Match(n, [z_case, s_case]),
                                               prelude.l(a), [a])


def define_nat_iterate(prelude):
    """Defines a function that takes a number n and a function f;
    returns a closure that takes an argument and applies f
    n times to its argument.

    Signature: fn<a>(fn(a) -> a, nat) -> fn(a) -> a
    """
    prelude.nat_iterate = GlobalVar("nat_iterate")
    a = TypeVar("a")
    f = Var("f", FuncType([a], a))
    x = Var("x", prelude.nat())
    y = Var("y", prelude.nat())

    z_case = Clause(PatternConstructor(prelude.z), prelude.id)
    s_case = Clause(PatternConstructor(prelude.s, [PatternVar(y)]),
                    prelude.compose(f, prelude.nat_iterate(f, y)))

    prelude.mod[prelude.nat_iterate] = Function([f, x],
                                                Match(x, [z_case, s_case]),
                                                FuncType([a], a),
                                                [a])


def add_nat_definitions(prelude):
    """Given a Relay prelude, adds a Peano nat ADT, as well as functions
    for adding nats and doubling nats. It also adds versions of
    update, nth, and iterate that take nats instead of scalars (the
    names are prefixed with `nat_`)."""
    define_nat_adt(prelude)
    define_nat_double(prelude)
    define_nat_add(prelude)
    define_nat_nth(prelude)
    define_nat_update(prelude)
    define_nat_iterate(prelude)


# helper functions for working with nats


def count(prelude, n):
    """Takes a ConstructorValue corresponding to a nat ADT
    and converts it into a Python integer. This is an example of
    using an ADT value in Python.
    """
    assert isinstance(n, ConstructorValue)
    if n.tag == prelude.z.tag:
        return 0
    assert n.tag == prelude.s.tag
    return 1 + count(prelude, n.fields[0])


def make_nat_value(prelude, n):
    """The inverse of count(): Given a non-negative Python integer,
    constructs a ConstructorValue representing that value as a nat.
    """
    if n == 0:
        return ConstructorValue(prelude.z.tag, [], None)
    return ConstructorValue(prelude.s.tag, [make_nat_value(prelude, n - 1)], None)


def make_nat_expr(prelude, n):
    """Given a non-negative Python integer, constructs a Python
    expression representing that integer's value as a nat.
    """
    assert n >= 0
    ret = prelude.z()
    while n > 0:
        ret = prelude.s(ret)
        n = n - 1
    return ret
