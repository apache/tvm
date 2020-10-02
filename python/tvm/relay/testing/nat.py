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
"""Defines a unary natural number (Peano natural number) abstract
data type for Relay and provides some utility functions for it.
Nats are useful for testing purposes, as they make it easy to write
test cases for recursion and pattern matching."""

from tvm.relay.backend.interpreter import ConstructorValue


def get_type(prelude, name):
    ty_var = prelude.mod.get_global_type_var(name)
    ty_data = prelude.mod.type_definitions[ty_var]
    return tuple([ty_var] + list(ty_data.constructors))


def count(prelude, n):
    """Takes a ConstructorValue corresponding to a nat ADT
    and converts it into a Python integer. This is an example of
    using an ADT value in Python.
    """
    assert isinstance(n, ConstructorValue)
    _, z, s = prelude.mod.get_type("nat")
    if n.tag == z.tag:
        return 0
    assert n.tag == s.tag
    return 1 + count(prelude, n.fields[0])


def make_nat_value(prelude, n):
    """The inverse of count(): Given a non-negative Python integer,
    constructs a ConstructorValue representing that value as a nat.
    """
    _, z, s = prelude.mod.get_type("nat")
    if n == 0:
        return ConstructorValue(z.tag, [], z)
    return ConstructorValue(s.tag, [make_nat_value(prelude, n - 1)], s)


def make_nat_expr(prelude, n):
    """Given a non-negative Python integer, constructs a Python
    expression representing that integer's value as a nat.
    """
    assert n >= 0
    _, z, s = prelude.mod.get_type("nat")
    ret = z()
    while n > 0:
        ret = s(ret)
        n = n - 1
    return ret
