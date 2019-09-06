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
"""Tests for module functionality."""
import tvm
from tvm import relay
from tvm.relay import Module
from tvm.relay.prelude import Prelude
from tvm.relay.testing import add_nat_definitions

def constructor_list(p):
    return [p.nil, p.cons, p.rose, p.some, p.none, p.z, p.s]


def adt_list(p):
    return [p.nat, p.l, p.optional, p.tree]


def test_constructor_tag_round_trip():
    mod1 = Module()
    p1 = Prelude(mod1)
    add_nat_definitions(p1)
    mod2 = Module()
    p2 = Prelude(mod2)
    add_nat_definitions(p2)

    # ensure hashes match across modules
    ctors1 = constructor_list(p1)
    ctors2 = constructor_list(p2)

    for i in range(len(ctors1)):
        tag = ctors1[i].tag
        ctor = mod2.get_constructor(tag)
        assert ctor == ctors2[i]
        assert ctor.name_hint == ctors1[i].name_hint


def test_constructor_tag_differences():
    # ensure that if we have the type data for a given ADT, the tags
    # for the constructors of the *same ADT* are simple offsets from
    # each other
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)

    adts = adt_list(p)
    for adt in adts:
        data = mod[adt]
        for i in range(len(data.constructors) - 1):
            ctor1 = data.constructors[i]
            ctor2 = data.constructors[i + 1]
            assert ctor2.tag - ctor1.tag == 1
            # make sure there is something present at the MSB
            assert ctor1.tag - i != 0
            assert ctor2.tag - (i + 1) != 0


def test_add_mutual_recursion():
    mod = Module()
    p = Prelude(mod)
    add_nat_definitions(p)

    # even and odd are mutually recursive
    even = relay.GlobalVar('even')
    odd = relay.GlobalVar('odd')

    x = relay.Var("x")
    v = relay.Var("v")
    odd_func = relay.Function(
        [x],
        relay.Match(x, [
            relay.Clause(relay.PatternConstructor(p.s, [relay.PatternVar(v)]), even(v)),
            relay.Clause(relay.PatternConstructor(p.z, []), relay.const(False))
        ]))

    y = relay.Var("y")
    w = relay.Var("w")
    even_func = relay.Function(
        [y],
        relay.Match(y, [
            relay.Clause(relay.PatternConstructor(p.s, [relay.PatternVar(w)]), odd(w)),
            relay.Clause(relay.PatternConstructor(p.z, []), relay.const(True))
        ]))

    mod.add_multiple({even: even_func, odd: odd_func})

    expected_type = relay.FuncType([p.nat()],
                                   relay.scalar_type('bool'))
    assert mod[odd].checked_type == expected_type
    assert mod[even].checked_type == expected_type


def test_initial_mutual_recursion():
    odd = relay.GlobalVar("odd")
    even = relay.GlobalVar("even")

    x = relay.Var("x", relay.scalar_type('int32'))
    odd_func = relay.Function(
        [x],
        relay.If(relay.equal(x, relay.const(0, 'int32')),
                 relay.const(True, 'bool'),
                 even(relay.subtract(x, relay.const(1, 'int32')))))
    y = relay.Var("y", relay.scalar_type('int32'))
    even_func = relay.Function(
        [y],
        relay.If(relay.equal(y, relay.const(1, 'int32')),
                 relay.const(True, 'bool'),
                 odd(relay.subtract(y, relay.const(1, 'int32')))))

    main = relay.GlobalVar('main')
    z = relay.Var('z')
    mapping = {odd: odd_func, even : even_func, main : relay.Function([z], odd(z))}
    mod = relay.Module(mapping)

    expected_type = relay.FuncType([relay.scalar_type('int32')],
                                   relay.scalar_type('bool'))

    assert mod[odd].checked_type == expected_type
    assert mod[even].checked_type == expected_type
    assert mod[main].checked_type == expected_type
