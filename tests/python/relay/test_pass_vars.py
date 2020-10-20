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
import tvm
from tvm import te
from tvm import relay
from tvm.relay.analysis import (
    free_vars,
    free_type_vars,
    bound_vars,
    bound_type_vars,
    all_vars,
    all_type_vars,
)


def assert_vars_match(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert actual[i] == expected[i]


def test_free_vars():
    ty = relay.TensorType([], "int32")
    x = relay.Var("x", ty)
    fvx = free_vars(x)
    assert len(fvx) == 1
    assert fvx[0] == x
    v = relay.Constant(tvm.nd.array(10))

    let = relay.Let(x, v, x)
    fvx = free_vars(let)
    assert len(free_vars(let)) == 0
    f = relay.Function([x], x, ty)
    assert len(free_vars(f)) == 0


def test_free_vars_tuple():
    t = relay.Var("t")
    fv = free_vars(relay.Tuple([t, t]))
    assert len(fv) == 1
    assert fv[0] == t
    fv = free_vars(relay.TupleGetItem(t, 123))
    assert len(fv) == 1
    assert fv[0] == t


def test_free_type_vars():
    tp = relay.TypeVar("")
    ty = relay.TupleType([tp, relay.TensorType([], "int32")])
    x = relay.Var("x", ty)
    y = relay.Var("y")
    let = relay.Let(x, y, x)
    fvl = free_vars(let)
    assert len(fvl) == 1
    assert fvl[0] == y
    ftvl = free_type_vars(let)
    assert len(ftvl) == 1
    assert ftvl[0] == tp


def test_bound_vars():
    x = relay.Var("x")
    y = relay.Var("y")
    z = relay.Var("z")
    a = relay.Var("a")

    f1 = relay.Function([x, y, z], relay.Let(a, x, relay.Tuple([])))
    assert_vars_match(bound_vars(f1), [x, y, z, a])

    tup = relay.Tuple([x, y, z, a])
    assert len(bound_vars(tup)) == 0

    f2 = relay.Function([x, y], relay.Tuple([x, y, z, a]))
    assert_vars_match(bound_vars(f2), [x, y])


def test_match_vars():
    mod = tvm.IRModule()
    p = relay.prelude.Prelude(mod)
    rlist, cons, nil = p.mod.get_type("List")

    x = relay.Var("x")
    y = relay.Var("y")
    z = relay.Var("z")

    match1 = relay.Match(
        nil(),
        [
            relay.Clause(relay.PatternConstructor(nil), z),
            relay.Clause(
                relay.PatternConstructor(cons, [relay.PatternVar(x), relay.PatternVar(y)]),
                cons(x, y),
            ),
        ],
    )

    match2 = relay.Match(
        nil(),
        [
            relay.Clause(
                relay.PatternConstructor(cons, [relay.PatternWildcard(), relay.PatternVar(x)]), y
            ),
            relay.Clause(relay.PatternWildcard(), z),
        ],
    )

    assert_vars_match(bound_vars(match1), [x, y])
    assert_vars_match(free_vars(match1), [z])
    assert_vars_match(all_vars(match1), [z, x, y])

    assert_vars_match(bound_vars(match2), [x])
    assert_vars_match(free_vars(match2), [y, z])
    assert_vars_match(all_vars(match2), [x, y, z])


def test_bound_type_vars():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    c = relay.TypeVar("c")

    ft1 = relay.FuncType([a], b, [a, b])
    bound_ft1 = bound_type_vars(ft1)
    assert_vars_match(bound_type_vars(ft1), [a, b])

    ft2 = relay.FuncType([], c, [a])
    assert_vars_match(bound_type_vars(ft2), [a])

    tup_ty = relay.TupleType([a, b, c])
    assert len(bound_type_vars(tup_ty)) == 0

    f1 = relay.Function([], relay.Tuple([]), type_params=[a, b])
    assert_vars_match(bound_type_vars(f1), [a, b])

    f2 = relay.Function([], relay.Tuple([]), c)
    assert len(bound_type_vars(f2)) == 0

    x = relay.Var("x", a)
    let1 = relay.Let(x, relay.Tuple([]), x)
    assert len(bound_type_vars(let1)) == 0

    let2 = relay.Let(x, relay.Function([], relay.Tuple([]), type_params=[b, c]), x)
    assert_vars_match(bound_type_vars(let2), [b, c])


def test_all_vars():
    x = relay.Var("x")
    y = relay.Var("y")
    z = relay.Var("z")

    f1 = relay.Function([x, y], z)
    assert_vars_match(all_vars(f1), [x, y, z])

    f2 = relay.Function([x], relay.Let(y, relay.Tuple([]), z))
    assert_vars_match(all_vars(f2), [x, y, z])

    f3 = relay.Function([x], relay.Tuple([y, z]))
    assert_vars_match(all_vars(f3), [x, y, z])

    tup = relay.Tuple([x, y, z])
    assert_vars_match(all_vars(tup), [x, y, z])


def test_all_type_vars():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    c = relay.TypeVar("c")

    ft1 = relay.FuncType([b], c, [a])
    assert_vars_match(all_type_vars(ft1), [a, b, c])

    ft2 = relay.FuncType([], relay.TupleType([a, b, c]), [])
    assert_vars_match(all_type_vars(ft2), [a, b, c])

    w = relay.Var("w")
    x = relay.Var("x", a)
    y = relay.Var("y", b)
    z = relay.Var("z", c)

    f1 = relay.Function([x], y, b, [a])
    assert_vars_match(all_type_vars(f1), [a, b])

    f2 = relay.Function([x], relay.Let(y, x, z))
    assert_vars_match(all_type_vars(f2), [a, b, c])

    f3 = relay.Function([], relay.Tuple([x, y, z]), ret_type=relay.TupleType([a, b, c]))
    assert_vars_match(all_type_vars(f3), [a, b, c])

    f4 = relay.Function([w], relay.Tuple([]), type_params=[a, b, c])
    assert_vars_match(all_type_vars(f4), [a, b, c])

    f5 = relay.Function([w], w)
    assert len(all_type_vars(f5)) == 0
