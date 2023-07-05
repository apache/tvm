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
import pytest
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.relay.backend.interpreter import ConstructorValue
from tvm.relay import create_executor
from tvm.relay.prelude import Prelude, StaticTensorArrayOps
from tvm.relay.testing import count as count_, make_nat_value, make_nat_expr

import numpy as np

prelude = p = Prelude(tvm.IRModule({}))
p.mod.import_from_std("nat.rly")


def count(e):
    return count_(p, e)


dev = tvm.device("llvm", 0)


def eval(expr):
    # CAUTION: These tests re-process the entire prelude for each test expression.
    # Hoisting the create_executor won't improve that since preprocessing won't begin
    # until the evaluate.
    return create_executor(mod=prelude.mod, device=dev, target="llvm").evaluate(expr)


nat, z, s = prelude.mod.get_type("nat")

double = p.mod.get_global_var("nat_double")
add = p.mod.get_global_var("nat_add")

optional, some, none = prelude.mod.get_type("Option")
rlist, cons, nil = prelude.mod.get_type("List")

hd = p.hd
tl = p.tl
nth = p.nth
update = p.update
length = p.length
map = p.map
foldl = p.foldl
foldr = p.foldr
foldr1 = p.foldr1
sum = p.sum

concat = p.concat
filter = p.filter
zip = p.zip
rev = p.rev
unfoldl = p.unfoldl
unfoldr = p.unfoldr
map_accumr = p.map_accumr
map_accuml = p.map_accuml

tree, rose = prelude.mod.get_type("Tree")

tmap = p.tmap
size = p.size

compose = p.compose
iterate = p.iterate


def to_list(l):
    assert isinstance(l, ConstructorValue)
    val = l
    ret = []
    while True:
        if val.tag == cons.tag:
            ret.append(val.fields[0])
            val = val.fields[1]
        else:
            assert val.tag == nil.tag
            break
    return ret


def tree_to_dict(t):
    assert isinstance(t, ConstructorValue)
    ret = {}
    assert t.tag == rose.tag
    ret["member"] = t.fields[0]
    ret["children"] = []
    for subtree in to_list(t.fields[1]):
        l = tree_to_dict(subtree)
        ret["children"].append(l)
    return ret


def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o.numpy().tolist()]
    elif isinstance(o, tvm.runtime.container.ADT):
        if len(o) == 0:
            tensor_nil = p.get_var("tensor_nil", dtype=dtype)
            if tensor_nil.tag == o.tag:
                return [0]
            return []

        result = []
        for f in o:
            result.extend(vmobj_to_list(f, dtype))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == "Cons":
            tl = vmobj_to_list(o.fields[1], dtype)
            hd = vmobj_to_list(o.fields[0], dtype)
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == "Nil":
            return []
        elif "tensor_nil" in o.constructor.name_hint:
            return [0]
        elif "tensor" in o.constructor.name_hint:
            return [o.fields[0].numpy()]
        else:
            raise RuntimeError("Unknown object type: %s" % o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


# turns a scalar-valued relay tensor value into a python number
def get_scalar(tv):
    return tv.numpy().item()


# @tvm.testing.uses_gpu
def test_nat_value():
    assert count(make_nat_value(p, 10)) == 10
    assert count(eval(s(s(z())))) == 2


@tvm.testing.uses_gpu
def test_nat_constructor():
    func = relay.Function([], z())
    test_z = relay.GlobalVar("test_z")
    test_sz = relay.GlobalVar("test_sz")
    prelude.mod[test_z] = func
    func = relay.Function([], s(z()))
    prelude.mod[test_sz] = func
    ck_mod = relay.transform.InferType()(prelude.mod)
    assert ck_mod[test_z].body.checked_type == nat()
    assert ck_mod[test_sz].body.checked_type == nat()


@tvm.testing.uses_gpu
def test_double():
    assert prelude.mod[double].checked_type == relay.FuncType([nat()], nat())
    res = eval(double(s(z())))
    assert count(res) == 2


@tvm.testing.uses_gpu
def test_add():
    assert prelude.mod[add].checked_type == relay.FuncType([nat(), nat()], nat())
    res = eval(add(s(z()), s(z())))
    assert count(res) == 2


@tvm.testing.uses_gpu
def test_list_constructor():
    test_consz = relay.GlobalVar("test_consz")
    func = relay.Function([], cons(z(), nil()))
    prelude.mod[test_consz] = func
    ck_mod = relay.transform.InferType()(prelude.mod)
    assert ck_mod[test_consz].body.checked_type == rlist(nat())


@tvm.testing.uses_gpu
def test_hd_tl():
    expected = list(range(10))
    l = nil()
    for i in reversed(expected):
        l = cons(make_nat_expr(prelude, i), l)

    got = []
    for i in range(len(expected)):
        got.append(count(eval(hd(l))))
        l = tl(l)

    assert got == expected


@tvm.testing.uses_gpu
def test_nth():
    expected = list(range(10))
    l = nil()
    for i in reversed(expected):
        l = cons(relay.const(i), l)

    for i in range(len(expected)):
        nth = prelude.mod.get_global_var("nth")
        item = eval(nth(l, relay.const(i)))
        assert get_scalar(item) == i


@tvm.testing.uses_gpu
def test_update():
    expected = list(range(10))
    l = nil()
    # create zero initialized list
    for i in range(len(expected)):
        l = cons(make_nat_expr(prelude, 0), l)

    # set value
    for i, v in enumerate(expected):
        l = update(l, relay.const(i), make_nat_expr(prelude, v))

    got = []
    for i in range(len(expected)):
        got.append(count(eval(nth(l, relay.const(i)))))

    assert got == expected


@tvm.testing.uses_gpu
def test_length():
    a = relay.TypeVar("a")
    assert prelude.mod[length].checked_type == relay.FuncType(
        [rlist(a)], relay.scalar_type("int32"), [a]
    )
    res = eval(length(cons(z(), cons(z(), cons(z(), nil())))))
    assert get_scalar(res) == 3


@tvm.testing.uses_gpu
def test_map():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = prelude.mod[map].checked_type
    rhs = relay.FuncType([relay.FuncType([a], b), rlist(a)], rlist(b), [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    add_one = relay.Function([x], s(x))
    res = eval(map(add_one, cons(z(), cons(z(), nil()))))
    ones = to_list(res)
    assert len(ones) == 2
    assert count(ones[0]) == 1 and count(ones[1]) == 1


@tvm.testing.uses_gpu
def test_foldl():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")

    lhs = prelude.mod[foldl].checked_type
    rhs = relay.FuncType([relay.FuncType([a, b], a), a, rlist(b)], a, [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    rev_dup = relay.Function([y, x], cons(x, cons(x, y)))
    res = eval(
        foldl(
            rev_dup,
            nil(),
            cons(
                make_nat_expr(prelude, 1),
                cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
            ),
        )
    )
    reversed = to_list(res)
    assert len(reversed) == 6
    assert count(reversed[0]) == 3 and count(reversed[1]) == 3
    assert count(reversed[2]) == 2 and count(reversed[3]) == 2
    assert count(reversed[4]) == 1 and count(reversed[5]) == 1


@tvm.testing.uses_gpu
def test_foldr():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = prelude.mod[foldr].checked_type
    rhs = relay.FuncType([relay.FuncType([a, b], b), b, rlist(a)], b, [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    identity = relay.Function([x, y], cons(x, y))
    res = eval(
        foldr(
            identity,
            nil(),
            cons(
                make_nat_expr(prelude, 1),
                cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
            ),
        )
    )
    same = to_list(res)
    assert len(same) == 3
    assert count(same[0]) == 1 and count(same[1]) == 2 and count(same[2]) == 3


@tvm.testing.uses_gpu
def test_foldr1():
    a = relay.TypeVar("a")
    lhs = prelude.mod[foldr1].checked_type
    rhs = relay.FuncType([relay.FuncType([a, a], a), rlist(a)], a, [a])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    f = relay.Function([x, y], add(x, y))
    res = eval(
        foldr1(
            f,
            cons(
                make_nat_expr(prelude, 1),
                cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
            ),
        )
    )

    assert count(res) == 6


@tvm.testing.uses_gpu
def test_sum():
    assert prelude.mod[sum].checked_type == relay.FuncType(
        [rlist(relay.scalar_type("int32"))], relay.scalar_type("int32")
    )
    res = eval(sum(cons(relay.const(1), cons(relay.const(2), nil()))))
    assert get_scalar(res) == 3


@tvm.testing.uses_gpu
def test_concat():
    a = relay.TypeVar("a")
    assert prelude.mod[concat].checked_type == relay.FuncType([rlist(a), rlist(a)], rlist(a), [a])

    l1 = cons(make_nat_expr(prelude, 1), cons(make_nat_expr(prelude, 2), nil()))
    l2 = cons(make_nat_expr(prelude, 3), cons(make_nat_expr(prelude, 4), nil()))
    res = eval(concat(l1, l2))

    catted = to_list(res)
    assert len(catted) == 4
    assert count(catted[0]) == 1
    assert count(catted[1]) == 2
    assert count(catted[2]) == 3
    assert count(catted[3]) == 4


@tvm.testing.uses_gpu
def test_filter():
    a = relay.TypeVar("a")
    expected_type = relay.FuncType(
        [relay.FuncType([a], relay.scalar_type("bool")), rlist(a)], rlist(a), [a]
    )
    assert prelude.mod[filter].checked_type == expected_type

    x = relay.Var("x", nat())
    greater_than_one = relay.Function(
        [x],
        relay.Match(
            x,
            [
                relay.Clause(
                    relay.PatternConstructor(
                        s, [relay.PatternConstructor(s, [relay.PatternWildcard()])]
                    ),
                    relay.const(True),
                ),
                relay.Clause(relay.PatternWildcard(), relay.const(False)),
            ],
        ),
    )
    res = eval(
        filter(
            greater_than_one,
            cons(
                make_nat_expr(prelude, 1),
                cons(
                    make_nat_expr(prelude, 1),
                    cons(
                        make_nat_expr(prelude, 3),
                        cons(
                            make_nat_expr(prelude, 1),
                            cons(make_nat_expr(prelude, 5), cons(make_nat_expr(prelude, 1), nil())),
                        ),
                    ),
                ),
            ),
        )
    )
    filtered = to_list(res)
    assert len(filtered) == 2
    assert count(filtered[0]) == 3
    assert count(filtered[1]) == 5


@tvm.testing.uses_gpu
def test_zip():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    expected_type = relay.FuncType([rlist(a), rlist(b)], rlist(relay.TupleType([a, b])), [a, b])
    assert prelude.mod[zip].checked_type == expected_type

    l1 = cons(
        make_nat_expr(prelude, 1),
        cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
    )
    l2 = cons(nil(), cons(cons(nil(), nil()), cons(cons(nil(), cons(nil(), nil())), nil())))

    res = eval(zip(l1, l2))
    zipped = to_list(res)
    assert len(zipped) == 3
    assert count(zipped[0][0]) == 1
    assert len(to_list(zipped[0][1])) == 0
    assert count(zipped[1][0]) == 2
    assert len(to_list(zipped[1][1])) == 1
    assert count(zipped[2][0]) == 3
    assert len(to_list(zipped[2][1])) == 2

    # test truncation
    l3 = cons(make_nat_expr(prelude, 4), cons(make_nat_expr(prelude, 5), nil()))
    shorter_res = eval(zip(l3, l2))
    truncated = to_list(shorter_res)
    assert len(truncated) == 2
    assert count(truncated[0][0]) == 4
    assert len(to_list(truncated[0][1])) == 0
    assert count(truncated[1][0]) == 5
    assert len(to_list(truncated[1][1])) == 1

    l4 = cons(nil(), nil())
    shortest_res = eval(zip(l3, l4))
    singleton = to_list(shortest_res)
    assert len(singleton) == 1
    assert count(singleton[0][0]) == 4
    assert len(to_list(singleton[0][1])) == 0


@tvm.testing.uses_gpu
def test_rev():
    a = relay.TypeVar("a")
    assert prelude.mod[rev].checked_type == relay.FuncType([rlist(a)], rlist(a), [a])

    res = eval(
        rev(
            cons(
                make_nat_expr(prelude, 1),
                cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
            )
        )
    )
    reversed = to_list(res)

    assert len(reversed) == 3
    assert count(reversed[0]) == 3
    assert count(reversed[1]) == 2
    assert count(reversed[2]) == 1


@tvm.testing.uses_gpu
def test_unfoldr():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    expected_type = relay.FuncType(
        [relay.FuncType([a], optional(relay.TupleType([a, b]))), a], rlist(b), [a, b]
    )

    x = relay.Var("x", nat())
    n = relay.Var("n", nat())
    count_down = relay.Function(
        [x],
        relay.Match(
            x,
            [
                relay.Clause(
                    relay.PatternConstructor(s, [relay.PatternVar(n)]), some(relay.Tuple([n, x]))
                ),
                relay.Clause(relay.PatternConstructor(z, []), none()),
            ],
        ),
    )

    res = eval(unfoldr(count_down, make_nat_expr(prelude, 3)))
    unfolded = to_list(res)

    assert len(unfolded) == 3
    assert count(unfolded[0]) == 3
    assert count(unfolded[1]) == 2
    assert count(unfolded[2]) == 1


@tvm.testing.uses_gpu
def test_unfoldl():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    expected_type = relay.FuncType(
        [relay.FuncType([a], optional(relay.TupleType([a, b]))), a], rlist(b), [a, b]
    )

    x = relay.Var("x", nat())
    n = relay.Var("n", nat())
    count_down = relay.Function(
        [x],
        relay.Match(
            x,
            [
                relay.Clause(
                    relay.PatternConstructor(s, [relay.PatternVar(n)]), some(relay.Tuple([n, x]))
                ),
                relay.Clause(relay.PatternConstructor(z, []), none()),
            ],
        ),
    )

    res = eval(unfoldl(count_down, make_nat_expr(prelude, 3)))
    unfolded = to_list(res)

    assert len(unfolded) == 3
    assert count(unfolded[0]) == 1
    assert count(unfolded[1]) == 2
    assert count(unfolded[2]) == 3


@tvm.testing.uses_gpu
def test_map_accumr():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    c = relay.TypeVar("c")
    expected_type = relay.FuncType(
        [relay.FuncType([a, b], relay.TupleType([a, c])), a, rlist(b)],
        relay.TupleType([a, rlist(c)]),
        [a, b, c],
    )
    assert prelude.mod[map_accumr].checked_type == expected_type

    acc = relay.Var("acc", nat())
    x = relay.Var("x", nat())
    add_acc_to_each = relay.Function([acc, x], relay.Tuple([add(x, acc), add(x, acc)]))

    vals = cons(
        make_nat_expr(prelude, 1),
        cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
    )
    res = eval(map_accumr(add_acc_to_each, z(), vals))

    sum = count(res[0])
    new_vals = to_list(res[1])

    assert sum == 6
    assert len(new_vals) == 3
    assert count(new_vals[0]) == 6
    assert count(new_vals[1]) == 5
    assert count(new_vals[2]) == 3


@tvm.testing.uses_gpu
def test_map_accuml():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    c = relay.TypeVar("c")
    expected_type = relay.FuncType(
        [relay.FuncType([a, b], relay.TupleType([a, c])), a, rlist(b)],
        relay.TupleType([a, rlist(c)]),
        [a, b, c],
    )
    assert prelude.mod[map_accuml].checked_type == expected_type

    acc = relay.Var("acc", nat())
    x = relay.Var("x", nat())
    add_to_acc = relay.Function([acc, x], relay.Tuple([add(x, acc), x]))

    vals = cons(
        make_nat_expr(prelude, 1),
        cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
    )
    res = eval(map_accuml(add_to_acc, z(), vals))

    sum = count(res[0])
    new_vals = to_list(res[1])

    assert sum == 6
    assert len(new_vals) == 3
    assert count(new_vals[0]) == 3
    assert count(new_vals[1]) == 2
    assert count(new_vals[2]) == 1


@tvm.testing.uses_gpu
def test_optional_matching():
    x = relay.Var("x")
    y = relay.Var("y")
    v = relay.Var("v")
    condense = relay.Function(
        [x, y],
        relay.Match(
            x,
            [
                relay.Clause(relay.PatternConstructor(some, [relay.PatternVar(v)]), cons(v, y)),
                relay.Clause(relay.PatternConstructor(none), y),
            ],
        ),
    )

    res = eval(
        foldr(
            condense,
            nil(),
            cons(
                some(make_nat_expr(prelude, 3)),
                cons(none(), cons(some(make_nat_expr(prelude, 1)), nil())),
            ),
        )
    )

    reduced = to_list(res)
    assert len(reduced) == 2
    assert count(reduced[0]) == 3
    assert count(reduced[1]) == 1


@tvm.testing.uses_gpu
def test_tmap():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = prelude.mod[tmap].checked_type
    rhs = relay.FuncType([relay.FuncType([a], b), tree(a)], tree(b), [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    add_one = relay.Function([x], s(x))
    res = eval(tmap(add_one, rose(z(), cons(rose(z(), nil()), cons(rose(z(), nil()), nil())))))

    tree_dict = tree_to_dict(res)
    assert count(tree_dict["member"]) == 1
    assert len(tree_dict["children"]) == 2
    for subtree in tree_dict["children"]:
        assert count(subtree["member"]) == 1
        assert len(subtree["children"]) == 0


@tvm.testing.uses_gpu
def test_size():
    a = relay.TypeVar("a")
    lhs = prelude.mod[size].checked_type
    rhs = relay.FuncType([tree(a)], relay.scalar_type("int32"), [a])
    assert lhs == rhs

    root = rose(z(), cons(rose(z(), nil()), cons(rose(z(), nil()), nil())))
    t = rose(z(), cons(root, cons(root, cons(root, nil()))))
    res = eval(size(t))
    assert get_scalar(res) == 10


@tvm.testing.uses_gpu
def test_wildcard_match_solo():
    x = relay.Var("x", nat())
    copy = relay.Function([x], relay.Match(x, [relay.Clause(relay.PatternWildcard(), x)]), nat())

    res = eval(copy(s(s(s(z())))))
    assert count(res) == 3


@tvm.testing.uses_gpu
def test_wildcard_match_order():
    x = relay.Var("x", rlist(nat()))
    y = relay.Var("y")
    a = relay.Var("a")
    return_zero = relay.Function(
        [x],
        relay.Match(
            x,
            [
                relay.Clause(relay.PatternWildcard(), z()),
                relay.Clause(
                    relay.PatternConstructor(cons, [relay.PatternVar(y), relay.PatternVar(a)]), y
                ),
                relay.Clause(relay.PatternConstructor(nil), s(z())),
            ],
        ),
        nat(),
    )

    res = eval(return_zero(cons(s(z()), nil())))
    # wildcard pattern is evaluated first
    assert count(res) == 0


@tvm.testing.uses_gpu
def test_nested_matches():
    a = relay.TypeVar("a")
    # TODO(@jroesch): inference should be able to handle this one
    x = relay.Var("x", type_annotation=rlist(rlist(a)))
    y = relay.Var("y")
    w = relay.Var("w")
    h = relay.Var("h")
    t = relay.Var("t")
    flatten = relay.GlobalVar("flatten")

    # flatten could be written using a fold, but this way has nested matches
    inner_match = relay.Match(
        y,
        [
            relay.Clause(relay.PatternConstructor(nil), flatten(w)),
            relay.Clause(
                relay.PatternConstructor(cons, [relay.PatternVar(h), relay.PatternVar(t)]),
                cons(h, flatten(cons(t, w))),
            ),
        ],
    )

    prelude.mod[flatten] = relay.Function(
        [x],
        relay.Match(
            x,
            [
                relay.Clause(relay.PatternConstructor(nil), nil()),
                relay.Clause(
                    relay.PatternConstructor(cons, [relay.PatternVar(y), relay.PatternVar(w)]),
                    inner_match,
                ),
            ],
        ),
        rlist(a),
        [a],
    )

    first_list = cons(
        make_nat_expr(prelude, 1),
        cons(make_nat_expr(prelude, 2), cons(make_nat_expr(prelude, 3), nil())),
    )
    second_list = cons(
        make_nat_expr(prelude, 4),
        cons(make_nat_expr(prelude, 5), cons(make_nat_expr(prelude, 6), nil())),
    )
    final_list = cons(first_list, cons(second_list, nil()))

    res = eval(flatten(final_list))

    flat = to_list(res)
    assert len(flat) == 6
    for i in range(6):
        assert count(flat[i]) == i + 1


@tvm.testing.uses_gpu
def test_match_full_var():
    x = relay.Var("x")
    v = relay.Var("v")
    id_func = relay.Function([x], relay.Match(x, [relay.Clause(relay.PatternVar(v), v)]))

    res1 = eval(id_func(nil()))
    res2 = eval(id_func(cons(z(), cons(z(), nil()))))

    empty = to_list(res1)
    assert len(empty) == 0

    zeroes = to_list(res2)
    assert len(zeroes) == 2
    assert count(zeroes[0]) == 0
    assert count(zeroes[1]) == 0


@tvm.testing.uses_gpu
def test_nested_pattern_match():
    x = relay.Var("x", rlist(nat()))
    h1 = relay.Var("h1")
    h2 = relay.Var("h2")
    t = relay.Var("t")
    match = relay.Match(
        x,
        [
            relay.Clause(
                relay.PatternConstructor(
                    cons,
                    [
                        relay.PatternVar(h1),
                        relay.PatternConstructor(cons, [relay.PatternVar(h2), relay.PatternVar(t)]),
                    ],
                ),
                h2,
            ),
            relay.Clause(relay.PatternWildcard(), z()),
        ],
    )
    get_second = relay.Function([x], match)

    res = eval(get_second(cons(s(z()), cons(s(s(z())), nil()))))

    assert count(res) == 2


@tvm.testing.uses_gpu
def test_compose():
    n = relay.Var("n")
    inc = relay.Function([n], s(n))
    x = relay.Var("x")
    res = eval(relay.Call(compose(inc, double), [s(s(z()))]))
    assert count(res) == 5


@tvm.testing.uses_gpu
def test_iterate():
    expr = relay.Call(iterate(double, relay.const(2)), [make_nat_expr(prelude, 3)])
    res = eval(relay.Function([], expr)())
    assert count(res) == 12


if __name__ == "__main__":
    tvm.testing.main()
