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
from tvm.relay.testing import add_nat_definitions, count as count_, make_nat_value, make_nat_expr

import numpy as np

mod = tvm.IRModule()
p = Prelude(mod)
add_nat_definitions(p)


def count(e):
    return count_(p, e)


ctx = tvm.context("llvm", 0)
intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

z = p.z
s = p.s
nat = p.nat
double = p.double
add = p.add

optional = p.optional
some = p.some
none = p.none

nil = p.nil
cons = p.cons
l = p.l
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

tree = p.tree
rose = p.rose
tmap = p.tmap
size = p.size

compose = p.compose
iterate = p.iterate

# this is an example of creating the adt value in python side
def make_nat(n):
    if n != 0:
        return ConstructorValue(s, [make_nat(n - 1)])
    else:
        return ConstructorValue(z, [])


def make_nat_expr(n):
    assert n >= 0
    ret = z()
    while n > 0:
        ret = s(ret)
        n = n - 1
    return ret


def to_list(l):
    assert isinstance(l, ConstructorValue)
    val = l
    ret = []
    while True:
        if val.tag == p.cons.tag:
            ret.append(val.fields[0])
            val = val.fields[1]
        else:
            assert val.tag == p.nil.tag
            break
    return ret


def tree_to_dict(t):
    assert isinstance(t, ConstructorValue)
    ret = {}
    assert t.tag == p.rose.tag
    ret["member"] = t.fields[0]
    ret["children"] = []
    for subtree in to_list(t.fields[1]):
        l = tree_to_dict(subtree)
        ret["children"].append(l)
    return ret


def vmobj_to_list(o, dtype="float32"):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy().tolist()]
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
            return [o.fields[0].asnumpy()]
        else:
            raise RuntimeError("Unknown object type: %s" % o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


# turns a scalar-valued relay tensor value into a python number
def get_scalar(tv):
    return tv.asnumpy().item()


@tvm.testing.uses_gpu
def test_nat_value():
    assert count(make_nat_value(p, 10)) == 10
    assert count(intrp.evaluate(s(s(z())))) == 2


@tvm.testing.uses_gpu
def test_nat_constructor():
    func = relay.Function([], z())
    test_z = relay.GlobalVar("test_z")
    mod[test_z] = func
    assert mod[test_z].body.checked_type == nat()
    test_sz = relay.GlobalVar("test_sz")
    func = relay.Function([], s(z()))
    mod[test_sz] = func
    assert mod[test_sz].body.checked_type == nat()


@tvm.testing.uses_gpu
def test_double():
    assert mod[double].checked_type == relay.FuncType([nat()], nat())
    res = intrp.evaluate(double(s(z())))
    assert count(res) == 2


@tvm.testing.uses_gpu
def test_add():
    assert mod[add].checked_type == relay.FuncType([nat(), nat()], nat())
    res = intrp.evaluate(add(s(z()), s(z())))
    assert count(res) == 2


@tvm.testing.uses_gpu
def test_list_constructor():
    test_consz = relay.GlobalVar("test_consz")
    func = relay.Function([], cons(z(), nil()))
    mod[test_consz] = func
    assert mod[test_consz].body.checked_type == l(nat())


@tvm.testing.uses_gpu
def test_hd_tl():
    expected = list(range(10))
    l = nil()
    for i in reversed(expected):
        l = cons(make_nat_expr(i), l)

    got = []
    for i in range(len(expected)):
        got.append(count(intrp.evaluate(hd(l))))
        l = tl(l)

    assert got == expected


@tvm.testing.uses_gpu
def test_nth():
    expected = list(range(10))
    l = nil()
    for i in reversed(expected):
        l = cons(relay.const(i), l)

    for i in range(len(expected)):
        item = intrp.evaluate(nth(l, relay.const(i)))
        assert get_scalar(item) == i


@tvm.testing.uses_gpu
def test_update():
    expected = list(range(10))
    l = nil()
    # create zero initialized list
    for i in range(len(expected)):
        l = cons(make_nat_expr(0), l)

    # set value
    for i, v in enumerate(expected):
        l = update(l, relay.const(i), make_nat_expr(v))

    got = []
    for i in range(len(expected)):
        got.append(count(intrp.evaluate(nth(l, relay.const(i)))))

    assert got == expected


@tvm.testing.uses_gpu
def test_length():
    a = relay.TypeVar("a")
    assert mod[length].checked_type == relay.FuncType([l(a)], relay.scalar_type("int32"), [a])
    res = intrp.evaluate(length(cons(z(), cons(z(), cons(z(), nil())))))
    assert get_scalar(res) == 3


@tvm.testing.uses_gpu
def test_map():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = mod[map].checked_type
    rhs = relay.FuncType([relay.FuncType([a], b), l(a)], l(b), [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    add_one = relay.Function([x], s(x))
    res = intrp.evaluate(map(add_one, cons(z(), cons(z(), nil()))))
    ones = to_list(res)
    assert len(ones) == 2
    assert count(ones[0]) == 1 and count(ones[1]) == 1


@tvm.testing.uses_gpu
def test_foldl():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = mod[foldl].checked_type
    rhs = relay.FuncType([relay.FuncType([a, b], a), a, l(b)], a, [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    rev_dup = relay.Function([y, x], cons(x, cons(x, y)))
    res = intrp.evaluate(
        foldl(
            rev_dup,
            nil(),
            cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil()))),
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
    lhs = mod[foldr].checked_type
    rhs = relay.FuncType([relay.FuncType([a, b], b), b, l(a)], b, [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    identity = relay.Function([x, y], cons(x, y))
    res = intrp.evaluate(
        foldr(
            identity,
            nil(),
            cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil()))),
        )
    )
    same = to_list(res)
    assert len(same) == 3
    assert count(same[0]) == 1 and count(same[1]) == 2 and count(same[2]) == 3


@tvm.testing.uses_gpu
def test_foldr1():
    a = relay.TypeVar("a")
    lhs = mod[p.foldr1].checked_type
    rhs = relay.FuncType([relay.FuncType([a, a], a), l(a)], a, [a])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    f = relay.Function([x, y], add(x, y))
    res = intrp.evaluate(
        foldr1(f, cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil()))))
    )

    assert count(res) == 6


@tvm.testing.uses_gpu
def test_sum():
    assert mod[sum].checked_type == relay.FuncType(
        [l(relay.scalar_type("int32"))], relay.scalar_type("int32")
    )
    res = intrp.evaluate(sum(cons(relay.const(1), cons(relay.const(2), nil()))))
    assert get_scalar(res) == 3


@tvm.testing.uses_gpu
def test_concat():
    a = relay.TypeVar("a")
    assert mod[concat].checked_type == relay.FuncType([l(a), l(a)], l(a), [a])

    l1 = cons(make_nat_expr(1), cons(make_nat_expr(2), nil()))
    l2 = cons(make_nat_expr(3), cons(make_nat_expr(4), nil()))
    res = intrp.evaluate(concat(l1, l2))

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
        [relay.FuncType([a], relay.scalar_type("bool")), l(a)], l(a), [a]
    )
    assert mod[filter].checked_type == expected_type

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
    res = intrp.evaluate(
        filter(
            greater_than_one,
            cons(
                make_nat_expr(1),
                cons(
                    make_nat_expr(1),
                    cons(
                        make_nat_expr(3),
                        cons(
                            make_nat_expr(1), cons(make_nat_expr(5), cons(make_nat_expr(1), nil()))
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
    expected_type = relay.FuncType([l(a), l(b)], l(relay.TupleType([a, b])), [a, b])
    assert mod[zip].checked_type == expected_type

    l1 = cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil())))
    l2 = cons(nil(), cons(cons(nil(), nil()), cons(cons(nil(), cons(nil(), nil())), nil())))

    res = intrp.evaluate(zip(l1, l2))
    zipped = to_list(res)
    assert len(zipped) == 3
    assert count(zipped[0][0]) == 1
    assert len(to_list(zipped[0][1])) == 0
    assert count(zipped[1][0]) == 2
    assert len(to_list(zipped[1][1])) == 1
    assert count(zipped[2][0]) == 3
    assert len(to_list(zipped[2][1])) == 2

    # test truncation
    l3 = cons(make_nat_expr(4), cons(make_nat_expr(5), nil()))
    shorter_res = intrp.evaluate(zip(l3, l2))
    truncated = to_list(shorter_res)
    assert len(truncated) == 2
    assert count(truncated[0][0]) == 4
    assert len(to_list(truncated[0][1])) == 0
    assert count(truncated[1][0]) == 5
    assert len(to_list(truncated[1][1])) == 1

    l4 = cons(nil(), nil())
    shortest_res = intrp.evaluate(zip(l3, l4))
    singleton = to_list(shortest_res)
    assert len(singleton) == 1
    assert count(singleton[0][0]) == 4
    assert len(to_list(singleton[0][1])) == 0


@tvm.testing.uses_gpu
def test_rev():
    a = relay.TypeVar("a")
    assert mod[rev].checked_type == relay.FuncType([l(a)], l(a), [a])

    res = intrp.evaluate(
        rev(cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil()))))
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
        [relay.FuncType([a], optional(relay.TupleType([a, b]))), a], l(b), [a, b]
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

    res = intrp.evaluate(unfoldr(count_down, make_nat_expr(3)))
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
        [relay.FuncType([a], optional(relay.TupleType([a, b]))), a], l(b), [a, b]
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

    res = intrp.evaluate(unfoldl(count_down, make_nat_expr(3)))
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
        [relay.FuncType([a, b], relay.TupleType([a, c])), a, l(b)],
        relay.TupleType([a, l(c)]),
        [a, b, c],
    )
    assert mod[map_accumr].checked_type == expected_type

    acc = relay.Var("acc", nat())
    x = relay.Var("x", nat())
    add_acc_to_each = relay.Function([acc, x], relay.Tuple([add(x, acc), add(x, acc)]))

    vals = cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil())))
    res = intrp.evaluate(map_accumr(add_acc_to_each, z(), vals))

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
        [relay.FuncType([a, b], relay.TupleType([a, c])), a, l(b)],
        relay.TupleType([a, l(c)]),
        [a, b, c],
    )
    assert mod[map_accuml].checked_type == expected_type

    acc = relay.Var("acc", nat())
    x = relay.Var("x", nat())
    add_to_acc = relay.Function([acc, x], relay.Tuple([add(x, acc), x]))

    vals = cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil())))
    res = intrp.evaluate(map_accuml(add_to_acc, z(), vals))

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

    res = intrp.evaluate(
        foldr(
            condense,
            nil(),
            cons(some(make_nat_expr(3)), cons(none(), cons(some(make_nat_expr(1)), nil()))),
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
    lhs = mod[tmap].checked_type
    rhs = relay.FuncType([relay.FuncType([a], b), tree(a)], tree(b), [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    add_one = relay.Function([x], s(x))
    res = intrp.evaluate(
        tmap(add_one, rose(z(), cons(rose(z(), nil()), cons(rose(z(), nil()), nil()))))
    )

    tree_dict = tree_to_dict(res)
    assert count(tree_dict["member"]) == 1
    assert len(tree_dict["children"]) == 2
    for subtree in tree_dict["children"]:
        assert count(subtree["member"]) == 1
        assert len(subtree["children"]) == 0


@tvm.testing.uses_gpu
def test_size():
    a = relay.TypeVar("a")
    lhs = mod[size].checked_type
    rhs = relay.FuncType([tree(a)], relay.scalar_type("int32"), [a])
    assert lhs == rhs

    root = rose(z(), cons(rose(z(), nil()), cons(rose(z(), nil()), nil())))
    t = rose(z(), cons(root, cons(root, cons(root, nil()))))
    res = intrp.evaluate(size(t))
    assert get_scalar(res) == 10


@tvm.testing.uses_gpu
def test_wildcard_match_solo():
    x = relay.Var("x", nat())
    copy = relay.Function([x], relay.Match(x, [relay.Clause(relay.PatternWildcard(), x)]), nat())

    res = intrp.evaluate(copy(s(s(s(z())))))
    assert count(res) == 3


@tvm.testing.uses_gpu
def test_wildcard_match_order():
    x = relay.Var("x", l(nat()))
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

    res = intrp.evaluate(return_zero(cons(s(z()), nil())))
    # wildcard pattern is evaluated first
    assert count(res) == 0


@tvm.testing.uses_gpu
def test_nested_matches():
    a = relay.TypeVar("a")
    x = relay.Var("x")
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

    mod[flatten] = relay.Function(
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
        l(a),
        [a],
    )

    first_list = cons(make_nat_expr(1), cons(make_nat_expr(2), cons(make_nat_expr(3), nil())))
    second_list = cons(make_nat_expr(4), cons(make_nat_expr(5), cons(make_nat_expr(6), nil())))
    final_list = cons(first_list, cons(second_list, nil()))

    res = intrp.evaluate(flatten(final_list))

    flat = to_list(res)
    assert len(flat) == 6
    for i in range(6):
        assert count(flat[i]) == i + 1


@tvm.testing.uses_gpu
def test_match_full_var():
    x = relay.Var("x")
    v = relay.Var("v")
    id_func = relay.Function([x], relay.Match(x, [relay.Clause(relay.PatternVar(v), v)]))

    res1 = intrp.evaluate(id_func(nil()))
    res2 = intrp.evaluate(id_func(cons(z(), cons(z(), nil()))))

    empty = to_list(res1)
    assert len(empty) == 0

    zeroes = to_list(res2)
    assert len(zeroes) == 2
    assert count(zeroes[0]) == 0
    assert count(zeroes[1]) == 0


@tvm.testing.uses_gpu
def test_nested_pattern_match():
    x = relay.Var("x", l(nat()))
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

    res = intrp.evaluate(get_second(cons(s(z()), cons(s(s(z())), nil()))))

    assert count(res) == 2


@tvm.testing.uses_gpu
def test_compose():
    n = relay.Var("n")
    inc = relay.Function([n], s(n))
    x = relay.Var("x")
    res = intrp.evaluate(relay.Call(compose(inc, double), [s(s(z()))]))
    assert count(res) == 5


@tvm.testing.uses_gpu
def test_iterate():
    expr = relay.Call(iterate(double, relay.const(2)), [make_nat_expr(3)])
    res = intrp.evaluate(relay.Function([], expr)())
    assert count(res) == 12


def check_tensor_array(ta_mod, ref_res, *args, dtype="float32", rtol=1e-5):
    for kind in ["debug", "vm"]:
        for target, ctx in testing.enabled_targets():
            if kind == "debug" and ctx.device_type != tvm.cpu().device_type:
                continue
            ex = relay.create_executor(kind, mod=ta_mod, ctx=ctx, target=target)
            result = ex.evaluate()(*args)
            got = vmobj_to_list(result, dtype)
            tvm.testing.assert_allclose(ref_res, got, rtol=rtol, atol=rtol)


@tvm.testing.uses_gpu
def test_tensor_expand_dims():
    def run(dtype):
        x = relay.var("x")
        mod = tvm.IRModule()
        p = Prelude(mod)
        expand_dims_func = p.get_var("tensor_expand_dims", dtype)
        tensor1 = p.get_var("tensor1", dtype)
        mod["main"] = relay.Function([x], expand_dims_func(tensor1(x)))
        x_np = np.random.uniform(low=0.0, high=8.0, size=(1,)).astype(dtype)
        expected = [np.expand_dims(x_np, axis=0)]
        check_tensor_array(mod, expected, x_np)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_constructor():
    def run(dtype):
        x = relay.var("x")
        mod = tvm.IRModule()
        p = Prelude(mod)
        tensor_array = p.get_var("tensor_array", dtype)
        mod["main"] = relay.Function([x], tensor_array(x))
        expected = np.array([0, 0, 0, 0, 0])
        check_tensor_array(mod, expected, 5, dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_read():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        l = relay.var("l")
        i = relay.var("i")
        read_func = p.get_var("tensor_array_read", dtype)
        tensor_array = p.get_var("tensor_array", dtype)
        mod["main"] = relay.Function([l, i], read_func(tensor_array(l), i))
        expected = [0]
        check_tensor_array(mod, expected, *(1, 0), dtype=dtype)
        check_tensor_array(mod, expected, *(5, 1), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_write():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        tensor_array = p.get_var("tensor_array", dtype)
        init_tensor_array = tensor_array(relay.const(2))
        write_func = p.get_var("tensor_array_write", dtype)
        tensor1 = p.get_var("tensor1", dtype)
        tensor_array1 = write_func(init_tensor_array, relay.const(0), tensor1(v1))
        tensor_array2 = write_func(tensor_array1, relay.const(1), tensor1(v2))
        mod["main"] = relay.Function([v1, v2], tensor_array2)
        expected = [3, 7]
        check_tensor_array(mod, expected, *(3, 7), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_stack():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        tensor_array = p.get_var("tensor_array", dtype)
        tensor1 = p.get_var("tensor1", dtype)
        write = p.get_var("tensor_array_write", dtype)
        stack = p.get_var("tensor_array_stack", dtype)
        v = relay.var("v")
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor1(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor1(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor1(v))
        tensor_array4 = stack(tensor_array3)
        mod["main"] = relay.Function([v], tensor_array4)
        t = np.random.uniform(low=0.0, high=8.0, size=(1,)).astype(dtype)
        expected = [np.stack([t, t, t])]
        check_tensor_array(mod, expected, t, dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_unstack():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        unstack_tensor1 = p.get_var("tensor_array_unstack_tensor1", dtype)
        v = relay.var("v")
        mod["main"] = relay.Function([v], unstack_tensor1(v))
        t = np.random.uniform(low=0.0, high=8.0, size=(1,)).astype(dtype)
        check_tensor_array(mod, t, t, dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_take():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        take = p.get_var("tensor_take", dtype)
        tensor2 = p.get_var("tensor2", dtype)
        v = relay.var("v")
        lower = relay.var("lower")
        upper = relay.var("upper")
        mod["main"] = relay.Function([v, lower, upper], take(tensor2(v), lower, upper))
        v_data = np.random.uniform(low=0.0, high=8.0, size=(10, 10)).astype(dtype)
        expected = [np.take(v_data, range(2, 5), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 2, 5), dtype=dtype)
        expected = [np.take(v_data, range(0, 9), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 0, 9), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_concatenate():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        concat = p.get_var("tensor_concatenate", dtype)
        tensor1 = p.get_var("tensor1", dtype)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        mod["main"] = relay.Function([v1, v2], concat(tensor1(v1), tensor1(v2)))
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(5,)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(5,)).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data))]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_concat():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        tensor_array = p.get_var("tensor_array", dtype)
        tensor_array1 = tensor_array(relay.const(2))
        write_func = p.get_var("tensor_array_write", dtype)
        concat_func = p.get_var("tensor_array_concat", dtype)
        tensor1 = p.get_var("tensor2", dtype)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor1(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor1(v2))
        tensor_array_concat = concat_func(tensor_array1)
        mod["main"] = relay.Function([v1, v2], tensor_array_concat)
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(1, 3)).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data), axis=0)]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_scatter():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")
        tensor_array = p.get_var("tensor_array", dtype)
        tensor_array1 = tensor_array(relay.const(3))
        write_func = p.get_var("tensor_array_write", dtype)
        scatter_func = p.get_var("tensor_array_scatter", dtype)
        tensor2 = p.get_var("tensor2", dtype)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor2(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor2(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor2(v3))

        # indices array
        index = relay.var("index")

        # values array
        value_0 = relay.var("value_0")
        value_1 = relay.var("value_1")
        values_array = tensor_array(relay.const(2))
        values_array = write_func(values_array, relay.const(0), tensor2(value_0))
        values_array = write_func(values_array, relay.const(1), tensor2(value_1))

        # create the scatter function
        tensor_array_scatter = scatter_func(tensor_array1, index, values_array)
        mod["main"] = relay.Function([v1, v2, v3, index, value_0, value_1], tensor_array_scatter)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        index_data = np.array([0, 1], dtype="int32")
        val1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        val2_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        expected = [val1_data, val2_data, v3_data]
        check_tensor_array(
            mod,
            expected,
            *(v1_data, v2_data, v3_data, index_data, val1_data, val2_data),
            dtype=dtype,
        )

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_tensor_array_split():
    def run(dtype):
        mod = tvm.IRModule()
        p = Prelude(mod)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")
        tensor_array = p.get_var("tensor_array", dtype)
        tensor_array1 = tensor_array(relay.const(3))
        write_func = p.get_var("tensor_array_write", dtype)
        split_func = p.get_var("tensor_array_split", dtype)
        tensor2 = p.get_var("tensor2", dtype)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor2(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor2(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor2(v3))

        # value tensor
        value = relay.var("value")

        # lengths tensor
        ta_len = relay.var("length")

        # create the scatter function
        tensor_array_split = split_func(tensor_array1, tensor2(value), ta_len)
        mod["main"] = relay.Function([v1, v2, v3, value, ta_len], tensor_array_split)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        value_data = np.random.uniform(low=0.0, high=8.0, size=(4, 3)).astype(dtype)
        length_data = np.array([2, 2], dtype="int32")
        expected = np.concatenate([value_data, v3_data])
        expected = np.split(expected, indices_or_sections=[2, 4])
        check_tensor_array(
            mod, expected, *(v1_data, v2_data, v3_data, value_data, length_data), dtype=dtype
        )

    run("float32")
    run("int32")


@tvm.testing.uses_gpu
def test_static_tensor_take():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        take = p.get_var_static("tensor_take", dtype, shape)
        tensor_constructor = p.get_var_static("tensor_constructor", dtype, shape)
        v = relay.var("v")
        lower = relay.var("lower")
        upper = relay.var("upper")
        mod["main"] = relay.Function([v, lower, upper], take(tensor_constructor(v), lower, upper))
        v_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.take(v_data, range(2, 5), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 2, 5), dtype=dtype)
        expected = [np.take(v_data, range(0, 9), axis=0)]
        check_tensor_array(mod, expected, *(v_data, 0, 9), dtype=dtype)

    run("float32", [10, 10])
    run("int32", [15, 11])


@tvm.testing.uses_gpu
def test_static_tensor_concatenate():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        concat = p.get_var_static("tensor_concatenate", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        mod["main"] = relay.Function([v1, v2], concat(tensor(v1), tensor(v2)))
        v1_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data))]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run(
        "float32",
        [
            5,
        ],
    )
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_expand_dims():
    def run(dtype, shape):
        x = relay.var("x")
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        expand_dims_func = p.get_var_static("tensor_expand_dims", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        mod["main"] = relay.Function([x], expand_dims_func(tensor(x)))
        x_np = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.expand_dims(x_np, axis=0)]
        check_tensor_array(mod, expected, x_np)

    run("float32", [])
    run(
        "int32",
        [
            2,
        ],
    )


@tvm.testing.uses_gpu
def test_static_tensor_array_constructor():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()
        tensor_constructor = p.get_name_static("tensor_constructor", dtype, shape)
        assert tensor_constructor != None

    run("float32", [1, 1])


@tvm.testing.uses_gpu
def test_static_tensor_array_read():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        np_data_list = []
        ta_length = 3
        for _ in range(ta_length):
            np_data_list.append(np.random.uniform(0, 10, size=shape).astype(dtype))

        v0 = relay.var("v0")
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        n = relay.var("n")
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        init_tensor_array = tensor_array(relay.const(ta_length))
        read_func = p.get_var_static("tensor_array_read", dtype, shape)
        write_func = p.get_var_static("tensor_array_write", dtype, shape)
        tensor_array0 = write_func(init_tensor_array, relay.const(0), tensor(v0))
        tensor_array1 = write_func(tensor_array0, relay.const(1), tensor(v1))
        tensor_array2 = write_func(tensor_array1, relay.const(2), tensor(v2))

        mod["main"] = relay.Function([v0, v1, v2, n], read_func(tensor_array2, n))
        expected = [np_data_list[0]]
        check_tensor_array(mod, expected, *list(np_data_list + [0]), dtype=dtype)
        expected = [np_data_list[1]]
        check_tensor_array(mod, expected, *list(np_data_list + [1]), dtype=dtype)
        expected = [np_data_list[2]]
        check_tensor_array(mod, expected, *list(np_data_list + [2]), dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_write():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        ta_length = 2
        np_data_list = [
            np.random.uniform(0, 10, size=shape).astype(dtype) for _ in range(ta_length)
        ]

        v0 = relay.var("v0")
        v1 = relay.var("v1")
        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        init_tensor_array = tensor_array(relay.const(ta_length))
        write_func = p.get_var_static("tensor_array_write", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        tensor_array0 = write_func(init_tensor_array, relay.const(0), tensor(v0))
        tensor_array1 = write_func(tensor_array0, relay.const(1), tensor(v1))
        mod["main"] = relay.Function([v0, v1], tensor_array1)
        expected = np_data_list
        check_tensor_array(mod, expected, *np_data_list, dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_unstack():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        unstack_tensor = p.get_var_static("tensor_array_unstack", dtype, shape)
        v = relay.var("v")
        mod["main"] = relay.Function([v], unstack_tensor(v))
        t = np.random.uniform(low=0, high=10, size=shape).astype(dtype)
        (*expected,) = t
        check_tensor_array(mod, expected, t, dtype=dtype)

    run("float32", [4])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_scatter():
    def run(dtype, shape, indices_shape=None):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()
        if indices_shape is not None:
            static_tensor_array_ops.define_tensor_array_scatter(indices_shape, True)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")
        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        tensor_array0 = tensor_array(relay.const(3))
        write_func = p.get_var_static("tensor_array_write", dtype, shape)
        scatter_func = p.get_var_static("tensor_array_scatter", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        tensor_array1 = write_func(tensor_array0, relay.const(0), tensor(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor(v3))

        # indices array
        index = relay.var("index")

        # values array
        value_0 = relay.var("value_0")
        value_1 = relay.var("value_1")
        values_array = tensor_array(relay.const(2))
        values_array = write_func(values_array, relay.const(0), tensor(value_0))
        values_array = write_func(values_array, relay.const(1), tensor(value_1))

        # create the scatter function
        tensor_array_scatter = scatter_func(tensor_array1, index, values_array)
        mod["main"] = relay.Function([v1, v2, v3, index, value_0, value_1], tensor_array_scatter)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        index_data = np.array([0, 1], dtype="int32")
        val1_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        val2_data = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [val1_data, val2_data, v3_data]
        check_tensor_array(
            mod,
            expected,
            *(v1_data, v2_data, v3_data, index_data, val1_data, val2_data),
            dtype=dtype,
        )

    run("float32", [2, 3])
    run("int32", [2, 3])
    run(
        "float32",
        [2, 3],
        [
            2,
        ],
    )


@tvm.testing.uses_gpu
def test_static_tensor_array_split():
    def run(dtype, shape, value_shape=None, lengths_shape=None):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()
        if value_shape is not None or lengths_shape is not None:
            static_tensor_array_ops.define_tensor_array_split(value_shape, lengths_shape, True)

        # tensor array
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        v3 = relay.var("v2")

        adt_shape = [
            relay.Any(),
        ] + shape[1:]
        origin_shape = static_tensor_array_ops.shape
        static_tensor_array_ops.shape = adt_shape
        static_tensor_array_ops.define_tensor_array()
        tensor_array = p.get_var_static("tensor_array", dtype, adt_shape)
        static_tensor_array_ops.shape = origin_shape
        tensor_array1 = tensor_array(relay.const(3))
        write_func = p.get_var_static("tensor_array_write", dtype, adt_shape)
        split_func = p.get_var_static("tensor_array_split", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, adt_shape)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor(v2))
        tensor_array1 = write_func(tensor_array1, relay.const(2), tensor(v3))

        # value tensor
        value = relay.var("value")

        # lengths tensor
        ta_len = relay.var("length")

        # create the split function
        if value_shape is None:
            tensor1 = p.get_var_static("tensor_constructor", dtype, shape)
        else:
            static_tensor_array_ops = StaticTensorArrayOps(p, dtype, value_shape)
            static_tensor_array_ops.register()
            tensor1 = p.get_var_static("tensor_constructor", dtype, value_shape)
        tensor_array_split = split_func(tensor_array1, tensor1(value), ta_len)
        mod["main"] = relay.Function([v1, v2, v3, value, ta_len], tensor_array_split)

        # initialize and check
        v1_data = np.random.uniform(low=0.0, high=8.0, size=[2, 3]).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=[2, 3]).astype(dtype)
        v3_data = np.random.uniform(low=0.0, high=8.0, size=[2, 3]).astype(dtype)
        value_data = np.random.uniform(low=0.0, high=8.0, size=value_shape or shape).astype(dtype)
        length_data = np.array([2, 2], dtype="int32")
        expected = np.concatenate([value_data, v3_data])
        expected = np.split(expected, indices_or_sections=[2, 4])
        check_tensor_array(
            mod, expected, *(v1_data, v2_data, v3_data, value_data, length_data), dtype=dtype
        )

    run("float32", [4, 3])
    run("int32", [4, 3])
    run(
        "int32",
        [relay.Any(), 3],
        [4, 3],
        [
            2,
        ],
    )


@tvm.testing.uses_gpu
def test_static_tensor_array_concat():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        v1 = relay.var("v1")
        v2 = relay.var("v2")
        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        tensor_array1 = tensor_array(relay.const(2))
        write_func = p.get_var_static("tensor_array_write", dtype, shape)
        concat_func = p.get_var_static("tensor_array_concat", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        tensor_array1 = write_func(tensor_array1, relay.const(0), tensor(v1))
        tensor_array1 = write_func(tensor_array1, relay.const(1), tensor(v2))
        tensor_array_concat = concat_func(tensor_array1)
        mod["main"] = relay.Function([v1, v2], tensor_array_concat)
        v1_data = np.random.uniform(low=0.0, high=8.0, size=(2, 3)).astype(dtype)
        v2_data = np.random.uniform(low=0.0, high=8.0, size=(1, 3)).astype(dtype)
        expected = [np.concatenate((v1_data, v2_data), axis=0)]
        check_tensor_array(mod, expected, *(v1_data, v2_data), dtype=dtype)

    run("float32", [relay.Any(), 3])
    run("int32", [relay.Any(), 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_gather():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        write = p.get_var_static("tensor_array_write", dtype, shape)
        gather = p.get_var_static("tensor_array_gather", dtype, shape)
        v = relay.var("v")
        indice = relay.var("indice")
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor(v))
        out = gather(tensor_array3, indice)
        mod["main"] = relay.Function([v, indice], out)
        t = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        indice_data = np.array([0, 2], dtype="int32")
        expected = [np.stack([t, t])]
        check_tensor_array(mod, expected, *(t, indice_data), dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_array_stack():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        write = p.get_var_static("tensor_array_write", dtype, shape)
        stack = p.get_var_static("tensor_array_stack", dtype, shape)
        v = relay.var("v")
        init_tensor_array = tensor_array(relay.const(3))
        tensor_array1 = write(init_tensor_array, relay.const(0), tensor(v))
        tensor_array2 = write(tensor_array1, relay.const(1), tensor(v))
        tensor_array3 = write(tensor_array2, relay.const(2), tensor(v))
        tensor_array4 = stack(tensor_array3)
        mod["main"] = relay.Function([v], tensor_array4)
        t = np.random.uniform(low=0.0, high=8.0, size=shape).astype(dtype)
        expected = [np.stack([t, t, t])]
        check_tensor_array(mod, expected, t, dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


@tvm.testing.uses_gpu
def test_static_tensor_get_data():
    def run(dtype, shape):
        mod = tvm.IRModule()
        p = Prelude(mod)
        static_tensor_array_ops = StaticTensorArrayOps(p, dtype, shape)
        static_tensor_array_ops.register()

        np_data_list = []
        ta_length = 3
        for _ in range(ta_length):
            np_data_list.append(np.random.uniform(0, 10, size=shape).astype(dtype))

        v0 = relay.var("v0")
        v1 = relay.var("v1")
        v2 = relay.var("v2")
        n = relay.var("n")
        tensor = p.get_var_static("tensor_constructor", dtype, shape)
        tensor_array = p.get_var_static("tensor_array", dtype, shape)
        init_tensor_array = tensor_array(relay.const(ta_length))
        read_func = p.get_var_static("tensor_array_read", dtype, shape)
        write_func = p.get_var_static("tensor_array_write", dtype, shape)
        get_data_func = p.get_var_static("tensor_get_data", dtype, shape)
        tensor_array0 = write_func(init_tensor_array, relay.const(0), tensor(v0))
        tensor_array1 = write_func(tensor_array0, relay.const(1), tensor(v1))
        tensor_array2 = write_func(tensor_array1, relay.const(2), tensor(v2))

        mod["main"] = relay.Function([v0, v1, v2, n], get_data_func(read_func(tensor_array2, n)))
        expected = [np_data_list[0]]
        check_tensor_array(mod, expected, *list(np_data_list + [0]), dtype=dtype)
        expected = [np_data_list[1]]
        check_tensor_array(mod, expected, *list(np_data_list + [1]), dtype=dtype)
        expected = [np_data_list[2]]
        check_tensor_array(mod, expected, *list(np_data_list + [2]), dtype=dtype)

    run("float32", [])
    run("int32", [2, 3])


if __name__ == "__main__":
    pytest.main([__file__])
