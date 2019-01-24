import tvm
from tvm import relay
from tvm.relay.ir_pass import infer_type
from tvm.relay.backend.interpreter import Value, TupleValue, ConValue
from tvm.relay import testing, create_executor
from tvm.relay.prelude import Prelude

mod = relay.Module()
p = Prelude(mod)
ctx = tvm.context("llvm", 0)
intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

z = p.z
s = p.s
nat = p.nat
double = p.double
add = p.add

nil = p.nil
cons = p.cons
l = p.l
length = p.length
map = p.map
foldl = p.foldl
foldr = p.foldr
sum = p.sum

tree = p.tree
rose = p.rose
tmap = p.tmap
size = p.size

# this is an example of using the adt value in python side
def count(n):
    assert isinstance(n, ConValue)
    if n.con.name_hint == 's':
        return 1 + count(n.fields[0])
    else:
        assert n.con.name_hint == 'z'
        return 0

# this is an example of creating the adt value in python side
def make_nat(n):
    if n != 0:
        return ConValue(s, [make_nat(n - 1)], [])
    else:
        return ConValue(z, [], [])

def build_nat(n):
    assert n >= 0
    ret = z()
    while n > 0:
        ret = s(ret)
        n = n - 1
    return ret

def to_list(l):
    assert isinstance(l, ConValue)
    val = l
    ret = []
    while True:
        if val.con.name_hint == 'cons':
            ret.append(val.fields[0])
            val = val.fields[1]
        else:
            assert val.con.name_hint == 'nil'
            break
    return ret

def tree_to_dict(t):
    assert isinstance(t, ConValue)
    ret = {}
    assert t.con.name_hint == 'rose'
    ret['member'] = t.fields[0]
    ret['children'] = []
    for subtree in to_list(t.fields[1]):
        l = tree_to_dict(subtree)
        ret['children'].append(l)
    return ret

def test_nat_value():
    assert count(make_nat(10)) == 10


def test_nat_constructor():
    assert relay.ir_pass.infer_type(z(), mod).checked_type == nat()
    assert relay.ir_pass.infer_type(s, mod).checked_type == relay.FuncType([nat()], nat())
    assert relay.ir_pass.infer_type(s(z()), mod).checked_type == nat()


def test_double():
    assert mod[double].checked_type == relay.FuncType([nat()], nat())
    res = intrp.evaluate(double(s(z())))
    assert count(res) == 2


def test_add():
    assert mod[add].checked_type == relay.FuncType([nat(), nat()], nat())
    res = intrp.evaluate(add(s(z()), s(z())))
    assert count(res) == 2


def test_list_constructor():
    a = relay.TypeVar("a")
    assert relay.ir_pass.infer_type(nil, mod).checked_type == relay.FuncType([], l(a), [a])
    assert relay.ir_pass.infer_type(cons(z(), nil()), mod).checked_type == l(nat())


def test_length():
    a = relay.TypeVar("a")
    assert mod[length].checked_type == relay.FuncType([l(a)], nat(), [a])
    res = intrp.evaluate(length(cons(z(), cons(z(), cons(z(), nil())))))
    assert count(res) == 3


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


def test_foldl():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = mod[foldl].checked_type
    rhs = relay.FuncType([relay.FuncType([a, b], a), a, l(b)], a, [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    rev = relay.Function([y, x], cons(x, y))
    res = intrp.evaluate(foldl(rev, nil(),
                               cons(build_nat(1),
                                    cons(build_nat(2),
                                         cons(build_nat(3), nil())))))
    reversed = to_list(res)
    assert len(reversed) == 3
    assert count(reversed[0]) == 3 and count(reversed[1]) == 2 and count(reversed[2]) == 1


def test_foldr():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = mod[foldr].checked_type
    rhs = relay.FuncType([relay.FuncType([a, b], b), b, l(a)], b, [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    y = relay.Var("y")
    identity = relay.Function([x, y], cons(x, y))
    res = intrp.evaluate(foldr(identity, nil(),
                               cons(build_nat(1),
                                    cons(build_nat(2),
                                         cons(build_nat(3), nil())))))
    same = to_list(res)
    assert len(same) == 3
    assert count(same[0]) == 1 and count(same[1]) == 2 and count(same[2]) == 3


def test_sum():
    assert mod[sum].checked_type == relay.FuncType([l(nat())], nat())
    res = intrp.evaluate(sum(cons(build_nat(1), cons(build_nat(2), nil()))))
    assert count(res) == 3


def test_tmap():
    a = relay.TypeVar("a")
    b = relay.TypeVar("b")
    lhs = mod[tmap].checked_type
    rhs = relay.FuncType([relay.FuncType([a], b), tree(a)], tree(b), [a, b])
    assert lhs == rhs

    x = relay.Var("x")
    add_one = relay.Function([x], s(x))
    res = intrp.evaluate(tmap(add_one,
                              rose(z(),
                                   cons(rose(z(), nil()),
                                        cons(rose(z(), nil()),
                                             nil())))))

    tree_dict = tree_to_dict(res)
    assert count(tree_dict['member']) == 1
    assert len(tree_dict['children']) == 2
    for subtree in tree_dict['children']:
        assert count(subtree['member']) == 1
        assert len(subtree['children']) == 0


def test_size():
    a = relay.TypeVar("a")
    lhs = mod[size].checked_type
    rhs = relay.FuncType([tree(a)], nat(), [a])
    assert lhs == rhs

    root = rose(z(), cons(rose(z(), nil()),
                                  cons(rose(z(), nil()),
                                       nil())))
    t = rose(z(), cons(root, cons(root, cons(root, nil()))))
    res = intrp.evaluate(size(t))
    assert count(res) == 10


def test_wildcard_match_solo():
    x = relay.Var('x', nat())
    copy = relay.Function([x],
                          relay.Match(x, [relay.Clause(relay.PatternWildcard(), x)]),
                          nat())

    res = intrp.evaluate(copy(s(s(s(z())))))
    assert count(res) == 3


def test_wildcard_match_order():
    x = relay.Var('x', l(nat()))
    y = relay.Var('y')
    a = relay.Var('a')
    return_zero = relay.Function(
        [x],
        relay.Match(x, [
            relay.Clause(relay.PatternWildcard(), z()),
            relay.Clause(
                relay.PatternConstructor(
                    cons, [relay.PatternVar(y), relay.PatternVar(a)]),
                y),
            relay.Clause(relay.PatternConstructor(nil), s(z()))
        ]),
        nat())

    res = intrp.evaluate(return_zero(cons(s(z()), nil())))
    # wildcard pattern is evaluated first
    assert count(res) == 0


def test_nested_matches():
    a = relay.TypeVar('a')
    x = relay.Var('x', l(l(a)))
    y = relay.Var('y')
    w = relay.Var('w')
    h = relay.Var('h')
    t = relay.Var('t')
    flatten = relay.GlobalVar('flatten')

    # flatten could be written using a fold, but this way has nested matches
    inner_match = relay.Match(
        y, [
            relay.Clause(relay.PatternConstructor(nil), flatten(w)),
            relay.Clause(relay.PatternConstructor(
                cons, [relay.PatternVar(h), relay.PatternVar(t)]),
                cons(h, flatten(cons(t, w))))
        ])

    mod[flatten] = relay.Function(
        [x],
        relay.Match(x, [
            relay.Clause(relay.PatternConstructor(nil), nil()),
            relay.Clause(relay.PatternConstructor(
                cons, [relay.PatternVar(y), relay.PatternVar(w)]),
                         inner_match)
        ]), l(a), [a])

    first_list = cons(build_nat(1), cons(build_nat(2),
                                         cons(build_nat(3), nil())))
    second_list = cons(build_nat(4), cons(build_nat(5),
                                          cons(build_nat(6), nil())))
    final_list = cons(first_list, cons(second_list, nil()))

    res = intrp.evaluate(flatten(final_list))

    flat = to_list(res)
    assert len(flat) == 6
    for i in range(6):
        assert count(flat[i]) == i + 1


if __name__ == "__main__":
    test_nat_constructor()
    test_double()
    test_add()
    test_list_constructor()
    test_length()
    test_map()
    test_foldl()
    test_foldr()
    test_sum()
    test_tmap()
    test_size()
