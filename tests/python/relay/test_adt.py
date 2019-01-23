import tvm
from tvm import relay
from tvm.relay.ty import GlobalTypeVar, TypeVar, FuncType
from tvm.relay.expr import Var, Function, GlobalVar
from tvm.relay.adt import Constructor, TypeData, Clause, Match
from tvm.relay.adt import PatternConstructor, PatternVar, PatternWildcard
from tvm.relay.ir_pass import infer_type
from tvm.relay.backend.interpreter import Value, TupleValue, ConValue
from tvm.relay import testing, create_executor
from tvm.relay.prelude import Prelude

mod = relay.Module()
p = Prelude(mod)
ctx = tvm.context("llvm", 0)
intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

# defines peano nats and related functions for testing purposes
def add_nat_definitions():
    p.nat = GlobalTypeVar("nat")
    p.z = Constructor("z", [], p.nat)
    p.s = Constructor("s", [p.nat()], p.nat)
    mod[p.nat] = TypeData(p.nat, [], [p.z, p.s])

    p.double = GlobalVar("double")
    x = Var("x", p.nat())
    y = Var("y")
    z_case = Clause(PatternConstructor(p.z), p.z())
    s_case = Clause(PatternConstructor(p.s, [PatternVar(y)]), p.s(p.s(p.double(y))))
    mod[p.double] = Function([x], Match(x, [z_case, s_case]))

    p.add = GlobalVar("add")
    x = Var("x", p.nat())
    y = Var("y", p.nat())
    a = Var("a")
    z_case = Clause(PatternConstructor(p.z), y)
    s_case = Clause(PatternConstructor(p.s, [PatternVar(a)]), p.s(p.add(a, y)))
    mod[p.add] = Function([x, y], Match(x, [z_case, s_case]))

    p.sum = GlobalVar("sum")
    a = Var("a", p.l(p.nat()))
    mod[p.sum] = Function([a], p.foldl(p.add, p.z(), a))

    p.length = GlobalVar("length")
    a = TypeVar("a")
    x = Var("x", p.l(a))
    y = Var("y")
    nil_case = Clause(PatternConstructor(p.nil), p.z())
    cons_case = Clause(PatternConstructor(p.cons, [PatternWildcard(), PatternVar(y)]),
                       p.s(p.length(y)))
    mod[p.length] = Function([x], Match(x, [nil_case, cons_case]), None, [a])

add_nat_definitions()

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
