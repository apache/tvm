import tvm
from tvm import relay
from tvm.relay.ir_pass import well_formed
from tvm.relay.prelude import Prelude

def test_let():
    x = relay.Var("x")
    assert well_formed(x)
    v = relay.Constant(tvm.nd.array(10))
    ty = None
    let = relay.Let(x, v, x)
    assert well_formed(let)
    assert not well_formed(relay.Let(x, v, let))
    f = relay.Function([x], x, ty)
    assert well_formed(f)
    assert well_formed(
        relay.Let(relay.Var("y"), f,
                  relay.Let(relay.Var("z"), f, v)))


def test_tuple():
    x = relay.Var("x")
    assert well_formed(x)
    v = relay.Constant(tvm.nd.array(10))
    let = relay.Let(x, v, x)
    assert well_formed(let)
    assert well_formed(relay.Tuple([v, v]))
    assert not well_formed(relay.Tuple([let, relay.Let(x, v, x)]))


def test_tuple_get_item():
    t = relay.Var("t")
    assert well_formed(relay.TupleGetItem(t, 2))


def test_adt():
    mod = relay.Module()
    p = Prelude(mod)
    x = relay.Var("x")
    s_case = relay.Clause(relay.PatternConstructor(p.s, [relay.PatternVar(x)]), x)
    default_case = relay.Clause(relay.PatternVar(x), x)
    m0 = relay.Match(p.z(), [default_case])
    m1 = relay.Match(p.z(), [s_case, default_case])
    assert well_formed(m0)
    assert not well_formed(m1)

if __name__ == "__main__":
    test_let()
    test_tuple()
    test_tuple_get_item()
    test_adt()
