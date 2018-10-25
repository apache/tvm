import tvm
from tvm import relay
from tvm.relay.ir_pass import well_formed

def test_well_formed():
    x = relay.Var('x')
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
    x = relay.Var('x')
    assert well_formed(x)
    v = relay.Constant(tvm.nd.array(10))
    let = relay.Let(x, v, x)
    assert well_formed(let)
    assert well_formed(relay.Tuple([v, v]))
    assert not well_formed(relay.Tuple([let, relay.Let(x, v, x)]))


def test_tuple_get_item():
    t = relay.Var('t')
    assert well_formed(relay.TupleGetItem(t, 2))
