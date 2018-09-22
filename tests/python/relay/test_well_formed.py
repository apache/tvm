import tvm
from tvm import relay
from tvm.relay.ir_pass import well_formed

def test_well_formed():
    x = relay.Var("x")
    assert well_formed(x)
    v = relay.Constant(tvm.nd.array(10))
    ty = None
    let = relay.Let(x, v, x, ty)
    assert well_formed(let)
    assert not well_formed(relay.Let(x, v, let, ty))
    f = relay.Function([relay.Param(x, ty)], ty, x)
    assert well_formed(f)
    # this test should pass in case of weak uniqueness (only test for shadowing)
    # but we want all binder to be distinct from each other.
    assert not well_formed(relay.Let(relay.Var("y"), f,
                                     relay.Let(relay.Var("z"), f, v, ty), ty))
