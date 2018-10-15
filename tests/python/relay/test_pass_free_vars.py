import tvm
from tvm import relay
from tvm.relay.ir_pass import free_vars, free_type_vars

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


def test_tuple():
    t = relay.Var('t')
    fv = free_vars(relay.Tuple([t, t]))
    assert len(fv) == 1
    assert fv[0] == t
    fv = free_vars(relay.TupleGetItem(t, 123))
    assert len(fv) == 1
    assert fv[0] == t


def test_free_type_vars():
    tp = relay.TypeParam("")
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
