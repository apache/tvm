import tvm
from tvm import relay
from tvm.relay.ir_pass import free_vars, free_type_vars

def test_free_vars():
    x = relay.Var("x")
    fvx = free_vars(x)
    assert len(fvx) == 1
    assert fvx[0] == x
    v = relay.Constant(tvm.nd.array(10))
    ty = relay.TensorType([], "int32")
    let = relay.Let(x, v, x, ty)
    fvx = free_vars(let)
    assert len(free_vars(let)) == 0
    f = relay.Function([relay.Param(x, ty)], ty, x)
    assert len(free_vars(f)) == 0

def test_free_type_vars():
    tp = relay.TypeParam("")
    ty = relay.TupleType([tp, relay.TensorType([], "int32")])
    x = relay.Var("x")
    y = relay.Var("y")
    let = relay.Let(x, y, x, ty)
    fvl = free_vars(let)
    assert len(fvl) == 1
    assert fvl[0] == y
    ftvl = free_type_vars(let)
    assert len(ftvl) == 1
    assert ftvl[0] == tp
