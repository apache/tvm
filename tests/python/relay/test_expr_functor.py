import tvm
from tvm import relay
from tvm.relay import ExprFunctor, ExprMutator

def check_visit(expr):
    ef = ExprFunctor()
    try:
        ef.visit(expr)
        assert False
    except NotImplementedError:
        pass

    em = ExprMutator()
    assert em.visit(expr)

def test_constant():
    check_visit(relay.const(1.0))

def test_tuple():
    t = relay.Tuple([relay.var('x', shape=())])
    check_visit(t)

def test_var():
    v = relay.var('x', shape=())
    check_visit(v)

def test_global():
    v = relay.GlobalVar('f')
    check_visit(v)

def test_function():
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    params = [x, y]
    body = x + y
    ret_type = relay.TensorType(())
    type_params = []
    attrs = None # How to build?
    f = relay.Function(
        params,
        body,
        ret_type,
        type_params,
        attrs
    )
    check_visit(f)

def test_call():
    x = relay.var('x', shape=())
    y = relay.var('y', shape=())
    call = relay.op.add(x, y)
    check_visit(call)

def test_let():
    x = relay.var('x', shape=())
    value = relay.const(2.0)
    body = x + x
    l = relay.Let(x, value, body)
    check_visit(l)

def test_ite():
    cond = relay.var('x', shape=(), dtype='bool')
    ite = relay.If(cond, cond, cond)
    check_visit(ite)

def test_get_item():
    t = relay.Tuple([relay.var('x', shape=())])
    t = relay.TupleGetItem(t, 0)
    check_visit(t)

def test_ref_create():
    r = relay.expr.RefCreate(relay.const(1.0))
    check_visit(r)

def test_ref_read():
    ref = relay.expr.RefCreate(relay.const(1.0))
    r = relay.expr.RefRead(ref)
    check_visit(r)

def test_ref_write():
    ref = relay.expr.RefCreate(relay.const(1.0))
    r = relay.expr.RefWrite(ref, relay.const(2.0))
    check_visit(r)

if __name__ == "__main__":
    test_constant()
    test_tuple()
    test_var()
    test_global()
    test_function()
    test_call()
    test_let()
    test_ite()
    test_ref_create()
    test_ref_read()
    test_ref_write()
