import tvm
from tvm import relay
from tvm.relay.expr import debug_print
from tvm.relay.ir_builder import IRBuilder

ib = IRBuilder()

def show(e):
    r = debug_print(ib.env, e)
    assert r is not None


def test_constant():
    arr = tvm.nd.array(10)
    const = relay.Constant(arr)
    show(const)
    # should print the array inside?


def test_tuple():
    fields = tvm.convert([])
    tup = relay.Tuple(fields)
    show(tup)


def test_local_var():
    name_hint = 's'
    lv = relay.Var(name_hint)
    show(lv)


def test_dup_var():
    lv = relay.Var('s')
    rv = relay.Var('s')
    show(relay.Tuple([lv, rv]))


def test_large_dup_var():
    av = relay.Var('s')
    bv = relay.Var('s')
    cv = relay.Var('s')
    show(relay.Tuple([av, bv, cv]))


def test_global_var():
    name_hint = 'g'
    gv = relay.GlobalVar(name_hint)
    gv.name_hint == name_hint
    show(gv)


def test_param():
    lv = relay.Var('x')
    ty = None
    param = relay.Param(lv, ty)
    show(lv)


def test_function():
    param_names = ['a', 'b', 'c', 'd']
    params = tvm.convert([relay.Param(relay.Var(n), None) for n in param_names])
    ret_type = None
    body = params[0].var
    type_params = tvm.convert([])
    fn = relay.Function(params, ret_type, body, type_params)
    show(fn)



def test_call():
    op = relay.Var('f')
    arg_names = ['a', 'b', 'c', 'd']
    args = tvm.convert([relay.Var(n) for n in arg_names])
    call = relay.Call(op, args, None, None)
    show(call)


def test_let():
    lv = relay.Var('x')
    ty = relay.ty.TensorType((10, 20), "float32")
    arr = tvm.nd.array(10)
    value = relay.Constant(arr)
    let = relay.Let(lv, value, lv, ty)
    show(let)


def test_if():
    cond = relay.Var('cond')
    left = relay.Var('left')
    right = relay.Var('right')
    ife = relay.If(cond, left, right)
    show(ife)
