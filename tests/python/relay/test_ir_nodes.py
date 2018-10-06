""" test ir"""
import tvm
from tvm import relay
from tvm.expr import *

def test_bad_constructor():
    try:
        x = relay.ty.TensorType("xx", "xx")
    except tvm.TVMError:
        pass


# Span
def test_span():
    span = relay.Span(None, 1, 1)
    assert span.source == None
    assert span.lineno == 1
    assert span.col_offset == 1
    assert span.same_as(span)
    assert span == span
    assert isinstance(span, relay.base.Span)
    str(span)

# Types

def test_tensor_type():
    shape = tvm.convert([1, 2, 3])
    dtype = 'float32'
    tt = relay.TensorType(shape, dtype)
    assert tt.dtype == dtype
    assert tt.shape == shape
    assert tt.span == None
    str(tt)


def test_type_param():
    tp = relay.TypeParam('name', relay.Kind.Type)
    assert tp.kind == relay.Kind.Type
    # assert tp.span  # TODO allow us to set span
    str(tp)


def test_func_type():
    type_params = tvm.convert([])
    type_constraints = tvm.convert([])  # TODO: fill me in
    arg_types = tvm.convert([])
    ret_type = None
    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)
    assert tf.type_params == type_params
    assert tf.type_constraints == type_constraints
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert tf.span == None
    # TODO make sure we can set
    str(tf)


def test_tuple_type():
    tp = relay.TypeParam('tp', relay.Kind.Type)
    tf = relay.FuncType(tvm.convert([]), None, tvm.convert([]), tvm.convert([]))
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    fields = tvm.convert([tp, tf, tt])

    tup_ty = relay.TupleType(fields)
    assert tup_ty.fields == fields


def test_type_relation():
    tp = relay.TypeParam('tp', relay.Kind.Type)
    tf = relay.FuncType(tvm.convert([]), None, tvm.convert([]), tvm.convert([]))
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    args = tvm.convert([tf, tt, tp])

    num_inputs = 2
    func = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    attrs = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))

    tr = relay.TypeRelation(func, args, num_inputs, attrs)
    assert tr.args == args
    assert tr.num_inputs == num_inputs


def test_constant():
    arr = tvm.nd.array(10)
    const = relay.Constant(arr)
    assert const.data == arr
    assert const.span == None
    str(const)


def test_tuple():
    fields = tvm.convert([])
    tup = relay.Tuple(fields)
    assert tup.fields == fields
    assert tup.span == None
    str(tup)


def test_local_var():
    name_hint = 's'
    lv = relay.Var(name_hint)
    lv.name_hint == name_hint
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(lv)


def test_global_var():
    name_hint = 'g'
    gv = relay.GlobalVar(name_hint)
    gv.name_hint == name_hint
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(gv)


def test_param():
    lv = relay.Var('x')
    ty = None
    param = relay.Param(lv, ty)
    assert param.var == lv
    assert param.type == ty
    assert param.span == None
    str(param)


def test_function():
    param_names = ['a', 'b', 'c', 'd']
    params = tvm.convert([relay.Param(relay.Var(n), None) for n in param_names])
    ret_type = None
    body = None
    type_params = tvm.convert([])
    fn = relay.Function(params, ret_type, body, type_params)
    assert fn.params == params
    assert fn.body == body
    assert fn.type_params == type_params
    assert fn.span == None
    str(fn)


def test_call():
    op = relay.Var('f')
    arg_names = ['a', 'b', 'c', 'd']
    args = tvm.convert([relay.Var(n) for n in arg_names])
    call = relay.Call(op, args, None, None)
    assert call.op == op
    assert call.args == args
    assert call.span == None
    str(call)


def test_let():
    lv = relay.Var('x')
    ty = None
    arr = tvm.nd.array(10)
    value = relay.Constant(arr)
    # I would prefer that the order of arguments
    # matches syntax let x: t = v in b
    let = relay.Let(lv, value, lv, ty)
    assert let.var == lv
    assert let.value == value
    assert let.value_type == ty
    assert let.body == lv
    assert let.span == None
    str(let)


def test_if():
    cond = relay.Var('cond')
    left = relay.Var('left')
    right = relay.Var('right')
    ife = relay.If(cond, left, right)
    assert ife.cond == cond
    assert ife.true_branch == left
    assert ife.false_branch == right
    assert ife.span == None
    str(ife)


if __name__ == "__main__":
    test_bad_constructor()
    test_span()
    test_tensor_type()
    test_type_param()
    test_func_type()
    test_tuple_type()
    test_type_relation()
    test_constant()
    test_tuple()
    test_local_var()
    test_global_var()
    test_param()
    test_function()
    test_call()
    test_let()
    test_if()
