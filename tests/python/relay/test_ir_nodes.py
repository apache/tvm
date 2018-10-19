""" test ir"""
import tvm
from tvm import relay
from tvm.expr import *
from tvm.relay.ir_pass import alpha_equal


def json_roundtrip(node):
    json_str = tvm.save_json(node)
    return tvm.load_json(json_str)


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

    back = json_roundtrip(span)
    assert back.source == span.source
    assert back.lineno == span.lineno
    assert back.col_offset == span.col_offset

# Types

def test_tensor_type():
    shape = tvm.convert([1, 2, 3])
    dtype = 'float32'
    tt = relay.TensorType(shape, dtype)
    assert tt.dtype == dtype
    assert tt.shape == shape
    assert tt.span == None
    str(tt)

    # roundtrip preserves alpha-equality
    back = json_roundtrip(tt)
    assert back == tt


def test_type_param():
    tp = relay.TypeVar('name', relay.Kind.Type)
    assert tp.kind == relay.Kind.Type
    # assert tp.span  # TODO allow us to set span
    str(tp)

    back = json_roundtrip(tp)
    # pointer equality will not be preserved so alpha-equality will fail
    assert back.kind == tp.kind


def test_func_type():
    type_params = tvm.convert([])
    type_constraints = tvm.convert([])  # TODO: fill me in
    arg_types = tvm.convert([])
    ret_type = relay.TensorType((1, 2, 3), 'float32')
    tf = relay.FuncType(arg_types, ret_type, type_params, type_constraints)
    assert tf.type_params == type_params
    assert tf.type_constraints == type_constraints
    assert tf.arg_types == arg_types
    assert tf.ret_type == ret_type
    assert tf.span == None
    # TODO make sure we can set span
    str(tf)

    back = json_roundtrip(tf)
    assert back == tf


def test_tuple_type():
    tp = relay.TypeVar('tp', relay.Kind.Type)
    tf = relay.FuncType(tvm.convert([]), None, tvm.convert([]), tvm.convert([]))
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    fields = tvm.convert([tp, tf, tt])

    tup_ty = relay.TupleType(fields)
    assert tup_ty.fields == fields

    back = json_roundtrip(tup_ty)
    assert back == tup_ty


def test_type_relation():
    tp = relay.TypeVar('tp', relay.Kind.Type)
    tf = relay.FuncType(tvm.convert([]), None, tvm.convert([]), tvm.convert([]))
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    args = tvm.convert([tp, tf, tt])

    num_inputs = 2
    func = tvm.get_env_func("tvm.relay.type_relation.Broadcast")
    attrs = tvm.make.node("attrs.TestAttrs", name="attr", padding=(3,4))

    tr = relay.TypeRelation(func, args, num_inputs, attrs)
    assert tr.args == args
    assert tr.num_inputs == num_inputs

    back = json_roundtrip(tr)
    # assert back == tr
    assert tr.num_inputs == back.num_inputs
    assert len(back.args) == len(tr.args)
    for i in range(len(back.args)):
        assert back.args[i] == tr.args[i]
    assert back.attrs.name == tr.attrs.name


def test_constant():
    arr = tvm.nd.array(10)
    const = relay.Constant(arr)
    assert const.data == arr
    assert const.span == None
    str(const)

    back = json_roundtrip(const)
    assert alpha_equal(const, back)


def test_tuple():
    fields = tvm.convert([])
    tup = relay.Tuple(fields)
    assert tup.fields == fields
    assert tup.span == None
    str(tup)

    back = json_roundtrip(tup)
    assert alpha_equal(tup, back)


def test_local_var():
    name_hint = 's'
    lv = relay.Var(name_hint)
    assert lv.name_hint == name_hint
    assert lv.type_annotation is None
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(lv)

    t1 = relay.ty.TensorType((), "float")
    lv = relay.Var(name_hint, t1)
    assert lv.name_hint == name_hint
    assert lv.type_annotation == t1

    back = json_roundtrip(lv)
    # assert alpha_equal(lv, back)
    assert back.name_hint == lv.name_hint
    assert back.type_annotation == lv.type_annotation


def test_global_var():
    name_hint = 'g'
    gv = relay.GlobalVar(name_hint)
    gv.name_hint == name_hint
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(gv)

    back = json_roundtrip(gv)
    # assert alpha_equal(gv, back)
    assert back.name_hint == gv.name_hint


def test_function():
    param_names = ['a', 'b', 'c', 'd']
    params = tvm.convert([relay.Var(n) for n in param_names])
    ret_type = relay.TupleType(tvm.convert([]))
    body = relay.Tuple(tvm.convert([]))
    type_params = tvm.convert([])
    fn = relay.Function(params, body, ret_type, type_params)
    assert fn.params == params
    assert fn.body == body
    assert fn.type_params == type_params
    assert fn.span == None
    str(fn)

    back = json_roundtrip(fn)
    assert alpha_equal(fn, back)


def test_call():
    op = relay.Var('f')
    arg_names = ['a', 'b', 'c', 'd']
    args = tvm.convert([relay.Var(n) for n in arg_names])
    call = relay.Call(op, args, None, None)
    assert call.op == op
    assert call.args == args
    assert call.span == None
    str(call)

    back = json_roundtrip(call)
    # assert alpha_equal(call, back)
    assert back.op.name_hint == call.op.name_hint
    assert len(back.args) == len(call.args)
    for i in range(len(call.args)):
        assert back.args[i].name_hint == call.args[i].name_hint


def test_let():
    lv = relay.Var('x')
    ty = None
    arr = tvm.nd.array(10)
    value = relay.Constant(arr)
    # I would prefer that the order of arguments
    # matches syntax let x: t = v in b
    let = relay.Let(lv, value, lv)
    assert let.var == lv
    assert let.value == value
    assert let.body == lv
    assert let.span == None
    str(let)

    back = json_roundtrip(let)
    # assert alpha_equal(let, back)
    assert back.var.name_hint == let.var.name_hint
    assert alpha_equal(back.value, let.value)
    assert back.body.name_hint == let.body.name_hint


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

    back = json_roundtrip(ife)
    #assert alpha_equal(ife, back)
    assert back.cond.name_hint == ife.cond.name_hint
    assert back.true_branch.name_hint == ife.true_branch.name_hint
    assert back.false_branch.name_hint == ife.false_branch.name_hint


def test_tuple_get_item():
    tup = relay.Var("tuple")
    get = relay.TupleGetItem(tup, 1)
    assert get.tuple_value == tup
    assert get.index == 1
    str(get)

    back = json_roundtrip(get)
    #assert alpha_equal(get, back)
    assert back.tuple.name_hint == get.tuple.name_hint
    assert back.index == get.index


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
    test_function()
    test_call()
    test_let()
    test_if()
    test_tuple_get_item()
