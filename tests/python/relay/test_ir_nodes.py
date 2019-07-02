# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" test ir"""
import tvm
from tvm import relay
from tvm.expr import *
from tvm.relay import op
from tvm.relay.analysis import graph_equal


def check_json_roundtrip(node):
    json_str = tvm.save_json(node)
    back = tvm.load_json(json_str)
    assert graph_equal(back, node)


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

    # span is not a node so we can't use graph_equal
    # to test the round trip
    back = tvm.load_json(tvm.save_json(span))
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
    check_json_roundtrip(tt)


def test_type_param():
    tp = relay.TypeVar('name', relay.Kind.Type)
    assert tp.kind == relay.Kind.Type
    # assert tp.span  # TODO allow us to set span
    str(tp)
    check_json_roundtrip(tp)


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
    check_json_roundtrip(tf)


def test_tuple_type():
    tp = relay.TypeVar('tp', relay.Kind.Type)
    tf = relay.FuncType(tvm.convert([]), None, tvm.convert([]), tvm.convert([]))
    tt = relay.TensorType(tvm.convert([1, 2, 3]), 'float32')
    fields = tvm.convert([tp, tf, tt])

    tup_ty = relay.TupleType(fields)
    assert tup_ty.fields == fields
    str(tup_ty)
    check_json_roundtrip(tup_ty)


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
    str(tr)
    check_json_roundtrip(tr)


def test_constant():
    arr = tvm.nd.array(10)
    const = relay.Constant(arr)
    assert const.data == arr
    assert const.span == None
    str(const)
    check_json_roundtrip(const)


def test_tuple():
    fields = tvm.convert([])
    tup = relay.Tuple(fields)
    assert tup.fields == fields
    assert tup.span == None
    str(tup)
    check_json_roundtrip(tup)


def test_local_var():
    name_hint = 's'
    lv = relay.Var(name_hint)
    assert lv.name_hint == name_hint
    assert lv.type_annotation is None
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(lv)
    check_json_roundtrip(lv)

    t1 = relay.ty.TensorType((), "float")
    lv = relay.Var(name_hint, t1)
    assert lv.name_hint == name_hint
    assert lv.type_annotation == t1


def test_global_var():
    name_hint = 'g'
    gv = relay.GlobalVar(name_hint)
    gv.name_hint == name_hint
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(gv)
    check_json_roundtrip(gv)


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
    check_json_roundtrip(fn)


def test_call():
    op = relay.Var('f')
    arg_names = ['a', 'b', 'c', 'd']
    args = tvm.convert([relay.Var(n) for n in arg_names])
    call = relay.Call(op, args, None, None)
    assert call.op == op
    assert call.args == args
    assert call.span == None
    str(call)
    check_json_roundtrip(call)


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
    check_json_roundtrip(let)


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
    check_json_roundtrip(ife)


def test_tuple_get_item():
    tup = relay.Var("tuple")
    get = relay.TupleGetItem(tup, 1)
    assert get.tuple_value == tup
    assert get.index == 1
    str(get)
    check_json_roundtrip(get)


def test_op():
    add = op.op.get("add")
    check_json_roundtrip(add)


def test_conv2d_attrs():
    data = relay.var('data', shape=(1, 3, 224, 224))
    param = relay.var('param', shape=(64, 3, 7, 7))
    out = op.nn.conv2d(
        data,
        param,
        strides=(2, 2),
        padding=(3, 3),
        channels=64,
        kernel_size=(7, 7))
    check_json_roundtrip(out)


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
    test_op()
    test_conv2d_attrs()
