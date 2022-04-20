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
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.tir.expr import *
from tvm.relay import op
import numpy as np


def check_json_roundtrip(node):
    json_str = tvm.ir.save_json(node)
    back = tvm.ir.load_json(json_str)
    assert tvm.ir.structural_equal(back, node, map_free_vars=True)


# Span
def test_span():
    span = relay.Span(None, 1, 2, 3, 4)
    assert span.source_name == None
    assert span.line == 1
    assert span.end_line == 2
    assert span.column == 3
    assert span.end_column == 4
    assert span.same_as(span)
    assert span == span
    assert isinstance(span, relay.base.Span)
    str(span)

    # span is not a node so we can't use graph_equal
    # to test the round trip
    back = tvm.ir.load_json(tvm.ir.save_json(span))
    assert back.source_name == span.source_name
    assert back.line == span.line
    assert back.end_line == span.end_line
    assert back.column == span.column
    assert back.end_column == span.end_column


def test_constant():
    arr = tvm.nd.array(10)
    const = relay.Constant(arr)
    assert const.data == arr
    assert const.span == None
    str(const)
    check_json_roundtrip(const)


def test_tuple():
    fields = tvm.runtime.convert([])
    tup = relay.Tuple(fields)
    assert tup.fields == fields
    assert tup.span == None
    str(tup)
    check_json_roundtrip(tup)


def test_local_var():
    name_hint = "s"
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
    name_hint = "g"
    gv = relay.GlobalVar(name_hint)
    gv.name_hint == name_hint
    # assert lv.span == None todo(@jroesch): what do we do about spans
    str(gv)
    check_json_roundtrip(gv)


def test_function():
    param_names = ["a", "b", "c", "d"]
    params = tvm.runtime.convert([relay.Var(n) for n in param_names])
    ret_type = relay.TupleType(tvm.runtime.convert([]))
    body = relay.Tuple(tvm.runtime.convert([]))
    type_params = tvm.runtime.convert([])
    fn = relay.Function(params, body, ret_type, type_params)
    fn = fn.with_attr("test_attribute", "value")
    fn = fn.with_attr("test_attribute1", "value1")
    assert fn.params == params
    assert fn.body == body
    assert fn.type_params == type_params
    assert fn.span == None
    assert fn.attrs["test_attribute"] == "value"
    assert fn.attrs["test_attribute1"] == "value1"
    str(fn)
    check_json_roundtrip(fn)


def test_function_attrs():
    param_names = ["a", "b", "c", "d"]
    params = tvm.runtime.convert([relay.var(n, shape=(5, 2)) for n in param_names])
    ret_type = relay.TupleType(tvm.runtime.convert([]))
    body = relay.Tuple(tvm.runtime.convert([]))
    type_params = tvm.runtime.convert([])
    fn = relay.Function(params, body, ret_type, type_params)
    model_params = {}
    for param in params[:1]:
        cty = param.type_annotation
        tensor = np.random.rand(*[int(sh) for sh in cty.shape]).astype(cty.dtype)
        model_params[param] = relay.Constant(tvm.nd.array(tensor))

    fn = fn.with_attr("__params__", model_params)

    assert fn.params == params
    assert fn.body == body
    assert fn.type_params == type_params
    assert fn.span == None
    str(fn)
    check_json_roundtrip(fn)
    json_str = tvm.ir.save_json(fn)
    fn_after = tvm.ir.load_json(json_str)
    model_params_after = fn_after.attrs["__params__"]
    after_keys = [item[0] for item in model_params_after.items()]
    for key1, key2 in zip(model_params, after_keys):
        assert key1.name_hint == key2.name_hint
        p1 = model_params[key1]
        p2 = model_params_after[key2]
        np.testing.assert_allclose(p1.data.numpy(), p2.data.numpy())


def test_call():
    op = relay.Var("f")
    arg_names = ["a", "b", "c", "d"]
    args = tvm.runtime.convert([relay.Var(n) for n in arg_names])
    call = relay.Call(op, args, None, None)
    assert call.op == op
    assert call.args == args
    assert call.span == None
    str(call)
    check_json_roundtrip(call)


def test_let():
    lv = relay.Var("x")
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
    cond = relay.Var("cond")
    left = relay.Var("left")
    right = relay.Var("right")
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
    data = relay.var("data", shape=(1, 3, 224, 224))
    param = relay.var("param", shape=(64, 3, 7, 7))
    out = op.nn.conv2d(data, param, strides=(2, 2), padding=(3, 3), channels=64, kernel_size=(7, 7))
    check_json_roundtrip(out)


# Commented due to weird memory allocation issue
# def test_large_grpah():
# Test large graphs to avoid stack overflow in serialize/deserialize
#    size = int(1e5)
#    var = [relay.var("var_" + str(i), shape=(2, 3)) for i in range(size)]
#    body = var[-1]
#    for i in range(size, 1, -1):
#        body = relay.Let(var[i - 1], op.add(var[i - 2], var[i - 2]), body)
#    func = relay.Function([var[0]], body)
#    check_json_roundtrip(func)


if __name__ == "__main__":
    test_span()
    test_constant()
    test_tuple()
    test_local_var()
    test_global_var()
    test_function()
    test_function_attrs()
    test_call()
    test_let()
    test_if()
    test_tuple_get_item()
    test_op()
    test_conv2d_attrs()
    # Commented due to weird memory allocation issue
    # test_large_grpah()
