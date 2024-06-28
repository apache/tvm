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
# pylint: disable=unused-wildcard-import
import numpy as np

import tvm
from tvm.script import tir as T
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_opt_pass

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 0
K_BROADCAST = 1
K_INJECTIVE = 2

## NODE TESTS
def test_expr_pattern():
    ep = is_expr(relay.var("x", shape=(4, 1)))
    assert isinstance(ep, ExprPattern)
    assert isinstance(ep.expr, relay.Var)


def test_var_pattern():
    v = is_var("x")
    assert isinstance(v, VarPattern)
    assert v.name == "x"


def test_constant_pattern():
    c = is_constant()
    assert isinstance(c, ConstantPattern)


def test_wildcard_pattern():
    wc = wildcard()
    assert isinstance(wc, WildcardPattern)


def test_CallPattern():
    wc1 = wildcard()
    wc2 = wildcard()
    c = is_op("add")(wc1, wc2)
    assert isinstance(c, CallPattern)
    assert isinstance(c.args[0], WildcardPattern)
    assert isinstance(c.args[1], WildcardPattern)


def test_FunctionPattern():
    wc1 = wildcard()
    wc2 = wildcard()
    c = is_op("add")(wc1, wc2)
    f = FunctionPattern([wc1, wc2], c)
    assert isinstance(f, FunctionPattern)
    assert isinstance(f.params[0], WildcardPattern)
    assert isinstance(f.params[1], WildcardPattern)
    assert isinstance(f.body, CallPattern)
    assert isinstance(f.body.args[0], WildcardPattern)
    assert isinstance(f.body.args[1], WildcardPattern)


def test_TuplePattern():
    wc1 = wildcard()
    wc2 = wildcard()
    t = is_tuple([wc1, wc2])
    assert isinstance(t, TuplePattern)
    assert isinstance(t.fields[0], WildcardPattern)
    assert isinstance(t.fields[1], WildcardPattern)


def test_TupleGetItemPattern():
    wc1 = wildcard()
    wc2 = wildcard()
    t = is_tuple([wc1, wc2])
    tgi = is_tuple_get_item(t, 1)
    assert isinstance(tgi, TupleGetItemPattern)
    assert isinstance(tgi.tuple, TuplePattern)
    assert isinstance(tgi.tuple.fields[0], WildcardPattern)
    assert isinstance(tgi.tuple.fields[1], WildcardPattern)


def test_AltPattern():
    is_add_or_sub = is_op("add") | is_op("subtract")
    assert isinstance(is_add_or_sub, AltPattern)


def test_TypePattern():
    ttype = relay.TensorType((10, 10), "float32")
    ty_pat = has_type(ttype)
    assert isinstance(ty_pat, TypePattern)
    assert ty_pat.type == ttype


def test_DataTypePattern():
    dtype = "float16"
    pattern = has_dtype(dtype)
    assert isinstance(pattern, DataTypePattern)
    assert pattern.dtype == dtype


def test_ShapePattern():
    shape = [T.int32(10), T.int32(10)]
    pattern = has_shape(shape)
    assert isinstance(pattern, ShapePattern)
    tvm.ir.assert_structural_equal(pattern.shape, shape)


def test_AttrPattern():
    op = is_op("add").has_attr({"TOpPattern": K_ELEMWISE})
    assert isinstance(op, AttrPattern)
    assert op.attrs["TOpPattern"] == K_ELEMWISE


def test_IfPattern():
    x = is_var("x")
    y = is_var("y")
    pat = is_if(is_op("less")(x, y), x, y)

    assert isinstance(pat, IfPattern)
    assert isinstance(pat.cond, CallPattern)
    assert isinstance(pat.true_branch, VarPattern)
    assert isinstance(pat.false_branch, VarPattern)


def test_LetPattern():
    x = is_var("x")
    y = is_var("y")
    let_var = is_var("let")
    pat = is_let(let_var, is_op("less")(x, y), let_var)

    assert isinstance(pat, LetPattern)
    assert isinstance(pat.var, VarPattern)
    assert isinstance(pat.value, CallPattern)
    assert isinstance(pat.body, VarPattern)


## MATCHER TESTS


def test_match_op():
    assert is_op("add").match(relay.op.op.get("add"))


def test_no_match_op():
    assert not is_op("add").match(relay.op.op.get("subtract"))


def test_match_op_or():
    is_add_or_sub = is_op("add") | is_op("subtract")
    assert is_add_or_sub.match(relay.op.op.get("add"))
    assert is_add_or_sub.match(relay.op.op.get("subtract"))


def test_match_call_commutive():
    x = relay.var("x")
    y = relay.var("y")
    add_pattern = is_op("add")(is_var("x"), is_var("y"))
    assert add_pattern.match(x + y)
    assert add_pattern.match(y + x)
    mul_pattern = is_op("multiply")(is_var("x"), is_var("y"))
    assert mul_pattern.match(x * y)
    assert mul_pattern.match(y * x)


def test_no_match_call_commutive():
    x = relay.var("x")
    y = relay.var("y")
    add_pattern = is_op("subtract")(is_var("x"), is_var("y"))
    assert add_pattern.match(x - y)
    assert not add_pattern.match(y - x)
    add_pattern = is_op("divide")(is_var("x"), is_var("y"))
    assert add_pattern.match(x / y)
    assert not add_pattern.match(y / x)


def test_match_call():
    x = relay.var("x")
    y = relay.var("y")
    add_pattern = is_op("add")(wildcard(), wildcard())
    assert add_pattern.match(x + y)

    # Match call with any number of inputs
    call_pattern = wildcard()(None)
    assert call_pattern.match(relay.op.nn.relu(x))
    assert call_pattern.match(relay.op.add(x, y))


def test_no_match_call():
    x = relay.var("x")
    y = relay.var("y")
    add_pattern = is_op("add")(wildcard(), wildcard())
    assert not add_pattern.match(x - y)


def test_match_func():
    x = relay.var("x")
    y = relay.var("y")
    wc1 = wildcard()
    wc2 = wildcard()
    func_pattern = FunctionPattern([wc1, wc2], wc1 + wc2)
    assert func_pattern.match(relay.Function([x, y], x + y))

    # Match Function with any number of inputs
    func_pattern = FunctionPattern(None, wildcard())
    assert func_pattern.match(relay.Function([x], x))
    assert func_pattern.match(relay.Function([x, y], x + y))


def test_no_match_func():
    x = relay.var("x")
    y = relay.var("y")
    wc1 = wildcard()
    wc2 = wildcard()
    func_pattern = FunctionPattern([wc1, wc2], wc1 + wc2)
    assert not func_pattern.match(relay.Function([x, y], x - y))


def test_match_if():
    x = is_var("x")
    y = is_var("y")
    pat = is_if(is_op("less")(x, y), x, y)

    x = relay.var("x")
    y = relay.var("y")
    cond = x < y

    assert pat.match(relay.expr.If(cond, x, y))


def test_no_match_if():
    x = is_var("x")
    y = is_var("y")
    pat = is_if(is_op("less")(x, y), x, y)

    x = relay.var("x")
    y = relay.var("y")

    assert not pat.match(relay.expr.If(x > y, x, y))
    assert not pat.match(relay.expr.If(x < y, y, x))


def test_match_let():
    x = is_var("x")
    y = is_var("y")
    let_var = is_var("let")
    pat = is_let(let_var, is_op("less")(x, y), let_var)

    x = relay.var("x")
    y = relay.var("y")
    lv = relay.var("let")
    cond = x < y
    assert pat.match(relay.expr.Let(lv, cond, lv))


def test_no_match_let():
    x = is_var("x")
    y = is_var("y")
    let_var = is_var("let")
    pat = is_let(let_var, is_op("less")(x, y), let_var)

    x = relay.var("x")
    y = relay.var("y")
    lv = relay.var("let")

    assert not pat.match(relay.expr.Let(lv, x > y, lv))
    assert not pat.match(relay.expr.Let(lv, x < y, lv * x))


def test_match_option():
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")
    pattern = is_op("nn.relu")(
        is_op("nn.conv2d")(wildcard(), wildcard()).optional(
            lambda x: is_op("nn.bias_add")(x, wildcard())
        )
    )

    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)
    assert pattern.match(relu)

    conv2d = relay.op.nn.conv2d(x, w)
    bias_add = relay.op.nn.bias_add(conv2d, b)
    relu = relay.op.nn.relu(bias_add)
    assert pattern.match(relu)

    pattern = is_op("nn.conv2d")(wildcard(), wildcard())
    pattern = pattern.optional(is_op("nn.relu")).optional(is_op("tanh"))

    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)
    tanh = relay.op.tanh(conv2d)
    tanh2 = relay.op.tanh(relu)
    relu2 = relay.op.nn.relu(tanh)
    assert pattern.match(conv2d)
    assert pattern.match(relu)
    assert pattern.match(tanh)
    assert pattern.match(tanh2)
    assert not pattern.match(relu2)


def test_no_match_option():
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")
    pattern = is_op("nn.relu")(
        is_op("nn.conv2d")(wildcard(), wildcard()).optional(
            lambda x: is_op("nn.bias_add")(x, wildcard())
        )
    )

    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.tanh(conv2d)
    assert not pattern.match(relu)

    conv2d = relay.op.nn.dense(x, w)
    relu = relay.op.tanh(conv2d)
    assert not pattern.match(relu)

    conv2d = relay.op.nn.dense(x, w)
    bias_add = relay.op.nn.bias_add(conv2d, b)
    relu = relay.op.nn.relu(bias_add)
    assert not pattern.match(relu)

    conv2d = relay.op.nn.conv2d(x, w)
    bias_add = conv2d + w
    relu = relay.op.nn.relu(bias_add)
    assert not pattern.match(relu)


def test_match_const():
    conv2d = is_op("nn.conv2d")(wildcard(), is_constant())
    pattern = is_op("nn.bias_add")(conv2d, wildcard())

    x = relay.var("x", shape=(1, 3, 224, 224))
    w = relay.var("w", shape=(3, 3, 3, 3))
    b = relay.var("b", shape=(3,))
    conv2d = relay.op.nn.conv2d(x, w)
    out = relay.op.nn.bias_add(conv2d, b)
    func = relay.Function([x, w, b], out)
    mod = tvm.IRModule.from_expr(func)

    assert not pattern.match(mod["main"].body)
    mod["main"] = bind_params_by_name(mod["main"], {"w": tvm.nd.array(np.ones(shape=(3, 3, 3, 3)))})
    assert pattern.match(mod["main"].body)


def test_match_tuple():
    x = relay.var("x")
    y = relay.var("y")
    z = relay.op.op.get("add")
    tuple_pattern = is_tuple((is_var("x"), wildcard(), is_op("add")))
    assert tuple_pattern.match(relay.expr.Tuple((x, y, z)))

    tuple_pattern = is_tuple((is_var("x"), wildcard(), is_op("add")))
    tuple_get_item_pattern = is_tuple_get_item(tuple_pattern, 1)
    assert tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x, y, z)), 1))

    tuple_get_item_pattern = is_tuple_get_item(tuple_pattern)  # Match any index
    assert tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x, y, z)), 0))
    assert tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x, y, z)), 1))
    assert tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x, y, z)), 2))

    # Match tuple with any inputs
    tuple_pattern = is_tuple(None)
    concat_pattern = is_op("concatenate")(tuple_pattern)
    assert concat_pattern.match(relay.op.concatenate(relay.expr.Tuple((x,)), axis=0))
    assert concat_pattern.match(relay.op.concatenate(relay.expr.Tuple((x, y)), axis=0))
    assert concat_pattern.match(relay.op.concatenate(relay.expr.Tuple((x, y, z)), axis=0))


def test_no_match_tuple():
    x = relay.var("x")
    y = relay.var("y")
    z = relay.op.op.get("add")
    tuple_pattern = is_tuple((is_var("x"), wildcard(), is_op("add"), wildcard()))
    assert not tuple_pattern.match(relay.expr.Tuple((x, y, z)))

    tuple_pattern = is_tuple((is_var("x"), wildcard(), is_op("add")))
    tuple_get_item_pattern = is_tuple_get_item(tuple_pattern, 1)
    assert not tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x, y, z)), 2))


def test_match_type():
    x = relay.var("x", shape=(10, 10), dtype="float32")
    ty_pat = has_type(relay.TensorType((10, 10), "float32"))
    assert ty_pat.match(x)


def test_no_match_type():
    x = relay.var("x", shape=(10, 10), dtype="int32")
    ty_pat = has_type(relay.TensorType((10, 10), "float32"))
    assert not ty_pat.match(x)


def test_match_dtype():
    x = relay.var("x", shape=(10, 10), dtype="float32")
    ty_pat = has_dtype("float32")
    assert ty_pat.match(x)


def test_no_match_dtype():
    x = relay.var("x", shape=(10, 10), dtype="int32")
    ty_pat = has_dtype("float32")
    assert not ty_pat.match(x)


def test_match_shape():
    x = relay.var("x", shape=(10, 10), dtype="float32")
    ty_pat = has_shape((10, 10))
    assert ty_pat.match(x)


def test_no_match_shape():
    x = relay.var("x", shape=(10, 10), dtype="int32")
    ty_pat = has_shape((10, 5))
    assert not ty_pat.match(x)


def test_match_op_attr():
    op = is_op("add").has_attr({"TOpPattern": K_BROADCAST})
    op_pat = op(wildcard(), wildcard())
    x = relay.var("x")
    y = relay.var("y")
    assert op_pat.match(x + y)


def test_no_match_op_attr():
    op = is_op("nn.dense").has_attr({"TOpPattern": K_ELEMWISE})
    op_pat = op(wildcard(), wildcard())
    x = relay.var("x")
    y = relay.var("y")
    assert not op_pat.match(relay.op.nn.dense(x, y))
    op = is_op("add").has_attr({"TOpPattern": K_BROADCAST})
    op_pat = op(wildcard(), wildcard())
    x = relay.var("x")
    y = relay.var("y")
    assert not op_pat.match(x - y)
    z = relay.var("z")
    assert not op_pat.match(relay.Let(z, x + y, z))


def test_match_func_attr():
    pattern = wildcard().has_attr({"Composite": "add"})
    x = relay.var("x")
    y = relay.var("y")
    f = relay.Function([x, y], x + y).with_attr("Composite", "add")
    assert pattern.match(f)


def test_no_match_func_attr():
    pattern = wildcard().has_attr({"Composite": "add"})
    x = relay.var("x")
    y = relay.var("y")

    f = relay.Function([x, y], x + y).with_attr("RandomTest", "add")
    assert not pattern.match(f)
    f = relay.Function([x, y], x + y).with_attr("Composite", "conv_bias")
    assert not pattern.match(f)


def test_match_call_attr():
    # String attr
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"data_layout": "NCHW"})
    x = relay.var("x")
    y = relay.var("y")
    assert is_conv2d.match(relay.op.nn.conv2d(x, y))

    # Array attr
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"kernel_size": [3, 3]})
    out = relay.op.nn.conv2d(x, y, kernel_size=[3, 3])
    assert is_conv2d.match(out)

    # non-operator call
    attr_dict = {"call_attr": "attr"}
    call_has_attr = wildcard()(wildcard()).has_attr(attr_dict)
    call_attr = tvm.ir.make_node("DictAttrs", **attr_dict)
    a = relay.Var("a")
    b = relay.Var("b")
    assert call_has_attr.match(relay.Call(a, [b], attrs=call_attr))

    # empty attrs should match anything
    empty_attrs = tvm.ir.make_node("DictAttrs", **{})
    call_has_empty_attrs = wildcard()(wildcard()).has_attr({})
    assert call_has_empty_attrs.match(relay.Call(a, [b], attrs=empty_attrs))
    assert call_has_empty_attrs.match(relay.Call(a, [b], attrs=call_attr))


def test_no_match_call_attr():
    x = relay.var("x")
    y = relay.var("y")

    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"data_layout": "NHWC"})
    assert not is_conv2d.match(relay.op.nn.conv2d(x, y))

    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"RandomAttr": "NCHW"})
    assert not is_conv2d.match(relay.op.nn.conv2d(x, y))

    # Array attr
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"kernel_size": [3, 3]})
    out = relay.op.nn.conv2d(x, y, kernel_size=[2, 1])
    assert not is_conv2d.match(out)

    # non-operator calls
    call_has_attr = wildcard()(wildcard()).has_attr({"call_attr": "attr"})
    wrong_key = tvm.ir.make_node("DictAttrs", **{"wrong": "attr"})
    wrong_value = tvm.ir.make_node("DictAttrs", **{"call_attr": "wrong"})
    empty_attrs = tvm.ir.make_node("DictAttrs", **{})

    a = relay.Var("a")
    b = relay.Var("b")
    # attrs left undefined
    assert not call_has_attr.match(relay.Call(a, [b]))
    # wrong attrs
    assert not call_has_attr.match(relay.Call(a, [b], attrs=wrong_key))
    assert not call_has_attr.match(relay.Call(a, [b], attrs=wrong_value))
    assert not call_has_attr.match(relay.Call(a, [b], attrs=empty_attrs))


def test_match_call_attr_dtype():
    is_cast = is_op("cast")(wildcard()).has_attr({"dtype": "float32"})
    x = relay.var("x")
    assert is_cast.match(relay.op.cast(x, "float32"))


def test_match_diamond():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    path1 = is_op("nn.relu")(is_conv2d)
    path2 = is_op("nn.leaky_relu")(is_conv2d)
    diamond = is_op("add")(path1, path2)

    # Expr
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)


def test_no_match_diamond():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    path1 = is_op("nn.relu")(is_conv2d)
    path2 = is_op("nn.leaky_relu")(is_conv2d)
    diamond = is_op("add")(path1, path2)

    # Expr
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)

    # Check
    assert not diamond.match(leaky_relu)
    assert not diamond.match(relu)


def test_match_fake_diamond():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    path1 = is_op("nn.relu")(is_conv2d)
    path2 = is_op("nn.leaky_relu")(is_conv2d)
    diamond = is_op("add")(path1, path2)

    # Expr
    input1 = relay.var("input1")
    weight1 = relay.var("weight1")
    conv2d1 = relay.op.nn.conv2d(input1, weight1)
    inp2 = relay.var("input2")
    weight2 = relay.var("weight2")
    conv2d2 = relay.op.nn.conv2d(inp2, weight2)
    relu = relay.op.nn.relu(conv2d1)
    leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)


def test_at_most_one_parent():
    # Pattern
    P = is_op("nn.conv2d")(wildcard(), wildcard())  # 'parent'
    I = is_op("nn.relu")(wildcard())  # 'intermediate' ('path' in the code)
    C = is_op("add")(wildcard(), wildcard())  # 'child'
    pattern = dominates(P, I, C)

    #       n6(P)
    #      /  \
    #     n7   \
    #    /      \
    #    n8(P)  n10(I)
    #    \      /
    #    n9(I) /
    #      \  /
    #      n11(C)

    x = relay.var("x")
    w = relay.var("w")
    n6 = relay.op.nn.conv2d(x, w)  # matches P
    n7 = relay.op.tanh(n6)  # does not match I
    n8 = relay.op.nn.conv2d(n7, w)  # matches P
    n9 = relay.op.nn.relu(n8)  # matches I
    n10 = relay.op.nn.relu(n6)  # matches I
    n11 = relay.add(n9, n10)  # matches C

    # Does not match: Can't match the parent pattern P at both 8 and 6.
    # Note that if we did allow P to be used twice the implementation would
    # need to be changed to not 'jump over' n7.
    assert not pattern.match(n11)


def test_match_dominator():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Classic Diamond
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

    # Deeper Branch
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    relu = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

    # Single Branch
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    tanh = relay.op.tanh(relu)
    out = relu + tanh

    # Check
    assert diamond.match(out)

    # Fuzzy path/nested Diamond
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard()) | is_op(
        "add"
    )(wildcard(), wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relu + relu
    tanh = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = tanh + leaky_relu

    assert diamond.match(out)


def test_match_dominator2():
    # Pattern
    conv2d_pat = is_op("nn.conv2d")(wildcard(), wildcard())
    eltwise_pat = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(None)
    broadcast_pat = (wildcard().has_attr({"TOpPattern": K_BROADCAST}))(None)
    path_pat = eltwise_pat | broadcast_pat
    injective_pat = (wildcard().has_attr({"TOpPattern": K_INJECTIVE}))(wildcard())
    pattern = injective_pat.dominates(conv2d_pat, path_pat)

    # Graph
    inp = relay.var("input")
    weight = relay.var("weight")
    bias = relay.var("bias")
    conv2d = relay.op.nn.conv2d(inp, weight)
    bias_add = relay.op.nn.bias_add(conv2d, bias)
    relu = relay.op.nn.relu(bias_add)
    reshape = relay.op.reshape(relu, newshape=[-1, 2, 8])

    # Check
    assert pattern.match(reshape)


def test_not_match_dominator():
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Fake Diamond
    input1 = relay.var("input1")
    weight1 = relay.var("weight1")
    conv2d1 = relay.op.nn.conv2d(input1, weight1)
    inp2 = relay.var("input2")
    weight2 = relay.var("weight2")
    conv2d2 = relay.op.nn.conv2d(inp2, weight2)
    relu = relay.op.nn.relu(conv2d1)
    leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)

    # Add op that doesn't match K_ELEMWISE
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relu + relu
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)

    # Relu on the input instead of the conv
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(inp)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)

    # No conv
    inp = relay.var("input")
    relu = relay.op.nn.relu(inp)
    relu = relay.op.nn.relu(relu)
    tanh = relay.op.tanh(relu)
    out = relu + tanh

    # Check
    assert not diamond.match(out)


def test_not_match_dominator2():
    # Pattern
    P = is_op("nn.conv2d")(wildcard(), wildcard())  # 'parent'
    I = is_op("nn.relu")(wildcard())  # 'intermediate' ('path' in the code)
    C = is_op("add")(wildcard(), wildcard())  # 'child'
    pattern = dominates(P, I, C)

    #       n6(P)
    #      /  \
    #     n7   \
    #    /      \
    #    n8(P)  n9(I)
    #    \      /
    #     \    /
    #      \  /
    #      n10(C)

    x = relay.var("x")
    w = relay.var("w")
    n6 = relay.op.nn.conv2d(x, w)  # matches P
    n7 = relay.op.tanh(n6)  # does not match I
    n8 = relay.op.nn.conv2d(n7, w)  # matches P
    n9 = relay.op.nn.relu(n6)  # matches I
    n10 = relay.add(n8, n9)  # matches C

    # Does not match: Can't match the parent pattern P at both 8 and 6.
    # Note that if we did allow P to be used twice the implementation would
    # need to be changed to not 'jump over' n7.
    assert not pattern.match(n10)


def test_match_typed_dominator():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard()).has_dtype(
        "float32"
    )
    reduction = is_op("add")(wildcard(), wildcard()).has_shape([1, 3, 10, 10])
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Classic Diamond
    inp = relay.var("input", relay.TensorType((1, 3, 12, 12), "float32"))
    weight = relay.var("weight", relay.TensorType((3, 3, 3, 3), "float32"))
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)


def test_no_match_typed_dominator():
    # Classic Diamond
    inp = relay.var("input", relay.TensorType((1, 3, 12, 12), "float32"))
    weight = relay.var("weight", relay.TensorType((3, 3, 3, 3), "float32"))
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard()).has_dtype(
        "float32"
    )
    reduction = is_op("add")(wildcard(), wildcard()).has_shape([1, 1, 10, 10])
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Check
    assert not diamond.match(out)

    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard()).has_dtype(
        "float16"
    )
    reduction = is_op("add")(wildcard(), wildcard()).has_shape([1, 3, 10, 10])
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Check
    assert not diamond.match(out)


def test_rewrite():
    x = relay.var("x")
    y = relay.var("y")
    add_pattern = is_op("add")(wildcard(), wildcard())
    sub_pattern = is_op("subtract")(wildcard(), wildcard())

    class TestRewrite(DFPatternCallback):
        def __init__(self):
            super(TestRewrite, self).__init__()
            self.pattern = add_pattern

        def callback(self, pre, post, node_map):
            return post.args[0] - post.args[1]

    out = rewrite(TestRewrite(), x + y)
    assert sub_pattern.match(out)


def test_rewrite_func():
    x = relay.var("x")
    w = relay.var("w")
    y = relay.var("y")
    add_pattern = is_op("add")(wildcard(), wildcard())
    sub_pattern = is_op("subtract")(wildcard(), wildcard())

    class TestRewrite(DFPatternCallback):
        def __init__(self):
            super(TestRewrite, self).__init__()
            self.pattern = add_pattern

        def callback(self, pre, post, node_map):
            return post.args[0] - post.args[1]

    inpf = relay.var("input")
    weightf = relay.var("weight")
    func = relay.Function(
        [inpf, weightf], relay.op.nn.relu(relay.op.nn.conv2d(inpf, weightf)), attrs=None
    )
    out = rewrite(TestRewrite(), func(x, w) + y)
    assert sub_pattern.match(out)


def test_rewrite_func_with_attr():
    x = relay.var("x")
    y = relay.var("y")
    f = relay.Function([x, y], x + y).with_attr("Composite", "add")

    a = relay.var("a")
    b = relay.var("b")
    c = relay.Call(f, [a, b])
    c_abs = relay.abs(c)

    class TestRewrite(DFPatternCallback):
        def __init__(self):
            super(TestRewrite, self).__init__()
            self.pattern = wildcard().has_attr({"Composite": "add"})(wildcard(), wildcard())

        def callback(self, pre, post, node_map):
            return post.args[0] + post.args[1]

    out = rewrite(TestRewrite(), c_abs)
    inlined_add_pattern = is_op("abs")(is_op("add")(wildcard(), wildcard()))
    assert inlined_add_pattern.match(out)


def test_nested_rewrite():
    class PatternCallback(DFPatternCallback):
        def __init__(self, pattern):
            super(PatternCallback, self).__init__()
            self.pattern = pattern

        def callback(self, pre, post, node_map):
            return post

    def gen():
        x = relay.var("x")
        y = relay.var("y")
        y_add = relay.add(y, y)
        n0 = relay.add(x, y_add)
        n1 = relay.add(x, n0)
        return relay.add(n1, n0)

    def pattern():
        a = wildcard()
        b = wildcard()
        n0 = is_op("add")(a, b)
        n1 = is_op("add")(n0, a)
        return is_op("add")(n0, n1)

    out = gen()
    pat = pattern()
    new_out = rewrite(PatternCallback(pat), out)

    tvm.ir.assert_structural_equal(out, new_out)


def test_not_fuse_multi_diamond():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    path1 = is_op("nn.relu")(is_conv2d)
    path2 = is_op("nn.leaky_relu")(is_conv2d)
    diamond = is_op("add")(path1, path2)

    # Expr
    inp = relay.var("input")
    weight = relay.var("weight")
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu
    out = out + conv2d
    # Check
    assert not diamond.match(out)


class BatchnormCallback(DFPatternCallback):
    def __init__(self):
        super(BatchnormCallback, self).__init__()
        self.x = wildcard()
        self.var = wildcard()
        self.mean = wildcard()
        self.beta = wildcard()
        self.gamma = wildcard()
        self.eps = is_constant()

        self.pattern = (
            self.gamma * (self.x - self.mean) / is_op("sqrt")(self.var + self.eps) + self.beta
        )

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        var = node_map[self.var][0]
        mean = node_map[self.mean][0]
        beta = node_map[self.beta][0]
        gamma = node_map[self.gamma][0]
        eps = node_map[self.eps][0]
        return relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=eps.data.numpy().item())[0]


def test_fuse_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    BN = gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN)
    tvm.ir.assert_structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )


def test_no_fuse_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    fake_BN = gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5)) - beta

    out = rewrite(BatchnormCallback(), fake_BN)
    tvm.ir.assert_structural_equal(out, fake_BN)


def test_fuse_double_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    BN = gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5)) + beta
    BN2 = gamma * (BN - mean) / relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN2)

    bn = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    bn2 = relay.op.nn.batch_norm(bn, gamma, beta, mean, var, epsilon=1e-5)[0]

    tvm.ir.assert_structural_equal(out, bn2)


def test_partial_fuse_double_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    BN = gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5)) - beta
    BN2 = gamma * (BN - mean) / relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN2)

    bn2 = relay.op.nn.batch_norm(BN, gamma, beta, mean, var, epsilon=1e-5)[0]

    tvm.ir.assert_structural_equal(out, bn2)


def test_fuse_batchnorm_commutation():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    # commute add
    BN = beta + gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5))
    out = rewrite(BatchnormCallback(), BN)
    tvm.ir.assert_structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )

    # associate divide/multiply
    BN = (gamma * (x - mean)) / relay.op.sqrt(var + relay.const(1e-5)) + beta
    out = rewrite(BatchnormCallback(), BN)
    tvm.ir.assert_structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )

    # associate multiply/divide
    BN = gamma * ((x - mean) / relay.op.sqrt(var + relay.const(1e-5))) + beta
    out = rewrite(BatchnormCallback(), BN)
    tvm.ir.assert_structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )


def test_quadruple_rewrite_dominator():
    class DominatorRemovalCallback(DFPatternCallback):
        def __init__(self):
            super(DominatorRemovalCallback, self).__init__()
            self.inp = wildcard()
            self.weight = wildcard()
            is_conv2d = is_op("nn.conv2d")(self.inp, self.weight)
            is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(
                wildcard()
            ) | is_op("add")(wildcard(), wildcard())
            reduction = is_op("add")(wildcard(), wildcard())
            self.pattern = dominates(is_conv2d, is_unary_elemwise, reduction)

        def callback(self, pre, post, node_map):
            inp = node_map[self.inp][0]
            weight = node_map[self.weight][0]
            return relay.op.nn.conv2d(inp, weight)

    inp = relay.var("input")
    weight = relay.var("weight")
    # Classic Diamond
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Deeper Branch
    conv2d = relay.op.nn.conv2d(out, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    relu = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Single Branch
    conv2d = relay.op.nn.conv2d(out, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    tanh = relay.op.tanh(relu)
    out = relu + tanh

    # Fuzzy path/nested Diamond
    conv2d = relay.op.nn.conv2d(out, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relu + relu
    tanh = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = tanh + leaky_relu
    one = relay.op.nn.conv2d(inp, weight)
    two = relay.op.nn.conv2d(one, weight)
    three = relay.op.nn.conv2d(two, weight)
    four = relay.op.nn.conv2d(three, weight)

    tvm.ir.assert_structural_equal(DominatorRemovalCallback().rewrite(out), four)


def algebraic_simplify(expr):
    zero = is_expr(relay.const(0)) | is_expr(relay.const(0.0))
    one = is_expr(relay.const(1)) | is_expr(relay.const(1.0))

    class ElwiseNullCallback(DFPatternCallback):
        def callback(self, pre, post, node_map):
            return node_map[self.x][0]  # pylint: disable=no-member

    class AddCallback(ElwiseNullCallback):
        def __init__(self):
            super(AddCallback, self).__init__()
            self.x = wildcard()
            self.pattern = self.x + zero

    class SubCallback(ElwiseNullCallback):
        def __init__(self):
            super(SubCallback, self).__init__()
            self.x = wildcard()
            self.pattern = self.x - zero

    class MulCallback(ElwiseNullCallback):
        def __init__(self):
            super(MulCallback, self).__init__()
            self.x = wildcard()
            self.pattern = self.x * one

    class DivCallback(ElwiseNullCallback):
        def __init__(self):
            super(DivCallback, self).__init__()
            self.x = wildcard()
            self.pattern = self.x / one

    class MulZeroCallback(ElwiseNullCallback):
        def __init__(self):
            super(MulZeroCallback, self).__init__()
            self.x = zero
            self.pattern = self.x * wildcard()

    class ZeroDivCallback(ElwiseNullCallback):
        def __init__(self):
            super(ZeroDivCallback, self).__init__()
            self.x = zero
            self.pattern = self.x / wildcard()

    return rewrite(
        [
            AddCallback(),
            SubCallback(),
            MulCallback(),
            DivCallback(),
            MulZeroCallback(),
            ZeroDivCallback(),
        ],
        expr,
    )


def test_algebraic_simplify():
    x = relay.Var("x")
    y = relay.Var("y")

    one = relay.const(1)
    zero = relay.const(0)
    onef = relay.const(1.0)
    zerof = relay.const(0.0)

    assert algebraic_simplify(x + zero) == x
    assert algebraic_simplify(x + zerof) == x
    assert algebraic_simplify(zero + x) == x
    assert algebraic_simplify(zerof + x) == x

    assert algebraic_simplify(x - zero) == x
    assert algebraic_simplify(x - zerof) == x

    assert algebraic_simplify(x * one) == x
    assert algebraic_simplify(x * onef) == x
    assert algebraic_simplify(one * x) == x
    assert algebraic_simplify(onef * x) == x
    assert algebraic_simplify(x * zero) == zero
    assert algebraic_simplify(x * zerof) == zerof

    assert algebraic_simplify(x / one) == x
    assert algebraic_simplify(x / onef) == x
    assert algebraic_simplify(zero / x) == zero
    assert algebraic_simplify(zerof / x) == zerof

    tvm.ir.assert_structural_equal(
        algebraic_simplify((x + zero * y) / one + (y * one) - zero / x), x + y
    )


def test_double_partition():
    # Pattern 1
    conv2d_p = is_op("nn.conv2d")(wildcard(), wildcard())
    bias_add_p = is_op("nn.bias_add")(conv2d_p, wildcard())
    relu_p = is_op("nn.relu")(bias_add_p)

    # Graph
    x = relay.var("input")
    w = relay.var("weight")
    b = relay.var("bias")
    w2 = relay.var("weight")
    b2 = relay.var("bias")
    conv2d = relay.op.nn.conv2d(x, w)
    bias_add = relay.op.nn.bias_add(conv2d, b)
    relu = relay.op.nn.relu(bias_add)
    conv2d2 = relay.op.nn.conv2d(relu, w2)
    bias_add2 = relay.op.nn.bias_add(conv2d2, b2)

    partitioned = bias_add2
    for pat, label in [(relu_p, "conv_bias_relu"), (bias_add_p, "conv_bias")]:
        partitioned = pat.partition(partitioned, {"Composite": label})

    inpf = relay.var("input")
    weightf = relay.var("weight")
    biasf = relay.var("bias")
    func0 = (
        relay.Function(
            [inpf, weightf, biasf],
            relay.op.nn.relu(relay.op.nn.bias_add(relay.op.nn.conv2d(inpf, weightf), biasf)),
        )
        .with_attr("Composite", "conv_bias_relu")
        .with_attr("PartitionedFromPattern", "nn.conv2d_nn.bias_add_nn.relu_")
    )
    inpf = relay.var("input")
    weightf = relay.var("weight")
    biasf = relay.var("bias")
    func1 = (
        relay.Function(
            [inpf, weightf, biasf], relay.op.nn.bias_add(relay.op.nn.conv2d(inpf, weightf), biasf)
        )
        .with_attr("Composite", "conv_bias")
        .with_attr("PartitionedFromPattern", "nn.conv2d_nn.bias_add_")
    )

    expected = func1(func0(x, w, b), w2, b2)
    tvm.ir.assert_structural_equal(partitioned, expected)


def test_partition_dominator():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Classic Diamond
    inp = relay.var("input")
    weight = relay.var("weight")

    def generate_diamond(inp, weight):
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        relu = relay.op.nn.relu(relu)
        leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
        return relu + leaky_relu

    out = generate_diamond(inp * inp, weight * weight)
    # Check
    partitioned = diamond.partition(out)

    i = relay.Var("input")
    w = relay.Var("weight")
    f = relay.Function([i, w], generate_diamond(i, w)).with_attr(
        "PartitionedFromPattern", "nn.conv2d_nn.relu_nn.relu_nn.leaky_relu_add_"
    )
    tvm.ir.assert_structural_equal(partitioned, f(inp * inp, weight * weight))


def test_quadruple_partition_dominator():
    # Pattern
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr({"TOpPattern": K_ELEMWISE}))(wildcard()) | is_op(
        "add"
    )(wildcard(), wildcard())
    reduction = is_op("add")(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    inp = relay.var("input")
    weight = relay.var("weight")

    # Classic Diamond
    def classic_diamond(inp, weight):
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        relu = relay.op.nn.relu(relu)
        leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
        return relu + leaky_relu

    # Deeper Branch
    def deeper_diamond(inp, weight):
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        relu = relay.op.nn.relu(relu)
        relu = relay.op.tanh(relu)
        leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
        return relu + leaky_relu

    # Single Branch
    def single_branch(inp, weight):
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        relu = relay.op.nn.relu(relu)
        tanh = relay.op.tanh(relu)
        return relu + tanh

    # Fuzzy path/nested Diamond
    def nested_diamond(inp, weight):
        conv2d = relay.op.nn.conv2d(inp, weight)
        relu = relay.op.nn.relu(conv2d)
        relu = relu + relu
        tanh = relay.op.tanh(relu)
        leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
        return tanh + leaky_relu

    partitioned = diamond.partition(
        nested_diamond(
            single_branch(deeper_diamond(classic_diamond(inp, weight), weight), weight), weight
        )
    )

    functions = []
    partition_names = [
        "nn.conv2d_nn.relu_nn.relu_nn.leaky_relu_add_",
        "nn.conv2d_nn.relu_nn.relu_tanh_nn.leaky_relu_add_",
        "nn.conv2d_nn.relu_nn.relu_tanh_add_",
        "nn.conv2d_nn.relu_add_tanh_nn.leaky_relu_add_",
    ]
    for i, f in enumerate([classic_diamond, deeper_diamond, single_branch, nested_diamond]):
        inpf = relay.var("input")
        weightf = relay.var("weight")
        functions.append(
            relay.Function([inpf, weightf], f(inpf, weightf)).with_attr(
                "PartitionedFromPattern", partition_names[i]
            )
        )

    reference = functions[3](
        functions[2](functions[1](functions[0](inp, weight), weight), weight), weight
    )
    tvm.ir.assert_structural_equal(partitioned, reference)


def get_BN(x, var, mean, beta, gamma, eps):
    return gamma * (x - mean) / relay.op.sqrt(var + eps) + beta


def test_partition_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")
    eps = relay.const(1e-5)
    BN = get_BN(x, var, mean, beta, gamma, eps)

    xf = relay.var("xf")
    varf = relay.var("varf")
    meanf = relay.var("meanf")
    betaf = relay.var("betaf")
    gammaf = relay.var("gammaf")
    # Put the arguments in toplogological order for the reference
    f = relay.Function(
        [gammaf, xf, meanf, varf, betaf], get_BN(xf, varf, meanf, betaf, gammaf, eps)
    ).with_attr("PartitionedFromPattern", "subtract_multiply_add_sqrt_divide_add_")

    partitioned = BatchnormCallback().pattern.partition(BN)
    reference = f(gamma, x, mean, var, beta)
    tvm.ir.assert_structural_equal(partitioned, reference)


def test_partition_double_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")
    eps = relay.const(1e-5)

    BN = gamma * (x - mean) / relay.op.sqrt(var + eps) + beta
    BN2 = gamma * (BN - mean) / relay.op.sqrt(var + eps) + beta

    xf = relay.var("xf")
    varf = relay.var("varf")
    meanf = relay.var("meanf")
    betaf = relay.var("betaf")
    gammaf = relay.var("gammaf")
    f1 = relay.Function(
        [gammaf, xf, meanf, varf, betaf], get_BN(xf, varf, meanf, betaf, gammaf, eps)
    ).with_attr("PartitionedFromPattern", "subtract_multiply_add_sqrt_divide_add_")
    # The partitioner doesn't replace duplicates, so we use two copies of the function
    xf2 = relay.var("xf2")
    varf2 = relay.var("varf2")
    meanf2 = relay.var("meanf2")
    betaf2 = relay.var("betaf2")
    gammaf2 = relay.var("gammaf2")
    f2 = relay.Function(
        [gammaf2, xf2, meanf2, varf2, betaf2], get_BN(xf2, varf2, meanf2, betaf2, gammaf2, eps)
    ).with_attr("PartitionedFromPattern", "subtract_multiply_add_sqrt_divide_add_")

    partitioned = BatchnormCallback().pattern.partition(BN2)
    reference = f2(gamma, f1(gamma, x, mean, var, beta), mean, var, beta)
    tvm.ir.assert_structural_equal(partitioned, reference)


def test_overlappting_partitions():
    x = wildcard()
    gamma = wildcard()
    beta = wildcard()
    moving_mean = wildcard()
    moving_var = wildcard()
    bn_node = is_op("nn.batch_norm")(x, gamma, beta, moving_mean, moving_var)
    tuple_get_item_node = TupleGetItemPattern(bn_node, 0)

    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")
    BN = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)
    T1 = BN[0]
    T2 = BN[0]
    add = T1 + T2

    assert tuple_get_item_node.partition(add) == add


def test_partition_overused():
    pattern = is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))

    x = relay.var("input")
    w = relay.var("weight")
    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)
    out = relu + conv2d

    assert pattern.partition(out) == out


def test_partition_fuzzy_tuple():
    x = relay.var("x")
    y = relay.var("y")
    z = x + y
    tuple_pattern = is_tuple(None)
    concat_pattern = is_op("concatenate")(tuple_pattern)

    xp = relay.var("xp")
    yp = relay.var("yp")
    zp = relay.var("zp")

    def create_func(args, body):
        return relay.Function(args, body).with_attr("PartitionedFromPattern", "Tuple_concatenate_")

    def concat(*args):
        return relay.op.concatenate(relay.expr.Tuple(args), axis=0)

    one = concat_pattern.partition(concat(x))
    tvm.ir.assert_structural_equal(one, create_func([xp], concat(xp))(x))
    two = concat_pattern.partition(concat(x, y))
    tvm.ir.assert_structural_equal(two, create_func([xp, yp], concat(xp, yp))(x, y))
    three = concat_pattern.partition(concat(x, y, z))
    tvm.ir.assert_structural_equal(three, create_func([xp, yp, zp], concat(xp, yp, zp))(x, y, z))


def test_partition_fuzzy_function_args():
    func_pattern = FunctionPattern(None, wildcard() + wildcard())(None) + wildcard()
    x = relay.var("x")
    y = relay.var("y")
    z = relay.var("z")
    b = relay.var("b")
    xp = relay.var("xp")
    yp = relay.var("yp")
    zp = relay.var("zp")

    def create_func(call):
        N = len(call.op.params)
        new_params = [relay.var(str(i)) for i in range(N + 1)]
        label = "add_FunctionCall_add_"
        if N == 3:
            label = "add_" + label
        return relay.Function(
            new_params, relay.Call(call.op, (new_params[0:-1])) + new_params[-1]
        ).with_attr("PartitionedFromPattern", label)(*([x, y, z][0:N] + [b]))

    f1 = relay.Function([xp], xp + xp)(x)
    one = func_pattern.partition(f1 + b)
    tvm.ir.assert_structural_equal(one, create_func(f1))
    f2 = relay.Function([xp, yp], xp + yp)(x, y)
    two = func_pattern.partition(f2 + b)
    tvm.ir.assert_structural_equal(two, create_func(f2))
    f3 = relay.Function([xp, yp, zp], xp + yp + zp)(x, y, z)
    three = func_pattern.partition(f3 + b)
    tvm.ir.assert_structural_equal(three, create_func(f3))


def test_partition_check():
    pattern = is_op("nn.relu")(is_op("nn.conv2d")(is_var("input"), wildcard()))

    def check(pre):
        return pre.args[0].attrs.data_layout == "NCHW"

    x = relay.var("input")
    w = relay.var("weight")
    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)

    xf = relay.var("input")
    wf = relay.var("weight")
    conv2df = relay.op.nn.conv2d(xf, wf)
    reluf = relay.op.nn.relu(conv2df)
    func = relay.Function([xf, wf], reluf).with_attr("PartitionedFromPattern", "nn.conv2d_nn.relu_")

    reference = func(x, w)
    partitioned = pattern.partition(relu, check=check)
    tvm.ir.assert_structural_equal(partitioned, reference)

    conv2d = relay.op.nn.conv2d(x, w, data_layout="NHWC")
    relu = relay.op.nn.relu(conv2d)
    assert relu == pattern.partition(relu, check=check)


def test_partition_check_types():
    pattern = is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))

    def check(pre):
        conv = pre.args[0]
        return (conv.attrs.data_layout == "NCHW") and bool(conv.checked_type.shape[0] == 1)

    x = relay.var("input", shape=(1, 10, 10, 10))
    w = relay.var("weight", shape=(10, 10, 3, 3))
    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)
    relu = run_opt_pass(relu, relay.transform.InferType())

    partitioned = pattern.partition(relu, check=check)
    assert partitioned.op.attrs["PartitionedFromPattern"] == "nn.conv2d_nn.relu_"

    conv2d = relay.op.nn.conv2d(x, w, data_layout="NHWC")
    relu = relay.op.nn.relu(conv2d)
    relu = run_opt_pass(relu, relay.transform.InferType())
    assert relu == pattern.partition(relu, check=check)

    x = relay.var("input", shape=(2, 10, 10, 10))
    w = relay.var("weight", shape=(10, 10, 3, 3))
    conv2d = relay.op.nn.conv2d(x, w)
    relu = relay.op.nn.relu(conv2d)
    relu = run_opt_pass(relu, relay.transform.InferType())
    assert relu == pattern.partition(relu, check=check)


def conv_bias_relu(x, w, b):
    conv2d = relay.op.nn.conv2d(x, w)
    bias_add = relay.op.nn.bias_add(conv2d, b)
    relu = relay.op.nn.relu(bias_add)
    return relu


def test_partition_option():
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")

    conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    bias = conv2d.optional(lambda x: is_op("nn.bias_add")(x, wildcard()))
    pattern1 = is_op("nn.relu")(bias)

    conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
    bias = is_op("nn.bias_add")(conv2d, wildcard())
    pattern2 = bias.optional(lambda x: is_op("nn.relu")(x))

    relu = conv_bias_relu(x, w, b)

    xf = relay.var("x")
    wf = relay.var("w")
    bf = relay.var("b")
    func = relay.Function([xf, wf, bf], conv_bias_relu(xf, wf, bf)).with_attr(
        "PartitionedFromPattern", "nn.conv2d_nn.bias_add_nn.relu_"
    )

    assert pattern1.match(relu)
    tvm.ir.assert_structural_equal(func(x, w, b), pattern1.partition(relu))

    assert pattern2.match(relu)
    tvm.ir.assert_structural_equal(func(x, w, b), pattern2.partition(relu))


def test_partition_function():
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")

    x1 = relay.var("x1")
    w1 = relay.var("w1")

    wc_x = wildcard()
    wc_w = wildcard()
    wc_b = wildcard()
    wc_x1 = wildcard()
    wc_w1 = wildcard()

    func_pattern = FunctionPattern([wc_x1, wc_w1], is_op("nn.conv2d")(wc_x1, wc_w1))
    pattern = func_pattern(wc_x, wc_w) + wc_b

    func = relay.Function([x1, w1], relay.nn.conv2d(x1, w1))
    expr = func(x, w) + b + b

    x2 = relay.var("x2")
    w2 = relay.var("w2")
    b2 = relay.var("b2")
    func2 = relay.Function([x2, w2, b2], func(x2, w2) + b2).with_attr(
        "PartitionedFromPattern", "nn.conv2d_FunctionCall_add_"
    )
    expr2 = func2(x, w, b) + b
    tvm.ir.assert_structural_equal(pattern.partition(expr), expr2)


def test_partition_optional_function():
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")

    x1 = relay.var("x1")
    w1 = relay.var("w1")

    wc_x = wildcard()
    wc_w = wildcard()
    wc_x1 = wildcard()
    wc_w1 = wildcard()

    func_pattern0 = FunctionPattern(
        [wc_x1, wc_w1], is_op("sigmoid")(is_op("nn.conv2d")(wc_x1, wc_w1))
    )
    func_pattern1 = FunctionPattern(
        [wc_x1, wc_w1], is_op("nn.relu")(is_op("nn.conv2d")(wc_x1, wc_w1))
    )
    pattern = func_pattern0(wc_x, wc_w) | func_pattern1(wc_x, wc_w)

    func = relay.Function([x1, w1], relay.nn.relu(relay.nn.conv2d(x1, w1)))
    expr = func(x, w) + b

    x2 = relay.var("x2")
    w2 = relay.var("w2")
    func2 = relay.Function([x2, w2], func(x2, w2)).with_attr(
        "PartitionedFromPattern", "nn.conv2d_nn.relu_FunctionCall_"
    )
    expr2 = func2(x, w) + b
    tvm.ir.assert_structural_equal(pattern.partition(expr), expr2)


def test_rewrite_function_with_fuzzy_body():
    """Allow Rewriting a function with a fuzzy body via dominator analysis"""
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")

    x1 = relay.var("x1")
    w1 = relay.var("w1")

    wc_x = wildcard()
    wc_w = wildcard()
    wc_b = wildcard()
    wc_x1 = wildcard()
    wc_w1 = wildcard()

    func_pattern = FunctionPattern([wc_x1, wc_w1], wildcard())
    pattern = func_pattern(wc_x, wc_w) + wc_b

    func = relay.Function([x1, w1], relay.nn.conv2d(x1, w1))
    expr = func(x, w) + b + b

    class TestRewrite(DFPatternCallback):
        def __init__(self):
            super(TestRewrite, self).__init__()
            self.pattern = pattern

        def callback(self, pre, post, node_map):
            return x + w

    out = rewrite(TestRewrite(), expr)
    tvm.ir.assert_structural_equal(out, x + w + b)


def test_partition_function_with_fuzzy_body():
    """
    Allow Rewriting a function with a fuzzy body via dominator analysis
    """
    x = relay.var("x")
    w = relay.var("w")
    b = relay.var("b")

    x1 = relay.var("x1")
    w1 = relay.var("w1")

    wc_x = wildcard()
    wc_w = wildcard()
    wc_b = wildcard()
    wc_x1 = wildcard()
    wc_w1 = wildcard()

    func_pattern = FunctionPattern([wc_x1, wc_w1], wildcard())
    pattern = func_pattern(wc_x, wc_w) + wc_b

    func = relay.Function([x1, w1], relay.nn.conv2d(x1, w1))
    expr = func(x, w) + b + b

    x2 = relay.var("x2")
    w2 = relay.var("w2")
    b2 = relay.var("b2")
    func2 = relay.Function([x2, w2, b2], func(x2, w2) + b2).with_attr(
        "PartitionedFromPattern", "nn.conv2d_FunctionCall_add_"
    )
    expr2 = func2(x, w, b) + b
    tvm.ir.assert_structural_equal(pattern.partition(expr), expr2)


def test_match_match():
    add_pattern = is_op("add")(wildcard(), wildcard())

    class TestRewrite(DFPatternCallback):
        def __init__(self):
            super(TestRewrite, self).__init__()
            self.pattern = add_pattern

        def callback(self, pre, post, node_map):
            return post.args[0] - post.args[1]

    mod = tvm.IRModule({})
    tvm.relay.prelude.Prelude(mod)
    # Apply rewrite on IR including relay.Match
    out = rewrite(TestRewrite(), mod["tensor_concatenate_int64"])
    tvm.ir.assert_structural_equal(mod["tensor_concatenate_int64"], out)


def test_partition_constant_embedding():
    x = relay.var("x")
    w = relay.var("w")
    wc = relay.const(1)
    b = relay.var("b")

    xf = relay.var("x")
    wf = relay.var("w")
    bf = relay.var("b")
    embeded_func = relay.Function([xf, bf], conv_bias_relu(xf, wc, bf)).with_attr(
        "PartitionedFromPattern", "nn.conv2d_nn.bias_add_nn.relu_"
    )
    xf = relay.var("x")
    wf = relay.var("w")
    bf = relay.var("b")
    lifted_func = relay.Function([xf, wf, bf], conv_bias_relu(xf, wf, bf)).with_attr(
        "PartitionedFromPattern", "nn.conv2d_nn.bias_add_nn.relu_"
    )
    relu = conv_bias_relu(x, w, b)
    reluc = conv_bias_relu(x, wc, b)

    # Check lifting of wildcard matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard())
    )
    tvm.ir.assert_structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    tvm.ir.assert_structural_equal(lifted_func(x, wc, b), pattern.partition(reluc))

    # Check lifting of input matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_var()), wildcard())
    )
    tvm.ir.assert_structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    tvm.ir.assert_structural_equal(reluc, pattern.partition(reluc))  # Constants are not Inputs

    # Check embedding of constant matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_constant()), wildcard())
    )
    tvm.ir.assert_structural_equal(relu, pattern.partition(relu))
    tvm.ir.assert_structural_equal(embeded_func(x, b), pattern.partition(reluc))

    # Check embedding of constant ExprPatterns
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_expr(wc)), wildcard())
    )
    tvm.ir.assert_structural_equal(relu, pattern.partition(relu))
    tvm.ir.assert_structural_equal(embeded_func(x, b), pattern.partition(reluc))

    # Check lifting/embedding of Alt matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_var() | is_constant()), wildcard())
    )
    tvm.ir.assert_structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    tvm.ir.assert_structural_equal(embeded_func(x, b), pattern.partition(reluc))

    # Check lifting/embedding of Alt matches with the other ordering
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_constant() | is_var()), wildcard())
    )
    tvm.ir.assert_structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    tvm.ir.assert_structural_equal(embeded_func(x, b), pattern.partition(reluc))


def test_rewrite_once():
    # This class recursively removes the arguments to concat until there is nothing left to concatenate.
    class ConcatRewriter(DFPatternCallback):
        def __init__(self, rewrite_once):
            super().__init__(rewrite_once=rewrite_once)
            self.pattern = is_op("concatenate")(None)

        def callback(self, pre, post, node_map):
            concat_args = post.args[0]
            # Remove the last argument
            new_args = [concat_args[i] for i in range(len(concat_args) - 1)]
            if new_args:
                return relay.op.concatenate(relay.expr.Tuple(new_args), axis=0)
            else:
                return concat_args[0]

    x = relay.var("x")
    y = relay.var("y")
    z = relay.var("z")
    concat = relay.op.concatenate(relay.expr.Tuple([x, y, z]), axis=0)

    def test_one_callback():
        # Let the rewriter run recursively
        out = rewrite(ConcatRewriter(False), concat)
        expected = x
        tvm.ir.assert_structural_equal(out, expected)

        # Run the rewriter once
        out = rewrite(ConcatRewriter(True), concat)
        expected = relay.op.concatenate(relay.expr.Tuple([x, y]), axis=0)
        tvm.ir.assert_structural_equal(out, expected)

    def test_multi_callbacks():
        # This class recursively add a nn.relu operator after nn.softmax
        class OneMoreReluRewriter(DFPatternCallback):
            def __init__(self, rewrite_once):
                super().__init__(rewrite_once=rewrite_once)
                self.pattern = is_op("nn.softmax")(None)

            def callback(self, pre, post, node_map):
                return relay.nn.relu(post)

        def before():
            # Before:
            #    x    y    z
            #    |    |    |
            #       concat
            #         |
            #      softmax
            return relay.nn.softmax(concat)

        def once_concat():
            # ConcatRewrite once, OneMoreReluRewrite once
            # Expected:
            #   x    y
            #   |    |
            #   concat
            #      |
            #   softmax
            #      |
            #    relu
            return relay.nn.relu(
                relay.nn.softmax(relay.op.concatenate(relay.expr.Tuple([x, y]), axis=0))
            )

        def recursive_concat():
            # ConcatRewrite recursively, OneMoreReluRewrite once
            # Expected:
            #      x
            #      |
            #   softmax
            #      |
            #    relu
            return relay.nn.relu(relay.nn.softmax(x))

        # Run ConcatRewriter once, OneMoreReluRewriter once
        out = rewrite(
            [OneMoreReluRewriter(True), ConcatRewriter(True)],
            before(),
        )
        tvm.ir.assert_structural_equal(out, once_concat())

        # Run ConcatRewriter recursively, OneMoreReluRewriter once
        out = rewrite(
            [OneMoreReluRewriter(True), ConcatRewriter(False)],
            before(),
        )
        tvm.ir.assert_structural_equal(out, recursive_concat())

    test_one_callback()
    test_multi_callbacks()


def test_matched_outside_but_dominated():
    """In this example the pattern matches the nn.conv2d/add/multiply flow. Even though the
    add output is consumed by the sigmoid, the sigmoid itself is dominated by the multiply.
    So partitioning can proceed, all be it with a duplication of the add."""
    in_mod = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(16, 16, 32, 32), float16], %weight: Tensor[(32, 16, 3, 3), float16], %bias: Tensor[(32), float32]) -> Tensor[(16, 32, 32, 32), float32] {
          %0 = layout_transform(%data, src_layout="NCHW", dst_layout="NHWC");
          %1 = layout_transform(%weight, src_layout="OIHW", dst_layout="OHWI");
          %2 = expand_dims(%bias, axis=1, num_newaxis=2);
          %3 = expand_dims(%2, axis=0);
          %4 = nn.conv2d(%0, %1, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="OHWI", out_dtype="float32");
          %5 = layout_transform(%3, src_layout="NCHW", dst_layout="NHWC");
          %6 = add(%4, %5);
          %7 = sigmoid(%6);
          %8 = multiply(%6, %7);
          layout_transform(%8, src_layout="NHWC", dst_layout="NCHW")
        }
        """
    )
    expected_mod = tvm.relay.parse(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(16, 16, 32, 32), float16], %weight: Tensor[(32, 16, 3, 3), float16], %bias: Tensor[(32), float32]) -> Tensor[(16, 32, 32, 32), float32] {
          %2 = expand_dims(%bias, axis=1, num_newaxis=2);
          %3 = expand_dims(%2, axis=0);
          %4 = layout_transform(%data, src_layout="NCHW", dst_layout="NHWC");
          %5 = layout_transform(%weight, src_layout="OIHW", dst_layout="OHWI");
          %6 = nn.conv2d(%4, %5, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="OHWI", out_dtype="float32");
          %7 = layout_transform(%3, src_layout="NCHW", dst_layout="NHWC");
          %8 = add(%6, %7);
          %9 = sigmoid(%8);
          %10 = fn (%FunctionVar_0_0, %FunctionVar_0_1, %FunctionVar_0_2, %FunctionVar_0_3, PartitionedFromPattern="nn.conv2d_add_multiply_") {
            %0 = nn.conv2d(%FunctionVar_0_0, %FunctionVar_0_1, padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="OHWI", out_dtype="float32");
            %1 = add(%0, %FunctionVar_0_2);
            multiply(%1, %FunctionVar_0_3)
          };
          %11 = %10(%4, %5, %7, %9);
          layout_transform(%11, src_layout="NHWC", dst_layout="NCHW")
        }
        """
    )
    pattern = is_op("multiply")(
        is_op("add")(is_op("nn.conv2d")(wildcard(), wildcard()), wildcard()), wildcard()
    )
    actual_mod = tvm.IRModule.from_expr(pattern.partition(in_mod["main"]))
    actual_mod = relay.transform.InferType()(actual_mod)
    tvm.ir.assert_structural_equal(actual_mod, expected_mod)


def test_partition_parallel_branch_with_same_input():
    """In this example, conv2d's two consumer(add and multiply) on two different branches are
    merged into one partition, make sure that the partitioned function has no redundant parameters"""
    # Pattern
    path1 = is_op("multiply")(wildcard(), wildcard())
    path2 = is_op("add")(wildcard(), wildcard())
    pattern = is_op("add")(path1, path2)

    i = relay.Var("input")
    w = relay.Var("weight")
    l = relay.Var("left")
    r = relay.Var("right")

    conv2d = relay.op.nn.conv2d(i, w)
    branch1 = relay.multiply(l, conv2d)
    branch2 = relay.add(conv2d, r)
    add = relay.add(branch1, branch2)

    lf = relay.Var("leftf")
    mf = relay.Var("midf")
    rf = relay.Var("rightf")
    f = relay.Function([lf, mf, rf], (lf * mf) + (mf + rf)).with_attr(
        "PartitionedFromPattern", "multiply_add_add_"
    )

    partitioned = pattern.partition(add)
    reference = f(l, conv2d, r)
    tvm.ir.assert_structural_equal(partitioned, reference)


def test_rewrite_with_pattern_recursion():
    data = relay.var("data", relay.TensorType((2, 8), "float32"))
    dense_weight = relay.const(np.zeros((4, 8)))
    feat = relay.nn.dense(data, dense_weight)
    feat = relay.cast(feat, "float32")
    feat = relay.cast(feat, "float32")
    feat = relay.cast(feat, "float32")
    feat = relay.cast(feat, "float32")
    feat = relay.cast(feat, "float32")
    oup = relay.cast(feat, "float32")

    expected = relay.nn.relu(oup)

    class TheRewrite(DFPatternCallback):
        def __init__(self, pattern):
            super(TheRewrite, self).__init__(rewrite_once=True)
            self.pattern = pattern

        def callback(self, pre, post, node_map):
            return relay.nn.relu(post)

    def test_reset_call_args():
        dense_pattern = is_op("nn.dense")(wildcard(), wildcard())
        wildcard_redirect = wildcard()
        the_pattern = is_op("cast")(wildcard_redirect)
        the_pattern2 = the_pattern | dense_pattern
        wildcard_redirect.redirect_to(the_pattern2)

        actual = rewrite(TheRewrite(the_pattern), oup)
        tvm.ir.assert_structural_equal(actual, expected)

    def test_reset_alt_left():
        dense_pattern = is_op("nn.dense")(wildcard(), wildcard())
        wildcard_redirect = wildcard()
        or_pattern = wildcard_redirect | dense_pattern
        the_pattern = is_op("cast")(or_pattern)
        wildcard_redirect.redirect_to(the_pattern)

        actual = rewrite(TheRewrite(the_pattern), oup)
        tvm.ir.assert_structural_equal(actual, expected)

    def test_reset_alt_right():
        dense_pattern = is_op("nn.dense")(wildcard(), wildcard())
        wildcard_redirect = wildcard()
        or_pattern = dense_pattern | wildcard_redirect
        the_pattern = is_op("cast")(or_pattern)
        wildcard_redirect.redirect_to(the_pattern)

        actual = rewrite(TheRewrite(the_pattern), oup)
        tvm.ir.assert_structural_equal(actual, expected)

    test_reset_call_args()
    test_reset_alt_left()
    test_reset_alt_right()


if __name__ == "__main__":
    tvm.testing.main()
