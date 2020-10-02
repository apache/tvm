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
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_opt_pass

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 0
K_BROADCAST = 1


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
    shape = [10, 10]
    pattern = has_shape(shape)
    assert isinstance(pattern, ShapePattern)
    assert tvm.ir.structural_equal(pattern.shape, shape)


def test_AttrPattern():
    op = is_op("add").has_attr({"TOpPattern": K_ELEMWISE})
    assert isinstance(op, AttrPattern)
    assert op.attrs["TOpPattern"] == K_ELEMWISE


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


def test_no_match_call():
    x = relay.var("x")
    y = relay.var("y")
    add_pattern = is_op("add")(wildcard(), wildcard())
    assert not add_pattern.match(x - y)


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
    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"data_layout": "NCHW"})
    x = relay.var("x")
    y = relay.var("y")
    assert is_conv2d.match(relay.op.nn.conv2d(x, y))


def test_no_match_call_attr():
    x = relay.var("x")
    y = relay.var("y")

    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"data_layout": "NHWC"})
    assert not is_conv2d.match(relay.op.nn.conv2d(x, y))

    is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard()).has_attr({"RandomAttr": "NCHW"})
    assert not is_conv2d.match(relay.op.nn.conv2d(x, y))


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

    assert tvm.ir.structural_equal(out, new_out)


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
        return relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=eps.data.asnumpy().item())[
            0
        ]


def test_fuse_batchnorm():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    BN = gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(
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
    assert tvm.ir.structural_equal(out, fake_BN)


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

    assert tvm.ir.structural_equal(out, bn2)


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

    assert tvm.ir.structural_equal(out, bn2)


def test_fuse_batchnorm_commutation():
    x = relay.var("x")
    var = relay.var("var")
    mean = relay.var("mean")
    beta = relay.var("beta")
    gamma = relay.var("gamma")

    # commute add
    BN = beta + gamma * (x - mean) / relay.op.sqrt(var + relay.const(1e-5))
    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )

    # associate divide/multiply
    BN = (gamma * (x - mean)) / relay.op.sqrt(var + relay.const(1e-5)) + beta
    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(
        out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon=1e-5)[0]
    )

    # associate multiply/divide
    BN = gamma * ((x - mean) / relay.op.sqrt(var + relay.const(1e-5))) + beta
    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(
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

    assert tvm.ir.structural_equal(DominatorRemovalCallback().rewrite(out), four)


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

    assert tvm.ir.structural_equal(
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
    assert tvm.ir.structural_equal(partitioned, expected)


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
    assert tvm.ir.structural_equal(partitioned, f(inp * inp, weight * weight))


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
    assert tvm.ir.structural_equal(partitioned, reference)


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
    assert tvm.ir.structural_equal(partitioned, reference)


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
    assert tvm.ir.structural_equal(partitioned, reference)


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


def test_partition_check():
    pattern = is_op("nn.relu")(is_op("nn.conv2d")(wildcard(), wildcard()))

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
    assert tvm.ir.structural_equal(partitioned, reference)

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
    assert tvm.ir.structural_equal(func(x, w, b), pattern1.partition(relu))

    assert pattern2.match(relu)
    assert tvm.ir.structural_equal(func(x, w, b), pattern2.partition(relu))


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
    assert tvm.ir.structural_equal(mod["tensor_concatenate_int64"], out)


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
    assert tvm.ir.structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    assert tvm.ir.structural_equal(lifted_func(x, wc, b), pattern.partition(reluc))

    # Check lifting of input matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_var()), wildcard())
    )
    assert tvm.ir.structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    assert tvm.ir.structural_equal(reluc, pattern.partition(reluc))  # Constants are not Inputs

    # Check embedding of constant matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_constant()), wildcard())
    )
    assert tvm.ir.structural_equal(relu, pattern.partition(relu))
    assert tvm.ir.structural_equal(embeded_func(x, b), pattern.partition(reluc))

    # Check embedding of constant ExprPatterns
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_expr(wc)), wildcard())
    )
    assert tvm.ir.structural_equal(relu, pattern.partition(relu))
    assert tvm.ir.structural_equal(embeded_func(x, b), pattern.partition(reluc))

    # Check lifting/embedding of Alt matches
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_var() | is_constant()), wildcard())
    )
    assert tvm.ir.structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    assert tvm.ir.structural_equal(embeded_func(x, b), pattern.partition(reluc))

    # Check lifting/embedding of Alt matches with the other ordering
    pattern = is_op("nn.relu")(
        is_op("nn.bias_add")(is_op("nn.conv2d")(wildcard(), is_constant() | is_var()), wildcard())
    )
    assert tvm.ir.structural_equal(lifted_func(x, w, b), pattern.partition(relu))
    assert tvm.ir.structural_equal(embeded_func(x, b), pattern.partition(reluc))


if __name__ == "__main__":
    test_expr_pattern()
    test_var_pattern()
    test_constant_pattern()
    test_wildcard_pattern()
    test_CallPattern()
    test_TuplePattern()
    test_TupleGetItemPattern()
    test_AltPattern()
    test_TypePattern()
    test_DataTypePattern()
    test_ShapePattern()
    test_AttrPattern()
    test_match_op()
    test_no_match_op()
    test_match_op_or()
    test_match_call_commutive()
    test_no_match_call_commutive()
    test_match_call()
    test_no_match_call()
    test_match_option()
    test_no_match_option()
    test_match_const()
    test_match_tuple()
    test_no_match_tuple()
    test_match_type()
    test_no_match_type()
    test_match_dtype()
    test_no_match_dtype()
    test_match_shape()
    test_no_match_shape()
    test_match_op_attr()
    test_no_match_op_attr()
    test_match_func_attr()
    test_no_match_func_attr()
    test_match_call_attr()
    test_no_match_call_attr()
    test_match_diamond()
    test_no_match_diamond()
    test_match_fake_diamond()
    test_match_dominator()
    test_not_match_dominator()
    test_rewrite()
    test_rewrite_func()
    test_nested_rewrite()
    test_not_fuse_multi_diamond()
    test_fuse_batchnorm()
    test_no_fuse_batchnorm()
    test_fuse_double_batchnorm()
    test_partial_fuse_double_batchnorm()
    test_fuse_batchnorm_commutation()
    test_quadruple_rewrite_dominator()
    test_algebraic_simplify()
    test_double_partition()
    test_partition_dominator()
    test_quadruple_partition_dominator()
    test_partition_batchnorm()
    test_partition_double_batchnorm()
    test_partition_check()
    test_partition_check_types()
    test_partition_option()
    test_match_match()
    test_partition_constant_embedding()
