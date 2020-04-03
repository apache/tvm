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
import tvm
from tvm import relay
from tvm.relay.df_pattern import *

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 1

## NODE TESTS
def test_expr_pattern():
    ep = ExprPattern(relay.var('x', shape=(4, 1)))
    print(ep)

def test_var_pattern():
    v = is_input("x")
    print(v)

def test_wildcard_pattern():
    wc = wildcard()
    print(wc)

def test_CallPattern():
    wc1 = wildcard()
    wc2 = wildcard()
    c = is_op("add")(wc1, wc2)
    print(c)

def test_TuplePattern():
    wc1 = wildcard()
    wc2 = wildcard()
    t = TuplePattern([wc1, wc2])
    print(t)

def test_TupleGetItemPattern():
    wc1 = wildcard()
    wc2 = wildcard()
    t = TuplePattern([wc1, wc2])
    tgi = TupleGetItemPattern(t, 1)
    print(tgi)

def test_AltPattern():
    is_add_or_sub = is_op('add') | is_op('subtract')
    print(is_add_or_sub)

def test_TypePattern():
    ty_pat = has_type(relay.TensorType((10, 10), "float32"))
    print(ty_pat)

def test_AttrPattern():
    op = is_op('add').has_attr("TOpPattern", K_ELEMWISE)
    op_pat = op(wildcard(), wildcard())
    print(op_pat)

## MATCHER TESTS

def test_match_op():
    assert is_op('add').match(relay.op.op.get("add"))

def test_no_match_op():
    assert not is_op('add').match(relay.op.op.get("subtract"))

def test_match_op_or():
    is_add_or_sub = is_op('add') | is_op('subtract')
    assert is_add_or_sub.match(relay.op.op.get("add"))
    assert is_add_or_sub.match(relay.op.op.get("subtract"))

def test_match_call_commutive():
    x = relay.var('x')
    y = relay.var('y')
    add_pattern = is_op('add')(is_input("x"), is_input("y"))
    assert add_pattern.match(x + y)
    assert add_pattern.match(y + x)
    mul_pattern = is_op('multiply')(is_input("x"), is_input("y"))
    assert mul_pattern.match(x * y)
    assert mul_pattern.match(y * x)

def test_no_match_call_commutive():
    x = relay.var('x')
    y = relay.var('y')
    add_pattern = is_op('subtract')(is_input("x"), is_input("y"))
    assert add_pattern.match(x - y)
    assert not add_pattern.match(y - x)
    add_pattern = is_op('divide')(is_input("x"), is_input("y"))
    assert add_pattern.match(x / y)
    assert not add_pattern.match(y / x)

def test_match_call():
    x = relay.var('x')
    y = relay.var('y')
    add_pattern = is_op('add')(wildcard(), wildcard())
    assert add_pattern.match(x + y)

def test_no_match_call():
    x = relay.var('x')
    y = relay.var('y')
    add_pattern = is_op('add')(wildcard(), wildcard())
    assert not add_pattern.match(x - y)

def test_match_tuple():
    x = relay.var('x')
    y = relay.var('y')
    z = relay.op.op.get("add")
    tuple_pattern = TuplePattern((is_input("x"), wildcard(), is_op("add")))
    assert tuple_pattern.match(relay.expr.Tuple((x,y,z)))

def test_no_match_tuple():
    x = relay.var('x')
    y = relay.var('y')
    z = relay.op.op.get("add")
    tuple_pattern = TuplePattern((is_input('x'), wildcard(), is_op("add"), wildcard()))
    assert not tuple_pattern.match(relay.expr.Tuple((x,y,z)))

def test_match_tuple():
    x = relay.var('x')
    y = relay.var('y')
    z = relay.op.op.get("add")
    tuple_pattern = TuplePattern((is_input("x"), wildcard(), is_op("add")))
    tuple_get_item_pattern = TupleGetItemPattern(tuple_pattern, 1)
    assert tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x,y,z)), 1))

def test_no_match_tuple():
    x = relay.var('x')
    y = relay.var('y')
    z = relay.op.op.get("add")
    tuple_pattern = TuplePattern((is_input('x'), wildcard(), is_op("add")))
    tuple_get_item_pattern = TupleGetItemPattern(tuple_pattern, 1)
    assert not tuple_get_item_pattern.match(relay.expr.TupleGetItem(relay.expr.Tuple((x,y,z)), 2))

def test_match_type():
    x = relay.var('x', shape=(10, 10), dtype="float32")
    ty_pat = has_type(relay.TensorType((10, 10), "float32"))
    assert ty_pat.match(x)

def test_no_match_type():
    x = relay.var('x', shape=(10, 10), dtype="int32")
    ty_pat = has_type(relay.TensorType((10, 10), "float32"))
    assert not ty_pat.match(x)

def test_match_attr():
    op = is_op('add').has_attr("TOpPattern", K_ELEMWISE)
    op_pat = op(wildcard(), wildcard())
    x = relay.var('x')
    y = relay.var('y')
    assert op_pat.match(x + y)

def test_no_match_attr():
    op = is_op('nn.dense').has_attr("TOpPattern", K_ELEMWISE)
    op_pat = op(wildcard(), wildcard())
    x = relay.var('x')
    y = relay.var('y')
    assert not op_pat.match(relay.op.nn.dense(x, y))

def test_match_diamond():
    # Pattern
    is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
    path1 = is_op('nn.relu')(is_conv2d)
    path2 = is_op('nn.leaky_relu')(is_conv2d)
    diamond = is_op('add')(path1, path2)

    # Expr
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

def test_no_match_diamond():
    # Pattern
    is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
    path1 = is_op('nn.relu')(is_conv2d)
    path2 = is_op('nn.leaky_relu')(is_conv2d)
    diamond = is_op('add')(path1, path2)

    # Expr
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(leaky_relu)
    assert not diamond.match(relu)

def test_match_fake_diamond():
    # Pattern
    is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
    path1 = is_op('nn.relu')(is_conv2d)
    path2 = is_op('nn.leaky_relu')(is_conv2d)
    diamond = is_op('add')(path1, path2)

    # Expr
    input1 = relay.var('input1')
    weight1 = relay.var('weight1')
    conv2d1 = relay.op.nn.conv2d(input1, weight1)
    inp2 = relay.var('input2')
    weight2 = relay.var('weight2')
    conv2d2 = relay.op.nn.conv2d(inp2, weight2)
    relu = relay.op.nn.relu(conv2d1)
    leaky_relu = relay.op.nn.leaky_relu(conv2d2, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)

def test_match_rewrite():
    x = relay.var('x')
    y = relay.var('y')
    add_pattern = is_op('add')(wildcard(), wildcard())
    sub_pattern = is_op('subtract')(wildcard(), wildcard())
    def add_to_sub(pre, post):
        return post.args[0] - post.args[1]
    out = rewrite([DFPatternCallback(add_pattern, add_to_sub)], x + y)
    assert sub_pattern.match(out)


if __name__ == "__main__":
    test_match_op()
    test_no_match_op()
    test_match_op_or()
    test_match_call()
    test_no_match_call()
    test_match_call_commutive()
    test_no_match_call_commutive()
    test_match_tuple()
    test_no_match_tuple()
    test_match_type()
    test_no_match_type()
    test_match_attr()
    test_no_match_attr()
    test_match_diamond()
    test_no_match_diamond()
    test_match_fake_diamond()
    test_match_rewrite()
