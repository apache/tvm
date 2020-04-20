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
import numpy as np

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 0
K_BROADCAST = 1

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
    op = is_op('add').has_attr("TOpPattern", K_BROADCAST)
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


def test_match_dominator():
    # Pattern
    is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr("TOpPattern", K_ELEMWISE))(wildcard())
    reduction = is_op('add')(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Classic Diamond
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

    # Deeper Branch
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    relu = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert diamond.match(out)

    # Single Branch
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relay.op.nn.relu(relu)
    tanh = relay.op.tanh(relu)
    out = relu + tanh

    # Check
    assert diamond.match(out)
    
    # Fuzzy path/nested Diamond
    is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr("TOpPattern", K_ELEMWISE))(wildcard()) | is_op('add')(wildcard(), wildcard())
    reduction = is_op('add')(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relu + relu
    tanh = relay.op.tanh(relu)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = tanh + leaky_relu

    assert diamond.match(out)

def test_not_match_dominator():
    is_conv2d = is_op('nn.conv2d')(wildcard(), wildcard())
    is_unary_elemwise = (wildcard().has_attr("TOpPattern", K_ELEMWISE))(wildcard())
    reduction = is_op('add')(wildcard(), wildcard())
    diamond = dominates(is_conv2d, is_unary_elemwise, reduction)

    # Fake Diamond
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

    # Add op that doesn't match K_ELEMWISE
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(conv2d)
    relu = relu + relu
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)

    # Relu on the input instead of the conv
    inp = relay.var('input')
    weight = relay.var('weight')
    conv2d = relay.op.nn.conv2d(inp, weight)
    relu = relay.op.nn.relu(inp)
    leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
    out = relu + leaky_relu

    # Check
    assert not diamond.match(out)

    # No conv
    inp = relay.var('input')
    relu = relay.op.nn.relu(inp)
    relu = relay.op.nn.relu(relu)
    tanh = relay.op.tanh(relu)
    out = relu + tanh

    # Check
    assert not diamond.match(out)

def test_rewrite():
    x = relay.var('x')
    y = relay.var('y')
    add_pattern = is_op('add')(wildcard(), wildcard())
    sub_pattern = is_op('subtract')(wildcard(), wildcard())
    class TestRewrite(DFPatternCallback):
        def __init__(self):
            self.pattern = add_pattern
        def callback(self, pre, post, node_map):
            return post.args[0] - post.args[1]
    out = rewrite(TestRewrite(), x + y)
    assert sub_pattern.match(out)

def test_not_fuse_multi_diamond():
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
    out = out + conv2d
    # Check
    assert not diamond.match(out)

class BatchnormCallback(DFPatternCallback):
    def __init__(self):
        self.x = wildcard()
        self.var = wildcard()
        self.mean = wildcard()
        self.beta = wildcard()
        self.gamma = wildcard()
        self.eps = wildcard()
        
        self.pattern = self.gamma * (self.x - self.mean)/is_op("sqrt")(self.var + self.eps) + self.beta

    def callback(self, pre, post, node_map):
        x = node_map[self.x][0]
        var = node_map[self.var][0]
        mean = node_map[self.mean][0]
        beta = node_map[self.beta][0]
        gamma = node_map[self.gamma][0]
        eps = node_map[self.eps][0]
        return relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = eps.data.asnumpy().item())[0]

def test_fuse_batchnorm():
    x = relay.var('x')
    var = relay.var('var')
    mean = relay.var('mean')
    beta = relay.var('beta')
    gamma = relay.var('gamma')
    
    BN = gamma * (x - mean)/relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = 1e-5)[0])

def test_no_fuse_batchnorm():
    x = relay.var('x')
    var = relay.var('var')
    mean = relay.var('mean')
    beta = relay.var('beta')
    gamma = relay.var('gamma')
    
    fake_BN = gamma * (x - mean)/relay.op.sqrt(var + relay.const(1e-5)) - beta

    out = rewrite(BatchnormCallback(), fake_BN)
    assert tvm.ir.structural_equal(out, fake_BN)

def test_fuse_double_batchnorm():
    x = relay.var('x')
    var = relay.var('var')
    mean = relay.var('mean')
    beta = relay.var('beta')
    gamma = relay.var('gamma')
    
    BN = gamma * (x - mean)/relay.op.sqrt(var + relay.const(1e-5)) + beta
    BN2 = gamma * (BN - mean)/relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN2)

    bn = relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = 1e-5)[0]
    bn2 = relay.op.nn.batch_norm(bn, gamma, beta, mean, var, epsilon = 1e-5)[0]

    assert tvm.ir.structural_equal(out, bn2)

def test_partial_fuse_double_batchnorm():
    x = relay.var('x')
    var = relay.var('var')
    mean = relay.var('mean')
    beta = relay.var('beta')
    gamma = relay.var('gamma')
    
    BN = gamma * (x - mean)/relay.op.sqrt(var + relay.const(1e-5)) - beta
    BN2 = gamma * (BN - mean)/relay.op.sqrt(var + relay.const(1e-5)) + beta

    out = rewrite(BatchnormCallback(), BN2)

    bn2 = relay.op.nn.batch_norm(BN, gamma, beta, mean, var, epsilon = 1e-5)[0]

    assert tvm.ir.structural_equal(out, bn2)

def test_fuse_batchnorm_commutation():
    x = relay.var('x')
    var = relay.var('var')
    mean = relay.var('mean')
    beta = relay.var('beta')
    gamma = relay.var('gamma')
    
    #commute add
    BN = beta + gamma * (x - mean)/relay.op.sqrt(var + relay.const(1e-5))
    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = 1e-5)[0])

    # associate divide/multiply
    BN = (gamma * (x - mean)) /relay.op.sqrt(var + relay.const(1e-5))  + beta
    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = 1e-5)[0])

    # associate multiply/divide
    BN = gamma * ((x - mean)/relay.op.sqrt(var + relay.const(1e-5))) + beta
    out = rewrite(BatchnormCallback(), BN)
    assert tvm.ir.structural_equal(out, relay.op.nn.batch_norm(x, gamma, beta, mean, var, epsilon = 1e-5)[0])

def algebraic_simplify(expr):
    zero = (ExprPattern(relay.const(0)) | ExprPattern(relay.const(0.0)))
    one = (ExprPattern(relay.const(1)) | ExprPattern(relay.const(1.0)))
    class ElwiseNullCallback(DFPatternCallback):
        def callback(self, pre, post, node_map):
            return node_map[self.x][0]

    class AddCallback(ElwiseNullCallback):
        def __init__(self):
            self.x = wildcard()
            self.pattern = self.x + zero
    
    class SubCallback(ElwiseNullCallback):
        def __init__(self):
            self.x = wildcard()
            self.pattern = self.x - zero

    class MulCallback(ElwiseNullCallback):
        def __init__(self):
            self.x = wildcard()
            self.pattern = self.x * one

    class DivCallback(ElwiseNullCallback):
        def __init__(self):
            self.x = wildcard()
            self.pattern = self.x / one

    class MulZeroCallback(ElwiseNullCallback):
        def __init__(self):
            self.x = zero
            self.pattern = self.x * wildcard()

    class ZeroDivCallback(ElwiseNullCallback):
        def __init__(self):
            self.x = zero
            self.pattern = self.x / wildcard()

    return rewrite([AddCallback(),
                    SubCallback(),
                    MulCallback(),
                    DivCallback(),
                    MulZeroCallback(),
                    ZeroDivCallback()
                    ], expr);

def test_algebraic_simplify():
    x = relay.Var('x')
    y = relay.Var('y')  

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

    assert tvm.ir.structural_equal(algebraic_simplify((x + zero * y) / one + (y * one) - zero / x), x + y)

if __name__ == "__main__":
    #test_match_op()
    #test_no_match_op()
    #test_match_op_or()
    #test_match_call()
    #test_no_match_call()
    #test_match_call_commutive()
    #test_no_match_call_commutive()
    #test_match_tuple()
    #test_no_match_tuple()
    #test_match_type()
    #test_no_match_type()
    #test_match_attr()
    #test_no_match_attr()
    #test_match_diamond()
    #test_no_match_diamond()
    #test_match_fake_diamond()
    #test_rewrite()
    #test_fuse_batchnorm()
    #test_no_fuse_batchnorm()
    #test_fuse_double_batchnorm()
    #test_partial_fuse_double_batchnorm()
    #test_fuse_batchnorm_commutation()
    #test_match_dominator()
    #test_not_match_dominator()
    test_algebraic_simplify()
