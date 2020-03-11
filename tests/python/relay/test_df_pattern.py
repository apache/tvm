import tvm
from tvm import relay
from tvm.relay.df_pattern import *


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

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 1
def test_AttrPattern():
    op = is_op('add').has_attr("TOpPattern", K_ELEMWISE)
    op_pat = op(wildcard(), wildcard())
    print(op_pat)

if __name__ == "__main__":
    test_expr_pattern()
    test_var_pattern()
    test_wildcard_pattern()
    test_CallPattern()
    test_TuplePattern()
    test_TupleGetItemPattern()
    test_AltPattern()
    test_TypePattern()
    test_AttrPattern()
