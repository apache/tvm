import tvm
from tvm import relay
from tvm.relay.df_pattern import ExprPattern


def test_expr_pattern():
    ep = ExprPattern(relay.var('x', shape=(4, 1)))
    print(ep)

test_expr_pattern()
