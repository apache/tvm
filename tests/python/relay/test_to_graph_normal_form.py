import numpy as np
import tvm
from tvm import relay
from tvm.relay.ir_pass import to_graph_normal_form, to_a_normal_form, alpha_equal
from tvm.relay import op, create_executor
from tvm.relay.backend.interpreter import Value, TupleValue


def check_eval(expr, args, expected_result, mod=None, rtol=1e-07):
    if mod is None:
        mod = relay.Module()

    ctx = tvm.context("llvm", 0)
    intrp = create_executor(mod=mod, ctx=ctx, target="llvm")

    result = intrp.evaluate(expr)(*args)
    np.testing.assert_allclose(result.asnumpy(), expected_result, rtol=rtol)


def test_implicit_share():
    x = relay.Var('x')
    y = relay.Var('y')
    z = relay.Var('z')
    body = relay.Let(z, op.add(y, y), op.add(z, z))
    body = relay.Let(y, op.add(x, x), body)
    f = relay.Function([], relay.Let(x, relay.const(1), body))
    g = to_graph_normal_form(f)
    assert "let" in f.astext()
    assert not "let" in g.astext()
    check_eval(f, [], 8.0)
    check_eval(g, [], 8.0)


def test_round_trip():
    x = relay.Var('x')
    y = relay.Var('y')
    z = relay.Var('z')
    body = relay.Let(z, op.add(y, y), op.add(z, z))
    body = relay.Let(y, op.add(x, x), body)
    f = relay.Function([], relay.Let(x, relay.const(1), body))
    g = to_graph_normal_form(f)
    h = to_a_normal_form(g)
    assert "let" in f.astext()
    assert not "let" in g.astext()
    check_eval(f, [], 8.0)
    check_eval(g, [], 8.0)
    check_eval(h, [], 8.0)

if __name__ == '__main__':
    test_implicit_share()
    test_round_trip()
