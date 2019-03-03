"""Test eliminate common subexpr pass"""
from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import ir_pass


def test_simple():
    def before():
        x = relay.var("x", shape=(1, 16))
        y1 = relay.nn.relu(x)
        y2 = relay.nn.relu(x)
        y1 = relay.add(y1, relay.const(1.0, "float32"))
        y2 = relay.add(y2, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16))
        y = relay.nn.relu(x)
        y = relay.add(y, relay.const(1.0, "float32"))
        y = relay.add(y, y)
        f = relay.Function([x], y)
        return f

    z = before()
    z = ir_pass.eliminate_common_subexpr(z)
    assert ir_pass.alpha_equal(z, expected())


def test_callback():
    def before():
        x = relay.var("x", shape=(1, 16))
        y1 = relay.nn.relu(x)
        y2 = relay.nn.relu(x)
        y1 = relay.add(y1, relay.const(1.0, "float32"))
        y2 = relay.add(y2, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def expected():
        x = relay.var("x", shape=(1, 16))
        y = relay.nn.relu(x)
        y1 = relay.add(y, relay.const(1.0, "float32"))
        y2 = relay.add(y, relay.const(1.0, "float32"))
        y = relay.add(y1, y2)
        f = relay.Function([x], y)
        return f

    def fskip(expr):
        if isinstance(expr, relay.expr.Call) and expr.op.name == 'add':
            return True
        return False

    z = before()
    z = ir_pass.eliminate_common_subexpr(z, fskip)
    assert ir_pass.alpha_equal(z, expected())


if __name__ == "__main__":
    test_simple()
    test_callback()
