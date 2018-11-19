"""Test alter op layout pass"""

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay.ir_pass import alter_op_layout, alpha_equal, infer_type

def test_alter_op():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, weight,
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    @register_alter_op_layout("nn.conv2d", level=100)
    def alter_conv2d(attrs, inputs):
        data, weight = inputs
        weight = relay.multiply(weight, relay.const(2.0))
        return relay.nn.conv2d(data, weight, **attrs)

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56))
        weight = relay.var('weight', shape=(64, 64, 3, 3))
        y = relay.nn.conv2d(x, relay.multiply(weight, relay.const(2.0)),
                            channels=64,
                            kernel_size=(3, 3),
                            padding=(1, 1))
        y = relay.nn.relu(y)
        y = relay.Function([x, weight], y)
        return y

    a = before()
    a = infer_type(a)
    a = alter_op_layout(a)

    b = expected()
    b = infer_type(b)

    assert(alpha_equal(a, b))


def test_alter_return_none():
    def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.nn.global_max_pool2d(x)
        y = relay.Function([x], y)
        return y

    called = [False]

    @register_alter_op_layout("nn.global_max_pool2d", level=101)
    def alter_conv2d(attrs, inputs):
        called[0] = True
        return None

    a = before()
    a = alter_op_layout(a)

    b = before()
    assert(alpha_equal(a, b))
    assert(called[0])


if __name__ == "__main__":
    test_alter_op()
    test_alter_return_none()
