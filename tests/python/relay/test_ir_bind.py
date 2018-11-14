""" test bind function."""
import tvm
from tvm import relay


def test_bind_params():
    x = relay.var("x")
    y = relay.var("y")
    z = relay.add(x, y)
    f = relay.Function([x, y], z)
    fbinded = relay.bind(f, {x : relay.const(1, "float32")})
    fexpected =relay.Function(
        [y],
        relay.add(relay.const(1, "float32"),  y))
    assert relay.ir_pass.alpha_equal(fbinded, fexpected)

    zbinded = relay.bind(z, {y: x})
    zexpected = relay.add(x, x)
    assert relay.ir_pass.alpha_equal(zbinded, zexpected)


if __name__ == "__main__":
    test_bind_params()
