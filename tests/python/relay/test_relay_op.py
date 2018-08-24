from tvm import relay

def test_op_level1():
    x = relay.Var("x")

    for op_name in ["log", "exp", "sqrt"]:
        y = getattr(relay, op_name)(x)
        assert y.op.name == op_name
        assert y.op.support_level == 1
        assert y.args[0] == x


if __name__ == "__main__":
    test_op_level1()
