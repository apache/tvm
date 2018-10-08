from tvm import relay

def test_op_attr():
    log_op = relay.op.get("log")

    @relay.op.register("exp", "ftest")
    def test(x):
        return x + 1

    assert log_op.num_inputs  == 1
    assert log_op.get_attr("ftest") is None
    assert relay.op.get("exp").get_attr("ftest")(1) == 2

def test_op_level1():
    x = relay.Var("x")

    for op_name in ["log", "exp", "sqrt", "tanh"]:
        y = getattr(relay, op_name)(x)
        assert y.op.name == op_name
        assert y.op.support_level == 1
        assert y.args[0] == x

def test_op_level3():
    x = relay.Var("x")

    for op_name in ["ceil", "floor", "trunc", "round", "abs", "negative"]:
        y = getattr(relay, op_name)(x)
        assert y.op.name == op_name
        assert y.op.support_level == 3
        assert y.args[0] == x

if __name__ == "__main__":
    test_op_attr()
    test_op_level1()
    test_op_level3()
