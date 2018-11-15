import numpy as np
import tvm
from tvm import relay


def test_fold_const():
    c_data = np.array([1, 2, 3]).astype("float32")
    def before():
        c = relay.const(c_data)
        x = relay.var("x")
        y = relay.add(c, c)
        y = relay.multiply(y, relay.const(2, "float32"))
        y = relay.add(x, y)
        z = relay.add(y, c)
        return relay.Function([x], z)

    def expected():
        x = relay.var("x")
        c_folded = (c_data + c_data) * 2
        y = relay.add(x, relay.const(c_folded))
        z = relay.add(y, relay.const(c_data))
        return relay.Function([x], z)

    def fail(x):
        raise RuntimeError()
    # the fold constant should work on any context.
    with tvm.build_config(add_lower_pass=[(0, fail)]):
        with tvm.target.create("cuda"):
            zz = relay.ir_pass.fold_constant(before())
    zexpected = expected()
    assert relay.ir_pass.alpha_equal(zz, zexpected)


def test_fold_let():
    c_data = np.array(1).astype("float32")
    def before():
        sb = relay.ScopeBuilder()
        x = relay.var("x")
        t1 = sb.let("t1", relay.const(c_data))
        t2 = sb.let("t2", relay.add(t1, t1))
        t3 = sb.let("t3", relay.add(t2, x))
        sb.ret(t3)
        return relay.Function([x], sb.get())

    def expected():
        sb = relay.ScopeBuilder()
        x = relay.var("x")
        c_folded = (c_data + c_data)
        t3 = sb.let("t3", relay.add(relay.const(c_folded), x))
        sb.ret(t3)
        return relay.Function([x], sb.get())

    zz = relay.ir_pass.fold_constant(before())
    zexpected = expected()
    assert relay.ir_pass.graph_equal(zz, zexpected)


def test_fold_tuple():
    c_data = np.array(1).astype("float32")
    def before():
        c = relay.const(c_data)
        x = relay.var("x")
        y = relay.Tuple([x, c])
        z = relay.add(y[1], c)
        z = relay.add(z, y[0])
        return relay.Function([x], z)

    def expected():
        c = relay.const(c_data + c_data)
        x = relay.var("x")
        z = relay.add(c, x)
        return relay.Function([x], z)

    zz = relay.ir_pass.fold_constant(before())
    zexpected = expected()
    assert relay.ir_pass.graph_equal(zz, zexpected)


if __name__ == "__main__":
    test_fold_const()
    test_fold_let()
    test_fold_tuple()
