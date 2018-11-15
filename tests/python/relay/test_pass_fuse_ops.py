import tvm
from tvm import relay

def test_fuse_simple():
    """Simple testcase."""
    def before():
        x = relay.var("x", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        return relay.Function([x], z)

    def expected():
        x = relay.var("p", shape=(10, 20))
        y = relay.add(x, relay.const(1, "float32"))
        z = relay.exp(y)
        f1 = relay.Function([x], z)
        x = relay.var("x", shape=(10, 20))
        y = relay.Call(f1, [x])
        return relay.Function([x], y)

    z = before()
    z = relay.ir_pass.infer_type(z)
    zz = relay.ir_pass.fuse_ops(z, opt_level=2)
    zz = relay.ir_pass.infer_type(zz)
    zz = relay.ir_pass.fuse_ops(zz)
    zz = relay.ir_pass.infer_type(zz)
    after = relay.ir_pass.infer_type(expected())
    assert relay.ir_pass.alpha_equal(zz, after)




def test_conv2d_fuse():
    """Test fusion case of conv2d"""
    def before(dshape):
        x = relay.var("x", shape=dshape)
        x = relay.add(x, relay.const(1, "float32"))
        y = relay.nn.conv2d(x, relay.var("w1"),
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            channels=16)
        # this is the next dominator.
        y1 = relay.add(relay.const(1, "float32"), y)
        y = relay.add(y, y1)
        # second path
        z2 = relay.nn.conv2d(y, relay.var("w2"),
                             kernel_size=(1, 1),
                             padding=(0,0),
                             channels=16)
        z3 = relay.nn.conv2d(y, relay.var("w3"),
                             kernel_size=(3, 3),
                             padding=(1,1),
                             channels=16)
        # add can only be fused to z1
        z = relay.add(z2, z3)
        return relay.Function(relay.ir_pass.free_vars(z), z)

    def expected(dshape):
        # segment 0
        x = relay.var("p0", shape=dshape)
        y = relay.add(x, relay.const(1, "float32"))
        f0 = relay.Function([x], y)
        # segment 1
        x = relay.var("p0", shape=dshape)
        w = relay.var("p1")
        y = relay.nn.conv2d(x, w,
                            kernel_size=(3, 3),
                            padding=(1, 1),
                            channels=16)
        y1 = relay.add(relay.const(1, "float32"), y)
        y = relay.add(y, y1)
        f1 = relay.Function([x, w], y)
        # segment 2
        x = relay.var("p0", shape=dshape)
        w = relay.var("p1")
        z2 = relay.nn.conv2d(x, w,
                             kernel_size=(3, 3),
                             padding=(1,1),
                             channels=16)
        f2 = relay.Function([x, w], z2)
        # segment 3
        x = relay.var("p0", shape=dshape)
        w = relay.var("p1")
        offset = relay.var("p2", shape=dshape)
        z3 = relay.nn.conv2d(x, w,
                             kernel_size=(1, 1),
                             padding=(0, 0),
                             channels=16)
        z3 = relay.add(z3, offset)
        f3 = relay.Function([x, w, offset], z3)
        # compose
        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        y = relay.Call(f1, [y, relay.var("w1")])
        z2 = relay.Call(f2, [y, relay.var("w3")])
        z3 = relay.Call(f3, [y, relay.var("w2"), z2])
        z = z3
        return relay.Function(relay.ir_pass.free_vars(z), z)

    dshape = (1, 16, 64, 64)
    z = before(dshape)
    z = relay.ir_pass.infer_type(z)
    zz = relay.ir_pass.fuse_ops(z, opt_level=2)
    zz = relay.ir_pass.infer_type(zz)
    after = relay.ir_pass.infer_type(expected(dshape))
    assert relay.ir_pass.alpha_equal(zz, after)



if __name__ == "__main__":
    test_fuse_simple()
    test_conv2d_fuse()
