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


def test_concatenate():
    """Test fusion case involving concat op and Tuple node"""

    def before(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        upsampled = relay.nn.upsampling(pooled, scale=2, layout="NCHW")
        concat = relay.concatenate((upsampled, x), axis=1)
        out = relay.add(concat, relay.const(1, "float32"))
        return relay.Function(relay.ir_pass.free_vars(out), out)

    def expected(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        f0 = relay.Function([x], pooled)

        p0 = relay.var("p0", shape=(dshape[0], dshape[1], dshape[2]//2, dshape[3]//2))
        p1 = relay.var("p1", shape=dshape)
        upsampled = relay.nn.upsampling(p0, scale=2, layout="NCHW")
        concat = relay.concatenate((upsampled, p1), axis=1)
        out = relay.add(concat, relay.const(1, "float32"))
        f1 = relay.Function([p0, p1], out)

        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        z = relay.Call(f1, [y, x])
        return relay.Function([x], z)

    dshape = (1, 16, 64, 64)
    z = before(dshape)
    z = relay.ir_pass.infer_type(z)
    zz = relay.ir_pass.fuse_ops(z, opt_level=0)
    assert not relay.ir_pass.free_vars(zz)
    zz = relay.ir_pass.fuse_ops(z, opt_level=2)
    zz = relay.ir_pass.infer_type(zz)
    assert not relay.ir_pass.free_vars(zz)
    after = relay.ir_pass.infer_type(expected(dshape))
    assert relay.ir_pass.alpha_equal(zz, after)


def test_tuple_root():
    """Test fusion case where Tuple node is the root in its group"""

    def before(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        upsampled = relay.nn.upsampling(pooled, scale=2, layout="NCHW")
        out = relay.Tuple((upsampled, x))
        return relay.Function(relay.ir_pass.free_vars(out), out)

    def expected(dshape):
        x = relay.var("x", shape=dshape)
        pooled = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
        f0 = relay.Function([x], pooled)

        p0 = relay.var("p0", shape=(dshape[0], dshape[1], dshape[2]//2, dshape[3]//2))
        p1 = relay.var("p1", shape=(dshape[0], dshape[1], dshape[2], dshape[3]))
        p1_copy = relay.copy(p1)
        upsampled = relay.nn.upsampling(p0, scale=2, layout="NCHW")
        out = relay.Tuple((upsampled, p1_copy))
        f1 = relay.Function([p0, p1], out)

        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        z = relay.Call(f1, [y, x])
        return relay.Function([x], z)

    dshape = (1, 16, 64, 64)
    z = before(dshape)
    z = relay.ir_pass.infer_type(z)
    zz = relay.ir_pass.fuse_ops(z, opt_level=0)
    assert not relay.ir_pass.free_vars(zz)
    zz = relay.ir_pass.fuse_ops(z, opt_level=2)
    zz = relay.ir_pass.infer_type(zz)
    assert not relay.ir_pass.free_vars(zz)
    after = relay.ir_pass.infer_type(expected(dshape))
    assert relay.ir_pass.alpha_equal(zz, after)


def test_tuple_strided_slice():
    """
    Test fusion case where the number of fields of tuple and
    the number of parameters to the function containing the tuple are different
    """

    def before(dshape):
        x = relay.var("x", shape=dshape)
        slice1 = relay.strided_slice(x, begin=[0, 0], end=[dshape[1]//2, dshape[1]], strides=[1,1])
        slice2 = relay.strided_slice(x, begin=[dshape[1]//2, 0], end=[dshape[0], dshape[1]], strides=[1,1])
        out = relay.Tuple((slice1, slice2))
        return relay.Function([x], out)

    def expected(dshape):
        x = relay.var("x", shape=dshape)
        slice1 = relay.strided_slice(x, begin=[0, 0], end=[dshape[1]//2, dshape[1]], strides=[1,1])
        slice2 = relay.strided_slice(x, begin=[dshape[1]//2, 0], end=[dshape[0], dshape[1]], strides=[1,1])
        out = relay.Tuple((slice1, slice2))
        f0 = relay.Function([x], out)

        x = relay.var("x", shape=dshape)
        y = relay.Call(f0, [x])
        return relay.Function([x], y)

    dshape = (64, 64)
    z = before(dshape)
    z = relay.ir_pass.infer_type(z)
    zz = relay.ir_pass.fuse_ops(z, opt_level=0)
    assert not relay.ir_pass.free_vars(zz)
    zz = relay.ir_pass.fuse_ops(z, opt_level=2)
    zz = relay.ir_pass.infer_type(zz)
    assert not relay.ir_pass.free_vars(zz)
    after = relay.ir_pass.infer_type(expected(dshape))
    assert relay.ir_pass.alpha_equal(zz, after)
    print(zz.astext())


def test_stop_fusion():
    def before(dshape):
        x = relay.var("x", shape=dshape)
        y = relay.add(x, relay.const(1, "float32"))
        y = relay.annotation.stop_fusion(y)
        z = relay.exp(y)
        return relay.Function([x], z)

    def expected(dshape):
        x = relay.var("p0", shape=dshape)
        y = relay.add(x, relay.const(1, "float32"))
        f1 = relay.Function([x], y)

        x = relay.var("p01", shape=dshape)
        y = relay.exp(x)
        f2 = relay.Function([x], y)

        x = relay.var("x", shape=dshape)
        y = relay.Call(f1, [x])
        z = relay.Call(f2, [y])
        return relay.Function([x], z)

    dshape = (10, 20)
    z = before(dshape)
    z = relay.ir_pass.infer_type(z)
    z = relay.ir_pass.fuse_ops(z)
    z = relay.ir_pass.infer_type(z)
    after = relay.ir_pass.infer_type(expected(dshape))
    assert relay.ir_pass.alpha_equal(z, after)


def test_fuse_myia_regression():
    def before(dshape, dtype):
        x = relay.var('x', shape=dshape, dtype=dtype)
        y = relay.var('y', shape=dshape, dtype=dtype)
        sb = relay.ScopeBuilder()
        with sb.if_scope(relay.op.greater(x, y)):
            sb.ret(relay.Function([], x))
        with sb.else_scope():
            sb.ret(relay.Function([], y))
        return relay.Function([x, y],
            relay.Call(sb.get(), []))

    def expected(dshape, dtype):
        x = relay.var('x', shape=dshape, dtype=dtype)
        y = relay.var('y', shape=dshape, dtype=dtype)
        sb = relay.ScopeBuilder()
        p1 = relay.var('p1', shape=dshape, dtype=dtype)
        p2 = relay.var('p2', shape=dshape, dtype=dtype)
        fused_gt = relay.Function([p1, p2],
            relay.op.greater(p1, p2))
        with sb.if_scope(fused_gt(x, y)):
            sb.ret(relay.Function([], x))
        with sb.else_scope():
            sb.ret(relay.Function([], y))
        return relay.Function([x, y],
            relay.Call(sb.get(), []))

    dshape = ()
    dtype = 'int64'
    f = before(dshape, dtype)
    f = relay.ir_pass.infer_type(f)
    f = relay.ir_pass.fuse_ops(f)
    after = relay.ir_pass.infer_type(expected(dshape, dtype))
    assert relay.ir_pass.alpha_equal(f, after)


if __name__ == "__main__":
    test_fuse_simple()
    test_conv2d_fuse()
    test_concatenate()
    test_tuple_root()
    test_tuple_strided_slice()
    test_stop_fusion()
    test_fuse_myia_regression()
