import tvm
from tvm import relay

def test_fuse_simple():
    """Simple testcase."""
    x = relay.var("x", shape=(10, 20))
    y = relay.add(x, x)
    z = relay.exp(y)
    z = relay.ir_pass.infer_type(z)
    zz = relay.ir_pass.fuse_ops(z)
    zz = relay.ir_pass.fuse_ops(zz)
    zz = relay.ir_pass.infer_type(zz)
    zz.astext()


if __name__ == "__main__":
    test_fuse_simple()
