import tvm
from tvm import relay


def test_expand_dims_infer_type():
    ib = relay.ir_builder.IRBuilder()
    n, t, d = tvm.var("n"), tvm.var("t"), 100
    # let's mimic a batch of sequences
    x = ib.param("x", relay.ty.TensorType((n, t, d), "float32"))
    with ib.function(x) as func:
        ib.ret(relay.expand_dims(x, axis=2))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type()
    assert ftype.ret_type == relay.ty.TensorType(
        (n, t, 1, 100), "float32")


if __name__ == "__main__":
    test_expand_dims_infer_type()
