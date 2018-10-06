import tvm
from tvm import relay

def test_unary_identity():
    for op in [relay.zeros_like, relay.ones_like]:
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((8, 9, 4), "int32"))
        with ib.function(x) as func:
            ib.ret(op(x.var))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type()
        assert ftype.ret_type == relay.TensorType((8, 9, 4), "int32")

def test_clip_type():
    ib = relay.ir_builder.IRBuilder()
    a = ib.param("a", relay.TensorType((10, 4), "float32"))
    with ib.function(a) as func:
        ib.ret(relay.clip(a.var, 1., 4.))
    ib.ret(func)
    func = relay.ir_pass.infer_type(ib.env, func.to_func())
    ftype = func.checked_type()
    assert ftype.ret_type == relay.TensorType((10, 4), "float32")

if __name__ == "__main__":
    test_unary_identity()
    test_clip_type()
