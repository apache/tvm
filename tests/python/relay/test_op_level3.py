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
        ftype = func.checked_type
        assert ftype.ret_type == relay.TensorType((8, 9, 4), "int32")
