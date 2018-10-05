import tvm
from tvm import relay


def test_cmp_type():
    for op in (relay.greater,
               relay.greater_equal,
               relay.less,
               relay.less_equal,
               relay.equal,
               relay.not_equal):
        ib = relay.ir_builder.IRBuilder()
        x = ib.param("x", relay.TensorType((10, 4), "float32"))
        y = ib.param("y", relay.TensorType((5, 10, 1), "float32"))
        with ib.function(x, y) as func:
            ib.ret(op(x.var, y.var))
        ib.ret(func)
        func = relay.ir_pass.infer_type(ib.env, func.to_func())
        ftype = func.checked_type()
        assert ftype.ret_type == relay.TensorType((5, 10, 4), "uint1")


if __name__ == "__main__":
    test_cmp_type()
