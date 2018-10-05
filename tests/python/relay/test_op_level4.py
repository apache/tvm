import tvm
from tvm import relay


def test_cmp_type(op):
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
    test_cmp_type(relay.greater)
    test_cmp_type(relay.greater_equal)
    test_cmp_type(relay.less)
    test_cmp_type(relay.less_equal)
    test_cmp_type(relay.equal)
    test_cmp_type(relay.not_equal)
