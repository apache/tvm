import tvm
from tvm import relay
from tvm.relay.ir_pass import fo_with_gradient
from tvm.relay.op import log, add, equal, subtract, multiply
from tvm.relay.ir_builder import IRBuilder, convert

def test_simple_gradient():
    b = IRBuilder()
    # f x = x * x
    ty = relay.TensorType(tvm.convert([]), "float32")
    x = relay.Var("x")
    f = relay.Function([relay.Param(x, ty)], ty, multiply(x, x), [])
    ret = fo_with_gradient(b.env, f)
    print(tvm.relay.expr.debug_print(b.env, ret))

def test_complex_gradient():
    b = IRBuilder()
    # f x = x * x + x
    ty = relay.TensorType(tvm.convert([]), "float32")
    x = relay.Var("x")
    f = relay.Function([relay.Param(x, ty)], ty, add(multiply(x, x), x), [])
    ret = fo_with_gradient(b.env, f)
    print(tvm.relay.expr.debug_print(b.env, ret))

if __name__ == "__main__":
    test_simple_gradient()
    #test_complex_gradient()
