import tvm
from tvm import tir

def test_scalar_add():
    a = tir.Var("a", "float32") 
    b = tir.Var("b", "float32") 
    c = a + b
    c = tir.call_intrin("float32", "tir.ret", c) 
    c = tir.Evaluate(c)
    func = tir.PrimFunc([a, b], c)
    mod = tvm.IRModule({'add': func})
    func = tvm.build(mod['add'])
    out = func(1.0, 2.0)
    assert out == 3.0

if __name__ == "__main__":
    test_scalar_add()
