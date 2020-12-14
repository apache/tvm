import tvm
from tvm import tir
from tvm.ir.transform import PassContext

def test_scalar_add():
    a = tir.Var("a", "float32")
    b = tir.Var("b", "float32")
    c = a + b
    c = tir.call_intrin("float32", "tir.ret", c)
    c = tir.Evaluate(c)
    func = tir.PrimFunc([a, b], c)
    func = func.with_attr("global_symbol", "main")
    pass_ctx = PassContext.current()
    if pass_ctx.config.get("tir.noalias", True):
        func = func.with_attr("tir.noalias", True)
    mod = tvm.IRModule({'main': func})
    func = tvm.build(mod)
    out = func(1.0, 2.0)
    assert out == 3.0

if __name__ == "__main__":
    test_scalar_add()
