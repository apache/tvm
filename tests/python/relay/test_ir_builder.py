import numpy as np
from tvm.relay.expr import Let, Constant
from tvm.relay.ir_builder import IRBuilder

def test_let():
    b = IRBuilder()
    x = b.let('x', 1)
    b.ret(x)
    prog, _ = b.get()
    assert isinstance(prog, Let)
    var = prog.var
    value = prog.value
    assert var.name_hint == 'x'
    assert var == prog.body
    assert isinstance(value, Constant)
    assert value.data.asnumpy() == np.array(1)
    assert prog.value_type == None

if __name__ == "__main__":
    test_let()
