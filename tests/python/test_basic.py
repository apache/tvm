import tvm

def test_const():
    x = tvm.const(1)
    assert x.type == 'int32'
    assert isinstance(x, tvm.expr.IntImm)

if __name__ == "__main__":
    test_const()
