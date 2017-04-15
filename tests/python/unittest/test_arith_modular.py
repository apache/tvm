import tvm

def test_basic():
    a = tvm.var()
    b = tvm.var()
    m = tvm.arith.EvalModular(a * 4 + b * 6 + 7)
    assert m.coeff == 2
    assert m.base == 1

    m = tvm.arith.EvalModular((a * 4 + 1) * (b * 8 + 3))
    assert m.coeff == 4
    assert m.base == 3

    m = tvm.arith.EvalModular((a * 4 + 1) / (b * 8 + 3))
    assert m.coeff == 1
    assert m.base == 0

    m = tvm.arith.EvalModular((a * 4 + 1) * (b * 8 / 4))
    assert m.coeff == 2
    assert m.base == 0

    m = tvm.arith.EvalModular((a * 12 + 1) - (b * 3 * 7  + 2))
    assert m.coeff == 3
    assert m.base == 2


    m = tvm.arith.EvalModular(a * 12 + tvm.min(b * 3 * 7, 2))
    assert m.coeff == 1
    assert m.base == 0

if __name__ == "__main__":
    test_basic()
