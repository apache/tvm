import tvm

def test_basic():
    a = tvm.var("a")
    b = tvm.var("b")
    m = tvm.arith.DetectLinearEquation(a * 4 + b * 6 + 7, a)
    assert m[1].value == 4
    assert tvm.ir_pass.Simplify(m[0] - (b * 6 + 7)).value == 0

    m = tvm.arith.DetectLinearEquation(a * 4 * (a+1) + b * 6 + 7, a)
    assert len(m) == 0

    m = tvm.arith.DetectLinearEquation(a * 4  + (a+1) + b * 6 + 7, a)
    assert m[1].value == 5
    assert tvm.ir_pass.Simplify(m[0] - (b * 6 + 7 + 1)).value == 0

    m = tvm.arith.DetectLinearEquation(a * b + 7, a)
    assert m[1] == b

    m = tvm.arith.DetectLinearEquation(b * 7, a)
    assert m[1].value == 0

if __name__ == "__main__":
    test_basic()
