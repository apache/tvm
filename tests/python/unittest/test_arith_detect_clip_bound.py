import tvm

def test_basic():
    a = tvm.var("a")
    b = tvm.var("b")
    c = tvm.var("c")
    m = tvm.arith.DetectClipBound(tvm.all(a * 1 < b * 6,
                                          a - 1 > 0), [a])
    assert tvm.ir_pass.Simplify(m[1] - (b * 6 - 1)).value == 0
    assert m[0].value == 2
    m = tvm.arith.DetectClipBound(tvm.all(a * 1 < b * 6,
                                          a - 1 > 0), [a, b])
    assert len(m) == 0
    m = tvm.arith.DetectClipBound(tvm.all(a + 10 * c <= 20,
                                          b - 1 > 0), [a, b])
    assert tvm.ir_pass.Simplify(m[1] - (20 - 10 * c)).value == 0
    assert tvm.ir_pass.Simplify(m[2] - 2).value == 0


if __name__ == "__main__":
    test_basic()
