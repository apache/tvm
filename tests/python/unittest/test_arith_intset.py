import tvm

def test_basic():
    s = tvm.arith.intset_interval(2, 3)
    assert s.min().value == 2
    assert s.max().value == 3

def test_deduce():
    a = tvm.Var('a')
    b = tvm.Var('b')
    c = tvm.Var('c')
    d = tvm.Var('d')

    b_s = tvm.arith.intset_interval(2, 3)
    c_s = tvm.arith.intset_interval(10, 15)
    d_s = tvm.arith.intset_interval(-3, -1)

    e0 = (-b)*a+c-d
    res0 = tvm.arith.DeduceBound(a, e0>=0, {b: b_s, c: c_s, d: d_s})
    ans0 = (d-c)/(-b)+(-1)
    assert str(tvm.ir_pass.Simplify(res0.max())) == str(ans0)

    e1 = (a*4+b < c)
    res1 = tvm.arith.DeduceBound(a, e1, {b: b_s, c: c_s, d: d_s})
    ans1 = (c-b)/4+(-2)
    assert str(tvm.ir_pass.Simplify(res1.max())) == str(ans1)

def test_check():
    a = tvm.Var('a')
    b = tvm.Var('b')
    c = tvm.Var('c')
    d = tvm.Var('d')

    b_s = tvm.arith.intset_interval(2, 3)
    c_s = tvm.arith.intset_interval(5, 7)
    d_s = tvm.arith.intset_interval(-3, -1)

    # no compare operator
    res1 = tvm.arith.DeduceBound(a, a+b, {b: b_s})
    assert res1.is_nothing()

    # multiple compare operators
    res2 = tvm.arith.DeduceBound(a, a+b>3>c , {b: b_s, c: c_s})
    assert res1.is_nothing()

    # multiple target variable
    res2 = tvm.arith.DeduceBound(a, a*2-a>b, {b: b_s})
    assert res1.is_nothing()

if __name__ == "__main__":
    test_basic()
    test_deduce()
    test_check()
