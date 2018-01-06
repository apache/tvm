import tvm

def test_basic():
    s = tvm.arith.intset_interval(2, 3)
    assert s.min().value == 2
    assert s.max().value == 3

def test_vector():
    base = 10
    stride = 3
    lanes = 2
    s = tvm.arith.intset_vector(tvm.make.Ramp(base, stride, lanes))
    assert s.min().value == base
    assert s.max().value == base + stride * lanes - 1

def test_deduce():
    a = tvm.var('a')
    b = tvm.var('b')
    c = tvm.var('c')
    d = tvm.var('d')

    b_s = tvm.arith.intset_interval(2, 3)
    c_s = tvm.arith.intset_interval(10, 15)
    d_s = tvm.arith.intset_interval(-3, -1)

    e0 = (-b)*a+c-d
    res0 = tvm.arith.DeduceBound(a, e0>=0, {b: b_s, c: c_s, d: d_s}, {})
    ans0 = ((d - c) /(b*-1))
    assert str(tvm.ir_pass.Simplify(res0.max())) == str(ans0)

    e1 = (a*4+b < c)
    res1 = tvm.arith.DeduceBound(a, e1, {b: b_s, c: c_s, d: d_s}, {})
    ans1 = (((c - b) + -1)/4) 
    assert str(tvm.ir_pass.Simplify(res1.max())) == str(ans1)

    e2 = (tvm.max(5, a * 4) < 0)
    res2 = tvm.arith.DeduceBound(a, e2, {b: b_s, c: c_s, d: d_s}, {})
    assert str(res2.max()) == "neg_inf"
    assert str(res2.min()) == "pos_inf"

    e3 = (-b)+a*c-d
    res3 = tvm.arith.DeduceBound(a, e3>=0, {b: b_s, c: c_s, d: d_s}, {b: b_s, d: d_s})
    ans3 = 2/c+1
    assert str(tvm.ir_pass.Simplify(res3.min())) == str(ans3)

def test_check():
    a = tvm.var('a')
    b = tvm.var('b')
    c = tvm.var('c')
    d = tvm.var('d')

    b_s = tvm.arith.intset_interval(2, 3)
    c_s = tvm.arith.intset_interval(5, 7)
    d_s = tvm.arith.intset_interval(-3, -1)

    # no compare operator
    res1 = tvm.arith.DeduceBound(a, a+b, {b: b_s}, {})
    assert res1.is_nothing()

    # multiple compare operators
    res2 = tvm.arith.DeduceBound(a, (a+b>3)>c , {b: b_s, c: c_s}, {})
    assert res2.is_nothing()

    # multiple target variable
    res2 = tvm.arith.DeduceBound(a, a*2-a>b, {b: b_s}, {})
    assert res2.is_nothing()

def test_deduce_relax():
    def test_pos(a1, a2):
        a = tvm.var('a')
        b = tvm.var('b')
        b_s = tvm.arith.intset_interval(a1, a2)
        e0 = b + a*4
        res1 = tvm.arith.DeduceBound(a, e0<17, {b: b_s}, {b: b_s})
        assert tvm.ir_pass.Simplify((res1.max() * 4 + b_s.max()) < 17).value == 1

        res1 = tvm.arith.DeduceBound(a, e0>17, {b: b_s}, {b: b_s})
        assert (tvm.ir_pass.Simplify((res1.min() * 4 + b_s.min()) > 17)).value == 1

        res1 = tvm.arith.DeduceBound(a, e0<=17, {b: b_s}, {b: b_s})
        assert (tvm.ir_pass.Simplify((res1.max() * 4 + b_s.max()) <= 17)).value == 1
      
        res1 = tvm.arith.DeduceBound(a, e0>=17, {b: b_s}, {b: b_s})
        assert (tvm.ir_pass.Simplify((res1.min() * 4 + b_s.min()) >= 17)).value == 1
   
    def test_neg(a1, a2):
        a = tvm.var('a')
        b = tvm.var('b')
        b_s = tvm.arith.intset_interval(a1, a2)
        e0 = b + a*(-4)
        res1 = tvm.arith.DeduceBound(a, e0<17, {b: b_s}, {b: b_s})
        assert tvm.ir_pass.Simplify((res1.min() * (-4) + b_s.min()) < 17).value == 1

        res1 = tvm.arith.DeduceBound(a, e0>17, {b: b_s}, {b: b_s})
        assert (tvm.ir_pass.Simplify((res1.max() * (-4) + b_s.max()) > 17)).value == 1

        res1 = tvm.arith.DeduceBound(a, e0<=17, {b: b_s}, {b: b_s})
        assert (tvm.ir_pass.Simplify((res1.min() * (-4) + b_s.min()) <= 17)).value == 1

        res1 = tvm.arith.DeduceBound(a, e0>=17, {b: b_s}, {b: b_s})
        assert (tvm.ir_pass.Simplify((res1.max() * (-4) + b_s.max()) >= 17)).value == 1
       
    test_pos(0, 4)
    test_pos(1, 5)
    test_pos(2, 6)
    test_neg(0, 4)
    test_neg(1, 5)
    test_neg(2, 6)

if __name__ == "__main__":
    test_basic()
    test_vector()
    test_deduce()
    test_check()
    test_deduce_relax()
