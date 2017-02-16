import tvm

def test_basic():
    s = tvm.arith.intset_range(2, 3)
    assert s.min().value == 2
    assert s.max().value == 3

def test_deduce():
    a = tvm.Var('a')
    b = tvm.Var('b')
    c = tvm.Var('c')
    d = tvm.Var('d')

    b_s = tvm.arith.intset_range(2, 3)
    c_s = tvm.arith.intset_range(5, 7)
    d_s = tvm.arith.intset_range(-3, -1)

    e0 = (-b)*a+c-d*b
    res = tvm.arith.DeduceBound(a, e0, {b: b_s, c: c_s, d: d_s})
    ans = ((0+d*b)-c)/(-b)-1
    print(res)
    print(ans)
    # assert print(res.max() == ans) # will print False

if __name__ == "__main__":
    test_basic()
    test_deduce()
