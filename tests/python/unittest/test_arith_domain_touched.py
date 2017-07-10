import tvm, numpy

def test_domain_touched():
    i = tvm.var('i')
    j = tvm.var('j')
    n = tvm.convert(100)
    m = tvm.var('m')
    a = tvm.placeholder((n, m), name = 'a')
    b = tvm.placeholder((n, m), name = 'b')
    ir = tvm.make.For(
            i, 0, n, 0, 0,
            tvm.make.For(j, 0, m, 0, 0,
                tvm.make.Provide(
                    a.op,
                    0,
                    tvm.make.Call(b.dtype, 'b', [i - 1, j + 1], 3, b.op, 0) + 
                    tvm.make.Call(a.dtype, 'a', [i - 1, j - 1], 3, a.op, 0),
                    [i, j]
                )
            )
    )
    a_domain_r = tvm.arith.DomainTouched(ir, a, True, False)
    assert str(a_domain_r) == "[range(min=-1, ext=100),range(min=-1, ext=m)]"
    a_domain_w = tvm.arith.DomainTouched(ir, a, False, True)
    assert str(a_domain_w) == "[range(min=0, ext=100),range(min=0, ext=m)]"
    a_domain_rw= tvm.arith.DomainTouched(ir, a, True, True)
    assert str(a_domain_rw) == "[range(min=-1, ext=101),range(min=-1, ext=(m + 1))]"
    b_domain_r = tvm.arith.DomainTouched(ir, b, True, False)
    assert str(b_domain_r) == "[range(min=-1, ext=100),range(min=1, ext=m)]"
    b_domain_w = tvm.arith.DomainTouched(ir, b, False, True)
    assert str(b_domain_w) == "[]"

if __name__ == "__main__":
    test_domain_touched()

