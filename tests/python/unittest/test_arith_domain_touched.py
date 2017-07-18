import tvm

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
    assert a_domain_r[0].min.value == -1
    assert a_domain_r[0].extent.value == 100
    assert a_domain_r[1].min.value == -1
    assert a_domain_r[1].extent.name == 'm'

    a_domain_w = tvm.arith.DomainTouched(ir, a, False, True)
    assert a_domain_w[0].min.value == 0
    assert a_domain_w[0].extent.value == 100
    assert a_domain_w[1].min.value == 0
    assert a_domain_w[1].extent.name == 'm'

    a_domain_rw= tvm.arith.DomainTouched(ir, a, True, True)
    assert a_domain_rw[0].min.value == -1
    assert a_domain_rw[0].extent.value == 101
    assert a_domain_rw[1].min.value == -1
    assert isinstance(a_domain_rw[1].extent, tvm.expr.Add)
    assert a_domain_rw[1].extent.a.name == 'm'
    assert a_domain_rw[1].extent.b.value == 1

    b_domain_r = tvm.arith.DomainTouched(ir, b, True, False)
    assert b_domain_r
    assert b_domain_r[0].min.value == -1
    assert b_domain_r[0].extent.value == 100
    assert b_domain_r[1].min.value == 1
    assert b_domain_r[1].extent.name == 'm'

    b_domain_w = tvm.arith.DomainTouched(ir, b, False, True)
    assert isinstance(b_domain_w, tvm.container.Array)
    assert len(b_domain_w) == 0

if __name__ == "__main__":
    test_domain_touched()

