import tvm

def csimplify(z):
    return tvm.ir_pass.CanonicalSimplify(
        tvm.make.Evaluate(z)).value

def test_simplify():
    x = tvm.var('n')
    z = x * 4  - x * 2
    zz = csimplify(z)
    assert zz.b.value == 2

    z = (x / 4) * 2  - (x / 4)
    zz = csimplify(z)
    assert zz.a == x and zz.b.value == 4

    z = (x % 4) * 3  + (x % 4)
    zz = csimplify(z)
    assert zz.b.value == 4
    zz = zz.a
    assert zz.a == x and zz.b.value == 4

    n = tvm.var('n')
    assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(n % (-1)), tvm.const(0, "int32"))
    assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(n % 1), tvm.const(0, "int32"))
    assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(n / 1), n)
    tvm.ir_pass.CanonicalSimplify(n / (-1))
    # This is not true in the current implementation
    #  assert tvm.ir_pass.Equal(tvm.ir_pass.CanonicalSimplify(n / (-1)),
    #                           tvm.ir_pass.CanonicalSimplify(-n))

def test_simplify_div():
    x = tvm.var('x')
    assert tvm.ir_pass.CanonicalSimplify((16+48*x)/16 - (1 + (x*3))).value == 0
    # (17+48*x)/16 is not simplifiable for arbitrary x because when 17+48*x<0
    # (17+48*x)/16 != 1+3*x
    r = tvm.ir_pass.CanonicalSimplify((17+48*x)/16)
    assert r.b.value == 16
    assert tvm.ir_pass.CanonicalSimplify(r.a - (17 + 48*x)).value == 0
    # However, when x >= 0, then 17+48*x >= 0 and (17+48*x)/16 can be simplified
    assert tvm.ir_pass.CanonicalSimplify((17+48*x)/16 - (1 + (x*3)), {x: tvm.Range(0,10)}).value == 0

    # Trying expressions that are not simplifiable for any values of the variables
    r = tvm.ir_pass.CanonicalSimplify((17+47*x)/16, {x: tvm.Range(0,10)})
    assert r.b.value == 16
    assert tvm.ir_pass.CanonicalSimplify(r.a - (17+47*x)).value == 0

    r = tvm.ir_pass.CanonicalSimplify((8*x - 17)/8, {x : tvm.Range(4,10)})
    assert tvm.ir_pass.CanonicalSimplify(r - (x-3)).value == 0

def test_simplify_mod():
    """Not yet working, mock design"""
    ib = tvm.ir_builder.create()
    n = tvm.var('n')
    j = tvm.var('j')
    A = ib.pointer("float32", name="A")
    with ib.for_range(0, 16, name="i") as i:
        A[i] = A[((n * 4 + j * 2) * 8 + i+1) % 16]
    body = ib.get()
    stmt = tvm.ir_pass.CanonicalSimplify(body)
    diff = tvm.ir_pass.CanonicalSimplify(stmt.body.value.index - (1 + i) % 16)
    assert diff.value == 0
    # if we can't prove that j+n*32 is non-negative, we can't prove that (j+n*32) % 16 is j%16
    index = tvm.ir_pass.CanonicalSimplify(
        (j + n * 32) % 16, {j: tvm.Range(0, 6)})
    assert index != j
    index = tvm.ir_pass.CanonicalSimplify(
        (j + n * 32) % 16, {j: tvm.Range(0, 6), n: tvm.Range(0, 10)})
    assert index == j

def test_simplify_minmax():
    x = tvm.var('x')
    e1 = tvm.max(x, 1) - tvm.max(x, 1)
    e1s = tvm.ir_pass.CanonicalSimplify(e1)
    assert e1s.value == 0

    e2 = tvm.min(x, 1) - tvm.min(x, 1)
    e2s = tvm.ir_pass.CanonicalSimplify(e2)
    assert e2s.value == 0

def test_mul():
    x = tvm.var('x')
    e = x * x - x * x
    es = tvm.ir_pass.CanonicalSimplify(e)
    assert es.value == 0

def test_modular():
    rx = tvm.var("rx")
    ry = tvm.var("ry")
    y = tvm.var("y")
    x = tvm.var("x")
    i32_const = lambda x: tvm.const(x, "int32")
    vmap = {rx: tvm.Range(i32_const(0), i32_const(3)),
            ry: tvm.Range(i32_const(0), i32_const(3)),
            y: tvm.Range(i32_const(0), i32_const(2)),
            x: tvm.Range(i32_const(0), i32_const(14))}
    idx = ry * 16 + rx + y * 16 + x
    z1 = tvm.ir_pass.CanonicalSimplify(idx // 16, vmap)
    z2 = tvm.ir_pass.CanonicalSimplify(idx % 16, vmap)
    assert tvm.ir_pass.CanonicalSimplify(z1 - (ry + y)).value == 0
    assert tvm.ir_pass.CanonicalSimplify(z2 - (rx + x)).value == 0

def test_const_propagation():
    x1 = tvm.const(4, "int32")
    x2 = x1 + 5
    assert isinstance(x2, tvm.expr.IntImm) and x2.value == 9
    x3 = x2 / 3
    assert isinstance(x3, tvm.expr.IntImm) and x3.value == 3
    x4 = x3 + 0.5
    assert isinstance(x4, tvm.expr.FloatImm) and x4.value == 3.5
    x5 = tvm.ceil(x4)
    assert isinstance(x5, tvm.expr.FloatImm) and x5.value == 4
    x6 = x5.astype('int')
    assert isinstance(x6, tvm.expr.IntImm) and x6.value == 4
    y = (tvm.round((tvm.const(6.5, 'float32') - 1) / 1.5) + 2).astype('int')
    assert isinstance(y, tvm.expr.IntImm) and y.value == 6


if __name__ == "__main__":
    test_simplify_div()
    test_simplify_mod()
    test_modular()
    test_simplify()
    test_mul()
    test_simplify_minmax()
    test_const_propagation()
