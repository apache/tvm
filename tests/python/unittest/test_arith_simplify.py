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
    index = tvm.ir_pass.CanonicalSimplify(
        (j + n * 32) % 16, {j: tvm.Range(0, 6)})
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

if __name__ == "__main__":
    test_simplify_mod()
    test_modular()
    test_simplify()
    test_mul()
    test_simplify_minmax()
