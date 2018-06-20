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


def test_modular():
    rx = tvm.var("rx")
    ry = tvm.var("ry")
    y = tvm.var("y")
    x = tvm.var("x")
    vmap = {rx: tvm.Range(tvm.const(0), tvm.const(3)),
            ry: tvm.Range(tvm.const(0), tvm.const(3)),
            y: tvm.Range(tvm.const(0), tvm.const(2)),
            x: tvm.Range(tvm.const(0), tvm.const(14))}
    idx = ry * 16 + rx + y * 16 + x
    z1 = tvm.ir_pass.CanonicalSimplify(idx // 16, vmap)
    z2 = tvm.ir_pass.CanonicalSimplify(idx % 16, vmap)
    assert tvm.ir_pass.CanonicalSimplify(z1 - (ry + y)).value == 0
    assert tvm.ir_pass.CanonicalSimplify(z2 - (rx + x)).value == 0





if __name__ == "__main__":
    test_simplify_mod()
    test_modular()
    test_simplify()
