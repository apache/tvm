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

if __name__ == "__main__":
    test_simplify()
