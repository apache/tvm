import tvm

def test_const_fold():
    def check(f, *args):
        x = f(*[tvm.const(x) for x in args])
        y = f(*args)
        if not isinstance(x, (tvm.expr.IntImm, tvm.expr.UIntImm)) or x.value != int(y):
            raise ValueError("check error: %s vs %s " % (x, y))

    check(lambda x, y: x + y, 3, 4)
    check(lambda x, y: x * y, 3, 12)
    check(lambda x, y: x * y - 10, 3, 12)
    check(lambda x, y: x - y % 10, 3, 12)
    check(lambda x, y: x // y + 10, 100, 12)
    check(lambda x, y: x & y + 10, 112, 128)
    check(lambda x, y: x > y, 112, 128)
    check(lambda x, y: x < y, 112, 128)
    check(lambda x, y: x <= y, 112, 128)
    check(lambda x, y: x >= y, 112, 128)
    check(lambda x, y: (x | y) ^ 10, 112, 128)


def test_const_fold2():
    x = tvm.var("x")
    assert (x + 0).same_as(x)
    assert (0 + x).same_as(x)
    assert (x - 0).same_as(x)
    assert (x % 1).value == 0
    assert (x * 1).same_as(x)
    assert (1 * x).same_as(x)
    assert isinstance((1 / x), tvm.expr.Div)

if __name__ == "__main__":
    test_const_fold()
    test_const_fold2()
