import tvm

class RewriteChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, expected):
        res = self.analyzer.rewrite_simplify(data)
        assert tvm.ir_pass.Equal(res, expected), "data={}, res={}, expected={}".format(
            data, res, expected)


def test_vector_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    # Add rules
    ck.verify(tvm.expr.Ramp(x, 1, 4) + tvm.expr.Ramp(y, 2, 4),
              tvm.expr.Ramp(x + y, 3, 4))
    ck.verify(tvm.expr.Ramp(x, 1, 2) + y,
              tvm.expr.Ramp(x + y, 1, 2))
    ck.verify(y + tvm.expr.Ramp(x, 1, 2) ,
              tvm.expr.Ramp(y + x, 1, 2))
    ck.verify(y.astype("int32x2") + x.astype("int32x2"),
              (y + x).astype("int32x2"))
    # Sub rules
    ck.verify(tvm.expr.Ramp(x, 4, 4) - tvm.expr.Ramp(y, 2, 4),
              tvm.expr.Ramp(x - y, 2, 4))
    ck.verify(tvm.expr.Ramp(x, 1, 2) - y,
              tvm.expr.Ramp(x - y, 1, 2))
    ck.verify(y - tvm.expr.Ramp(x, 1, 2) ,
              tvm.expr.Ramp(y - x, -1, 2))
    ck.verify(y.astype("int32x2") - x.astype("int32x2"),
              (y - x).astype("int32x2"))

    # Mul rules
    ck.verify(y.astype("int32x2") * x.astype("int32x2"),
              (y * x).astype("int32x2"))
    ck.verify(tvm.expr.Ramp(x, 4, 4) * 2,
              tvm.expr.Ramp(x * 2, 8, 4))
    ck.verify(2 * tvm.expr.Ramp(x, 4, 4),
              tvm.expr.Ramp(x * 2, 8, 4))

    ## Div rules
    ck.verify(y.astype("int32x2") / x.astype("int32x2"),
              (y / x).astype("int32x2"))
    ck.verify(tvm.expr.Ramp(x, 4, 4) / 2,
              tvm.expr.Ramp(x/ 2, 2, 4))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(tvm.expr.Ramp(x * 8 + 1, 1, 4) / 8,
              (x).astype("int32x4"))
    ck.verify(tvm.expr.Ramp(x * 8 + 15, 1, 4) / 8,
              tvm.expr.Ramp(x * 8 + 15, 1, 4) / 8)

    ## Mod rules
    ck.verify(y.astype("int32x2") % x.astype("int32x2"),
              (y % x).astype("int32x2"))
    ck.verify(tvm.expr.Ramp(x, 4, 4) % 2,
              tvm.expr.Broadcast(x % 2, 4))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(tvm.expr.Ramp(x * 8 + 1, 1, 4) % 8,
              tvm.expr.Ramp(1, 1, 4))
    ck.verify(tvm.expr.Ramp(x * 8 + 1, 15, 4) % 8,
              tvm.expr.Ramp(1, 15, 4) % 8)



def test_select_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    # Add rules
    ck.verify(tvm.expr.Select(x > 0, y, 0) + tvm.expr.Select(x > 0, 1, z),
              tvm.expr.Select(x > 0, y + 1, z))
    ck.verify(tvm.expr.Select(x > 0, y, 1) - tvm.expr.Select(x > 0, 1, z),
              tvm.expr.Select(x > 0, y + (-1), 1 - z))
    ck.verify(tvm.expr.Select(x > 0, y, z) - y,
              tvm.expr.Select(x > 0, 0, z - y))
    ck.verify(tvm.expr.Select(x > 0, y, z) - z,
              tvm.expr.Select(x > 0, y - z, 0))


def test_add_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(x + (y - x), y)
    ck.verify(x - (y + 1) + (y + 1), x)
    ck.verify((x - 10) + (10 - z), x - z)
    ck.verify((x - y) + (z - x), z - y)

    ck.verify(tvm.min(x, y - z) + z, tvm.min(x + z, y))
    ck.verify(tvm.min(x - z, y) + z, tvm.min(x, y + z))
    ck.verify(tvm.max(x, y - 10) + 10, tvm.max(x + 10, y))
    ck.verify(tvm.max(x - 11, y) + 11, tvm.max(x, y + 11))

    ck.verify(tvm.max(x, y * 2) + tvm.min(x, y * 2), x + y * 2);
    ck.verify(tvm.min(x, y * 2) + tvm.max(x, y * 2), x + y * 2);

    ck.verify(tvm.max(x, y + 2) + (-2), tvm.max(x + (-2), y));
    ck.verify(tvm.min(x, y + 2) + (-2), tvm.min(x + (-2), y));
    ck.verify(tvm.min(x + 2, y + 3) + (-2), tvm.min(x, y + 1));

    ck.verify(x * y + x * 10, x * (y + 10))
    ck.verify(y * x + x * 10, x * (y + 10))
    ck.verify(y * x + 10 * x, x * (y + 10))
    ck.verify(x * y + 10 * x, x * (y + 10))

    ck.verify(y * (x % 8) + 10 * (x % 8), (x % 8) * (y + 10))
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify((x / 8) * 8 + x % 8, x)

    # canonicalization
    ck.verify(x + 2 + 3 + 4 + x, x * 2 + 9);
    ck.verify(x + 2 + 3 + 4 + x * 3, x * 4 + 9);

    # conservative bound
    try:
        ck.analyzer.update(x, tvm.arith.ConstIntBound(-1, 1000), override=True)
        ck.verify((x / 8) * 8 + x % 8, x)
        raise RuntimeError("bad")
    except AssertionError:
        pass


def test_sub_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(x + y - y, x)
    ck.verify(x + y - x, y)
    ck.verify(x - (y + x), 0 - y)
    ck.verify(x - (x + y), 0 - y)

    ck.verify(tvm.min(x, y) - x, tvm.min(0, y - x))
    ck.verify(tvm.min(x, y) - y, tvm.min(x - y, 0))
    ck.verify(tvm.max(x, y) - x, tvm.max(0, y - x))
    ck.verify(tvm.max(x, y) - y, tvm.max(x - y, 0))

    ck.verify(x - tvm.min(x, y), tvm.max(0, x - y))
    ck.verify(y - tvm.min(x, y), tvm.max(y - x, 0))
    ck.verify(x - tvm.max(x, y), tvm.min(0, x - y))
    ck.verify(y - tvm.max(x, y), tvm.min(y - x, 0))

    # mul co-efficient foldng
    ck.verify(x - x, 0)
    ck.verify(x * y - x, x * (y + (-1)))
    ck.verify(x * y - 10 * x, x * (y + (-10)))
    ck.verify(y * x - x * z, x * (y - z))
    ck.verify(y * x - z * x, x * (y - z))

    ck.verify(x + 10 - 20, x + (-10))

    # 4-operands pattern
    ck.verify((x + y) - (x + z), y - z)
    ck.verify((y + x) - (x + z), y - z)
    ck.verify((x + y) - (z + x), y - z)
    ck.verify((y + x) - (z + x), y - z)

    ck.verify(tvm.min(x + y, z) - x, tvm.min(y, z - x))
    ck.verify(tvm.min(y + x, z) - x, tvm.min(y, z - x))
    ck.verify(tvm.min(z, x + y) - x, tvm.min(z - x, y))
    ck.verify(tvm.min(z, y + x) - x, tvm.min(z - x, y))

    ck.verify(x - tvm.min(x + y, z), tvm.max(0 - y, x - z))
    ck.verify(x - tvm.min(y + x, z), tvm.max(0 - y, x - z))
    ck.verify(x - tvm.min(z, x + y), tvm.max(x - z, 0 - y))
    ck.verify(x - tvm.min(z, y + x), tvm.max(x - z, 0 - y))

    ck.verify(tvm.min(x, y) - tvm.min(y, x), 0)
    ck.verify(tvm.max(x, y) - tvm.max(y, x), 0)
    ck.verify(tvm.min(x, y) - tvm.min(x + 10, y + 10), -10)
    ck.verify(tvm.min(x + 10, y + 1) - tvm.min(x, y - 9), 10)

    # div pattern
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.verify(x - (x / 3) * 3, x % 3)
    ck.verify((x + 5) / 3 - x / 3, (((x + 2) % 3) + 5)/ 3)


def test_mul_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    ck.verify((x + 2) * 3, x * 3 + 6)
    ck.verify((x * 2) * 3, x * 6)
    ck.verify(tvm.min(x, y) * tvm.max(x, y), x * y)
    ck.verify(tvm.max(x, y) * tvm.min(x, y), x * y)
    ck.verify((x - y) * (-2), (y - x) * 2)


def test_div_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(z, tvm.arith.ConstIntBound(0, 1000), override=True)

    ck.verify(x / 2 / 3, x / 6)
    ck.verify((x / 2 + 1) / 3, (x + 2) / 6)
    ck.verify(x * 2 / 4, x / 2)
    ck.verify(x * 4 / 2, x * 2)

    ck.verify((x * 4 + y) / 2, x * 2 + y / 2)
    ck.verify(tvm.min(x * 6, y) / 2, tvm.min(x * 3, y / 2))
    ck.verify(tvm.max(x * 6, y) / 2, tvm.max(x * 3, y / 2))

    ck.verify((y + x * 4) / 2, y / 2 + x * 2)
    ck.verify(tvm.min(y, x * 6) / 2, tvm.min(y / 2, x * 3))
    ck.verify(tvm.max(y, x * 6) / 2, tvm.max(y / 2, x * 3))

    # 3-operands
    ck.verify((x * 6 + y + z) / 2, x * 3 + (y + z) / 2)
    ck.verify((x * 6 - y + (y + 3)) / 2, x * 3 + 1)
    ck.verify((x * 6 + (y + 3) - y) / 2, x * 3 + 1)
    ck.verify((y + x * 6 + z) / 2, x * 3 + (y + z) / 2)
    ck.verify((x + 4) / 2, x / 2 + 2)

    ck.verify((x + y) / x, y / x + 1)
    ck.verify((y + x) / x, y / x + 1)
    ck.verify(((x + y) + z) / x, (y + z) / x + 1)
    ck.verify(((y + x) + z) / x, (y + z) / x + 1)
    ck.verify((y + (x + z)) / x, (y + z) / x + 1)
    ck.verify((y + (z + x)) / x, (y + z) / x + 1)

    ck.verify((x * y) / y, x)
    ck.verify((y * x) / y, x)

    ck.verify((x * z + y) / z, x + y / z)
    ck.verify((z * x + y) / z, x + y / z)
    ck.verify((y + x * z) / z, y / z + x)
    ck.verify((y + z * x) / z, y / z + x)


def test_mod_index_simplify():
    ck = RewriteChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000), override=True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000), override=True)

    ck.verify(x * 10 % 2, 0)
    ck.verify((x * 10 + y) % 2, y % 2)
    ck.verify((x + 10) % 2, x % 2)
    ck.verify((x + y * 10) % 2, x % 2)
    ck.verify((x* 10 + 1 + y * 2 + 2) % 2, 1)


if __name__ == "__main__":
    test_mod_index_simplify()
    test_vector_simplify()
    test_add_index_simplify()
    test_sub_index_simplify()
    test_mul_index_simplify()
    test_div_index_simplify()
    test_select_simplify()
