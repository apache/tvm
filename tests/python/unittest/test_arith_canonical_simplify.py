# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import tvm

class CanonicalChecker:
    def __init__(self):
        self.analyzer = tvm.arith.Analyzer()

    def verify(self, data, expected):
        res = self.analyzer.canonical_simplify(data)
        assert tvm.ir_pass.Equal(res, expected), "\ndata={}\nres={}\nexpected={}".format(data, res, expected)


def test_mul_sum_simplify():
    ck = CanonicalChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    ck.verify(2 + (3 * x + z + y + 1) * 4 + x,
              x * 13 + z * 4 + y * 4 +6)
    ck.verify(x * 3 - 4 * x + 1, 1 - x)
    ck.verify(y + x * 3 - 5 * x + 1 + y, y * 2 + 1 - x * 2)
    # trucdiv
    ck.verify((x + y + x + y * 3) / 2, y * 2 + x)
    ck.verify((x + y + x + y * 3) % 2, 0)

    # floordiv
    fld = tvm.floordiv
    flm = tvm.floormod
    ck.verify(flm(x + x + y * 3, 2), flm(y * 3, 2))
    ck.verify(fld(x + y + x + y * 3, 2), y * 2 + x)
    ck.verify(flm(x + y + x + y * 3, 2), 0)
    ck.verify(fld(x + x + y * 3, 2), fld(y * 3, 2) + x)


def test_split_index_simplify():
    ck = CanonicalChecker()
    x, y, z = tvm.var("x"), tvm.var("y"), tvm.var("z")

    # trucdiv
    # split div const
    ck.verify((x/3) *3 + x % 3, x)
    ck.verify((x/6) * 6 + ((x/3) % 2) * 3 + x % 3, x)
    ck.verify(((x % 16) / 2) * 2 / 4, (x % 16) / 4)
    ck.verify((x % 2) / 8, 0)
    ck.verify((x % 2) / 7, 0)
    ck.verify(((x % 16) / 2) * 2 / 6, (x % 16) / 6)

    # split mod const
    ck.verify((x * 8) % 16, (x % 2) * 8)
    ck.verify((x * 8) % 2, 0)

    # simplify then fold
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 1000))
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1000))
    ck.verify((x * 4 + y) / 2 * 2 + (x * 4 + y) % 2, x * 4 + y)
    # complex fold
    ck.verify((z * 9 + y) / 2 * 2 + (z * 9 + y) % 2, z * 9 + y)

    ck.analyzer.update(x, tvm.arith.ConstIntBound(-100, 1000), True)
    ck.analyzer.update(y, tvm.arith.ConstIntBound(-100, 1000), True)
    ck.verify((x * 4 + y) / 2 * 2 + (x * 4 + y) % 2, x * 4 + y)

    # floordiv
    fld = tvm.floordiv
    flm = tvm.floormod
    ck.verify(fld(x, 3) * 3 + flm(x, 3), x)
    ck.verify(fld(x, 6) * 6 + flm(fld(x, 3), 2) * 3 + flm(x, 3), x)
    ck.verify(fld(fld(flm(x, 16), 2) * 2, 4), fld(flm(x, 16), 4))
    ck.verify(fld(flm(x, 2), 8), 0)
    ck.verify(fld(flm(x, 2), 7), 0)
    ck.verify(fld(fld(flm(x, 16), 2) * 2, 6), fld(flm(x, 16), 6))

    # cannot simplify mixed case, unless we canonicalize into one mode.
    ck.verify((x/6) * 2 + fld(x,3) % 2, (x/6) * 2 + fld(x,3) % 2)


def test_div_simplify():
    ck = CanonicalChecker()
    x = tvm.var("x")

    # truc div
    ck.verify((16+48*x)/16, x*3 + 1)
    # (17+48*x)/16 is not simplifiable for arbitrary x because when 17+48*x<0
    # (17+48*x)/16 != 1+3*x
    ck.verify((17+48*x)/16, (x * 48 + 17) / 16)
    # However, when x >= 0, then 17+48*x >= 0 and (17+48*x)/16 can be simplified
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 10))
    ck.verify((17+48*x)/16, x * 3 + 1)
    # Trying expressions that are not simplifiable for any values of the variables
    ck.verify((17+47*x)/16, (x * 47 + 17) / 16)

    # floordiv
    fld = tvm.floordiv
    ck.analyzer.update(x, tvm.arith.ConstIntBound(-1000, 10000), True)
    ck.verify(fld(16+48*x, 16), x*3 + 1)
    ck.verify(fld(17+48*x, 16), x * 3 + 1)
    ck.verify(fld(17+47*x, 16), fld(x * 47 + 17, 16))


def test_floormod_simplify():
    ck = CanonicalChecker()
    flm = tvm.floormod
    x, y = tvm.var("x"), tvm.var("y")
    ck.verify(flm(flm((x*4) + y  - 466036, 24528) - 24512,  16),
              flm((x*4) + y  + 12, 16))



def test_canonical_mixed():
    ck = CanonicalChecker()
    x = tvm.var("x")
    z = tvm.const(3, "int32")
    ck.verify(x / (z*z) - x / (z*z), 0)
    ck.verify(x / (z+z) - x / (z+z), 0)
    ck.verify(x - 2 < 3, x < 5)
    ck.verify(tvm.max(x, 1) - tvm.max(x, 1), 0)
    ck.verify(tvm.min(x, 1) - tvm.min(x, 1), 0)
    ck.verify(x * x - x * x, 0)

    fld = tvm.floordiv
    ck.verify(fld(x, (z*z)) - fld(x, (z*z)), 0)
    ck.verify(fld(x, (z+z)) - fld(x, (z+z)), 0)


def test_reduce_combiner_simplify():
    ck = CanonicalChecker()
    dummy = tvm.var('dummy')
    comm_reducer = tvm.comm_reducer
    prod = comm_reducer(lambda x, y: x*y, lambda t0: tvm.const(1, t0))

    sum_or_prod = comm_reducer(
        lambda x, y: tvm.expr.Select(dummy < 0,
                                     x + y, x*y),
        lambda t0: tvm.expr.Select(dummy < 0,
                                   tvm.const(0, t0), tvm.const(1, t0)))
    sum_and_prod = comm_reducer(
        lambda x, y: (x[0] + y[0],
                      x[1]*y[1]),
        lambda t0, t1: (tvm.const(0, t0),
                        tvm.const(5, t0) - tvm.const(4, t0)))
    some_reducer1 = comm_reducer(
        lambda x, y: (x[0] + y[0],
                      x[0] + y[0] + x[1] + y[1],
                      x[0]*y[2] + y[0]*x[2],
                      x[1] + y[2],
                    4.0),
        lambda t0, t1, t2, t3, t4: (tvm.const(0, t0),
                                    tvm.const(1, t1),
                                    tvm.const(2, t2),
                                    tvm.const(3, t3),
                                    tvm.const(4, t4)))

    k = tvm.reduce_axis((0, 10), name="k")
    A = tvm.placeholder((10,), name='A')
    # Test that SimplifyCombiner makes use of vranges
    ck.analyzer.update(dummy, tvm.arith.ConstIntBound(-10, -4))
    ck.verify(sum_or_prod(A[k], k), tvm.sum(A[k], k))
    ck.analyzer.update(dummy, tvm.arith.ConstIntBound(5, 9), True)
    ck.verify(sum_or_prod(A[k], k), prod(A[k], k))
    ck.analyzer.update(dummy, tvm.arith.ConstIntBound(-10, 100), True)
    ck.verify(sum_and_prod((A[k], A[10-k]), k)[0], tvm.sum(A[k], k))
    ck.verify(sum_and_prod((A[k], A[10-k]), k)[1], prod(A[10-k], k))

    reference_simplified_sources = [[A[0]],
                                    [A[0], A[1]],
                                    [A[0], A[2]],
                                    [A[0], A[1], A[2], A[3]],
                                    [A[4]]]
    for j in range(5):
        # Here we use the j-th component of the result, so only it and the components it
        # depends on are left.
        simplified = ck.analyzer.canonical_simplify(
            some_reducer1((A[0], A[1], A[2], A[3], A[4]), k)[j])

        # Check that the remaining components are the expected ones.
        for lhs, rhs in zip(simplified.source, reference_simplified_sources[j]):
            assert tvm.ir_pass.Equal(lhs, rhs)

    # Test that components with side effects are not removed
    side_effect = lambda *xs: tvm.make.Call("int32", "dummy", xs, tvm.expr.Call.Intrinsic, None, 0)
    ck.verify(sum_and_prod((A[k], side_effect(A[10-k])), k)[0],
             sum_and_prod((A[k], side_effect(A[10-k])), k)[0])
    ck.verify(sum_and_prod((side_effect(A[k]), A[10-k]), k)[0],
              tvm.sum(side_effect(A[k]), k))


def test_reduce_simplify():
    ck = CanonicalChecker()
    k = tvm.reduce_axis((0, 10), name="k")
    j = tvm.reduce_axis((-5, 3), name="j")
    A = tvm.placeholder((10,), name='A')
    ck.verify(tvm.sum(tvm.expr.Select(k + j < 12, k + j, 0), [k, j]),
              tvm.sum(k + j, [k, j]))
    ck.verify(tvm.sum(A[3], []), A[3])
    # The rule below is not typical, removed for now
    ck.verify(tvm.sum(k / 10, k), tvm.sum(tvm.const(0, "int32"), k))


def test_simplify_if_then_else():
    ck = CanonicalChecker()
    x = tvm.var("x")
    y = tvm.var("y")
    # simplification that takes condition into account.
    res = tvm.if_then_else((x * 4 + y) >= 466036,
                           tvm.if_then_else(24512 <= ((((x*4) + y) - 466036) % 24528),
                                            (((((x*4) + y)  - 466036) % 24528) -24512) % 16,
                                            x), y)

    res2 = tvm.if_then_else((x * 4) >= 466036 - y,
                           tvm.if_then_else(24512 <= ((((x*4) + y) - 466036) % 24528),
                                            (((((x*4) + y)  - 466036) % 24528) -24512) % 16,
                                            x), y)
    expected = tvm.if_then_else(
        tvm.expr.LE(466036, (x * 4 + y)),
        tvm.if_then_else(tvm.expr.LE(24512, ((((x*4) + y) - 4) % 24528)),
                         (((x*4) + y)  - 4) % 16,
                         x), y)
    ck.verify(res, expected)
    ck.verify(res2, expected)
    # can only simplify if condition
    res = tvm.expr.Select(tvm.all(x >= -1, y >= 0), (x + y + 100) % 3, (x + 100) % 3)
    expected = tvm.expr.Select(tvm.all(x >= -1, y >= 0), (x + y + 1) % 3, (x + 100) % 3)
    ck.verify(res, ck.analyzer.canonical_simplify(expected))

    res = tvm.expr.Select(x >= 10,
                          tvm.if_then_else(x / 3 > 2, x, 0), 0)
    expected = tvm.expr.Select(x >= 10, x, 0)
    ck.verify(res, ck.analyzer.canonical_simplify(expected))

    res = tvm.expr.Select(x >= 10,
                          tvm.if_then_else(x / 3 < 2, x, 0), 0)
    ck.verify(res, 0)


def test_complex_cases():
    ck = CanonicalChecker()
    x = tvm.var("x")
    y = tvm.var("y")
    res2 = (((((((((((x*128) + y) % 1296)/36)*2) + 1)/2)*36) +
              ((((((x*128) + y) % 36)*2) + 1)/2))
             - (((x*128) + y) % 1296)) + 1)
    ck.analyzer.update(x, tvm.arith.ConstIntBound(0, 5))
    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 127))
    ck.verify(res2, 1)

    ck.analyzer.update(y, tvm.arith.ConstIntBound(0, 1024), True)
    res3 = ((((((((((x*1024) + y)/65536) + ((((x*1024) + y) % 65536)/256))
                 + ((((x*1024) + y) % 256)/16)) + (((x*1024) + y) % 16)) - (y/256)) -
              ((y % 256)/16))  - (y % 16)) - (x*4))
    ck.verify(res3, ((((x*1024) + y)/256) - (y/256)) - (x*4))




if __name__ == "__main__":
    test_floormod_simplify()
    test_mul_sum_simplify()
    test_simplify_if_then_else()
    test_div_simplify()
    test_reduce_simplify()
    test_reduce_combiner_simplify()

    test_split_index_simplify()
    test_canonical_mixed()
    test_complex_cases()
