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
import itertools

import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def test_nested_byte_pointer_offset():
    @I.ir_module
    class Before:
        @T.function
        def main(data: T.handle("float32")):
            T.func_attr({"target": T.target("llvm")})
            T.call_extern(
                "consume",
                T.ptr_byte_offset(T.ptr_byte_offset(data, 8, ty=data.ty), 12, ty=data.ty),
                ty="void",
            )

    After = tvm_tirx.tirx.transform.LowerIntrin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)
    for target in ["c"] + (["llvm"] if env.has_llvm() else []):
        tvm_tirx.tirx.compile(Before, target=target).mod


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("dtype, byte_offset", [("float32x2", 16), ("float32x3", 24)])
def test_vector_byte_pointer_offset(dtype, byte_offset):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor((8,), dtype)):
            T.func_attr({"target": T.target("llvm")})
            T.call_extern(
                "consume", T.ptr_byte_offset(A.data, byte_offset, ty=A.data.ty), ty="void"
            )

    After = tvm_tirx.tirx.transform.LowerIntrin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Before)
    tvm_tirx.tirx.compile(Before, target="llvm").mod


@pytest.mark.parametrize("shape, indices", [((), []), ((16,), [3]), ((2, 4), [1, 2])])
def test_logical_pointer_preserves_tensor_identity(shape, indices):
    @I.ir_module
    class Before:
        @T.function
        def main(A: T.Tensor(shape, "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(A.ptr_to(indices))

    @I.ir_module
    class Expected:
        @T.function
        def main(A: T.Tensor(shape, "float32")):
            T.func_attr({"target": T.target("llvm")})
            T.evaluate(T.address_of(A[tuple(indices)]))

    After = tvm_tirx.tirx.transform.LowerIntrin()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    assert After["main"].body[0].value.args[0].source.same_as(Before["main"].params[0])


def _divmod_expression(x, y, dtype, kind, variant, lowered):
    """The signed remainder correction shared by floor division and modulo."""
    scalar = getattr(T, dtype)
    zero = scalar(0)
    operation = T.floordiv if kind == "div" else T.floormod
    if not lowered:
        if variant == "pow2":
            return operation(x, scalar(8))
        if variant == "offset":
            return operation(x + scalar(4), scalar(5))
        value = operation(x, y)
        if variant == "max":
            value = T.max(value, zero)
        if variant in ["positive_rhs", "max"]:
            return T.Select(y >= zero, value, zero)
        if variant == "positive_both":
            return T.Select(T.And(y >= zero, x >= zero), value, zero)
        return value

    bits = int(dtype[3:])
    if variant == "pow2":
        return x >> scalar(3) if kind == "div" else x & scalar(7)
    if variant == "plain":
        rmod = T.dynamic("rmod", dtype)
        same_sign = T.Or(T.And(y >= zero, rmod >= zero), T.And(y < zero, rmod <= zero))
        if kind == "mod":
            return T.Let(T.Select(same_sign, rmod, rmod + y), where={rmod: T.truncmod(x, y)})
        rdiv = T.dynamic("rdiv", dtype)
        return T.Let(
            T.Let(T.Select(same_sign, rdiv, rdiv - scalar(1)), where={rdiv: T.Div(x, y)}),
            where={rmod: T.truncmod(x, y)},
        )
    if variant == "offset":
        x, y = x + scalar(4), scalar(5)
    quotient, remainder = T.Div(x, y), T.truncmod(x, y)
    if variant == "max":
        value = T.max(quotient, zero)
    elif variant == "positive_both":
        value = quotient if kind == "div" else remainder
    elif dtype == "int16":
        value = (
            T.Select(remainder >= zero, quotient, quotient - scalar(1))
            if kind == "div"
            else T.Select(remainder >= zero, remainder, remainder + y)
        )
    elif kind == "div":
        value = quotient + (remainder >> scalar(bits - 1))
    else:
        value = remainder + (y & (remainder >> scalar(bits - 1)))
    if variant == "offset":
        return value
    condition = T.And(zero <= y, zero <= x) if variant == "positive_both" else zero <= y
    return T.Select(condition, value, zero)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("dtype", ["int32", "int64", "int16"])
@pytest.mark.parametrize(
    "kind,variant",
    [
        (kind, variant)
        for kind in ["div", "mod"]
        for variant in ["plain", "positive_rhs", "positive_both", "pow2", "offset"]
    ]
    + [("div", "max")],
)
def test_lower_floordiv_and_floormod(dtype, kind, variant):
    scalar = getattr(T, dtype)

    @I.ir_module
    class Before:
        @T.function
        def main(x: scalar, y: scalar) -> scalar:
            T.func_attr({"target": T.target("llvm")})
            return _divmod_expression(x, y, dtype, kind, variant, False)

    @I.ir_module
    class Expected:
        @T.function
        def main(x: scalar, y: scalar) -> scalar:
            T.func_attr({"target": T.target("llvm")})
            return _divmod_expression(x, y, dtype, kind, variant, True)

    After = tvm_tirx.ir.transform.Sequential(
        [tvm_tirx.tirx.transform.StmtSimplify(), tvm_tirx.tirx.transform.LowerIntrin()]
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    compiled = tvm_tirx.tirx.compile(After, "llvm")
    data = itertools.product(range(-10, 10), [y for y in range(-10, 10) if y])
    for x, y in data:
        if (variant == "pow2" and y != 8) or (variant == "offset" and y != 5):
            continue
        numerator = x + 4 if variant == "offset" else x
        expected = numerator // y if kind == "div" else numerator % y
        if variant == "max":
            expected = max(expected, 0)
        if (variant in ["positive_rhs", "max"] and y < 0) or (
            variant == "positive_both" and (y < 0 or x < 0)
        ):
            expected = 0
        assert compiled(x, y) == expected


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize(
    "dtype,clamp,data",
    [
        ("int64", False, [-(2**63), 0, 100]),
        ("int16", False, [-32768, 0, 100]),
        ("int16", True, [-10, 0, 32758, 32757]),
    ],
)
def test_lower_floordiv_overflow_checks(dtype, clamp, data):
    scalar = getattr(T, dtype)
    bits = int(dtype[3:])

    def numerator(x):
        return T.min(T.max(x, scalar(-10)), scalar(32758)) if clamp else x

    @I.ir_module
    class Before:
        @T.function
        def main(x: scalar) -> scalar:
            T.func_attr({"target": T.target("llvm")})
            return T.floordiv(numerator(x), scalar(3))

    def corrected(x):
        q, r = T.Div(numerator(x), scalar(3)), T.truncmod(numerator(x), scalar(3))
        return (
            T.Select(r >= scalar(0), q, q - scalar(1))
            if dtype == "int16"
            else q + (r >> scalar(bits - 1))
        )

    @I.ir_module
    class Expected:
        @T.function
        def main(x: scalar) -> scalar:
            T.func_attr({"target": T.target("llvm")})
            return corrected(x)

    After = tvm_tirx.ir.transform.Sequential(
        [tvm_tirx.tirx.transform.StmtSimplify(), tvm_tirx.tirx.transform.LowerIntrin()]
    )(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
    compiled = tvm_tirx.tirx.compile(After, "llvm")
    for value in data:
        expected = min(max(value, -10), 32758) if clamp else value
        assert compiled(value) == expected // 3


if __name__ == "__main__":
    tvm_tirx.testing.main()
