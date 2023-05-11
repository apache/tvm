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
import math
import random
import numpy as np
import tvm
import tvm.testing
import pytest
from tvm import tir
from tvm.script import tir as T
import pytest


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["int8", [-128, 0, 127]],
        ["uint8", [0, 255]],
        ["int32", [-2147483648, 2147483647]],
        ["uint32", [0, 4294967295]],
        ["int64", [-9223372036854775808, 9223372036854775807]],
        ["uint64", [0, 9223372036854775807]],
    ],
)
def test_tir_make_intimm(dtype, literals):
    for l in literals:
        imm = tir.const(l, dtype)
        assert imm.value == l, imm


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["int8", [-129, 128]],
        ["uint8", [-1, 256]],
        ["int32", [-2147483650, 2147483648]],
        ["uint32", [-1, 4294967296]],
        ["uint64", [-1, 18446744073709551616]],
    ],
)
def test_tir_invalid_intimm(dtype, literals):
    for l in literals:
        with pytest.raises(tvm.TVMError):
            tir.const(l, dtype)


@pytest.mark.parametrize(
    "dtype, literals",
    [
        [
            "uint64",
            {
                9223372036854775807: 9223372036854775807,
                18446744073709551615: 18446744073709551615,
            },
        ],
    ],
)
def test_tir_large_py_int_literals(dtype, literals):
    """
    For large uint value, use LargeUIntImm intrin,
    """
    for l in literals:
        x = tir.const(l, dtype)
        if isinstance(x, (tir.IntImm, tir.FloatImm)):
            assert x.value == literals[l]
        else:
            # LargeUIntImm(low32, hi32)
            assert (int(x.args[1]) << 32) + int(x.args[0]) == literals[l]


def test_tir_intimm_overflow():
    assert int(tir.const(255, "uint8") + tir.const(1, "uint8")) == 0
    assert int(tir.const(2**31 - 1, "int32") + tir.const(1, "int32")) == -(2**31)
    assert int(tir.const(2**32 - 1, "uint32") + tir.const(1, "uint32")) == 0
    assert int(tir.const(2**63 - 1, "int64") + tir.const(1, "int64")) == -(2**63)
    assert int(tir.const(2**32, "uint64") * tir.const(2**32, "uint64")) == 0
    # customized int types
    assert int(tir.const(7, "int4") + tir.const(1, "int4")) == -8
    assert int(tir.const(2**39 - 1, "int40") + tir.const(1, "int40")) == -(2**39)


def compare_float_value(value, expect, msg):
    if math.isfinite(value):
        assert np.abs(value - expect) < 1e-5, f"{value} vs {expect}, {msg}"
    elif math.isnan(value):
        assert math.isnan(expect), f"{value} vs {expect}, {msg}"
    elif math.isinf(value):
        assert math.isinf(expect), f"{value} vs {expect}, {msg}"


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["float16", [-65504.0, 3.14, 65504.0, np.inf, np.nan]],
        ["bfloat16", [-3.38953139e38, 3.38953139e38, 3.14]],
        ["float32", [np.finfo("float32").min, 3.14, np.finfo("float32").max, np.inf, np.nan]],
        ["float64", [np.finfo("float64").min, 3.14, np.finfo("float64").max, np.inf, np.nan]],
    ],
)
def test_tir_make_floatimm(dtype, literals):
    for l in literals:
        imm = tir.const(l, dtype)
        compare_float_value(imm.value, l, "imm value should match feed value")


@pytest.mark.parametrize(
    "dtype, literals",
    [
        ["float16", [-65505.0, 65505.0]],
        ["float32", [-3.402e39, 3.402e39]],
    ],
)
def test_tir_invalid_floatimm(dtype, literals):
    """Currently only fp16 and fp32 have range check."""
    for l in literals:
        with pytest.raises(tvm.TVMError):
            tir.const(l, dtype)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("literal", [3.14, np.nan, np.inf])
def test_tir_special_floatimms(dtype, literal):
    x = tir.const(literal, dtype)
    compare_float_value(x.value, literal, "imm value should match feed value")


@tvm.testing.requires_llvm()
def test_tir_too_large_literal_f64():
    # Behavior check: if literal f64 value is out of dtype range, the
    # object is still constructed, and eval to infinity.
    @T.prim_func
    def imm_overflow_fp64() -> T.float64:
        T.evaluate(T.ret(T.float64(1.7976e309), dtype="float64"))

    f = tvm.build(imm_overflow_fp64, target="llvm")
    assert math.isinf(f())


@pytest.mark.parametrize(
    "literal, expect_dtype",
    [
        (256, "int32"),
        (2147483647, "int32"),
        (-2147483648, "int32"),
        (2147483648, "int64"),
        (-2147483649, "int64"),
        (3.14159, "float32"),
        (np.finfo("float32").min, "float32"),
        (np.finfo("float32").max, "float32"),
        (-3.402e39, "float64"),
        (3.402e39, "float64"),
    ],
)
def test_tir_const_auto_dtype(literal, expect_dtype):
    x = tir.const(literal, dtype=None)
    assert x.dtype == expect_dtype
    assert x.value == literal


def check_tir_const_fold(
    dtype, foldf, calcf, x_range=None, y_range=None, expect=None, skip_overflow=False
):
    """Helper to check constant folding behavior

    Parameters
    ----------
    dtype: str
        Datatype of constants

    foldf: (x, y) -> z
        Folding function to call

    calcf: (x, y) -> z
        Compiled calculation function to call

    x_range: Union[int, float, tuple]
        Single value or value range [min, max]

    y_range: Union[int, float, tuple]
        Single value or value range [min, max]

    expect: Union[int, float]
        Expected calculation result

    skip_overflow: bool
        Skip assertion if the overflow happens
    """
    seed = random.randint(0, 2147483648)
    np.random.seed(seed)
    ninfo = np.finfo(dtype) if dtype.startswith("float") else np.iinfo(dtype)

    if x_range is None:
        x_range = (ninfo.min, ninfo.max)
    if isinstance(x_range, (int, float)):
        x = x_range
    elif dtype.startswith("int") or dtype.startswith("uint"):
        x = np.random.randint(x_range[0], x_range[1] + 1, dtype=dtype)
    else:
        x = np.random.uniform(x_range[0], x_range[1])

    if y_range is None:
        y_range = (ninfo.min, ninfo.max)
    if isinstance(y_range, (int, float)):
        y = y_range
    elif dtype.startswith("int") or dtype.startswith("uint"):
        y = np.random.randint(y_range[0], y_range[1] + 1, dtype=dtype)
    else:
        y = np.random.uniform(y_range[0], y_range[1])

    if skip_overflow:
        py_res = foldf(x, y)
        if isinstance(py_res, (tir.IntImm, tir.FloatImm)):
            py_res = py_res.value
        if not (ninfo.min <= py_res <= ninfo.max):
            # If the result overflow, certain arithmetics is non-defined
            # thus we intentionally do not make the test failed.
            return

    fold_res = foldf(tir.const(x, dtype), tir.const(y, dtype))
    calc_res = calcf(x, y)

    flaky_msg = (
        f"{dtype} ({x}, {y}, {expect}) const folding check failed.\n"
        + "This test is intentionally non-deterministic, "
        + f"if it fails please report it in github issue together with this seed {seed}\n"
    )
    if dtype.startswith("float"):
        compare_float_value(calc_res, fold_res.value, flaky_msg)
        if expect:
            compare_float_value(expect, calc_res, flaky_msg)
    else:
        assert calc_res == fold_res.value, flaky_msg
        if expect:
            assert expect == calc_res, flaky_msg


@tvm.testing.requires_llvm()
def test_tir_floatimm_const_fold():
    """Behavior check: folding fp32 match platform f32 arithmetic"""

    @T.prim_func
    def float_imm_multiply(x: T.float32, y: T.float32, z: T.Buffer((), "float32")):
        z[()] = x * y

    @T.prim_func
    def float_imm_add(x: T.float32, y: T.float32, z: T.Buffer((), "float32")):
        z[()] = x + y

    @T.prim_func
    def float_imm_sub(x: T.float32, y: T.float32, z: T.Buffer((), "float32")):
        z[()] = x - y

    @T.prim_func
    def float_imm_div(x: T.float32, y: T.float32, z: T.Buffer((), "float32")):
        z[()] = x / y

    def __wrap_build(f):
        lib = tvm.build(f, target="llvm")
        z = tvm.nd.array(np.zeros([]).astype("float32"))

        def _func(x, y):
            lib(x, y, z)
            return z.numpy()

        return _func

    fmul = __wrap_build(float_imm_multiply)
    fadd = __wrap_build(float_imm_add)
    fsub = __wrap_build(float_imm_sub)
    fdiv = __wrap_build(float_imm_div)

    # overflow
    check_tir_const_fold("float32", lambda x, y: x * y, fmul, 3.0e30, 3.0e30, np.inf)
    check_tir_const_fold("float32", lambda x, y: x * y, fmul, 3.0e30, -3.0e30, -np.inf)
    check_tir_const_fold("float32", lambda x, y: x / y, fdiv, 3.0e30, 3.0e-30, np.inf)

    # divide by zero
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("float32", lambda x, y: x / y, fdiv, 1.0, 0.0)

    # nan and inf
    check_tir_const_fold("float32", lambda x, y: x + y, fadd, 1.0, np.nan, np.nan)
    check_tir_const_fold("float32", lambda x, y: x + y, fadd, 1.0, np.inf, np.inf)
    check_tir_const_fold("float32", lambda x, y: x + y, fadd, 1.0, -np.inf, -np.inf)

    # randomized check
    check_tir_const_fold("float32", lambda x, y: x * y, fmul)
    check_tir_const_fold("float32", lambda x, y: x + y, fadd)
    check_tir_const_fold("float32", lambda x, y: x - y, fsub)
    check_tir_const_fold(
        "float32", lambda x, y: x / y, fdiv, y_range=(0.01, np.finfo("float32").max)
    )


@tvm.testing.requires_llvm()
def test_tir_int8_const_fold():
    """Behavior check: folding i8 operation match platform i8 arithmetic"""

    @T.prim_func
    def imm_multiply(x: T.int8, y: T.int8) -> T.int8:
        T.evaluate(T.ret(x * y, dtype="int8"))

    @T.prim_func
    def imm_add(x: T.int8, y: T.int8) -> T.int8:
        T.evaluate(T.ret(x + y, dtype="int8"))

    @T.prim_func
    def imm_sub(x: T.int8, y: T.int8) -> T.int8:
        T.evaluate(T.ret(x - y, dtype="int8"))

    @T.prim_func
    def imm_truncdiv(x: T.int8, y: T.int8) -> T.int8:
        T.evaluate(T.ret(T.truncdiv(x, y), dtype="int8"))

    @T.prim_func
    def imm_floordiv(x: T.int8, y: T.int8) -> T.int8:
        T.evaluate(T.ret(T.floordiv(x, y), dtype="int8"))

    fmul = tvm.build(imm_multiply, target="llvm")
    fadd = tvm.build(imm_add, target="llvm")
    fsub = tvm.build(imm_sub, target="llvm")
    ffloordiv = tvm.build(imm_floordiv, target="llvm")
    ftruncdiv = tvm.build(imm_truncdiv, target="llvm")

    # overflow
    check_tir_const_fold("int8", lambda x, y: x + y, fadd, 127, 1, -128)
    check_tir_const_fold("int8", lambda x, y: x * y, fmul, 127, 127, 1)

    # divide by zero
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("int8", lambda x, y: tir.floordiv(x, y), ffloordiv, 1, 0)
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("int8", lambda x, y: tir.truncdiv(x, y), ftruncdiv, 1, 0)

    # i8 mod folding is not implemented
    assert not isinstance(tir.floormod(tir.const(7, "int8"), tir.const(3, "int8")), tir.IntImm)
    assert not isinstance(tir.truncmod(tir.const(7, "int8"), tir.const(3, "int8")), tir.IntImm)

    # randomized check
    check_tir_const_fold("int8", lambda x, y: x * y, fmul)
    check_tir_const_fold("int8", lambda x, y: x + y, fadd)
    check_tir_const_fold("int8", lambda x, y: x - y, fsub)
    check_tir_const_fold(
        "int8", lambda x, y: tir.floordiv(x, y), ffloordiv, y_range=(1, np.iinfo("int8").max)
    )
    check_tir_const_fold(
        "int8", lambda x, y: tir.truncdiv(x, y), ftruncdiv, y_range=(1, np.iinfo("int8").max)
    )


@tvm.testing.requires_llvm()
def test_tir_uint8_const_fold():
    """Behavior check: folding u8 operation match platform u8 arithmetic"""

    @T.prim_func
    def imm_multiply(x: T.uint8, y: T.uint8) -> T.uint8:
        T.evaluate(T.ret(x * y, dtype="uint8"))

    @T.prim_func
    def imm_add(x: T.uint8, y: T.uint8) -> T.uint8:
        T.evaluate(T.ret(x + y, dtype="uint8"))

    @T.prim_func
    def imm_sub(x: T.uint8, y: T.uint8) -> T.uint8:
        T.evaluate(T.ret(x - y, dtype="uint8"))

    @T.prim_func
    def imm_truncdiv(x: T.uint8, y: T.uint8) -> T.uint8:
        T.evaluate(T.ret(T.truncdiv(x, y), dtype="uint8"))

    @T.prim_func
    def imm_floordiv(x: T.uint8, y: T.uint8) -> T.uint8:
        T.evaluate(T.ret(T.floordiv(x, y), dtype="uint8"))

    fmul = tvm.build(imm_multiply, target="llvm")
    fadd = tvm.build(imm_add, target="llvm")
    fsub = tvm.build(imm_sub, target="llvm")
    ffloordiv = tvm.build(imm_floordiv, target="llvm")
    ftruncdiv = tvm.build(imm_truncdiv, target="llvm")

    # overflow
    check_tir_const_fold("uint8", lambda x, y: x + y, fadd, 255, 1, 0)

    # zero sub
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("uint8", lambda x, y: x - y, fsub, 0, 10)

    # divide by zero
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("uint8", lambda x, y: tir.floordiv(x, y), ffloordiv, 1, 0)
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("uint8", lambda x, y: tir.truncdiv(x, y), ftruncdiv, 1, 0)

    # u8 mod folding is not implemented
    assert not isinstance(tir.floormod(tir.const(7, "uint8"), tir.const(3, "uint8")), tir.IntImm)
    assert not isinstance(tir.truncmod(tir.const(7, "uint8"), tir.const(3, "uint8")), tir.IntImm)

    # randomized check
    check_tir_const_fold("uint8", lambda x, y: x * y, fmul)
    check_tir_const_fold("uint8", lambda x, y: x + y, fadd)
    check_tir_const_fold("uint8", lambda x, y: x - y, fsub)
    check_tir_const_fold(
        "uint8", lambda x, y: tir.floordiv(x, y), ffloordiv, y_range=(1, np.iinfo("uint8").max)
    )
    check_tir_const_fold(
        "uint8", lambda x, y: tir.truncdiv(x, y), ftruncdiv, y_range=(1, np.iinfo("uint8").max)
    )


@tvm.testing.requires_llvm()
def test_tir_int32_const_fold():
    """Behavior check: folding i32 operation match platform i32 arithmetic"""

    @T.prim_func
    def imm_multiply(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(x * y, dtype="int32"))

    @T.prim_func
    def imm_add(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(x + y, dtype="int32"))

    @T.prim_func
    def imm_sub(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(x - y, dtype="int32"))

    @T.prim_func
    def imm_truncdiv(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(T.truncdiv(x, y), dtype="int32"))

    @T.prim_func
    def imm_truncmod(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(T.truncmod(x, y), dtype="int32"))

    @T.prim_func
    def imm_floordiv(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(T.floordiv(x, y), dtype="int32"))

    @T.prim_func
    def imm_floormod(x: T.int32, y: T.int32) -> T.int32:
        T.evaluate(T.ret(T.floormod(x, y), dtype="int32"))

    fmul = tvm.build(imm_multiply, target="llvm")
    fadd = tvm.build(imm_add, target="llvm")
    fsub = tvm.build(imm_sub, target="llvm")
    ffloordiv = tvm.build(imm_floordiv, target="llvm")
    ffloormod = tvm.build(imm_floormod, target="llvm")
    ftruncdiv = tvm.build(imm_truncdiv, target="llvm")
    ftruncmod = tvm.build(imm_truncmod, target="llvm")

    # i32 overflow is not specified, only check for range
    assert -(2**31) <= int(tir.const(2**31 - 1, "int32") + tir.const(1, "int32")) < 2**31
    assert -(2**31) <= int(tir.const(-(2**31), "int32") - tir.const(1, "int32")) < 2**31

    # divide by zero
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("int32", lambda x, y: tir.floordiv(x, y), ffloordiv, 1, 0)
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("int32", lambda x, y: tir.floormod(x, y), ffloormod, 1, 0)
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("int32", lambda x, y: tir.truncdiv(x, y), ftruncdiv, 1, 0)
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("int32", lambda x, y: tir.truncmod(x, y), ftruncmod, 1, 0)

    # randomized check
    check_tir_const_fold("int32", lambda x, y: x * y, fmul, skip_overflow=True)
    check_tir_const_fold("int32", lambda x, y: x + y, fadd, skip_overflow=True)
    check_tir_const_fold("int32", lambda x, y: x - y, fsub, skip_overflow=True)
    check_tir_const_fold(
        "int32",
        lambda x, y: tir.floordiv(x, y),
        ffloordiv,
        y_range=(1, np.iinfo("int32").max),
        skip_overflow=True,
    )
    check_tir_const_fold(
        "int32",
        lambda x, y: tir.truncdiv(x, y),
        ftruncdiv,
        y_range=(1, np.iinfo("int32").max),
        skip_overflow=True,
    )
    check_tir_const_fold(
        "int32",
        lambda x, y: tir.floormod(x, y),
        ffloormod,
        y_range=(1, np.iinfo("int32").max),
        skip_overflow=False,
    )
    check_tir_const_fold(
        "int32",
        lambda x, y: tir.truncmod(x, y),
        ftruncmod,
        y_range=(1, np.iinfo("int32").max),
        skip_overflow=False,
    )


@tvm.testing.requires_llvm()
def test_tir_uint32_const_fold():
    """Behavior check: folding u32 operation match platform u32 arithmetic"""

    @T.prim_func
    def imm_multiply(x: T.uint32, y: T.uint32) -> T.uint32:
        T.evaluate(T.ret(x * y, dtype="uint32"))

    @T.prim_func
    def imm_add(x: T.uint32, y: T.uint32) -> T.uint32:
        T.evaluate(T.ret(x + y, dtype="uint32"))

    @T.prim_func
    def imm_sub(x: T.uint32, y: T.uint32) -> T.uint32:
        T.evaluate(T.ret(x - y, dtype="uint32"))

    @T.prim_func
    def imm_truncdiv(x: T.uint32, y: T.uint32) -> T.uint32:
        T.evaluate(T.ret(T.truncdiv(x, y), dtype="uint32"))

    @T.prim_func
    def imm_floordiv(x: T.uint32, y: T.uint32) -> T.uint32:
        T.evaluate(T.ret(T.floordiv(x, y), dtype="uint32"))

    fmul = tvm.build(imm_multiply, target="llvm")
    fadd = tvm.build(imm_add, target="llvm")
    fsub = tvm.build(imm_sub, target="llvm")
    ffloordiv = tvm.build(imm_floordiv, target="llvm")
    ftruncdiv = tvm.build(imm_truncdiv, target="llvm")

    # u32 overflow is not specified, only check for range
    assert 0 <= int(tir.const(2**32 - 1, "uint32") + tir.const(1, "uint32")) < 2**32

    # divide by zero
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("uint32", lambda x, y: tir.floordiv(x, y), ffloordiv, 1, 0)
    with pytest.raises(tvm.TVMError):
        check_tir_const_fold("uint32", lambda x, y: tir.truncdiv(x, y), ftruncdiv, 1, 0)

    # u8 mod folding is not implemented
    assert not isinstance(tir.floormod(tir.const(7, "uint32"), tir.const(3, "uint32")), tir.IntImm)
    assert not isinstance(tir.truncmod(tir.const(7, "uint32"), tir.const(3, "uint32")), tir.IntImm)

    # randomized check
    check_tir_const_fold("uint32", lambda x, y: x * y, fmul, skip_overflow=True)
    check_tir_const_fold("uint32", lambda x, y: x + y, fadd, skip_overflow=True)
    check_tir_const_fold("uint32", lambda x, y: x - y, fsub, skip_overflow=True)
    check_tir_const_fold(
        "uint32",
        lambda x, y: tir.floordiv(x, y),
        ffloordiv,
        y_range=(1, np.iinfo("uint32").max),
        skip_overflow=False,
    )
    check_tir_const_fold(
        "uint32",
        lambda x, y: tir.truncdiv(x, y),
        ftruncdiv,
        y_range=(1, np.iinfo("uint32").max),
        skip_overflow=False,
    )


if __name__ == "__main__":
    tvm.testing.main()
