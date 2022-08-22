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
    assert int(tir.const(127, "int8") + tir.const(1, "int8")) == -128
    assert int(tir.const(127, "int8") + tir.const(2, "int8")) == -127
    assert int(tir.const(255, "uint8") + tir.const(1, "uint8")) == 0
    assert int(tir.const(2**31 - 1, "int32") + tir.const(1, "int32")) == -(2**31)
    assert int(tir.const(2**32 - 1, "uint32") + tir.const(1, "uint32")) == 0
    assert int(tir.const(2**63 - 1, "int64") + tir.const(1, "int64")) == -(2**63)
    assert int(tir.const(2**32, "uint64") * tir.const(2**32, "uint64")) == 0


def compare_float_value(value, expect):
    if math.isfinite(value):
        assert value == expect
    elif math.isnan(value):
        assert math.isnan(expect)
    elif math.isinf(value):
        assert math.isinf(expect)


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("literal", [3.14, np.nan, np.inf])
def test_tir_special_floatimms(dtype, literal):
    x = tir.const(literal, dtype)
    compare_float_value(x.value, literal)


@tvm.testing.requires_llvm()
def test_tir_floatimm_overflow():
    # Behavior check: if literal value is out of dtype range, the
    # object is still constructed, and eval to infinity.
    @T.prim_func
    def imm_overflow_fp16() -> T.float16:
        T.evaluate(T.ret(T.float16(65536), dtype="float16"))

    f = tvm.build(imm_overflow_fp16, target="llvm")
    assert math.isinf(f())

    @T.prim_func
    def imm_overflow_fp32() -> T.float32:
        T.evaluate(T.ret(T.float32(3.4028e39), dtype="float32"))

    f = tvm.build(imm_overflow_fp32, target="llvm")
    assert math.isinf(f())

    @T.prim_func
    def imm_overflow_fp64() -> T.float64:
        T.evaluate(T.ret(T.float64(1.7976e309), dtype="float64"))

    f = tvm.build(imm_overflow_fp64, target="llvm")
    assert math.isinf(f())

    # Behavior check: disable fp16 folding
    assert float(tir.const(1.0, "float32") * tir.const(2.0, "float32")) == 2.0
    assert not isinstance(tir.const(1.0, "float16") * tir.const(2.0, "float16"), tir.FloatImm)

    # Behavior check: folding when fp32 overflow get infinity
    x = np.float32(3.4028235e37)
    y = np.float32(3.4028235e37)
    assert math.isinf(float(tir.const(x, "float32") * tir.const(y, "float32")))


if __name__ == "__main__":
    tvm.testing.main()
