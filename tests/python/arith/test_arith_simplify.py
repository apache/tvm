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

import pytest

import tvm
import tvm.testing
from tvm import tir
from tvm.script import tir as T


def test_simplify_reshape_flattened_index():
    ana = tvm.arith.Analyzer()

    i0 = tir.Var("i0", "int64")
    i1 = tir.Var("i1", "int64")
    ana.bind(i0, tvm.ir.Range(0, 8))
    ana.bind(i1, tvm.ir.Range(0, 3))

    i_flattened = i0 * 3 + i1
    tvm.ir.assert_structural_equal(
        ana.simplify((i_flattened) // 12 * 12 + (i_flattened) % 12 // 4 * 4 + (i_flattened) % 4),
        i_flattened,
    )


dtype = tvm.testing.parameter(
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
)


def test_can_prove_self_identity(dtype):
    ana = tvm.arith.Analyzer()

    n = tir.Var("n", dtype)
    assert ana.can_prove(n == n)


def test_can_prove_self_equal_to_self(dtype):
    ana = tvm.arith.Analyzer()

    n = tir.Var("n", dtype)
    assert ana.can_prove_equal(n, n)


def test_simplify_symbolic_comparison():
    ana = tvm.arith.Analyzer()

    i0 = tir.Var("i0", "int64")
    i1 = tir.Var("i1", "int64")
    n, m = tvm.tir.SizeVar("n", "int64"), tvm.tir.SizeVar("m", "int64")
    outer = (n + 31) // 32
    ana.bind(i0, tvm.ir.Range(0, outer))
    ana.bind(i1, tvm.ir.Range(0, 32))
    PS = tvm.arith.ProofStrength

    assert not ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32, PS.DEFAULT)
    assert ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32, PS.SYMBOLIC_BOUND)
    assert ana.can_prove(i0 * 32 + i1 < (n + 31) // 32 * 32 + m, PS.SYMBOLIC_BOUND)
    assert ana.can_prove(i0 * 32 + i1 + 1 <= (n + 31) // 32 * 32, PS.SYMBOLIC_BOUND)
    assert ana.can_prove((n + 31) // 32 * 32 >= i0 * 32 + i1 + 1, PS.SYMBOLIC_BOUND)
    assert ana.can_prove((n + 31) // 32 * 32 >= i0 * 32 + i1, PS.SYMBOLIC_BOUND)


@pytest.mark.parametrize(
    "expression",
    [
        T.vscale() * 32 < T.vscale() * 64,
        T.vscale() * 2 * (T.vscale() * 2) >= T.vscale() * 4,
        (T.vscale() * 4 + 114) // (T.vscale() * 4) * (T.vscale() * 4) >= 115,
        64 % T.vscale() <= T.vscale(),
    ],
)
def test_simplify_vscale_comparison_with_sve_target(expression):
    ana = tvm.arith.Analyzer()

    with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+sve"):
        assert ana.can_prove(expression)


def test_simplify_vscale_comparison_without_sve_target(capfd):
    ana = tvm.arith.Analyzer()
    vs = tvm.tir.vscale()

    with pytest.raises(AssertionError):
        with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu"):
            assert ana.can_prove(vs * 32 < vs * 64)

    warning_msg = (
        "Warning: The expression contains scalable values. An attempt to prove by substituting "
        "with known values of vscale was not performed. This proof currently only supports "
        "AArch64 SVE targets, but the target was llvm -keys=arm_cpu,cpu -mtriple=aarch64-linux-gnu"
    )
    capture = capfd.readouterr().err
    assert warning_msg in capture


def test_regression_simplify_inf_recursion():
    ana = tvm.arith.Analyzer()
    cond = tir.Var("cond", "int32")

    res = (tvm.tir.NE(cond, 0).astype("int8") - tvm.tir.NE(cond, 0).astype("int8")).astype(
        "int32"
    ) == 0
    # regression in a previous case
    # try compare and int set recursive call can cause infinite loop
    ana.rewrite_simplify(res)


if __name__ == "__main__":
    tvm.testing.main()
