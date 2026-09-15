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

"""Regression tests for the default fixed-point multiply legalization rules."""

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import tirx


def _const(value, lanes, dtype="int32"):
    value = tirx.const(value, dtype)
    return value if lanes == 1 else tirx.Broadcast(value, lanes)


def _legalize(name, lanes, *args):
    dtype = "int32" if lanes == 1 else f"int32x{lanes}"
    call = tirx.call_intrin(dtype, name, *args)
    return tvm.ir.Op.get(name).get_attr("default.FLegalize")(call)


def _check_value(expr, bindings, expected):
    substituted = tvm_ffi.structural_map(
        expr, (tirx.Var, lambda var: bindings.get(var, var)), order="post"
    )
    actual = tvm.arith.Analyzer().simplify(substituted)
    tvm.ir.assert_structural_equal(actual, expected)


def _reference(x, y, q, left_shift, right_shift):
    total_shift = q + right_shift
    return (((x << left_shift) * y) + (1 << (total_shift - 1))) >> total_shift


@pytest.mark.parametrize(
    ("multiplier", "shift", "q"),
    [
        (None, 1, 31),
        (None, None, 31),
        (12345, None, 31),
        (1 << 30, None, 31),
        (1 << 30, -1, 31),
        (1 << 30, 0, 31),
        (1 << 30, 1, 31),
        (1 << 30, 2, 31),
        (1 << 30, 1, 30),
    ],
)
def test_q_multiply_shift(multiplier, shift, q):
    lanes = 1
    x, y, s = [tirx.Var(name, "int32") for name in ("x", "y", "s")]
    lowered = _legalize(
        "tirx.q_multiply_shift",
        lanes,
        x,
        y if multiplier is None else _const(multiplier, lanes),
        _const(q, lanes),
        s if shift is None else _const(shift, lanes),
    )
    if multiplier == 1 << 30 and q == 31 and shift == 1:
        tvm.ir.assert_structural_equal(lowered, x)
    for x_value, y_value, s_value in [
        (-1001, 1 << 30, -1),
        (-3, 12345, 0),
        (3, -(1 << 29), 1),
        (1001, (1 << 30) + 1, 2),
    ]:
        effective_y = y_value if multiplier is None else multiplier
        effective_s = s_value if shift is None else shift
        expected = _reference(x_value, effective_y, q, max(effective_s, 0), max(-effective_s, 0))
        bindings = {x: _const(x_value, lanes), y: _const(y_value, lanes), s: _const(s_value, lanes)}
        _check_value(lowered, bindings, _const(expected, lanes))


@pytest.mark.parametrize("flag_dtype", ["int32", "bool"])
@pytest.mark.parametrize("flag", [0, 1])
def test_q_multiply_shift_per_axis(flag, flag_dtype):
    lanes = 1
    x, y = [tirx.Var(name, "int32") for name in ("x", "y")]
    lowered = _legalize(
        "tirx.q_multiply_shift_per_axis",
        lanes,
        x,
        y,
        _const(2, lanes),
        _const(1, lanes),
        _const(31, lanes),
        _const(flag, lanes, flag_dtype),
        _const(1, lanes, flag_dtype),
    )
    for x_value in [-1001, -3, 0, 3, 1001]:
        expected = _reference(x_value, 1 << 30, 31, 2 if flag else 0, 1)
        _check_value(
            lowered, {x: _const(x_value, lanes), y: _const(1 << 30, lanes)}, _const(expected, lanes)
        )


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("case", ["runtime_y", "runtime_s", "broadcast_y", "zero_exponent"])
def test_q_multiply_shift_vector(case, lanes):
    x, y, s = [tirx.Var(name, f"int32x{lanes}") for name in ("x", "y", "s")]
    if case == "broadcast_y":
        y = tirx.Broadcast(tirx.Var("multiplier", "int32"), lanes)
    elif case != "runtime_y":
        y = _const(1 << 30, lanes)
    if case != "runtime_s":
        s = _const(1, lanes)
    lowered = _legalize("tirx.q_multiply_shift", lanes, x, y, _const(31, lanes), s)
    tvm.ir.assert_structural_equal(lowered.ty, x.ty)
    if case == "zero_exponent":
        tvm.ir.assert_structural_equal(lowered, x)


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("flag", [0, 1])
def test_q_multiply_shift_per_axis_vector(flag, lanes):
    x, y = [tirx.Var(name, f"int32x{lanes}") for name in ("x", "y")]
    args = [x, y, _const(2, lanes), _const(1, lanes), _const(31, lanes)]
    lowered = _legalize(
        "tirx.q_multiply_shift_per_axis", lanes, *args, _const(flag, lanes), _const(1, lanes)
    )
    expected = _legalize(
        "tirx.q_multiply_shift_per_axis",
        lanes,
        *args,
        _const(flag, lanes, "bool"),
        _const(1, lanes, "bool"),
    )
    analyzer = tvm.arith.Analyzer()
    tvm.ir.assert_structural_equal(analyzer.simplify(lowered), analyzer.simplify(expected))


if __name__ == "__main__":
    tvm.testing.main()
