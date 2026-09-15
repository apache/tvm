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

"""Minimal regressions for fixed-point multiply legalization."""

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm import tirx


def _legalize(name, *args):
    call = tirx.call_intrin("int32", name, *args)
    return tvm.ir.Op.get(name).get_attr("default.FLegalize")(call)


@pytest.mark.parametrize("runtime_arg", ["multiplier", "shift"])
def test_q_multiply_shift_runtime_argument(runtime_arg):
    value = tirx.Var("value", "int32")
    multiplier = value if runtime_arg == "multiplier" else tirx.const(1 << 30, "int32")
    shift = value if runtime_arg == "shift" else tirx.const(1, "int32")
    lowered = _legalize("tirx.q_multiply_shift", 3, multiplier, 31, shift)
    replacement = tirx.const(1 << 30 if runtime_arg == "multiplier" else 1, "int32")
    lowered = tvm_ffi.structural_map(
        lowered, (tirx.Var, lambda var: replacement if var.same_as(value) else var), order="post"
    )
    tvm.ir.assert_structural_equal(tvm.arith.Analyzer().simplify(lowered), tirx.const(3, "int32"))


def test_q_multiply_shift_zero_exponent():
    x = tirx.Var("x", "int32")
    lowered = _legalize("tirx.q_multiply_shift", x, 1 << 30, 31, 1)
    tvm.ir.assert_structural_equal(lowered, x)


def test_q_multiply_shift_non_q31():
    lowered = _legalize("tirx.q_multiply_shift", 3, 1 << 30, 30, 2)
    tvm.ir.assert_structural_equal(tvm.arith.Analyzer().simplify(lowered), tirx.const(12, "int32"))


def test_q_multiply_shift_per_axis_integer_flag():
    lowered = _legalize("tirx.q_multiply_shift_per_axis", 3, 1 << 30, 2, 1, 31, 1, 1)
    tvm.ir.assert_structural_equal(tvm.arith.Analyzer().simplify(lowered), tirx.const(3, "int32"))


if __name__ == "__main__":
    tvm.testing.main()
