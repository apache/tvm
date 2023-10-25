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
"""Tests to validate relax fast math tranform pass."""

import numpy as np
import pytest
import tvm.testing
from tvm import relax, topi
from tvm.ir.base import assert_structural_equal
from tvm.relax.transform import FastMathTransform
from tvm.script import ir as I, relax as R


def _run_pass_compare_output(Before, Expected):
    fast_mod = FastMathTransform()(Before)
    if not relax.analysis.well_formed(fast_mod):
        print("IRModule is not well-formed")
    assert_structural_equal(Expected, fast_mod)


def test_optimize_transform_layout_pass_one_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor((16,), dtype="float32")) -> R.Tensor((16,), dtype="float32"):
            lv1: R.Tensor((16,), dtype="float32") = R.nn.softmax(x)
            lv2: R.Tensor((16,), dtype="float32") = R.exp(lv1)
            lv3: R.Tensor((16,), dtype="float32") = R.erf(lv2)
            lv4: R.Tensor((16,), dtype="float32") = R.tanh(lv3)
            return lv4

    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((16,), "float32"))
    with bb.function("main", [x]):
        lv1 = bb.emit_te(topi.nn.fast_softmax, x)
        lv2 = bb.emit_te(topi.fast_exp, lv1)
        lv3 = bb.emit_te(topi.fast_erf, lv2)
        lv4 = bb.emit_te(topi.fast_tanh, lv3)
        bb.emit_func_output(lv4)
    Expected = bb.get()

    _run_pass_compare_output(Before, Expected)


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_fastmath(target, dev):
    def test_apply(low, high, step, alpha, dtype="float32"):
        a_np = np.arange(low, high, step).astype(dtype)
        b_np = np.power(a_np, alpha)

        @I.ir_module
        class ConstPower:
            @R.function
            def main(x: R.Tensor(a_np.shape, dtype=dtype)) -> R.Tensor(a_np.shape, dtype=dtype):
                lv: R.Tensor(a_np.shape, dtype=dtype) = R.power(x, R.const(alpha, dtype=dtype))
                return lv
        fast_mod = FastMathTransform()(ConstPower)
        ex = relax.build(fast_mod, target=target)
        vm = relax.VirtualMachine(ex, dev)
        x_tvm = tvm.nd.array(a_np)
        tvm_output = vm["main"](x_tvm)
        
        tvm.testing.assert_allclose(tvm_output.numpy(), b_np, rtol=1e-5, atol=1e-5)

        @I.ir_module
        class Power:
            @R.function
            def main(x: R.Tensor(a_np.shape, dtype=dtype), y: R.Tensor((), dtype=dtype)) -> R.Tensor(a_np.shape, dtype=dtype):
                lv: R.Tensor(a_np.shape, dtype=dtype) = R.power(x, y)
                return lv
        fast_mod = FastMathTransform()(Power)
        ex = relax.build(fast_mod, target=target)
        vm = relax.VirtualMachine(ex, dev)
        x_tvm = tvm.nd.array(a_np)
        y_tvm = tvm.nd.array(np.array(alpha).astype(dtype))
        tvm_output = vm["main"](x_tvm, y_tvm)
        
        tvm.testing.assert_allclose(tvm_output.numpy(), b_np, rtol=1e-5, atol=1e-5)


    test_apply(low=1, high=88, step=0.01, alpha=0.5)
    test_apply(low=-88, high=88, step=0.01, alpha=10)


if __name__ == "__main__":
    tvm.testing.main()
