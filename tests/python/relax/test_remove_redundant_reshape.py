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

"""
Test relax transform - Eliminate redundant reshape operations
"""
import tvm.testing
from tvm import relax
from tvm.relax.transform import DeadCodeElimination
from tvm.relax.transform import RemoveRedundantReshape
from tvm.script import ir as I, relax as R


def _run_pass_compare_output(Before, Expected):
    fused_mod = RemoveRedundantReshape()(Before)
    fused_mod = DeadCodeElimination()(fused_mod)
    tvm.ir.assert_structural_equal(Expected, fused_mod)


def test_remove_redundant_reshape_pass_one_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1001, 1, 1), dtype="float16")
        ) -> R.Tensor((1, 1001), dtype="float16"):
            with R.dataflow():
                lv: R.Tensor((1, 1001), dtype="float16") = R.reshape(x, R.shape([1, 1001]))
                lv1: R.Tensor((1, 1001), dtype="float16") = R.reshape(lv, R.shape([1, 1001]))
                gv: R.Tensor((1, 1001), dtype="float16") = R.reshape(lv1, R.shape([1, 1001]))
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 1001, 1, 1), dtype="float16")
        ) -> R.Tensor((1, 1001), dtype="float16"):
            with R.dataflow():
                gv: R.Tensor((1, 1001), dtype="float16") = R.reshape(x, R.shape([1, 1001]))
                R.output(gv)
            return gv

    _run_pass_compare_output(Before, Expected)


def test_remove_redundant_reshape_pass_two_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1001, 1, 1), dtype="float16")
        ) -> R.Tensor((1, 1001), dtype="float16"):
            with R.dataflow():
                lv: R.Tensor((1, 1001, 1), dtype="float16") = R.reshape(x, R.shape([1, 1001, 1]))
                lv1: R.Tensor((1, 1001), dtype="float16") = R.reshape(lv, R.shape([1, 1001]))
                R.output(lv1)
            return lv1

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 1001, 1, 1), dtype="float16")
        ) -> R.Tensor((1, 1001), dtype="float16"):
            with R.dataflow():
                lv1: R.Tensor((1, 1001), dtype="float16") = R.reshape(x, R.shape([1, 1001]))
                R.output(lv1)
            return lv1

    _run_pass_compare_output(Before, Expected)


def test_remove_redundant_reshape_pass_three_arg():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((1, 1001, 1, 1), dtype="float16")
        ) -> R.Tensor((1, 1001, 1, 1), dtype="float16"):
            with R.dataflow():
                lv: R.Tensor((1, 1001, 1, 1), dtype="float16") = R.reshape(
                    x, R.shape([1, 1001, 1, 1])
                )
                R.output(lv)
            return lv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((1, 1001, 1, 1), dtype="float16")
        ) -> R.Tensor((1, 1001, 1, 1), dtype="float16"):
            return x

    _run_pass_compare_output(Before, Expected)


if __name__ == "__main__":
    tvm.testing.main()
