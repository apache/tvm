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
import tvm.testing
from tvm.relax.transform import FoldDataflowBlockOutput
from tvm.script.parser import ir as I, relax as R, tir as T


def verify(input, expected):
    tvm.ir.assert_structural_equal(FoldDataflowBlockOutput()(input), expected)


def test_basic_example():
    @tvm.script.ir_module
    class Input:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1)
                n = y
                R.output(n)
            return n

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                n = R.const(1)
                R.output(n)
            return n

    verify(Input, Expected)


def test_match_cast():
    @tvm.script.ir_module
    class Input:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1)
                n = R.match_cast(y, R.Tensor((), "int32"))
                R.output(n)
            return n

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                n = R.match_cast(R.const(1), R.Tensor((), "int32"))
                R.output(n)
            return n

    verify(Input, Expected)


def test_unable_to_fold():
    @tvm.script.ir_module
    class MultipleUse:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1)
                # multiple uses -> cannot coalesce
                m = R.add(y, y)
                n = y
                R.output(n)
            return n

    @tvm.script.ir_module
    class ComplexExpr:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                y = R.const(1)
                # y does not appear by itself -> cannot coalesce
                n = R.add(y, y)
                R.output(n)
            return n

    verify(MultipleUse, MultipleUse)
    verify(ComplexExpr, ComplexExpr)


def test_multiple_outputs():
    @tvm.script.ir_module
    class Input:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                x = R.const(1)
                y = R.const(1)
                z = R.const(1)
                l = x
                m = y
                n = z
                R.output(l, m, n)
            return n

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                l = R.const(1)
                m = R.const(1)
                n = R.const(1)
                R.output(l, m, n)
            return n

    verify(Input, Expected)


def test_multiply_used_in_outputs():
    # cannot fold in this case
    @tvm.script.ir_module
    class UsedInMultipleOutputs:
        @R.function
        def main() -> R.Tensor((), "int32"):
            with R.dataflow():
                x = R.const(1)
                l = x
                m = x
                n = x
                R.output(l, m, n)
            return n

    verify(UsedInMultipleOutputs, UsedInMultipleOutputs)


if __name__ == "__main__":
    tvm.testing.main()
