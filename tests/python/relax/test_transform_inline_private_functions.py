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
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_inline_simple():
    """Simple case of inlining

    Inlining applies to all private functions
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            B = A * A
            C = Before.subroutine(B)
            D = C + C
            return D

        @R.function(private=True)
        def subroutine(B: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            C = R.concat([B, B], axis=1)
            return C

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([16, 16], "int32")) -> R.Tensor([16, 32], "int32"):
            B = A * A
            C = R.concat([B, B], axis=1)
            D = C + C
            return D

    After = tvm.relax.transform.InlinePrivateFunctions()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


def test_skip_inline_of_recursive_functions():
    """Recursively-defined functions

    This behavior is deliberately different between the
    `relax.transform.InlinePrivateFunctions` pass, and the
    `relax.Function.inline_functions` utility.

    For a user-facing utility, such as `func.inline_functions(...)`,
    the functions to be inlined are specifically listed, and must not
    be ignored.  If it is unable to inline the user-requested
    function, it should return an appropriate error.

    For a generic utility to be used in optimization pipelines, the
    framework is tasked with selecting the functions to be inlined,
    and should avoid selecting any function that cannot be inlined.
    This includes recursively-defined functions.
    """

    @I.ir_module
    class Before:
        @R.function
        def main():
            B = Before.subroutine()
            return B

        @R.function(private=True)
        def subroutine() -> R.Tensor([], "int64"):
            R.func_attr({"relax.force_pure": True})
            cond = R.call_packed("dummy_function", sinfo_args=R.Tensor([], "bool"))
            if cond:
                Out = Before.subroutine()
            else:
                Out = R.const(0, "int64")

            return Out

    Expected = Before

    After = tvm.relax.transform.InlinePrivateFunctions()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    tvm.testing.main()
