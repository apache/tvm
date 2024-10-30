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
from tvm.script import ir as I, relax as R, tir as T


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.relax.transform.RemoveUnusedParameters()


class TestRemoveUnusedRelaxParameter(BaseCompare):
    """A relax parameter may be removed

    This is only allowed for internal function calls, where all
    callsites can be updated.  For externally-exposed functions, the
    signature may not be modified.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Before.func(A, B)

        @R.function(private=True)
        def func(A: R.Tensor, B: R.Tensor) -> R.Tensor:
            return A

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor, B: R.Tensor):
            return Expected.func(A)

        @R.function(private=True)
        def func(A: R.Tensor) -> R.Tensor:
            return A


class TestReplaceSymbolicVariables(BaseCompare):
    """If a parameter is only required for its symbolic variables, provide them directly

    The relax parameter `A` isn't used by the subroutine.  However,
    its shape defines the symbolic variables `m` and `n`.  When
    removing the `R.Tensor` argument, we may need to provide
    additional parameters to define the symbolic variables.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            return Before.func(A)

        @R.function(private=True)
        def func(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            return R.zeros(R.shape([m, n]), dtype="float32")

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            out: R.Tensor([m, n], "float32") = Expected.func(R.prim_value(n), R.prim_value(m))
            return out

        @R.function(private=True)
        def func(
            param_n: R.Prim(value="n"), param_m: R.Prim(value="m")
        ) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            return R.zeros(R.shape([m, n]), dtype="float32")


class TestNoExtraSymbolicVariables(BaseCompare):
    """Don't add symbolic variables if they can be inferred.

    Even though some cases require adding new parameters to provide
    symbolic variables, not every symbolic variable requires a
    distinct parameter.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            return Before.func(A)

        @R.function(private=True)
        def func(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            zeros = R.zeros(R.shape([m, n]), dtype="float32")
            out = R.add(A, zeros)
            return out

    Expected = Before


class TestRemoveExtraPrimVariables(BaseCompare):
    """Remove parameters that only serve to define existing symbolic variables

    If a `R.Prim` parameter provies a definition of a symbolic
    variable, but that symbolic variable can be determined from a
    different parameter, then the `R.Prim` parameter can be removed.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            return Before.func(A, R.prim_value(m), R.prim_value(n))

        @R.function(private=True)
        def func(
            A: R.Tensor(["m", "n"], "float32"), _m: R.Prim(value="m"), _n: R.Prim(value="n")
        ) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            zeros = R.zeros(R.shape([m, n]), dtype="float32")
            out = R.add(A, zeros)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            return Expected.func(A)

        @R.function(private=True)
        def func(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            zeros = R.zeros(R.shape([m, n]), dtype="float32")
            out = R.add(A, zeros)
            return out


class TestRemoveExtraShapeVariables(BaseCompare):
    """Remove parameters that only serve to define existing symbolic variables

    If a `R.Shape` parameter provides a definition of a symbolic
    variable, but that symbolic variable can be determined from a
    different parameter, then the `R.Shape` parameter can be removed.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            return Before.func(A, R.shape([m, n]))

        @R.function(private=True)
        def func(
            A: R.Tensor(["m", "n"], "float32"),
            _: R.Shape(["m", "n"]),
        ) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            zeros = R.zeros(R.shape([m, n]), dtype="float32")
            out = R.add(A, zeros)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            return Expected.func(A)

        @R.function(private=True)
        def func(A: R.Tensor(["m", "n"], "float32")) -> R.Tensor(["m", "n"], "float32"):
            m = T.int64()
            n = T.int64()
            zeros = R.zeros(R.shape([m, n]), dtype="float32")
            out = R.add(A, zeros)
            return out


if __name__ == "__main__":
    tvm.testing.main()
