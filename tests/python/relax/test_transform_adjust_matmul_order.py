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

import inspect

import pytest

import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R


class Base:
    def test_compare(self):
        transform = relax.transform.AdjustMatmulOrder()

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm.ir.assert_structural_equal(self.Expected, after)


class TestSimple(Base):
    """Prefer A*(B*x) instead of (A*B)*x"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 2], "float32"),
            B: R.Tensor([2, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            weight = R.matmul(A, B)
            out = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 2], "float32"),
            B: R.Tensor([2, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            x = R.matmul(x, A)
            x = R.matmul(x, B)
            return x


class TestPreserveCompileTimeMatmul(Base):
    """Prefer (A*B)*x if (A*B) can be pre-computed

    If both `A` and `B` are known at compile-time, they can be lifted
    out and computed at compile-time.  Therefore, optimization should
    avoid breaking apart the `(A*B)` expression.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16], "float32"),
            A: R.Tensor([16, 2], "float32"),
            B: R.Tensor([2, 32], "float32"),
        ) -> R.Tensor([32], "float32"):
            R.func_attr({"num_input": 1})

            weight = R.matmul(A, B)
            out = R.matmul(x, weight)
            return out

    Expected = Before


if __name__ == "__main__":
    tvm.testing.main()
