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
from tvm.script import ir as I, relax as R, tir as T


class Base:
    def test_compare(self):
        transform = relax.transform.AdjustMatmulOrder()

        if inspect.isclass(self.Expected) and issubclass(self.Expected, Exception):
            with pytest.raises(self.Expected):
                transform(self.Before)
        else:
            after = transform(self.Before)
            tvm.ir.assert_structural_equal(self.Expected, after)


class TestLHS(Base):
    """Prefer (x*A)*B instead of x*(A*B)"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([16, 2]),
            B: R.Tensor([2, 32]),
        ) -> R.Tensor([32]):
            weight: R.Tensor([16, 32]) = R.matmul(A, B)
            out: R.Tensor([32]) = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([16, 2]),
            B: R.Tensor([2, 32]),
        ) -> R.Tensor([32]):
            x: R.Tensor([2]) = R.matmul(x, A)
            x: R.Tensor([32]) = R.matmul(x, B)
            return x


class TestRHS(Base):
    """Prefer A*(B*x) instead of (A*B)*x"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, 2]),
            B: R.Tensor([2, 16]),
        ) -> R.Tensor([32]):
            weight: R.Tensor([32, 16]) = R.matmul(A, B)
            out: R.Tensor([32]) = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, 2]),
            B: R.Tensor([2, 16]),
        ) -> R.Tensor([32]):
            x: R.Tensor([2]) = R.matmul(B, x)
            x: R.Tensor([32]) = R.matmul(A, x)
            return x


class TestIdempotentLHS(Base):
    """The transform shouldn't undo itself if re-applied"""

    Before = TestLHS.Expected
    Expected = TestLHS.Expected


class TestIdempotentRHS(Base):
    """The transform shouldn't undo itself if re-applied"""

    Before = TestRHS.Expected
    Expected = TestRHS.Expected


class TestPreserveCompileTimeMatmulOnLHS(Base):
    """Prefer x*(A*B) if (A*B) can be pre-computed

    If both `A` and `B` are known at compile-time, they can be lifted
    out and computed at compile-time.  Therefore, optimization should
    avoid breaking apart the `(A*B)` expression.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, 2]),
            B: R.Tensor([2, 16]),
        ) -> R.Tensor([32]):
            R.func_attr({"num_input": 1})

            weight = R.matmul(A, B)
            out = R.matmul(weight, x)
            return out

    Expected = Before


class TestPreserveCompileTimeMatmulOnRHS(Base):
    """Prefer (A*B)*x if (A*B) can be pre-computed

    If both `A` and `B` are known at compile-time, they can be lifted
    out and computed at compile-time.  Therefore, optimization should
    avoid breaking apart the `(A*B)` expression.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([16, 2]),
            B: R.Tensor([2, 32]),
        ) -> R.Tensor([32]):
            R.func_attr({"num_input": 1})

            weight = R.matmul(A, B)
            out = R.matmul(x, weight)
            return out

    Expected = Before


class TestLHSDynamic(Base):
    """Prefer (x*A)*B instead of x*(A*B)

    This case appears when evaluating LoRA-tuned models with a dynamic
    rank.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor([32]):
            weight: R.Tensor([16, 32]) = R.matmul(A, B)
            out: R.Tensor([32]) = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor([32]):
            lora_r = T.int64()
            x: R.Tensor([lora_r]) = R.matmul(x, A)
            x: R.Tensor([32]) = R.matmul(x, B)
            return x


class TestRHSDynamic(Base):
    """Prefer A*(B*x) instead of (A*B)*x"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor([32]):
            weight: R.Tensor([32, 16]) = R.matmul(A, B)
            out: R.Tensor([32]) = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor([32]):
            lora_r = T.int64()
            x: R.Tensor([lora_r]) = R.matmul(B, x)
            x: R.Tensor([32]) = R.matmul(A, x)
            return x


class TestIdempotentLHSDynamic(Base):
    """The transform shouldn't undo itself if re-applied"""

    Before = TestLHSDynamic.Expected
    Expected = TestLHSDynamic.Expected


class TestIdempotentRHSDynamic(Base):
    """The transform shouldn't undo itself if re-applied"""

    Before = TestRHSDynamic.Expected
    Expected = TestRHSDynamic.Expected


class TestLHSDynamicWithBatch(Base):
    """Prefer (x*A)*B instead of x*(A*B)"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor(["batch_size", 1, 32]):
            batch_size = T.int64()
            weight: R.Tensor([16, 32]) = R.matmul(A, B)
            out: R.Tensor([batch_size, 1, 32]) = R.matmul(x, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor(["batch_size", 1, 32]):
            lora_r = T.int64()
            batch_size = T.int64()
            x: R.Tensor([batch_size, 1, lora_r]) = R.matmul(x, A)
            x: R.Tensor([batch_size, 1, 32]) = R.matmul(x, B)
            return x


class TestRHSDynamicWithBatch(Base):
    """Prefer A*(B*x) instead of (A*B)*x"""

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor(["batch_size", 32, 1]):
            batch_size = T.int64()
            weight: R.Tensor([32, 16]) = R.matmul(A, B)
            out: R.Tensor([batch_size, 32, 1]) = R.matmul(weight, x)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor(["batch_size", 32, 1]):
            lora_r = T.int64()
            batch_size = T.int64()
            x: R.Tensor([batch_size, lora_r, 1]) = R.matmul(B, x)
            x: R.Tensor([batch_size, 32, 1]) = R.matmul(A, x)
            return x


class TestNoOpForFullyDynamicOnLHS(Base):
    """Keep existing order if no benefit can be proven

    Like `TestNoOpForFullyDynamicOnRHS`, except the input has the LHS
    computed first.

    Here, it is uncertain whether a reordering would improve the
    number of operations.

    LHS first: (M+P)*N*Q
    RHS first: (N+Q)*M*P

    After simplifying, the LHS should be performed first if the
    following inequality holds:

        1/M + 1/P < 1/N + 1/Q
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor(["M", "N"]),
            B: R.Tensor(["N", "P"]),
            C: R.Tensor(["P", "Q"]),
        ):
            out = R.matmul(R.matmul(A, B), C)
            return out

    Expected = Before


class TestNoOpForFullyDynamicOnRHS(Base):
    """Keep existing order if no benefit can be proven

    Like `TestNoOpForFullyDynamicOnLHS`, except the input has the RHS
    computed first.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor(["M", "N"]),
            B: R.Tensor(["N", "P"]),
            C: R.Tensor(["P", "Q"]),
        ):
            out = R.matmul(A, R.matmul(B, C))
            return out

    Expected = Before


if __name__ == "__main__":
    tvm.testing.main()
