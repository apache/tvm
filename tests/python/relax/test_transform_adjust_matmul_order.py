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

import numpy as np
import pytest
import torch

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


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
    """Prefer (x*A)*B instead of x*(A*B)

    LHS first - (x*A)*B:
        ops = 1*16*2 + 1*2*32 = 96
    RHS first - x*(A*B):
        ops = 16*2*32 + 1*16*32 = 1536
    """

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
    """Prefer A*(B*x) instead of (A*B)*x

    LHS first - (A*B)*x:
        ops = 32*2*16 + 32*16*1 = 1536
    RHS first - A*(B*x):
        ops = 2*16*1 + 32*2*1 = 96
    """

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

    LHS first - (x*A)*B:
        ops = 1*16*lora_r + 1*lora_r*32 = 48*lora_r
    RHS first - x*(A*B):
        ops = 16*lora_r*32 + 1*16*32 = 512*lora_r + 512

    48*lora_r can be proved to be less than 512*lora_r + 512, so the LHS first is preferred.
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
    """Prefer A*(B*x) instead of (A*B)*x

    LHS first - (A*B)*x:
        ops = 32*lora_r*16 + 32*16*1 = 512*lora_r + 512
    RHS first - A*(B*x):
        ops = lora_r*16*1 + 32*lora_r*1 = 48*lora_r

    48*lora_r can be proved to be less than 512*lora_r + 512, so the RHS first is preferred.
    """

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


class TestDynamicWithBatchSymbolic1(Base):
    """When both batch_size and lora_r are symbolic and it cannot be proven which
    is cheaper, LHS or RHS, maintain the existing order.

    `Before` computes `x * (A * B)` with
    `x: [batch_size, 1, 16]`, `A: [16, lora_r]`, `B: [lora_r, 32]`.

    RHS first - x * (A * B):
        16*lora_r*32 + batch_size*1*16*32 = 512*(lora_r + batch_size)

    LHS first - (x * A) * B:
        batch_size*1*16*lora_r + batch_size*1*lora_r*32 = 48*batch_size*lora_r

    When `batch_size` and `lora_r` are known at compile-time:
        - satisfy the inequality 48*batch_size*lora_r < 512*(lora_r + batch_size),
          the LHS first is preferred.
        - satisfy the inequality 512*(lora_r + batch_size) < 48*batch_size*lora_r,
          the RHS first is preferred.

    Without bounds on `batch_size` and `lora_r`, neither side is provably cheaper.
    """

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

    Expected = Before


class TestDynamicWithBatchConcrete1LHSFirst(Base):
    """With concrete shapes, LHS first is provably cheaper.

    batch_size=4, lora_r=16:
        LHS first: 48*4*16 = 3072
        RHS first: 512*(16 + 4) = 10240
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor(["batch_size", 1, 32]):
            batch_size = T.int64(4)
            lora_r = T.int64(16)  # noqa: F841
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
            batch_size = T.int64(4)
            lora_r = T.int64(16)
            weight: R.Tensor([batch_size, 1, lora_r]) = R.matmul(x, A)
            out: R.Tensor([batch_size, 1, 32]) = R.matmul(weight, B)
            return out


class TestDynamicWithBatchConcrete1RHSFirst(Base):
    """With concrete shapes, RHS first is provably cheaper.

    batch_size=64, lora_r=16:
        LHS first: 48*64*16 = 49152
        RHS first: 512*(16 + 64) = 40960
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor(["batch_size", 1, 32]):
            batch_size = T.int64(64)
            lora_r = T.int64(16)
            weight: R.Tensor([batch_size, 1, lora_r]) = R.matmul(x, A)
            out: R.Tensor([batch_size, 1, 32]) = R.matmul(weight, B)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1, 16]),
            A: R.Tensor([16, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor(["batch_size", 1, 32]):
            batch_size = T.int64(64)
            lora_r = T.int64(16)  # noqa: F841
            weight: R.Tensor([16, 32]) = R.matmul(A, B)
            out: R.Tensor([batch_size, 1, 32]) = R.matmul(x, weight)
            return out


class TestDynamicWithBatchSymbolic2(Base):
    """When both batch_size and lora_r are symbolic and it cannot be proven which
    is cheaper, LHS or RHS, maintain the existing order.

    `Before` computes `(A * B) * x` with
    `A: [32, lora_r]`, `B: [lora_r, 16]`, `x: [batch_size, 16, 1]`.

    LHS first - (A * B) * x:
        32*lora_r*16 + batch_size*32*16*1 = 512*(lora_r + batch_size)

    RHS first - A * (B * x):
        batch_size*lora_r*16*1 + batch_size*32*lora_r*1 = 48*batch_size*lora_r

    When `batch_size` and `lora_r` are known at compile-time:
        - satisfy the inequality 48*batch_size*lora_r < 512*(lora_r + batch_size),
          the RHS first is preferred.
        - satisfy the inequality 512*(lora_r + batch_size) < 48*batch_size*lora_r,
          the LHS first is preferred.

    Without bounds on `batch_size` and `lora_r`, neither side is provably cheaper.
    """

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

    Expected = Before


class TestDynamicWithBatchConcrete2RHSFirst(Base):
    """With concrete shapes, RHS first is provably cheaper.

    batch_size=4, lora_r=16:
        RHS first: 48*4*16 = 3072
        LHS first: 512*(16 + 4) = 10240
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor(["batch_size", 32, 1]):
            batch_size = T.int64(4)
            lora_r = T.int64(16)  # noqa: F841
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
            batch_size = T.int64(4)
            lora_r = T.int64(16)
            weight: R.Tensor([batch_size, lora_r, 1]) = R.matmul(B, x)
            out: R.Tensor([batch_size, 32, 1]) = R.matmul(A, weight)
            return out


class TestDynamicWithBatchConcrete2LHSFirst(Base):
    """With concrete shapes, LHS first is provably cheaper.

    batch_size=64, lora_r=16:
        RHS first: 48*64*16 = 49152
        LHS first: 512*(16 + 64) = 40960
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor(["batch_size", 32, 1]):
            batch_size = T.int64(64)
            lora_r = T.int64(16)
            weight: R.Tensor([batch_size, lora_r, 1]) = R.matmul(B, x)
            out: R.Tensor([batch_size, 32, 1]) = R.matmul(A, weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 16, 1]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor(["batch_size", 32, 1]):
            batch_size = T.int64(64)
            lora_r = T.int64(16)  # noqa: F841
            weight: R.Tensor([32, 16]) = R.matmul(A, B)
            out: R.Tensor([batch_size, 32, 1]) = R.matmul(weight, x)
            return out


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


class TestRHSPermuteDims(Base):
    """Prefer (x*A)*B instead of x*(A*B)

    Like `TestRHS`, but the weights on the RHS are transposed.

    Before: x * (BT * AT)
        ops = 16*2*32 + 1*16*32 = 1536
    After: (x * BT) * AT
        ops = 1*16*2 + 1*2*32 = 96
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, 2]),
            B: R.Tensor([2, 16]),
        ) -> R.Tensor([32]):
            linear_weight: R.Tensor([32, 16]) = R.matmul(A, B)
            matmul_weight: R.Tensor([16, 32]) = R.permute_dims(linear_weight)
            out: R.Tensor([32]) = R.matmul(x, matmul_weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, 2]),
            B: R.Tensor([2, 16]),
        ) -> R.Tensor([32]):
            B_transpose = R.permute_dims(B)
            x: R.Tensor([2]) = R.matmul(x, B_transpose)
            A_transpose = R.permute_dims(A)
            x: R.Tensor([32]) = R.matmul(x, A_transpose)
            return x


class TestRHSPermuteDimsDynamic(Base):
    """Prefer (x*A)*B instead of x*(A*B)

    Like `TestRHSPermuteDims`, but the weights on the RHS have a
    dynamic shape.

    Before: x * (BT * AT)
        ops = 16*lora_r*32 + 1*16*32 = 512*lora_r + 512
    After: (x * BT) * AT
        ops = 1*16*lora_r + 1*lora_r*32 = 48*lora_r

    48*lora_r can be proved to be less than 512*lora_r + 512, so the After is preferred.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([16]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 16]),
        ) -> R.Tensor([32]):
            linear_weight: R.Tensor([32, 16]) = R.matmul(A, B)
            matmul_weight: R.Tensor([16, 32]) = R.permute_dims(linear_weight)
            out: R.Tensor([32]) = R.matmul(x, matmul_weight)
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
            B_transpose = R.permute_dims(B)
            x: R.Tensor([lora_r]) = R.matmul(x, B_transpose)
            A_transpose = R.permute_dims(A)
            x: R.Tensor([32]) = R.matmul(x, A_transpose)
            return x


class TestRHSPermuteDimsWithDynamicBatch(Base):
    """Prefer (x*A)*B instead of x*(A*B)

    Like `TestRHSPermuteDims`, but both the weights on the RHS and the
    activations on the LHS have a dynamic dimension.

    Unlike most of the tests for this transform, the
    `tir_vars_upper_bound` attribute is required.  In order to make a
    change, `AdjustMatmulOrder` must first prove that the modified
    execution order reduces the number of computations.

        ops_left_to_right = (batch_size + lora_r)*4096*4096
        ops_right_to_left = (4096 + 4096)*batch_size*lora_r

    Without an upper bound on batch_size and`lora_r`, we cannot prove which
    of these is the preferred execution order.

    With the upper bound, TVM can determine the preferred order using
    the following arithmetic reasoning.

        (batch_size + lora_r)*4096*4096 > (4096 + 4096)*batch_size*lora_r
        (batch_size + lora_r)*2048 > batch_size*lora_r
        1/batch_size + 1/lora_r > 1/2048
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 4096]),
            A: R.Tensor([4096, "lora_r"]),
            B: R.Tensor(["lora_r", 4096]),
        ) -> R.Tensor(["batch_size", 4096]):
            R.func_attr(
                {
                    "tir_var_upper_bound": {"lora_r": 2048, "batch_size": 2048},
                }
            )
            lora_r = T.int64()  # noqa: F841
            batch_size = T.int64()
            linear_weight: R.Tensor([4096, 4096]) = R.matmul(A, B)
            matmul_weight: R.Tensor([4096, 4096]) = R.permute_dims(linear_weight)
            out: R.Tensor([batch_size, 4096]) = R.matmul(x, matmul_weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 4096]),
            A: R.Tensor([4096, "lora_r"]),
            B: R.Tensor(["lora_r", 4096]),
        ) -> R.Tensor(["batch_size", 4096]):
            R.func_attr(
                {
                    "tir_var_upper_bound": {"lora_r": 2048, "batch_size": 2048},
                }
            )
            lora_r = T.int64()
            batch_size = T.int64()
            B_transpose = R.permute_dims(B)
            x: R.Tensor([batch_size, lora_r]) = R.matmul(x, B_transpose)
            A_transpose = R.permute_dims(A)
            x: R.Tensor([batch_size, 4096]) = R.matmul(x, A_transpose)
            return x


class TestRHSPermuteDimsDynamicWithSquareMatrix(Base):
    """Prefer (x*A)*B instead of x*(A*B)

    Like `TestRHSPermuteDims`, but the weights on the RHS have a
    dynamic shape.

    Before: x * (BT * AT)
        ops = 32*lora_r*32 + 1*32*32 = 1024*lora_r + 1024
    After: (x * BT) * AT
        ops = 1*32*lora_r + 1*lora_r*32 = 64*lora_r
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([32]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor([32]):
            linear_weight: R.Tensor([32, 32]) = R.matmul(A, B)
            matmul_weight: R.Tensor([32, 32]) = R.permute_dims(linear_weight)
            out: R.Tensor([32]) = R.matmul(x, matmul_weight)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([32]),
            A: R.Tensor([32, "lora_r"]),
            B: R.Tensor(["lora_r", 32]),
        ) -> R.Tensor([32]):
            lora_r = T.int64()
            B_transpose = R.permute_dims(B)
            x: R.Tensor([lora_r]) = R.matmul(x, B_transpose)
            A_transpose = R.permute_dims(A)
            x: R.Tensor([32]) = R.matmul(x, A_transpose)
            return x


class TestBatchedBroadcastPreferLHSFirst(Base):
    """Use broadcasted batch prefix per matmul, not independent prefix products.

    Example with broadcast batch axes: A:[2,1,1], B:[2,1,2], C:[2,2,3].

    LHS first: (A * B) * C
        ops = 2*1*1*2 + 2*1*2*3 = 16
    RHS first: A * (B * C)
        ops = 2*1*2*3 + 2*1*1*3 = 18
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([2, 1, 1]),
            B: R.Tensor([2, 1, 2]),
            C: R.Tensor([2, 2, 3]),
        ) -> R.Tensor([2, 1, 3]):
            out: R.Tensor([2, 1, 3]) = R.matmul(A, R.matmul(B, C))
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([2, 1, 1]),
            B: R.Tensor([2, 1, 2]),
            C: R.Tensor([2, 2, 3]),
        ) -> R.Tensor([2, 1, 3]):
            temp: R.Tensor([2, 1, 2]) = R.matmul(A, B)
            out: R.Tensor([2, 1, 3]) = R.matmul(temp, C)
            return out


class TestBatchedSharedPrefixPreferLHSFirst(Base):
    """All operands share a nontrivial batch prefix [2, 3].

    Shapes: A:[2,3,4,5], B:[2,3,5,6], C:[2,3,6,7]

    LHS first:
        ops = 6*4*5*6 + 6*4*6*7 = 1728
    RHS first:
        ops = 6*5*6*7 + 6*4*5*7 = 2100
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            A: R.Tensor([2, 3, 4, 5]),
            B: R.Tensor([2, 3, 5, 6]),
            C: R.Tensor([2, 3, 6, 7]),
        ) -> R.Tensor([2, 3, 4, 7]):
            out: R.Tensor([2, 3, 4, 7]) = R.matmul(A, R.matmul(B, C))
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            A: R.Tensor([2, 3, 4, 5]),
            B: R.Tensor([2, 3, 5, 6]),
            C: R.Tensor([2, 3, 6, 7]),
        ) -> R.Tensor([2, 3, 4, 7]):
            temp: R.Tensor([2, 3, 4, 6]) = R.matmul(A, B)
            out: R.Tensor([2, 3, 4, 7]) = R.matmul(temp, C)
            return out


class TestAdjustMatmulOrderAttentionBlock:
    """AdjustMatmulOrder preserves numerics on a batched attention block.

    Covers ND `permute_dims` (swap last two axes) inside `matmul(q, kt)`,
    regression for issue #19576.
    """

    def _build_attention_module(self, batch, seq, dim):
        """Minimal batched attention block exercising ND permute_dims + matmul."""
        bb = relax.BlockBuilder()
        x = relax.Var("x", relax.TensorStructInfo((batch, seq, dim), "float32"))
        wq = relax.Var("wq", relax.TensorStructInfo((dim, dim), "float32"))
        wk = relax.Var("wk", relax.TensorStructInfo((dim, dim), "float32"))
        wv = relax.Var("wv", relax.TensorStructInfo((dim, dim), "float32"))
        wo = relax.Var("wo", relax.TensorStructInfo((dim, dim), "float32"))
        with bb.function("main", [x, wq, wk, wv, wo]):
            with bb.dataflow():
                q = bb.emit(relax.op.matmul(x, wq))
                k = bb.emit(relax.op.matmul(x, wk))
                v = bb.emit(relax.op.matmul(x, wv))
                kt = bb.emit(relax.op.permute_dims(k, axes=[0, 2, 1]))
                scores = bb.emit(relax.op.matmul(q, kt))
                scale = bb.emit(relax.const(1.0 / np.sqrt(dim), "float32"))
                scores = bb.emit(relax.op.multiply(scores, scale))
                attn = bb.emit(relax.op.nn.softmax(scores, axis=-1))
                out = bb.emit(relax.op.matmul(attn, v))
                proj = bb.emit_output(relax.op.matmul(out, wo))
            bb.emit_func_output(proj)
        return bb.finalize()

    def _run_relax_main(self, mod, inputs):
        exe = relax.build(mod, target="llvm")
        vm = relax.VirtualMachine(exe, device=tvm.cpu())
        args = [tvm.runtime.tensor(arr, device=tvm.cpu()) for arr in inputs]
        return vm["main"](*args).numpy()

    def _torch_attention_ref(self, x_np, w_np, dim):
        x = torch.from_numpy(x_np)
        w = torch.from_numpy(w_np)
        with torch.no_grad():
            q = torch.matmul(x, w)
            k = torch.matmul(x, w)
            v = torch.matmul(x, w)
            scores = torch.matmul(q, k.transpose(-2, -1))
            scores = scores * (1.0 / np.sqrt(dim))
            attn = torch.nn.functional.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = torch.matmul(out, w)
        return out.detach().numpy()

    @pytest.mark.parametrize("batch,seq,dim", [(2, 16, 64)])
    def test_attention_block_numerics(self, batch, seq, dim):
        mod = self._build_attention_module(batch, seq, dim)
        mod_opt = relax.transform.AdjustMatmulOrder()(mod)

        x_np = np.random.randn(batch, seq, dim).astype("float32")
        w_np = np.random.randn(dim, dim).astype("float32")
        inputs = [x_np, w_np, w_np, w_np, w_np]

        ref = self._torch_attention_ref(x_np, w_np, dim)
        out_before = self._run_relax_main(mod, inputs)
        out_after = self._run_relax_main(mod_opt, inputs)

        tvm.testing.assert_allclose(out_before, ref, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(out_after, ref, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(out_before, out_after, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
