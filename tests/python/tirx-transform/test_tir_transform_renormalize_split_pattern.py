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
from tvm.script import tirx as T


@tvm.script.ir_module
class Before:
    @T.function
    def main(A: T.Tensor((768,), "int32"), B: T.Tensor((96,), "int32")):
        for i in T.serial(24):
            for j in T.serial(32):
                A[i * 32 + j] = T.if_then_else(
                    128 <= i * 32 + j and i * 32 + j < 640,
                    (i * 32 + j) % 128 // 32,
                    0,
                )
            B[T.ramp(i * 4, 1, 4)] = T.ramp(i * 128, 1, 4) % 256 // 8


@tvm.script.ir_module
class After:
    @T.function
    def main(A: T.Tensor((768,), "int32"), B: T.Tensor((96,), "int32")):
        for i in T.serial(24):
            for j in T.serial(32):
                A[i * 32 + j] = T.if_then_else(
                    1 <= (i + j // 32) // 4 and (i + j // 32) // 20 < 1,
                    (i + j // 32) % 4,
                    0,
                )
            B[T.ramp(i * 4, 1, 4)] = T.ramp(i * 128, 1, 4) // 8 % 32


@tvm.script.ir_module
class After_simplified:
    @T.function
    def main(A: T.Tensor((768,), "int32"), B: T.Tensor((96,), "int32")):
        for i in T.serial(24):
            for j in T.serial(32):
                A[i * 32 + j] = T.if_then_else(4 <= i and i < 20, i % 4, 0)
            B[T.ramp(i * 4, 1, 4)] = T.broadcast(i % 2 * 16, 4)


def test_renormalize_split_pattern():
    after = tvm.tirx.transform.RenormalizeSplitPattern()(Before)
    tvm.ir.assert_structural_equal(after, After)
    after = tvm.tirx.transform.StmtSimplify()(after)
    tvm.ir.assert_structural_equal(after, After_simplified)


@T.function
def impossible_equality(n: T.int32):
    # Prior to bugfix, this conditional defined the expression "2" as
    # equal to zero within the then_case. [min_value=2, max_value=0]
    if 2 == 0:
        # Then this expression evaluates n/2, using the min/max values
        # of "2", which is caught as a divide by zero error.
        if n // 2 >= 16:
            T.evaluate(0)


@T.function
def impossible_inequality(n: T.int32):
    # Prior to bugfix, this conditional set up a range of possible
    # values for the expression "-2" as [0, kPosInf].
    if -1 < -2:
        if n // (-2) >= 16:
            T.evaluate(0)


integer_condition = tvm.testing.parameter(
    impossible_equality,
    impossible_inequality,
)


def test_analyze_inside_integer_conditional(integer_condition):
    """Avoid crash occurring in ConstIntBoundAnalyzer.

    Crash occurred when simplifying some expressions with provably
    false integer expressions.  If the expressions were renormalized
    before calling Simplify, conditional statements could assign a
    range of possible values to integers, as if they were variables.
    This would result in divide by zero throwing an exception,
    followed by a second exception during stack unwinding causing the
    program to crash.
    """

    # Similar issue would occur in most transformations that subclass
    # IRMutatorWithAnalyzer.  tirx.transform.StmtSimplify() is an
    # exception, as it rewrites the integer conditionals first.  These
    # tests are written using RenormalizeSplitPattern as it is the
    # first case identified.
    transform = tvm.tirx.transform.RenormalizeSplitPattern()

    # Issue would result in an error through while applying the transformation.
    mod = tvm.IRModule.from_expr(integer_condition)
    transform(mod)


if __name__ == "__main__":
    tvm.testing.main()
