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
from tvm.script import tir as T, ir as I


def test_introduce_no_op():
    """Remove a conditional by introducing a no-op

    If the else_case can have a no-op added in order to be identical
    to the then_case, then the conditional can be removed.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                if i < 14:
                    A[i] = 1
                    T.evaluate(0)
                else:
                    A[i] = 1

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 1
                T.evaluate(0)

    After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_introduce_addition_of_zero():
    """Insert a conditionally no-op statement

    Overcompute doesn't need to explicitly be a no-op, and can be
    something that simplifies to a no-op.  Here, when i==0, the
    expression simplifies to ``A[0] = A[0]``, which is a no-op.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(1, "int32")):
            for i in T.serial(16):
                if i > 0:
                    A[0] = A[0] + i * i

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(1, "int32")):
            for i in T.serial(16):
                A[0] = A[0] + i * i

    config = {
        "tir.ReduceBranchingThroughOvercompute": {
            "use_dataflow_analysis": True,
        }
    }
    with tvm.transform.PassContext(config=config):
        After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_introduce_addition_of_known_zero_in_buffer():
    """Insert a conditionally no-op statement

    Proving that the overcompute is a no-op may use known values that
    are present in a buffer.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32"), B: T.Buffer(1, "int32")):
            for i in T.serial(16):
                T.evaluate(T.assume(i < 14 or A[i] == 0))

            B[0] = 0
            for i in T.serial(16):
                if i < 14:
                    B[0] = B[0] + A[i]

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "int32"), B: T.Buffer(1, "int32")):
            for i in T.serial(16):
                T.evaluate(T.assume(i < 14 or A[i] == 0))

            B[0] = 0
            for i in T.serial(16):
                B[0] = B[0] + A[i]

    config = {
        "tir.ReduceBranchingThroughOvercompute": {
            "use_dataflow_analysis": True,
        }
    }
    with tvm.transform.PassContext(config=config):
        After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_introduce_overwritten_write():
    """Insert a write that is later overwritten.

    Given two sequential writes to the same location without a read
    occurring in-between, the first is a no-op.  Therefore, the
    conditional in the first loop can be removed, with any temporary
    values overwritten by the second loop.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                if i < 14:
                    A[i] = 1

            for i in T.serial(16):
                if i >= 14:
                    A[i] = 2

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 1

            for i in T.serial(16):
                if i >= 14:
                    A[i] = 2

    config = {
        "tir.ReduceBranchingThroughOvercompute": {
            "use_dataflow_analysis": True,
        }
    }
    with tvm.transform.PassContext(config=config):
        After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_maintain_values_used_later():
    """Do not insert writes that would be used later.

    As TestIntroduceOverwrittenWrite, except that the values stored at
    A[14] and A[15] are used by the second loop.  Overwriting them in
    the first loop would change the result, so the overcompute would
    not be valid.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                if i < 14:
                    A[i] = 1

            for i in T.serial(16):
                if i >= 14:
                    A[i] = A[i] + 1

    Expected = Before

    After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_identify_overwritten_write_from_equivalent_expressions():
    """Insert a write that is later overwritten.

    As TestIntroduceOverwrittenWrite, but the conditionals used in the
    first and second loop have different structures while referring to
    the same elements.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                if i < 14:
                    A[i] = 1

            for io, ii in T.grid(4, 4):
                if io == 3 and ii >= 2:
                    A[4 * io + ii] = 2

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 1

            for io, ii in T.grid(4, 4):
                if io == 3 and ii >= 2:
                    A[4 * io + ii] = 2

    config = {
        "tir.ReduceBranchingThroughOvercompute": {
            "use_dataflow_analysis": True,
        }
    }
    with tvm.transform.PassContext(config=config):
        After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_introduce_superset_overwritten_write():
    """Insert a write that is later overwritten.

    As TestIntroduceOverwrittenWrite, but the elements written in the
    second loop are not distinct from the elements in the first loop.
    So long as the writes introduced by overcompute in the first loop
    are a subset of the writes present in the second loop, the
    overcompute can be introduced.
    """

    @I.ir_module
    class Before:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                if i < 14:
                    A[i] = 1

            for i in T.serial(16):
                if i >= 14:
                    A[i] = 2

    @I.ir_module
    class Expected:
        @T.prim_func
        def main(A: T.Buffer(16, "int32")):
            for i in T.serial(16):
                A[i] = 1

            for i in T.serial(16):
                if i >= 14:
                    A[i] = 2

    config = {
        "tir.ReduceBranchingThroughOvercompute": {
            "use_dataflow_analysis": True,
        }
    }
    with tvm.transform.PassContext(config=config):
        After = tvm.tir.transform.ReduceBranchingThroughOvercompute()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
