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
from tvm.script import tir as T

import pytest


class BaseBeforeAfter(tvm.testing.CompareBeforeAfter):
    use_dataflow_analysis = False

    def transform(self):
        def inner(mod):
            config = {
                "tir.ReduceBranchingThroughOvercompute": {
                    "use_dataflow_analysis": self.use_dataflow_analysis,
                }
            }
            with tvm.transform.PassContext(config=config):
                mod = tvm.tir.transform.ReduceBranchingThroughOvercompute()(mod)
            return mod

        return inner


class TestIntroduceNoOp(BaseBeforeAfter):
    """Remove a conditional by introducing a no-op

    If the else_case can have a no-op added in order to be identical
    to the then_case, then the conditional can be removed.
    """

    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 14:
                A[i] = 1
                T.evaluate(0)
            else:
                A[i] = 1

    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 1
            T.evaluate(0)


class TestIntroduceAdditionOfZero(BaseBeforeAfter):
    """Insert a conditionally no-op statement

    Overcompute doesn't need to explicitly be a no-op, and can be
    something that simplifies to a no-op.  Here, when i==0, the
    expression simplifies to ``A[0] = A[0]``, which is a no-op.
    """

    use_dataflow_analysis = True

    def before(A: T.Buffer(1, "int32")):
        for i in T.serial(16):
            if i > 0:
                A[0] = A[0] + i * i

    def expected(A: T.Buffer(1, "int32")):
        for i in T.serial(16):
            A[0] = A[0] + i * i


class TestIntroduceAdditionOfKnownZeroInBuffer(BaseBeforeAfter):
    """Insert a conditionally no-op statement

    Proving that the overcompute is a no-op may use known values that
    are present in a buffer.
    """

    use_dataflow_analysis = True

    def before(A: T.Buffer(16, "int32"), B: T.Buffer(1, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(i < 14 or A[i] == 0))

        B[0] = 0
        for i in T.serial(16):
            if i < 14:
                B[0] = B[0] + A[i]

    def expected(A: T.Buffer(16, "int32"), B: T.Buffer(1, "int32")):
        for i in T.serial(16):
            T.evaluate(T.assume(i < 14 or A[i] == 0))

        B[0] = 0
        for i in T.serial(16):
            B[0] = B[0] + A[i]


class TestIntroduceOverwrittenWrite(BaseBeforeAfter):
    """Insert a write that is later overwritten.

    Given two sequential writes to the same location without a read
    occurring in-between, the first is a no-op.  Therefore, the
    conditional in the first loop can be removed, with any temporary
    values overwritten by the second loop.
    """

    use_dataflow_analysis = True

    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 14:
                A[i] = 1

        for i in T.serial(16):
            if i >= 14:
                A[i] = 2

    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 1

        for i in T.serial(16):
            if i >= 14:
                A[i] = 2


class TestMaintainValuesUsedLater(BaseBeforeAfter):
    """Do not insert writes that would be used later.

    As TestIntroduceOverwrittenWrite, except that the values stored at
    A[14] and A[15] are used by the second loop.  Overwriting them in
    the first loop would change the result, so the overcompute would
    not be valid.
    """

    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 14:
                A[i] = 1

        for i in T.serial(16):
            if i >= 14:
                A[i] = A[i] + 1

    expected = before


class TestIdentifyOverwrittenWriteFromEquivalentExpressions(BaseBeforeAfter):
    """Insert a write that is later overwritten.

    As TestIntroduceOverwrittenWrite, but the conditionals used in the
    first and second loop have different structures while referring to
    the same elements.
    """

    use_dataflow_analysis = True

    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 14:
                A[i] = 1

        for io, ii in T.grid(4, 4):
            if io == 3 and ii >= 2:
                A[4 * io + ii] = 2

    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 1

        for io, ii in T.grid(4, 4):
            if io == 3 and ii >= 2:
                A[4 * io + ii] = 2


class TestIntroduceSupersetOverwrittenWrite(BaseBeforeAfter):
    """Insert a write that is later overwritten.

    As TestIntroduceOverwrittenWrite, but the elements written in the
    second loop are not distinct from the elements in the first loop.
    So long as the writes introduced by overcompute in the first loop
    are a subset of the writes present in the second loop, the
    overcompute can be introduced.
    """

    use_dataflow_analysis = True

    def before(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            if i < 14:
                A[i] = 1

        for i in T.serial(16):
            if i >= 14:
                A[i] = 2

    def expected(A: T.Buffer(16, "int32")):
        for i in T.serial(16):
            A[i] = 1

        for i in T.serial(16):
            if i >= 14:
                A[i] = 2


if __name__ == "__main__":
    tvm.testing.main()
