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
from tvm.relax.transform.transform import CanonicalizeBindings
import tvm.script
import tvm.testing
import pytest
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script import relax as R, tir as T


def verify(input, expected):
    tvm.ir.assert_structural_equal(CanonicalizeBindings()(input), expected)


def test_simple_assignments():
    @tvm.script.ir_module
    class TestChainAssignments:
        @R.function
        def main(x: R.Tensor):
            y = x
            z = y
            q = z
            p = q
            o = p
            return o

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            return x

    verify(TestChainAssignments, Expected)


def test_dataflow_block():
    @tvm.script.ir_module
    class TestDataflowAssignments:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.const(1)
                z = y
                o = z
                p = o
                m = p
                n = m
                R.output(n)
            return n

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                n = R.const(1)
                R.output(n)
            return n

    verify(TestDataflowAssignments, Expected)


def test_assign_to_output_in_dataflow_block():
    @tvm.script.ir_module
    class TestDataflowAssignments:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = x  # is not a dataflow var
                z = y
                o = z
                p = o
                m = p
                n = m
                R.output(n)
            return n

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            # we get a dataflow block where the
            # only assignment is n = x, which we can eliminate,
            # resulting in an empty block that is normalized away
            return x

    verify(TestDataflowAssignments, Expected)


def test_ops():
    @tvm.script.ir_module
    class TestOps:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            w = y
            q = x
            z = R.add(w, q)
            return R.add(q, z)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            z = R.add(y, x)
            return R.add(x, z)

    verify(TestOps, Expected)


@pytest.mark.xfail(reason="The lhs and rhs of an assignment should have the same struct info.")
def test_casting():
    @tvm.script.ir_module
    class TestCasting:
        @R.function
        def main(x: R.Tensor) -> R.Object:
            y = x
            # z will be treated as object type even though it's a tensor
            z: R.Object = y
            return z

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor) -> R.Object:
            # Cannot unify because the cast indicates user intent
            z: R.Object = x
            return z

    verify(TestCasting, Expected)


def test_match_cast():
    @tvm.script.ir_module
    class TestMatchCast:
        @R.function
        def main(x: R.Tensor):
            q = x
            m, n = T.int64(), T.int64()
            z = R.match_cast(q, R.Tensor((m, n)))
            w = z
            return w

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            # can't get rid of z because its struct_info is different from x's
            m, n = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((m, n)))
            return z

    verify(TestMatchCast, Expected)


def test_same_shape():
    @tvm.script.ir_module
    class TestSameShape:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            m, n = T.int64(), T.int64()
            y = x
            # trivial check
            z = R.match_cast(x, R.Tensor((m, n), "float32"))
            w = z
            q = R.add(w, y)
            return R.add(q, w)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            # the trivial check is canonicalized into a var binding
            # and then eliminated
            q = R.add(x, x)
            return R.add(q, x)

    verify(TestSameShape, Expected)


def test_change_shape():
    @tvm.script.ir_module
    class TestChangeShape:
        @R.function
        def main(x: R.Tensor(("m", "n"))):
            y = x
            # not trivial: introduces new shape vars
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((o, p)))
            w = z
            q = R.add(w, y)
            return R.add(q, w)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"))):
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((o, p)))
            # the struct_info field on q will need to be updated
            q = R.add(z, x)
            return R.add(q, z)

    verify(TestChangeShape, Expected)


def test_unwrap_tuple():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            tuple_var = (x, y)
            w = tuple_var[0]
            q = tuple_var[1]
            z = R.add(w, q)
            return R.add(q, z)

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor, y: R.Tensor):
            tuple_var = (x, y)
            z = R.add(x, y)
            return R.add(y, z)

    verify(Before, Expected)


def test_basic_folding_example():
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


def test_fold_match_cast():
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
                # the cast is trivial, so it is removed
                n = R.const(1)
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
    # cannot fold output in this case
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
