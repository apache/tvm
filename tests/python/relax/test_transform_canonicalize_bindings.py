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
import tvm.script
import tvm.testing
import pytest
from tvm import relax
from tvm.ir.base import assert_structural_equal
from tvm.script import relax as R, tir as T


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

    # a little annoying to have these unused bindings around
    # but they can be eliminated in a separate pass
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            y = x
            z = x
            q = x
            p = x
            o = x
            return x

    new_mod = relax.transform.CanonicalizeBindings()(TestChainAssignments)
    assert_structural_equal(new_mod, Expected)


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

    # a little annoying to have these unused bindings around
    # but they can be eliminated in a separate pass
    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            with R.dataflow():
                y = R.const(1)
                z = y
                o = y
                p = y
                m = y
                # we can't get rid of n because it leaves the block
                n = y
                R.output(n)
            return n

    new_mod = relax.transform.CanonicalizeBindings()(TestDataflowAssignments)
    assert_structural_equal(new_mod, Expected)


def test_assign_to_output_indataflow_block():
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
            with R.dataflow():
                y = x
                z = x
                o = x
                p = x
                m = x
                # we can't get rid of n because it leaves the block
                n = x
                R.output(n)
            return x

    new_mod = relax.transform.CanonicalizeBindings()(TestDataflowAssignments)
    assert_structural_equal(new_mod, Expected)


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
            w = y
            q = x
            z = R.add(y, x)
            return R.add(x, z)

    new_mod = relax.transform.CanonicalizeBindings()(TestOps)
    assert_structural_equal(new_mod, Expected)


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
            y = x
            # Cannot unify because the cast indicates user intent
            z: R.Object = x
            return z

    new_mod = relax.transform.CanonicalizeBindings()(TestCasting)
    assert_structural_equal(new_mod, Expected)


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
            q = x
            # can't get rid of z because its shape_ is different from x's
            m, n = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((m, n)))
            w = z
            return z

    new_mod = relax.transform.CanonicalizeBindings()(TestMatchCast)
    assert_structural_equal(new_mod, Expected)


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
            m, n = T.int64(), T.int64()
            y = x
            # canonicalized into a var binding
            z = x
            w = x
            q = R.add(x, x)
            return R.add(q, x)

    new_mod = relax.transform.CanonicalizeBindings()(TestSameShape)
    assert_structural_equal(new_mod, Expected)


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
            y = x
            o, p = T.int64(), T.int64()
            z = R.match_cast(x, R.Tensor((o, p)))
            w = z
            # the shape_ field on q will need to be updated
            q = R.add(z, x)
            return R.add(q, z)

    new_mod = relax.transform.CanonicalizeBindings()(TestChangeShape)
    assert_structural_equal(new_mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
