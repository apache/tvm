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
"""Test fixpoint simplification pass"""
import tvm
import tvm.testing
from tvm.relax.transform import CanonicalizeBindings, DeadCodeElimination, FixpointSimplification
from tvm.script.parser import ir as I, relax as R, tir as T


def verify(input: tvm.IRModule, expected: tvm.IRModule) -> None:
    actual = FixpointSimplification()(input)
    tvm.ir.assert_structural_equal(actual, expected, map_free_vars=True)


def test_chain_assignment():
    # test case from binding canonicalization, except it will simplify all the way
    @I.ir_module
    class TestChainAssignments:
        @R.function
        def main(x: R.Tensor):
            # need the dataflow block for DCE to work
            with R.dataflow():
                y = x
                z = y
                q = z
                p = q
                o = p
                R.output(o)
            return o

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor):
            return x

    verify(TestChainAssignments, Expected)


def test_eliminate_trivial_check():
    # another case from canonicalize bindings that can be further simplified
    @I.ir_module
    class TestSameShape:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            # need the dataflow block for DCE to work
            with R.dataflow():
                m, n = T.int64(), T.int64()
                y = x
                # trivial check, eliminated by canonicalize bindings
                z = R.match_cast(x, R.Tensor((m, n), "float32"))
                w = z
                q = R.add(w, y)
                r = R.add(q, w)
                R.output(r)
            return r

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            with R.dataflow():
                q = R.add(x, x)
                r = R.add(q, x)
                R.output(r)
            return r

    verify(TestSameShape, Expected)


if __name__ == "__main__":
    tvm.testing.main()
