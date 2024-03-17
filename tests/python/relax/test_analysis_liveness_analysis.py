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

from typing import Set
import tvm
import tvm.testing
from tvm.script import ir as I, relax as R
from tvm.relax import Var
from tvm.relax.analysis import liveness_analysis


def assert_live_set(live_set: Set[Var], var_names: Set[str]) -> None:
    assert len(live_set) == len(var_names)
    for var in live_set:
        assert var.name_hint in var_names


def test_simple_liveness():
    @I.ir_module
    class SimpleFunc:
        @R.function
        def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            y = R.add(x, x)  # live: x
            z = R.add(y, y)  # live: y
            return z  # live: z

    live_sets = liveness_analysis(SimpleFunc["main"])
    assert_live_set(live_sets[0], {"x"})
    assert_live_set(live_sets[1], {"y"})
    assert_live_set(live_sets[2], {"z"})


def test_liveness_with_branches():
    @I.ir_module
    class BranchingFunc:
        @R.function
        def main(
            x: R.Tensor((), dtype="int32"),
            y: R.Tensor((), dtype="int32"),
            cond: R.Tensor((), dtype="bool"),
        ) -> R.Tensor((), dtype="int32"):
            z = R.add(x, x)  # live: x, y, cond
            q = R.add(z, z)  # live: y, z, cond
            if cond:  # live: q, y, cond
                r = R.subtract(q, y)  # live: q, y
                s = R.multiply(r, r)  # live: r
                # end of seq: the R.multiply will actually be bound to a fresh var
                #   and s will be used as the binding for the entire If node
            else:
                r = R.multiply(q, q)  # live: q, y
                s = R.subtract(r, y)  # live: r, y
                # end of seq: the R.subtract will actually be bound to a fresh var
                #   and s will be used as the binding for the entire If node
            # merge point: nothing is live (s is the variable bound at the merge)
            t = R.add(s, s)  # live: s
            u = R.multiply(t, s)  # live: t, s
            return u  # live: u

    live_sets = liveness_analysis(BranchingFunc["main"])
    assert_live_set(live_sets[0], {"x", "y", "cond"})
    assert_live_set(live_sets[1], {"y", "z", "cond"})
    assert_live_set(live_sets[2], {"q", "y", "cond"})
    assert_live_set(live_sets[3], {"q", "y"})
    assert_live_set(live_sets[4], {"r"})
    # the name is created by the parser and will be a placeholder so this is the best we can do
    assert len(live_sets[5]) == 1 and (
        BranchingFunc["main"].body.blocks[0].bindings[2].value.true_branch.body in live_sets[5]
    )
    assert_live_set(live_sets[6], {"q", "y"})
    assert_live_set(live_sets[7], {"r", "y"})
    assert len(live_sets[8]) == 1 and (
        BranchingFunc["main"].body.blocks[0].bindings[2].value.false_branch.body in live_sets[8]
    )
    assert_live_set(live_sets[9], {})
    assert_live_set(live_sets[10], {"s"})
    assert_live_set(live_sets[11], {"t", "s"})
    assert_live_set(live_sets[12], {"u"})


def test_liveness_inner_func():
    @I.ir_module
    class InnerFunc:
        @R.function
        def main(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            y = R.add(x, x)  # live: x
            z = R.add(y, y)  # live: x, y

            # the inner func captures x and y and so counts as a use of both
            # live: x, y, z
            @R.function
            def inner(q: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
                # (note: we would need to do liveness analysis of the inner func
                # separately to get liveness info for these locations)
                r = R.add(x, q)  # live: x, y, q (and z from outside)
                s = R.multiply(y, r)  # live: y, r (and z from outside)
                return s  # live: s (and z from outside)

            w = inner(z)  # live: inner, z
            return w  # live: w

    live_sets = liveness_analysis(InnerFunc["main"])
    assert_live_set(live_sets[0], {"x"})
    assert_live_set(live_sets[1], {"x", "y"})
    assert_live_set(live_sets[2], {"x", "y", "z"})
    assert_live_set(live_sets[3], {"inner", "z"})
    assert_live_set(live_sets[4], {"w"})

    # let's also analyze the inner func (note: we don't have a way to indicate
    # that z is live from outside the func)
    inner_live = liveness_analysis(InnerFunc["main"].body.blocks[0].bindings[2].value)
    assert_live_set(inner_live[0], {"x", "y", "q"})
    assert_live_set(inner_live[1], {"y", "r"})
    assert_live_set(inner_live[2], {"s"})


if __name__ == "__main__":
    tvm.testing.main()
