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
from typing import List
import tvm
import tvm.testing
from tvm import relax as rx
from tvm.script import relax as R

from tvm.relax.analysis import find_mutual_recursion


def assert_groups(groups: List[List[rx.GlobalVar]], expected: List[List[str]]) -> None:
    assert len(groups) == len(expected)

    # disregard order, search only by name for convenience
    expected_sets = [set(expected_group) for expected_group in expected]
    actual_sets = [set(map(lambda gv: gv.name_hint, actual_group)) for actual_group in groups]

    for expected_set in expected_sets:
        assert expected_set in actual_sets


def test_no_recursion():
    @tvm.script.ir_module
    class NoRecursion:
        @R.function
        def a(x: R.Object) -> R.Object:
            return x

        @R.function
        def b(x: R.Object) -> R.Object:
            return x

    groups = find_mutual_recursion(NoRecursion)
    assert len(groups) == 0


def test_simple_recursion():
    @tvm.script.ir_module
    class SimpleRecursion:
        @R.function
        def c(x: R.Object) -> R.Object:
            return c(x)

    groups = find_mutual_recursion(SimpleRecursion)
    assert len(groups) == 0


def test_tree():
    # no cycle!
    @tvm.script.ir_module
    class Tree:
        @R.function
        def a(x: R.Object) -> R.Object:
            return b(x)

        @R.function
        def b(x: R.Object) -> R.Object:
            return c(x)

        @R.function
        def c(x: R.Object) -> R.Object:
            z: R.Object = c(x)
            z: R.Object = d(x)
            return e(z)

        @R.function
        def d(x: R.Object) -> R.Object:
            return e(x)

        @R.function
        def e(x: R.Object) -> R.Object:
            return x

    groups = find_mutual_recursion(Tree)
    assert len(groups) == 0


def test_two_function_case():
    @tvm.script.ir_module
    class TwoFunctionCase:
        @R.function
        def a(x: R.Object) -> R.Object:
            return b(x)

        @R.function
        def b(x: R.Object) -> R.Object:
            return a(x)

        # not part of the group, shouldn't be reported
        @R.function
        def c(x: R.Object) -> R.Object:
            return x

    groups = find_mutual_recursion(TwoFunctionCase)
    assert_groups(groups, [["a", "b"]])


def test_two_groups_of_two():
    @tvm.script.ir_module
    class TwoGroupsOfTwo:
        @R.function
        def a(x: R.Object) -> R.Object:
            return b(x)

        @R.function
        def b(x: R.Object) -> R.Object:
            return a(x)

        @R.function
        def c(x: R.Object) -> R.Object:
            return d(x)

        @R.function
        def d(x: R.Object) -> R.Object:
            return c(x)

        # not part of either group, shouldn't be reported
        @R.function
        def e(x: R.Object) -> R.Object:
            return x

    groups = find_mutual_recursion(TwoGroupsOfTwo)
    assert_groups(groups, [["a", "b"], ["c", "d"]])


def test_three_function_case():
    @tvm.script.ir_module
    class ThreeFunctionCase:
        @R.function
        def a(x: R.Object) -> R.Object:
            return b(x)

        @R.function
        def b(x: R.Object) -> R.Object:
            return c(x)

        @R.function
        def c(x: R.Object) -> R.Object:
            return a(x)

    groups = find_mutual_recursion(ThreeFunctionCase)
    assert_groups(groups, [["a", "b", "c"]])


def test_call_from_outside_of_group():
    @tvm.script.ir_module
    class CallFromOutOfGroup:
        # A calls into a group of mutually recursive functions,
        # but is not part of the cycle
        @R.function
        def a(x: R.Object) -> R.Object:
            return d(x)

        @R.function
        def b(x: R.Object) -> R.Object:
            return c(x)

        @R.function
        def c(x: R.Object) -> R.Object:
            return d(x)

        @R.function
        def d(x: R.Object) -> R.Object:
            return b(x)

        # E also calls into the cycle but isn't part of it
        @R.function
        def e(x: R.Object) -> R.Object:
            return b(x)

    groups = find_mutual_recursion(CallFromOutOfGroup)
    assert_groups(groups, [["b", "c", "d"]])


def test_group_with_two_cycles():
    """
    a -> b <- f
    ^    |    ^
    |    v    |
    d <- c -> e

    There are two smaller cycles in this group,
    but you can have one big cycle
    B -> C -> D -> A -> B -> C -> E -> F -> B
    """

    @tvm.script.ir_module
    class GroupWithTwoCycles:
        @R.function
        def a(x: R.Object) -> R.Object:
            return b(x)

        @R.function
        def b(x: R.Object) -> R.Object:
            return c(x)

        @R.function
        def c(x: R.Object) -> R.Object:
            y = d(x)
            return e(y)

        @R.function
        def d(x: R.Object) -> R.Object:
            return a(x)

        @R.function
        def e(x: R.Object) -> R.Object:
            return f(x)

        @R.function
        def f(x: R.Object) -> R.Object:
            return b(x)

    groups = find_mutual_recursion(GroupWithTwoCycles)
    assert_groups(groups, [["a", "b", "c", "d", "e", "f"]])


def test_multicycle_example():
    """
    Example from the documentation
    A <-> B <-> C
    ^     |     ^
    |     v     |
    |     D     |
    |     |     |
    v     v     v
    E <-> F <-> G
    """

    @tvm.script.ir_module
    class MulticycleExample:
        @R.function
        def a(x: R.Object) -> R.Object:
            y = b(x)
            return e(y)

        @R.function
        def b(x: R.Object) -> R.Object:
            y = a(x)
            z = c(y)
            return d(z)

        @R.function
        def c(x: R.Object) -> R.Object:
            return b(x)

        @R.function
        def d(x: R.Object) -> R.Object:
            return f(x)

        @R.function
        def e(x: R.Object) -> R.Object:
            y = f(x)
            return a(y)

        @R.function
        def f(x: R.Object) -> R.Object:
            y = g(x)
            return e(y)

        @R.function
        def g(x: R.Object) -> R.Object:
            y = f(x)
            return c(y)

    groups = find_mutual_recursion(MulticycleExample)
    assert_groups(groups, [["a", "b", "c", "d", "e", "f", "g"]])


def test_control_flow():
    @tvm.script.ir_module
    class ControlFlowExample:
        @R.function
        def a(x: R.Object) -> R.Object:
            y: R.Tensor((), dtype="bool") = R.const(True, dtype="bool")
            if y:
                ret = b(x)
            else:
                ret = c(x)
            return ret

        @R.function
        def b(x: R.Object) -> R.Object:
            return a(x)

        @R.function
        def c(x: R.Object) -> R.Object:
            return a(x)

    groups = find_mutual_recursion(ControlFlowExample)
    assert_groups(groups, [["a", "b", "c"]])


if __name__ == "__main__":
    tvm.testing.main()
