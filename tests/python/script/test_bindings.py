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
"""Binding and store syntax preserves values, identity, and effect order."""

from __future__ import annotations

# Script-local bindings and failing decorated definitions are intentionally observable in IR.
# ruff: noqa: F841
import pytest

from tvm.script import ir as I


def test_ordinary_tuple_and_outer_branch_assignments_still_store(language):
    # Unpacking and branch writes must update existing mutable cells rather than replace them.
    M = language.M
    first, second = object(), object()
    calls = []

    def values():
        calls.append("values")
        return first, second

    @M.function
    def main():
        a = M.cell()
        b = M.cell()
        a, b = values()
        if M.value():
            a = first
        else:
            a = second
        x = 0
        x = 1
        x += 2
        b += 3
        M.record(x)

    declarations = dict(operands for kind, operands in main.body if kind == "declare")
    stores = [operands for kind, operands in main.body if kind == "set"]
    assert calls == ["values"]
    assert stores[:-1] == [
        (declarations["a"], first),
        (declarations["b"], second),
        (declarations["a"], first),
        (declarations["a"], second),
    ]
    assert main.body[-1] == ("emit", 3)
    target, addition = stores[-1]
    assert target is declarations["b"]
    assert addition.op == "add" and addition.args == (target, 3)


def test_ordinary_callable_aliases_update_mutable_targets(language):
    # A local callable must shadow its ambient namesake and preserve ordinary mutable stores.
    M = language.M
    marker, calls = object(), []

    def axis_alias():
        pytest.fail("the shadowed ambient callable ran")

    def ordinary():
        calls.append("ordinary")
        return marker

    @M.function
    def main():
        axis_alias = ordinary
        cell = M.cell()
        cell = axis_alias()
        M.record(cell)

    declaration = next(operands[1] for kind, operands in main.body if kind == "declare")
    stores = [operands for kind, operands in main.body if kind == "set"]
    assert calls == ["ordinary"]
    assert len(stores) == 1 and stores[0][0] is declaration and stores[0][1] is marker
    assert main.body[-1] == ("emit", declaration)


def test_loop_targets_preserve_names_and_local_rebinding(language):
    # Scalar and unpacked loop targets must retain names, extents and ordinary rebinding semantics.
    M = language.M

    @M.function
    def main():
        for i in M.grid(4):
            M.record(i)
            i = 5
            M.record(i)
        for iters in M.grid(4, 5):
            M.record(iters)
        for head, *tail in M.grid(2, 3, 4):
            M.record((head, *tail))

    scalar, rebound, packed, unpacked = [value for _, value in main.body]
    assert scalar.op == "loop" and scalar.args == (4,) and scalar.name == "i"
    assert rebound == 5
    assert [value.name for value in packed] == ["iters_0", "iters_1"]
    assert [value.args for value in packed] == [(4,), (5,)]
    assert [value.name for value in unpacked] == ["head", "tail_0", "tail_1"]
    assert [value.args for value in unpacked] == [(2,), (3,), (4,)]


def test_body_annotation_reads_a_preceding_ordinary_local(language):
    # A body annotation must read the current local value, not a stale enclosing capture.
    M = language.M
    annotations = []

    def annotation(shape):
        annotations.append(shape)
        return M.Tensor(shape)

    M.annotation = annotation

    @M.function
    def main():
        shape = (4,)
        value: M.annotation(shape) = 1
        M.record(value)
        shape = (8,)
        M.record(shape)

    assert annotations == [(4,)]
    assert main.body == [("emit", 1), ("emit", (8,))]


def test_conditional_binding_can_be_assigned_after_skipped_branch(language):
    # A skipped constexpr assignment must not prevent a later ordinary assignment.
    M = language.M

    @M.function
    def main():
        if I.constexpr(False):
            x = 1
        x = 2
        M.record(x)

    assert main.body[-1] == ("emit", 2)


def test_ir_optional_binding_survives_skipped_host_assignment(language):
    # A skipped host assignment must not corrupt an optional binding from an IR branch.
    M = language.M

    @M.function
    def main(condition: M.value):
        if condition:
            x = M.value(1)
        if I.constexpr(False):
            x = M.value(2)
        x = M.value(3)
        M.record(x)

    assert main.body[-1][1].args == (3,)


def test_missing_host_binding_cannot_be_truth_tested(language):
    # Reading an unexecuted constexpr binding must raise before truth testing it.
    M = language.M
    with pytest.raises(NameError):

        @M.function
        def main():
            if I.constexpr(False):
                x = 1
            M.record(1 if I.constexpr(x) else 0)


def test_missing_host_if_binding_raises_before_branch_assignments(language):
    # An if-condition must reject its missing incoming value before either arm assigns that name.
    M = language.M
    with pytest.raises(NameError):

        @M.function
        def main():
            if I.constexpr(False):
                x = 1
            if I.constexpr(x):
                x = 2
            else:
                x = 3


def test_host_lambda_local_does_not_capture_optional_binding(language):
    # A lambda parameter must shadow an outer optional binding during host selection.
    M = language.M

    @M.function
    def main():
        if I.constexpr(False):
            x = 1
        if I.constexpr((lambda x: x)(True)):
            M.record(222)

    assert main.body[-1] == ("emit", 222)


def test_ir_branch_incoming_read_and_rebinding_obeys_python_scope(language):
    # A branch-local write must not read the outer binding before its own initialization.
    M = language.M
    with pytest.raises(NameError, match="y"):

        @M.function
        def main(condition: M.value(), x: M.value()):
            y = x
            if condition:
                y = y + x
            else:
                y = x
            return y


def test_mutating_stores_and_loop_control_keep_effect_order(language):
    # Stores evaluate RHS before target/index; augmented stores load the target before their RHS.
    M = language.M
    effects = []

    class Storage:
        def __init__(self):
            self.value = 0
            self.items = [0]

        def __getitem__(self, key):
            effects.append("read")
            return self.items[key]

        def __setitem__(self, key, value):
            self.items[key] = value

    storage = Storage()

    def receiver():
        effects.append("receiver")
        return storage

    def key():
        effects.append("key")
        return 0

    def increment(value=3):
        effects.append("increment")
        return value

    @M.function
    def main(condition: M.value()):
        first, second = (2, 4)
        receiver().value = first
        receiver()[key()] = increment(second)
        receiver().value += increment()
        receiver()[key()] += increment()
        while condition:
            assert not condition, "keep the assertion in the loop"
            for i in range(2):
                if condition:
                    continue
                else:
                    break
            break
        return

    assert storage.value == 5 and storage.items == [7]
    assert effects == [
        "receiver",
        "increment",
        "receiver",
        "key",
        "receiver",
        "increment",
        "receiver",
        "key",
        "read",
        "increment",
    ]
    assert [kind for kind, _ in main.body] == [
        "setattr",
        "setitem",
        "setattr",
        "setitem",
        "assert",
        "continue",
        "break",
        "break",
        "return",
    ]
    assertion, message = main.body[4][1]
    assert assertion.op == "not" and assertion.args[0] is main.params[0]
    assert message == "keep the assertion in the loop"
    assert main.body[-1] == ("return", None)
