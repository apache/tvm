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
"""Registered constructor arguments resolve module names during construction.
A failed constructor must not change the next function's interpretation of literals.
"""

from types import SimpleNamespace

import pytest

from tvm.script.ir_builder import resolve_global_info_args


def test_constructor_policy_survives_a_failed_definition(language):
    # Aliased calls use the same builder lookup, even after a constructor failure.
    M = language.M
    calls = []
    dialect_infos = {}
    first, invalid, second = object(), object(), object()
    failure = RuntimeError("constructor failed")

    def resolve(name):
        return dialect_infos[name]

    @resolve_global_info_args("device", resolver=resolve)
    def constructor(device):
        calls.append(device)
        if device is invalid:
            raise failure
        return device

    M.constructor = constructor

    def build():
        @M.function
        def main():
            M.record(M.constructor(device="mesh"))
            M.record(constructor(device="mesh"))

        return main

    dialect_infos["mesh"] = first
    assert build().body == [("emit", first), ("emit", first)]
    dialect_infos["mesh"] = invalid
    with pytest.raises(RuntimeError, match="constructor failed") as caught:
        build()
    assert caught.value is failure
    dialect_infos["mesh"] = second
    assert build().body == [("emit", second), ("emit", second)]
    assert calls == [first, first, invalid, second, second]


def test_external_expression_preserves_symbol_dtype(language):
    # Ordinary expressions retain the externally declared int32 symbol dtype.
    M = language.M

    def shape(values):
        return values

    M.shape = shape
    n = M.dynamic("n", "int32")

    @M.function
    def main():
        M.shape((n, n + 1))

    n, increment = main.body[0][1]
    assert n.args == ("int32",)
    assert increment.op == "add"
    assert increment.args[0] is n
    assert increment.args[1] == 1


def test_ordinary_calls_preserve_literal_arguments(language):
    M = language.M
    Alias = M
    seen = []

    M.special = lambda value: seen.append(value)
    M.nested = SimpleNamespace(special=M.special)
    special = M.special

    @M.function
    def main():
        Alias.special("n")
        M.nested.special("n")
        special("n")
        M.record(int("3"))

    assert seen == ["n", "n", "n"]
    assert main.body[-1] == ("emit", 3)
