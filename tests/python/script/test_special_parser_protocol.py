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

import inspect
import traceback

import pytest

from tvm.script import tirx as T
from tvm.script.parser import protocol_registry as registry


def test_constructor_policy_survives_a_failed_definition(language):
    # A registered argument uses the dialect lookup, even after a constructor failure.
    M = language.M
    calls = []
    dialect_infos = {}
    first, invalid, second = object(), object(), object()
    failure = RuntimeError("constructor failed")

    @registry.args_policy("M.constructor", {"device": "global_info"})
    def constructor(device):
        calls.append(device)
        if device is invalid:
            raise failure
        return device

    def resolve(name):
        return dialect_infos[name]

    M.resolve_global_info_ = resolve
    M.constructor = constructor

    def build():
        @M.function
        def main():
            M.record(M.constructor(device="mesh"))
            M.record(constructor(device="mesh"))

        return main

    dialect_infos["mesh"] = first
    assert build().body == [("emit", first), ("emit", "mesh")]
    dialect_infos["mesh"] = invalid
    with pytest.raises(RuntimeError, match="constructor failed") as caught:
        build()
    assert caught.value is failure
    dialect_infos["mesh"] = second
    assert build().body == [("emit", second), ("emit", "mesh")]
    assert calls == [first, "mesh", invalid, second, "mesh"]


def test_argument_policy_preserves_expression_dtype(language):
    # Shape-expression parsing must retain the declared int32 symbol dtype.
    M = language.M

    @registry.args_policy("M.shape", {"values": "expr_str"}, dtype="int32")
    def shape(values):
        return values

    M.shape = shape

    @M.function
    def main():
        M.shape(("n", "n + 1"))

    n, increment = main.body[0][1]
    assert n.args == ("int32",)
    assert increment.op == "add"
    assert increment.args[0] is n
    assert increment.args[1] == 1


def test_tirx_rejects_global_info_at_the_source_argument(monkeypatch):
    # A custom global-info argument must report TIRx's unsupported policy at its source.
    @registry.args_policy("T.global_annotation", {"device": "global_info"})
    def global_annotation(device):
        return T.int32

    monkeypatch.setattr(T, "global_annotation", global_annotation, raising=False)
    with pytest.raises(
        NotImplementedError, match="TIRx does not support global-info lookup"
    ) as caught:

        @T.prim_func
        def main(value: T.global_annotation(device="cuda:0")):
            T.evaluate(value)

    lines, first = inspect.getsourcelines(test_tirx_rejects_global_info_at_the_source_argument)
    index, line = next((i, line) for i, line in enumerate(lines) if "def main(value:" in line)
    location = first + index
    frames = traceback.extract_tb(caught.value.__traceback__)
    source_frames = [
        frame for frame in frames if frame.filename == __file__ and frame.lineno == location
    ]
    assert source_frames
    if getattr(source_frames[-1], "colno", None) is not None:
        column = line.index('"cuda:0"')
        assert (
            source_frames[-1].colno,
            source_frames[-1].end_lineno,
            source_frames[-1].end_colno,
        ) == (
            column,
            location,
            column + len('"cuda:0"'),
        )
