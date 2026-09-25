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
"""TIRX parser integration for special parser protocol."""

import inspect
import traceback

import pytest

from tvm.script import tirx as T
from tvm.script.ir_builder import resolve_global_info_args
from tvm.script.parser import entry


def test_tirx_rejects_global_info_at_the_call_site(monkeypatch):
    # A custom builder reports its resolver failure at the ordinary call site.
    @resolve_global_info_args("device", resolver=T.resolve_global_info_)
    def global_annotation(device):
        return T.int32

    monkeypatch.setattr(T, "global_annotation", global_annotation, raising=False)
    with pytest.raises(
        NotImplementedError, match="TIRx does not support global-info lookup"
    ) as caught:

        @T.prim_func
        def main(value: T.global_annotation(device="cuda:0")):
            T.evaluate(value)

    lines, first = inspect.getsourcelines(test_tirx_rejects_global_info_at_the_call_site)
    index, line = next((i, line) for i, line in enumerate(lines) if "def main(value:" in line)
    location = first + index
    frames = traceback.extract_tb(caught.value.__traceback__)
    source_frames = [
        frame for frame in frames if frame.filename == __file__ and frame.lineno == location
    ]
    assert source_frames
    if getattr(source_frames[-1], "colno", None) is not None:
        column = line.index("T.global_annotation(")
        assert (
            source_frames[-1].colno,
            source_frames[-1].end_lineno,
            source_frames[-1].end_colno,
        ) == (
            column,
            location,
            column + len('T.global_annotation(device="cuda:0")'),
        )


def test_initial_import_alias_and_symbolic_range():
    source = """from tvm.script import tirx as Script
@Script.prim_func
def main(n: Script.int32):
    for i in range(n):
        Script.evaluate(i)
"""
    function = entry.parse(source)
    assert function.body.extent.same_as(function.params[0])
    assert function.body.body.value.same_as(function.body.loop_var)
