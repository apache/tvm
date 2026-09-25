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

"""TIRx script error handling."""

from __future__ import annotations

import inspect
import traceback

import pytest

import tvm
from tvm.script import tirx as T
from tvm.script.ir_builder import resolve_global_info_args


def test_optional_annotation_requires_jit_at_the_source_parameter():
    # Ordinary argument validation must reject Optional at its original parameter location.
    with pytest.raises(TypeError, match="^T.Optional is only supported by @T.jit$") as caught:

        @T.prim_func
        def invalid(value: T.Optional(T.handle)):
            T.evaluate(0)

    assert type(caught.value) is TypeError
    lines, first = inspect.getsourcelines(
        test_optional_annotation_requires_jit_at_the_source_parameter
    )
    index, line = next((i, text) for i, text in enumerate(lines) if "def invalid(" in text)
    frames = traceback.extract_tb(caught.value.__traceback__)
    source = [
        frame for frame in frames if frame.filename == __file__ and frame.lineno == first + index
    ]
    assert source
    if getattr(source[-1], "colno", None) is not None:
        column = line.index("value:")
        assert (source[-1].colno, source[-1].end_lineno, source[-1].end_colno) == (
            column,
            first + index,
            column + len("value: T.Optional(T.handle)"),
        )


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


def test_scalar_assign_error_not_swallowed():
    """Regression: genuine errors (non-TypeError) from buffer_store during
    scalar-assignment sugar must propagate, not be silently swallowed.

    Before the fix, both eval_expr and buffer_store were wrapped in a single
    broad ``except Exception: pass``, so any error from buffer_store would be
    swallowed and the assignment would silently fall through to eval_assign."""
    from unittest.mock import patch

    original = tvm.tirx.script.ir_builder.parser_protocol.buffer_store

    def bomb(*args, **kwargs):
        # Intercept only the scalar-assignment path (indices == [0])
        if args[2] == [0]:
            raise ValueError("boom")
        return original(*args, **kwargs)

    src = """
# from tvm.script import tirx as T

@T.prim_func
def func():
    T.device_entry()
    v: T.int32
    v = v + T.int32(1)
"""
    # The ValueError propagates unchanged. A broad ``except Exception`` here
    # previously swallowed it and fell through to eval_assign.
    with patch("tvm.tirx.script.ir_builder.parser_protocol.buffer_store", side_effect=bomb):
        with pytest.raises(ValueError, match="boom"):
            tvm.script.from_source(src, extra_vars={"I": tvm.script.ir, "T": T})
