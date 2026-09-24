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
"""Recording builder fixtures for the production source-to-builder parser."""

import pytest
from minilang import Language, RecordingSpanEntry

from tvm.script.ir_builder import base
from tvm.script.parser import entry, protocol_registry


@pytest.fixture
def language(monkeypatch):
    monkeypatch.setattr(
        entry,
        "register_namespace",
        lambda alias, namespace: monkeypatch.setitem(entry._NAMESPACES, alias, namespace),
    )
    language = Language()
    monkeypatch.setattr(entry, "builder_ir", language.I)
    monkeypatch.setattr(entry, "SpanEntry", lambda span: RecordingSpanEntry(language, span))
    return language


@pytest.fixture
def spanned_language(language, monkeypatch):
    monkeypatch.setattr(entry, "SpanEntry", base.SpanEntry)
    language.I.at_ = base.at_
    language.I.with_at_group_ = base.with_at_group_
    language.M.inline = protocol_registry.declaration_kind("M.inline", "helper")(
        entry.make_macro_decorator(language.M)
    )
    return language


@pytest.fixture
def primitive_language(spanned_language):
    """Use native primitive operators for minilang expression examples."""
    from functools import reduce

    from tvm.ir import prim

    language = spanned_language
    for kind in ("LT", "LE", "GT", "GE", "EQ", "NE"):
        constructor = getattr(prim._ffi_api, "_Op" + kind)
        setattr(
            language.M, kind.lower() + "_", lambda lhs, rhs, make=constructor: make(lhs, rhs, None)
        )
    language.M.and_ = lambda *conditions: reduce(
        lambda rhs, lhs: prim.And(lhs, rhs), reversed(conditions)
    )
    return language
