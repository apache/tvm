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
"""Configure pytest for TVM's Python test suite."""

import os
from pathlib import Path

import pytest


def pytest_sessionstart():
    if os.getenv("CI", "") == "true":
        from tvm.testing.utils import (
            install_request_hook,  # pylint: disable=import-outside-toplevel
        )

        install_request_hook(Path(__file__).with_name("request_hook.py"))


@pytest.fixture
def language(monkeypatch):
    """Use the shared recording language without selecting a native dialect."""
    from minilang import Language, RecordingSpanEntry

    from tvm.script.parser import entry

    monkeypatch.setattr(
        entry,
        "register_namespace",
        lambda alias, namespace: monkeypatch.setitem(entry._NAMESPACES, alias, namespace),
    )
    language = Language()
    monkeypatch.setattr(entry, "builder_ir", language.I)
    monkeypatch.setattr(entry, "SpanEntry", lambda span: RecordingSpanEntry(language, span))
    return language
