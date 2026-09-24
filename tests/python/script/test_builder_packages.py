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
"""Canonical builder packages, compatibility aliases, and external language variants."""

import os
import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.parametrize("language", ["tirx", "relax"])
@pytest.mark.parametrize("legacy_first", [False, True])
def test_builder_aliases_share_canonical_modules(language, legacy_first):
    # A fresh interpreter also checks bootstrap ordering and native registration.
    script = r"""
import importlib
import sys

language, legacy_first = sys.argv[1:]
canonical = "tvm.script.ir_builder." + language
legacy = "tvm." + language + ".script.builder"
first, second = (legacy, canonical) if legacy_first == "True" else (canonical, legacy)
first_module = importlib.import_module(first)
assert importlib.import_module(second) is first_module
assert first_module.__name__ == canonical
assert first_module.__spec__.name == canonical

children = ["ir", "frame", "_ffi_api", "parser_protocol"]
children += ["distributed.ir"] if language == "relax" else ["tirx", "utils"]
for child in children:
    module = importlib.import_module(first + "." + child)
    assert importlib.import_module(second + "." + child) is module
    assert importlib.import_module("tvm.script." + language + ".builder." + child) is module
    assert module.__name__ == canonical + "." + child
    assert module.__spec__.name == canonical + "." + child

for prefix in (canonical, legacy):
    try:
        importlib.import_module(prefix + ".missing_builder_module")
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("Missing modules must not redirect back into an alias")

from tvm.script import ir_builder
from tvm.script import ir as I

assert getattr(ir_builder, language) is first_module
assert importlib.import_module("tvm." + language + ".script").builder is first_module
assert I.IRBuilder is ir_builder.IRBuilder
assert I.IRModuleFrame is ir_builder.IRModuleFrame
assert I.meta_var is ir_builder.meta_var
assert not any(hasattr(I, name) for name in ("importlib", "Any", "tirx", "relax"))
with ir_builder.IRBuilder() as builder:
    with ir_builder.ir_module():
        ir_builder.module_attrs({"builder_package": language})
assert str(builder.get().attrs["builder_package"]) == language
"""
    subprocess.run([sys.executable, "-c", script, language, str(legacy_first)], check=True)


def test_registered_external_builder_package(tmp_path):
    package = tmp_path / "external_language"
    builder = package / "builder"
    builder.mkdir(parents=True)
    (package / "__init__.py").write_text("marker = object()\n")
    (builder / "__init__.py").write_text("marker = object()\n")
    (builder / "nested.py").write_text("marker = object()\n")
    script = dedent(
        """
        import importlib
        import tvm.script
        from tvm.script import ir_builder

        tvm.script.register_dialect("external_test", "external_language")
        external = importlib.import_module("external_language")
        builder = importlib.import_module("external_language.builder")
        nested = importlib.import_module("external_language.builder.nested")
        assert tvm.script.external_test is external
        assert ir_builder.external_test is builder
        assert importlib.import_module("tvm.script.parser.external_test") is external
        assert importlib.import_module("tvm.script.ir_builder.external_test") is builder
        assert importlib.import_module("tvm.script.ir_builder.external_test.nested") is nested
        assert importlib.import_module("tvm.script.external_test.builder.nested") is nested
        assert nested.__spec__.name == "external_language.builder.nested"
        """
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), *sys.path]))
    subprocess.run([sys.executable, "-c", script], env=env, check=True)
