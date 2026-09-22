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
"""Backend expression namespaces use the canonical script builder."""

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

import tvm
from tvm.backend import loader
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.tirx import script as T
from tvm.tirx.script import builder as builder
from tvm.tirx.script.builder import ir as builder_ir


def _preserve_namespace_registration(monkeypatch, name):
    for namespace in (builder_ir, builder, T):
        exports = vars(namespace).get("__all__")
        if isinstance(exports, list):
            monkeypatch.setattr(namespace, "__all__", list(exports))
        monkeypatch.setitem(vars(namespace), name, None)


@pytest.mark.parametrize(
    "backend, expression",
    [
        ("cuda", "T.cuda.warp_sync()"),
        ("cuda", "T.ptx.cp.async_.commit_group()"),
        ("metal", "T.metal.simd_shuffle(T.int32(1), 0)"),
        ("trn", "T.nki.memset(T.int32(0), T.int32(1))"),
    ],
)
def test_backend_namespace_constructs_expected_expression(backend, expression):
    tvm.backend.load(backend)
    source = f"@T.prim_func\ndef main():\n    T.evaluate({expression})\n"
    evaluate = builder.evaluate
    namespace = getattr(builder, expression.split(".")[1])
    with IRBuilder() as context:
        with builder.function():
            builder.func_name("main")
            builder.evaluate(eval(expression, {"T": builder}))
    expected = context.get()
    actual = parser.parse(source)
    tvm.ir.assert_structural_equal(expected, actual)
    assert T.evaluate is builder.evaluate is evaluate
    assert getattr(T, expression.split(".")[1]) is namespace
    assert isinstance(actual.body, tvm.tirx.Evaluate)


def test_backend_loaded_after_parser_initialization():
    # A fresh process leaves backend autoload disabled, so the first parse
    # cannot accidentally inherit the extension this test loads afterward.
    code = """
import tvm
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.tirx.script import builder
parse = parser.parse
parse('@T.prim_func\\ndef warmup():\\n    T.evaluate(0)\\n')
assert not tvm.backend.is_loaded('cuda')
try:
    builder.cuda
except AttributeError:
    pass
else:
    raise AssertionError('namespace access must not implicitly load CUDA')
assert not tvm.backend.is_loaded('cuda')
tvm.backend.load('cuda')
source = '@T.prim_func\\ndef main():\\n    T.evaluate(T.cuda.warp_sync())\\n'
with IRBuilder() as context:
    with builder.function():
        builder.func_name('main')
        builder.evaluate(builder.cuda.warp_sync())
tvm.ir.assert_structural_equal(context.get(), parse(source))
assert builder.cuda is not None
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TVM_DEVICE_BACKEND_AUTOLOAD": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_explicit_namespace_registered_after_parser_initialization(monkeypatch):
    parse = parser.parse
    parse("@T.prim_func\ndef warmup():\n    T.evaluate(0)\n")
    name = "extension_test"
    namespace = SimpleNamespace(value=lambda: tvm.tirx.IntImm("int32", 7))
    assert not hasattr(builder, name)
    _preserve_namespace_registration(monkeypatch, name)
    monkeypatch.setitem(builder_ir._SCRIPT_NAMESPACES, name, None)
    assert builder_ir.register_script_namespace(name, namespace) is namespace
    function = parse("@T.prim_func\ndef main():\n    T.evaluate(T.extension_test.value())\n")
    assert function.body.value.value == 7
    assert getattr(builder, name) is namespace
    replacement = SimpleNamespace(value=lambda: tvm.tirx.IntImm("int32", 8))
    builder_ir.register_script_namespace(name, replacement)
    updated = parse("@T.prim_func\ndef main():\n    T.evaluate(T.extension_test.value())\n")
    assert updated.body.value.value == 8
    assert getattr(T, name) is replacement


def test_backend_namespace_factory_errors_propagate_and_retry(monkeypatch):
    backend_name = "namespace_error_test"
    name = "error_recovery_test"
    backend = ModuleType("tvm.backend." + backend_name)
    seen = []

    def factory():
        seen.append("called")
        if len(seen) == 1:
            raise RuntimeError("namespace registration failed")
        return {name: object()}

    def register_backend():
        for namespace_name, namespace in backend.script_namespaces().items():
            builder_ir.register_script_namespace(namespace_name, namespace)

    def import_backend(requested):
        assert requested == backend_name
        return backend

    backend.script_namespaces = factory
    backend.register_backend = register_backend
    monkeypatch.setattr(loader, "_import_backend", import_backend)
    monkeypatch.setattr(loader, "_LOADED_BACKENDS", dict(loader._LOADED_BACKENDS))
    monkeypatch.setattr(tvm.tirx, backend_name, None, raising=False)
    monkeypatch.setitem(sys.modules, f"tvm.tirx.{backend_name}", None)
    _preserve_namespace_registration(monkeypatch, name)
    try:
        with pytest.raises(RuntimeError, match="namespace registration failed"):
            loader.load(backend_name)
        assert not loader.is_loaded(backend_name)
        with pytest.raises(AttributeError):
            builder_ir._get_script_namespace(name)
        loader.load(backend_name)
        namespace = builder_ir._get_script_namespace(name)
        assert namespace is getattr(builder, name) is getattr(T, name)
        assert loader.is_loaded(backend_name)
        loader.load(backend_name)
        assert seen == ["called", "called"]
    finally:
        builder_ir._SCRIPT_NAMESPACES.pop(name, None)
