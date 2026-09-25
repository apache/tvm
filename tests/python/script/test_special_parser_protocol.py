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
from types import SimpleNamespace

import pytest

from tvm.script import tirx as T
from tvm.script.ir_builder import resolve_global_info_args
from tvm.script.parser import entry


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


def test_imported_namespace_alias_keeps_canonical_mutable_policy(language, monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("parser_policy_namespace")
    module.language = language.M
    monkeypatch.setitem(sys.modules, module.__name__, module)
    function = entry.parse(
        """from parser_policy_namespace import language as Y
@Y.function
def main():
    value = Y.cell()
    value = 3
    Y.record(value)
"""
    )
    declaration = next(operands[1] for kind, operands in function.body if kind == "declare")
    stores = [operands for kind, operands in function.body if kind == "set"]
    assert stores == [(declaration, 3)]
    assert function.body[-1] == ("emit", declaration)


def test_mutable_member_alias_keeps_scalar_dtype():
    function = entry.parse(
        """from tvm.script import tirx as Script
@Script.prim_func
def main():
    value: Script.i64 = 1
    value = 2
    Script.evaluate(value)
"""
    )
    declaration, initial, updated, evaluated = function.body.seq
    assert declaration.buffer.ty.dtype == "int64"
    assert initial.value.value == 1
    assert updated.value.value == 2
    assert initial.buffer.same_as(declaration.buffer)
    assert updated.buffer.same_as(declaration.buffer)
    assert evaluated.value.ty.dtype == "int64"


def test_global_info_selectors_share_module_context_and_preserve_identity():
    from tvm.relax.script.ir_builder.distributed.ir import _lookup_device_mesh
    from tvm.script import ir as I
    from tvm.script import ir_builder as I_builder
    from tvm.script import relax as R

    mesh = R.device_mesh((2,), I.Range(0, 2))
    device = R.vdevice("llvm")
    second = R.vdevice("llvm", 1)
    # Concrete constructor inputs remain usable without any source/module context.
    assert R.Tensor((2,), "float32", vdevice=device).vdevice.is_(device)
    assert R.DTensor((2,), "float32", mesh, "R").device_mesh.is_(mesh)
    for resolve in (I.resolve_global_info_, R.resolve_global_info_):
        with pytest.raises(TypeError, match="selectors must be strings"):
            resolve(mesh)
        with pytest.raises(ValueError):
            resolve("mesh[0]")

    with I.IRBuilder():
        with pytest.raises(ValueError, match="enclosing module frame"):
            I.resolve_global_info_("mesh[0]")
        with pytest.raises(ValueError, match="enclosing module frame"):
            R.lookup_vdevice("llvm", 0)
        with I_builder.ir_module():
            I.module_global_infos({"mesh": [mesh], "vdevice": [device, second]})
            assert I.resolve_global_info_("mesh[0]").is_(mesh)
            assert R.resolve_global_info_("mesh[0]").is_(mesh)
            assert _lookup_device_mesh("mesh[0]").is_(mesh)
            assert R.resolve_global_info_("llvm:1:global").is_(second)
            assert R.resolve_global_info_("vdevice:0").is_(device)
            assert R.lookup_vdevice("llvm", 0).is_(device)
            with pytest.raises(KeyError):
                I.resolve_global_info_("missing[0]")
            with pytest.raises(IndexError):
                R.resolve_global_info_("mesh[2]")
            with pytest.raises(ValueError):
                R.resolve_global_info_("mesh[-1]")
            with pytest.raises(ValueError):
                R.resolve_global_info_("cuda:0")
            with pytest.raises(TypeError, match="must be a DeviceMesh"):
                _lookup_device_mesh("vdevice[0]")


def test_global_info_resolution_does_not_capture_executing_classes():
    from tvm.script import ir as I
    from tvm.script import relax as R

    mesh = R.device_mesh((2,), I.Range(0, 2))

    class Declaration:
        I.module_global_infos({"mesh": [mesh]})
        with pytest.raises(ValueError, match="enclosing module frame"):
            I.resolve_global_info_("mesh[0]")
        with pytest.raises(ValueError, match="enclosing module frame"):
            R.lookup_vdevice("llvm", 0)

    assert "__tvm_script_global_infos__" not in vars(Declaration)
    with pytest.raises(ValueError, match="enclosing module frame"):
        I.resolve_global_info_("mesh[0]")
