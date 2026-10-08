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
"""Public script entry points and diagnostic rendering."""

import re

import pytest
import tvm_ffi
from tvm_ffi.access_path import AccessPath

import tvm
from tvm.script.printer.scriptable import PrinterConfig, _script


@pytest.mark.parametrize(
    "entry",
    ["node.TVMScriptPrinterScript", "script.printer.Script", "script.printer.ReprPrintRelax"],
)
def test_script_entry_points(entry):
    render = tvm_ffi.get_global_func(entry)
    config = PrinterConfig(extra_config={"ir.prefix": "IR"})
    assert render(tvm.ir.Range(0, 4), config) == "IR.Range(0, 4)"
    fallback_config = PrinterConfig(path_to_underline=[AccessPath.root().attr("missing")])
    assert render(tvm.runtime.ShapeTuple([1, 2]), fallback_config) == (
        "Access path: <root>.missing\n"
        "Note: No visible object for this path is rendered in TVMScript.\n\n"
        "Shape(1, 2)"
    )


@pytest.mark.parametrize(
    "diagnostic", ["path_to_underline", "obj_to_underline", "path_to_annotate", "obj_to_annotate"]
)
def test_diagnostics_without_invisible_path_info(diagnostic):
    value = tvm.ir.PrimType("int32")
    target = AccessPath.root() if diagnostic.startswith("path") else value
    annotated = diagnostic.endswith("annotate")
    config = PrinterConfig(
        **{diagnostic: {target: "type note"} if annotated else [target]},
        extra_config={"render_invisible_path_info": False},
    )
    text = _script(value, config)
    assert "T.int32" in text
    assert "type note" in text if annotated else "^^^^^^^" in text
    assert "Access path:" not in text


def test_plain_render_without_path_mapping():
    mod = tvm.IRModule(attrs={"constant": tvm.runtime.tensor([1, 2, 3])})
    options = {"ir.prefix": "IR", "ir.module_name": "Example"}
    expected = mod.script(show_meta=True, extra_config=options)
    actual = mod.script(
        show_meta=True, extra_config={**options, "render_invisible_path_info": False}
    )
    assert actual == expected
    assert "load_json" in actual
    assert "from tvm.script import ir as IR" in actual


def test_redirected_repr_translation_failure():
    value = tvm.tirx.For(tvm.tirx.Var("i", "int32"), 0, 1, 99, tvm.tirx.Evaluate(0))
    with pytest.raises(TypeError, match="unknown loop kind"):
        value.script()
    assert re.fullmatch(r"tirx\.For\((?:0x)?[0-9a-fA-F]+\)", repr(value))
