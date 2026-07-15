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

from tvm_ffi.access_path import AccessPath

import tvm
import tvm.testing
from tvm.runtime.script_printer import PrinterConfig, _script


def test_render_invisible_path_info_defaults_to_true():
    config = PrinterConfig()
    assert config.render_invisible_path_info is True

    result = tvm.ir.PrimType("int32").script(path_to_underline=[AccessPath.root().attr("dtype")])
    assert result == (
        "Access path: <root>.dtype\n"
        "Note: The underlined object is the nearest visible parent of this path.\n\n"
        "T.int32\n"
        "^^^^^^^"
    )


def test_render_invisible_path_info_without_target_path_is_unchanged():
    assert tvm.ir.PrimType("int32").script() == "T.int32"


def test_render_invisible_path_info_explicit_false_uses_legacy_output():
    result = tvm.ir.PrimType("int32").script(
        path_to_underline=[AccessPath.root().attr("dtype")],
        extra_config={"render_invisible_path_info": False},
    )
    assert result == "T.int32\n^^^^^^^"


def test_render_invisible_path_info_exact_visible_path_omits_note():
    result = tvm.ir.PrimType("int32").script(path_to_underline=[AccessPath.root()])
    assert result == "Access path: <root>\n\nT.int32\n^^^^^^^"


def test_render_invisible_path_info_without_visible_path():
    result = _script(
        tvm.runtime.ShapeTuple([1, 2]),
        PrinterConfig(path_to_underline=[AccessPath.root().attr("missing")]),
    )
    assert result == (
        "Access path: <root>.missing\n"
        "Note: No visible object for this path is rendered in TVMScript.\n\n"
        "Shape(1, 2)"
    )


if __name__ == "__main__":
    tvm.testing.main()
