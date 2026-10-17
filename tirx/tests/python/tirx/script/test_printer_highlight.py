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

"""TIRx script printer highlight."""

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T


def test_highlight_script():
    @tvm_tirx.script.ir_module
    class Module:
        @T.function
        def main(  # type: ignore
            A: T.Tensor([16, 128, 128], "float32"),
            B: T.Tensor([16, 128, 128], "float32"),
            C: T.Tensor([16, 128, 128], "float32"),
        ) -> None:  # pylint: disable=no-self-argument
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for n, i, j in T.grid(16, 128, 128):
                C[n, i, j] = 0.0  # type: ignore
                for k in T.serial(128):
                    C[n, i, j] = C[n, i, j] + A[n, i, k] * B[n, j, k]

    Module.show()
    Module["main"].show()
    Module["main"].show(style="light")
    Module["main"].show(style="dark")
    Module["main"].show(style="ansi")


def test_show_preserves_annotations(monkeypatch, capsys):
    from tvm_tirx.script.printer import highlight

    @T.function
    def main(a: T.int32):
        T.evaluate(a)

    monkeypatch.setattr(highlight, "_get_pygments_style", lambda *_: None)
    monkeypatch.setenv("TVM_BLACK_FORMAT", "1")
    options = {
        "print_line_numbers": True,
        "obj_to_underline": [main.params[0]],
        "obj_to_annotate": {main.params[0]: "argument"},
        "extra_config": {"render_invisible_path_info": False},
    }
    expected = main.script(**options)
    assert "^" in expected
    assert "argument" in expected
    main.show(**options)
    assert capsys.readouterr().out == expected + "\n"


def test_cprint_plain_output(monkeypatch, capsys):
    from tvm_tirx.script.printer import highlight

    monkeypatch.setattr(highlight, "_get_pygments_style", lambda *_: None)
    source = "x=  1\n"
    highlight.cprint(source)
    assert capsys.readouterr().out == source + "\n"
    value = tvm_tirx.ir.IntImm("int32", 1)
    highlight.cprint(value)
    assert capsys.readouterr().out == value.script() + "\n"
