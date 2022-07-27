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
import pytest

import tvm
from tvm.script import tir as T
from tvm.script.printer.doc_printer import to_python_script
from tvm.script.printer.doc import LiteralDoc


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")
    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)
    return "\n".join(line[spaces_to_remove:] for line in s.splitlines())


@pytest.mark.parametrize(
    "doc,expected",
    [
        (LiteralDoc(None), "None"),
        (LiteralDoc(True), "True"),
        (LiteralDoc(False), "False"),
        (LiteralDoc("test"), '"test"'),
        (LiteralDoc(""), '""'),
        (LiteralDoc('""'), r'"\"\""'),
        (LiteralDoc("\n\t\\test\r"), r'"\n\t\\test\r"'),
        # TODO: fix the roundatrippable problem caused by utf8
        pytest.param(LiteralDoc("\x88"), r'"\x88"', marks=pytest.mark.xfail),
        (LiteralDoc(0), "0"),
        (LiteralDoc(-1), "-1"),
        (LiteralDoc(3.25), "3.25"),
        (LiteralDoc(-0.5), "-0.5"),
    ],
)
def test_print_literal_doc(doc, expected):
    assert to_python_script(doc).rstrip("\n") == format_script(expected)


def test_highlight_script():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(  # type: ignore
            a: T.handle,
            b: T.handle,
            c: T.handle,
        ) -> None:  # pylint: disable=no-self-argument
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [16, 128, 128])
            B = T.match_buffer(b, [16, 128, 128])
            C = T.match_buffer(c, [16, 128, 128])
            for n, i, j, k in T.grid(16, 128, 128, 128):
                with T.block("matmul"):
                    vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                    with T.init():
                        C[vn, vi, vj] = 0.0  # type: ignore
                    C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]

    Module.show()
    Module["main"].show()
