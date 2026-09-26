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

"""TIRx script printer underlining."""

from tvm_ffi.access_path import AccessPath

from tvm.script import ir as I
from tvm.script import tirx as T


def test_underline_from_obj():
    @T.prim_func
    def func(a: T.int32, b: T.int32):
        T.evaluate(a)
        T.evaluate(b)
        T.evaluate(a)
        T.evaluate(b)
        T.evaluate(a)
        T.evaluate(b)

    result = func.with_attr("global_symbol", "main").script(
        obj_to_underline=[func.params[0]],
        extra_config={"render_invisible_path_info": False},
    )
    assert result == format_script(
        """
        # from tvm.script import tirx as T
        # from tvm.tirx.layout import Axis

        @T.prim_func
        def main(a: T.int32, b: T.int32):
            T.evaluate(a)
                       ^
            T.evaluate(b)
            T.evaluate(a)
                       ^
            T.evaluate(b)
            T.evaluate(a)
                       ^
            T.evaluate(b)
    """
    )


def format_script(s: str) -> str:
    """
    Remove leading and trailing blank lines, and make the minimum idention 0
    """
    s = s.strip("\n")

    non_empty_lines = [line for line in s.splitlines() if line and not line.isspace()]
    if not non_empty_lines:
        # no actual content
        return "\n"

    line_indents = [len(line) - len(line.lstrip(" ")) for line in non_empty_lines]
    spaces_to_remove = min(line_indents)

    cleaned_lines = "\n".join(line[spaces_to_remove:] for line in s.splitlines())
    if not cleaned_lines.endswith("\n"):
        cleaned_lines += "\n"
    return cleaned_lines.strip()


def test_underline_from_multi_obj():
    @T.prim_func
    def func():
        T.evaluate(-1)
        T.evaluate(1)
        T.evaluate(2)
        T.evaluate(3)
        T.evaluate(4)
        T.evaluate(5)
        T.evaluate(6)
        T.evaluate(7)

    result = func.with_attr("global_symbol", "main").script(
        obj_to_underline=[
            func.body.seq[1],
            func.body.seq[3],
            func.body.seq[5],
            func.body.seq[7],
        ],
        extra_config={"render_invisible_path_info": False},
    )
    assert result == format_script(
        """
        # from tvm.script import tirx as T
        # from tvm.tirx.layout import Axis

        @T.prim_func
        def main():
            T.evaluate(-1)
            T.evaluate(1)
            ^^^^^^^^^^^^^
            T.evaluate(2)
            T.evaluate(3)
            ^^^^^^^^^^^^^
            T.evaluate(4)
            T.evaluate(5)
            ^^^^^^^^^^^^^
            T.evaluate(6)
            T.evaluate(7)
            ^^^^^^^^^^^^^
    """
    )


def test_underline_func():
    @T.prim_func
    def func():
        T.evaluate(0)

    result = func.with_attr("global_symbol", "main").script(
        path_to_underline=[
            AccessPath.root(),
        ],
        extra_config={"render_invisible_path_info": False},
    )
    assert result == format_script(
        """
        # from tvm.script import tirx as T
        # from tvm.tirx.layout import Axis

        @T.prim_func
        ^^^^^^^^^^^^
        def main():
        ^^^^^^^^^^^
            T.evaluate(0)
            ^^^^^^^^^^^^^
    """
    )


def test_underline_func_in_irmodule():
    @I.ir_module
    class irmodule:
        @T.prim_func
        def func():
            T.evaluate(0)

    result = irmodule.script(
        path_to_underline=[
            AccessPath.root().attr("functions").map_item(irmodule.get_global_var("func")),
        ],
        extra_config={"render_invisible_path_info": False},
    )
    assert result == format_script(
        """
        # from tvm.script import ir as I
        # from tvm.script import tirx as T
        # from tvm.tirx.layout import Axis

        @I.ir_module
        class Module:
            @T.prim_func
            ^^^^^^^^^^^^
            def func():
            ^^^^^^^^^^^
                T.evaluate(0)
                ^^^^^^^^^^^^^
    """
    )


def test_underline_irmodule():
    @I.ir_module
    class irmodule:
        @T.prim_func
        def func():
            T.evaluate(0)

    result = irmodule.script(
        path_to_underline=[
            AccessPath.root(),
        ],
        extra_config={"render_invisible_path_info": False},
    )
    assert result == format_script(
        """
        # from tvm.script import ir as I
        # from tvm.script import tirx as T
        # from tvm.tirx.layout import Axis

        @I.ir_module
        ^^^^^^^^^^^^
        class Module:
        ^^^^^^^^^^^^^
            @T.prim_func
            ^^^^^^^^^^^^
            def func():
            ^^^^^^^^^^^
                T.evaluate(0)
                ^^^^^^^^^^^^^
    """
    )
