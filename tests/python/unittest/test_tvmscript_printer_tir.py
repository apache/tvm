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

from tvm.ir import PointerType, PrimType, TupleType
from tvm.script.printer import script
from tvm.tir import SizeVar, Var, decl_buffer


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
    return cleaned_lines


def as_tir_script(node):
    return script(node, "tir", {"tir": "T"})


@pytest.mark.parametrize(
    "ty, expected",
    [
        pytest.param(
            PrimType("int8"),
            """
            T.int8
            """,
            id="int",
        ),
        pytest.param(
            PrimType("float32"),
            """
            T.float32
            """,
            id="float",
        ),
        pytest.param(
            PointerType(PrimType("int32")),
            """
            T.Ptr(T.int32)
            """,
            id="pointer",
        ),
        pytest.param(
            PointerType(PrimType("int32"), "global"),
            """
            T.Ptr(T.int32, "global")
            """,
            id="with_scope",
        ),
        pytest.param(
            TupleType([]),
            """
            None
            """,
            id="none",
        ),
    ],
)
def test_type(ty, expected):
    assert format_script(expected) == as_tir_script(ty)


@pytest.mark.parametrize("var_type", [Var, SizeVar])
def test_var(var_type):
    var = var_type("x", "int8")

    assert as_tir_script(var) == format_script(
        """
        x: T.int8
        x
        """
    )


@pytest.mark.parametrize(
    "buffer, expected",
    [
        pytest.param(
            decl_buffer((5, 10), name="b"),
            """
            b: T.Buffer(shape=(5, 10))
            b
            """,
            id="simple",
        ),
        pytest.param(
            decl_buffer((5), name=""),
            """
            buf: T.Buffer(shape=(5,))
            buf
            """,
            id="no_name",
        ),
        pytest.param(
            decl_buffer((SizeVar("m", "int"), SizeVar("n", "int")), dtype="int8"),
            """
            m: T.int32
            n: T.int32
            buffer: T.Buffer("int8", shape=(m, n))
            buffer
            """,
            id="symbolic_shape",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="int8",
                data=Var("p", PointerType(PrimType("int16"), "local")),
                strides=[2, 5],
                elem_offset=2,
                data_alignment=16,
                offset_factor=2,
                scope="local",
            ),
            """
            p: T.Ptr(T.int16, "local")
            buffer: T.Buffer("int8", shape=(4, 10), data=p, strides=(2, 5), elem_offset=2, scope="local", align=16, offset_factor=2)
            buffer
            """,
            id="all_param",
        ),
        pytest.param(
            decl_buffer(
                (4, 10),
                dtype="bool",
            ),
            """
            buffer: T.Buffer("bool", shape=(4, 10))
            buffer
            """,
            id="bool_different_ptr_type",
        ),
    ],
)
def test_buffer(buffer, expected):
    assert as_tir_script(buffer) == format_script(expected)
