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
from tvm.tir import SizeVar, Var


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


@pytest.mark.parametrize(
    "ty, expected",
    [
        (
            PrimType("int8"),
            """
            T.int8
            """,
        ),
        (
            PrimType("float32"),
            """
            T.float32
            """,
        ),
        (
            PointerType(PrimType("int32")),
            """
            T.Ptr(T.int32)
            """,
        ),
        (
            PointerType(PrimType("int32"), "global"),
            """
            T.Ptr(T.int32, "global")
            """,
        ),
        (
            TupleType([]),
            """
            None
            """,
        ),
    ],
)
def test_type(ty, expected):
    assert format_script(expected) == script(ty, "tir", {"tir": "T"})


@pytest.mark.parametrize("var_type", [Var, SizeVar])
def test_var(var_type):
    var = var_type("x", "int8")

    assert script(var, "tir", {"tir": "T"}) == format_script(
        """
        x: T.int8
        x
        """
    )
