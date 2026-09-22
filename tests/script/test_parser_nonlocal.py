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
"""Enclosing lexical declarations retain captured values during construction."""

import pytest

from tvm import ir
from tvm.script import parser
from tvm.script import relax as R
from tvm.script import tirx as T


@pytest.mark.parametrize(
    "source,captures,expected_source",
    [
        (
            """
@R.function
def main(x: R.Tensor((2,), dtype)):
    nonlocal dtype
    return x
""",
            {"R": R, "dtype": "float32"},
            '@R.function\ndef main(x: R.Tensor((2,), "float32")):\n    return x\n',
        ),
        (
            """
@T.prim_func
def main():
    nonlocal value
    T.evaluate(value + 1)
""",
            {"T": T, "value": 3},
            "@T.prim_func\ndef main():\n    T.evaluate(4)\n",
        ),
    ],
)
def test_nonlocal_declaration_preserves_captured_values(source, captures, expected_source):
    original_captures = dict(captures)
    expected = parser.parse(expected_source)
    actual = parser.parse(source, extra_vars=captures)
    ir.assert_structural_equal(expected, actual)
    assert captures == original_captures
