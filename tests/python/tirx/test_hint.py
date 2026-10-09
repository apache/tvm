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
"""Tests for hint configuration on tile primitive calls."""

import tvm
import tvm.script
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.script import tirx as T


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_hint_keyword_arg_on_tx_op():
    """Tx.op(..., hint="msg") stores hint in TileOpCall.config."""
    from tvm.tirx.stmt import TileOpCall
    from tvm.tirx.tensor import decl_tensor

    A = decl_tensor((64, 64), "float32", scope="global")
    A_sm = decl_tensor((64, 64), "float32", scope="shared")

    op_call = TileOpCall(
        A[0:64, 0:64],
        A_sm[0:64, 0:64],
        op=tvm.ir.Op.get("tirx.tile.copy"),
        workspace={},
        config={"hint": "3-input ptx"},
    )
    assert "hint" in op_call.config
    assert op_call.config["hint"].value == "3-input ptx"


def test_hint_keyword_arg_on_tx_op_roundtrip():
    """Tx.op(..., hint="msg") roundtrips through printer/parser."""
    from tvm.script.tirx import tile as Tx

    @T.function
    def func(
        A: T.Tensor([10], "float32", scope="global"), B: T.Tensor([10], "float32", scope="global")
    ):
        Tx.add(B, A, T.float32(1), hint="use_fast_math")

    code = func.script()
    assert 'hint="use_fast_math"' in code
    reparsed = from_source(code)
    assert reparsed.script() == code
    assert_structural_equal(func, reparsed)


if __name__ == "__main__":
    tvm.testing.main()
