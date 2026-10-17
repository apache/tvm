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
"""Tests for tirx.analysis.undefined_vars."""

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import tirx


def test_decl_tensor_data_is_use():
    """DeclTensor's data var should be reported as undefined (USE), not defined.

    When UndefinedVars encounters a DeclTensor, the data pointer references
    an existing variable from the enclosing scope.  It must appear in the
    undefined list so that callers (e.g., CreateComputeScope) capture it.
    """
    n = tirx.Var("n", "int32")
    from tvm_tirx.ir import PrimType, PtrType

    data_ptr = tirx.Var("buf_data", PtrType(PrimType("float32")))
    buf = tirx.Var("buf", tirx.TensorType((n,), "float32", data=data_ptr))

    body = tvm_tirx.ir.Evaluate(tvm_tirx.ir.TensorLoad(buf, [0]))
    decl = tvm_tirx.ir.Bind(
        buf,
        tvm_tirx.ir.Call(
            "tirx.decl_tensor",
            [data_ptr],
            ty_args=[buf.ty],
        ),
    )
    stmt = tvm_tirx.ir.SeqStmt([decl, body])

    undef = tvm_tirx.tirx.analysis.undefined_vars(stmt, [])
    undef_names = {v.name for v in undef}
    # data_ptr must be undefined (it comes from outside the DeclTensor)
    assert "buf_data" in undef_names, f"Expected buf_data in undefined vars, got {undef_names}"


def test_decl_tensor_elem_offset_is_use():
    """DeclTensor's elem_offset var should be reported as undefined (USE).

    After FlattenTensor, DeclTensor nodes carry elem_offset vars from
    match_tensor entries.  These must appear in the undefined list.
    """
    from tvm_tirx.ir import PrimType, PtrType

    n = tirx.Var("n", "int32")
    data_ptr = tirx.Var("buf_data", PtrType(PrimType("float32")))
    elem_off = tirx.Var("buf_elem_offset", "int32")
    buf = tirx.Var("buf", tirx.TensorType((n,), "float32", data=data_ptr, elem_offset=elem_off))

    body = tvm_tirx.ir.Evaluate(tvm_tirx.ir.TensorLoad(buf, [0]))
    decl = tvm_tirx.ir.Bind(
        buf,
        tvm_tirx.ir.Call(
            "tirx.decl_tensor",
            [data_ptr],
            ty_args=[buf.ty],
        ),
    )
    stmt = tvm_tirx.ir.SeqStmt([decl, body])

    undef = tvm_tirx.tirx.analysis.undefined_vars(stmt, [])
    undef_names = {v.name for v in undef}
    assert "buf_data" in undef_names, f"Expected buf_data in undefined vars, got {undef_names}"
    assert "buf_elem_offset" in undef_names, (
        f"Expected buf_elem_offset in undefined vars, got {undef_names}"
    )


def test_alloc_tensor_data_is_def():
    """AllocTensor's data var should NOT be reported as undefined (it's a DEF).

    AllocTensor allocates new storage — the data pointer is a new definition,
    not a reference to an external variable.
    """
    n = tirx.Var("n", "int32")
    buf = tirx.Var("buf", tirx.TensorType((n,), "float32"))

    body = tvm_tirx.ir.Evaluate(tvm_tirx.ir.TensorLoad(buf, [0]))
    alloc = tvm_tirx.ir.Bind(
        buf,
        tvm_tirx.ir.Call(
            "tirx.alloc_tensor",
            [],
            attrs=tvm_tirx.ir.DictAttrs({}),
            ty_args=[buf.ty],
        ),
    )
    stmt = tvm_tirx.ir.SeqStmt([alloc, body])

    undef = tvm_tirx.tirx.analysis.undefined_vars(stmt, [])
    undef_names = {v.name for v in undef}
    # The tensor Var itself is defined by AllocTensor.
    assert buf.name not in undef_names
    # shape var n should be undefined (comes from enclosing scope)
    assert "n" in undef_names, f"Expected shape var 'n' in undefined vars, got {undef_names}"


def test_tensor_data_projection_is_tensor_use():
    """An opaque data projection must retain the TensorVar identity."""
    buf = tirx.Var("buf", tirx.TensorType((16,), "float32"))
    stmt = tvm_tirx.ir.Evaluate(buf.data)

    undef = tvm_tirx.tirx.analysis.undefined_vars(stmt, [])
    assert len(undef) == 1
    assert undef[0].same_as(buf)


if __name__ == "__main__":
    tvm_tirx.testing.main()
