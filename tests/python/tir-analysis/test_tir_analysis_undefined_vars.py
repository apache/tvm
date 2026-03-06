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
"""Tests for tir.analysis.undefined_vars (VarUseDefAnalyzer)."""

import tvm
import tvm.testing
from tvm import tir


def test_decl_buffer_data_is_use():
    """DeclBuffer's data var should be reported as undefined (USE), not defined.

    When UndefinedVars encounters a DeclBuffer, the data pointer references
    an existing variable from the enclosing scope.  It must appear in the
    undefined list so that callers (e.g., CreateComputeScope) capture it.
    """
    n = tir.SizeVar("n", "int32")
    from tvm.ir import PointerType, PrimType

    data_ptr = tir.Var("buf_data", PointerType(PrimType("float32")))
    buf = tir.decl_buffer((n,), "float32", "buf", data=data_ptr)

    body = tir.Evaluate(tir.BufferLoad(buf, [0]))
    decl = tir.DeclBuffer(buf)
    stmt = tir.SeqStmt([decl, body])

    undef = tvm.tir.analysis.undefined_vars(stmt, [])
    undef_names = {v.name for v in undef}
    # data_ptr must be undefined (it comes from outside the DeclBuffer)
    assert "buf_data" in undef_names, f"Expected buf_data in undefined vars, got {undef_names}"


def test_decl_buffer_elem_offset_is_use():
    """DeclBuffer's elem_offset var should be reported as undefined (USE).

    After FlattenBuffer, DeclBuffer nodes carry elem_offset vars from
    match_buffer entries.  These must appear in the undefined list.
    """
    from tvm.ir import PointerType, PrimType

    n = tir.SizeVar("n", "int32")
    data_ptr = tir.Var("buf_data", PointerType(PrimType("float32")))
    elem_off = tir.Var("buf_elem_offset", "int32")
    buf = tir.decl_buffer((n,), "float32", "buf", data=data_ptr, elem_offset=elem_off)

    body = tir.Evaluate(tir.BufferLoad(buf, [0]))
    decl = tir.DeclBuffer(buf)
    stmt = tir.SeqStmt([decl, body])

    undef = tvm.tir.analysis.undefined_vars(stmt, [])
    undef_names = {v.name for v in undef}
    assert "buf_data" in undef_names, f"Expected buf_data in undefined vars, got {undef_names}"
    assert "buf_elem_offset" in undef_names, (
        f"Expected buf_elem_offset in undefined vars, got {undef_names}"
    )


def test_alloc_buffer_data_is_def():
    """AllocBuffer's data var should NOT be reported as undefined (it's a DEF).

    AllocBuffer allocates new storage — the data pointer is a new definition,
    not a reference to an external variable.
    """
    n = tir.SizeVar("n", "int32")
    buf = tir.decl_buffer((n,), "float32", "buf")

    body = tir.Evaluate(tir.BufferLoad(buf, [0]))
    alloc = tir.AllocBuffer(buf)
    stmt = tir.SeqStmt([alloc, body])

    undef = tvm.tir.analysis.undefined_vars(stmt, [])
    undef_names = {v.name for v in undef}
    # data should NOT be undefined — AllocBuffer defines it
    assert buf.data.name not in undef_names, (
        f"AllocBuffer data should be defined, but found {buf.data.name} in {undef_names}"
    )
    # shape var n should be undefined (comes from enclosing scope)
    assert "n" in undef_names, f"Expected shape var 'n' in undefined vars, got {undef_names}"


if __name__ == "__main__":
    tvm.testing.main()
