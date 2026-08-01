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


def _import_and_register():
    # Ensure all schedule registrations (legacy + dispatcher variants) are loaded
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401


class _DummyKind:
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:  # used in messages
        return self.name


class _DummyTarget:
    def __init__(self, kind_name: str):
        self.kind = _DummyKind(kind_name)


class _DummyExecScope:
    def __init__(self, name: str):
        self.name = name


class _DummySctx:
    def __init__(self, target_kind: str, exec_scope: str):
        self.target = _DummyTarget(target_kind)
        self.exec_scope = _DummyExecScope(exec_scope)
        self.scope_kind = exec_scope


def test_dispatch_prints_predicate_reasons():
    """Validate TRACE mode prints per-variant predicate failure reasons."""
    _import_and_register()
    from tvm.ir import Op
    from tvm.tirx.operator.tile_primitive.dispatcher import run_dispatch

    class _OpCall:
        def __init__(self, op):
            self.op = op
            self.args = []  # not used by the tested predicates

    # Use TRN copy; predicate requires exec_scope == "thread".
    op_call = _OpCall(Op.get("tirx.tile.copy"))
    sctx = _DummySctx(target_kind="trn", exec_scope="warp")  # intentionally wrong

    with pytest.raises(RuntimeError) as e:
        run_dispatch(op_call, sctx)

    out = str(e.value)
    print(out)
    # Header + per-variant reason must be printed in table format
    assert "TIRx schedule dispatch failed: op=tirx.tile.copy target=trn" in out
    assert "Variant" in out  # table header present
    assert "default" in out  # variant name present
    assert "rejected: exec_scope" in out
    # opcall object IR should be printed in the table
    assert "opcall:" in out


def test_dispatch_forced_variant_missing_table_and_message():
    _import_and_register()
    from tvm.ir import Op
    from tvm.tirx.operator.tile_primitive.dispatcher import run_dispatch

    class _OpCall:
        def __init__(self, op):
            self.op = op
            self.dispatch = "__nonexistent__"
            self.args = []

    op_call = _OpCall(Op.get("tirx.tile.copy"))
    sctx = _DummySctx(target_kind="trn", exec_scope="thread")

    with pytest.raises(RuntimeError) as e:
        run_dispatch(op_call, sctx)

    msg = str(e.value)
    print(msg)
    assert "TIRx schedule dispatch failed: op=tirx.tile.copy target=trn" in msg
    assert "no variant named '__nonexistent__' is registered" in msg


def test_dispatch_raises_with_aggregated_reasons():
    """Validate STRICT mode raises aggregated error message with reasons."""
    _import_and_register()
    from tvm.ir import Op
    from tvm.tirx.operator.tile_primitive.dispatcher import run_dispatch

    class _OpCall:
        def __init__(self, op):
            self.op = op
            self.args = []

    # Use TRN compose_op; variant implementation raises NotImplementedError
    op_call = _OpCall(Op.get("tirx.tile.compose_op"))
    sctx = _DummySctx(target_kind="trn", exec_scope="thread")

    with pytest.raises(RuntimeError) as e:
        run_dispatch(op_call, sctx)

    msg = str(e.value)
    print(msg)
    assert "TIRx schedule dispatch failed: op=tirx.tile.compose_op target=trn" in msg
    assert "default" in msg
    assert "exception — NotImplementedError" in msg
    # opcall content and backtrace should be included inside the table
    assert "opcall:" in msg
    assert "Traceback (most recent call last):" in msg


def test_dispatch_prints_real_opcall_ir():
    """Create a real TilePrimitiveCall via BufferRegions and ensure its IR is in the table."""
    _import_and_register()
    from tvm.ir import Op
    from tvm.tirx.buffer import decl_buffer
    from tvm.tirx.operator.tile_primitive.dispatcher import run_dispatch
    from tvm.tirx.tile_primitive import TilePrimitiveCall

    # Build a real TIRx TilePrimitiveCall: tirx.tile.copy(A[0:64], B[0:64])
    A = decl_buffer((64,), "float32", scope="global")
    B = decl_buffer((64,), "float32", scope="shared")
    real_opcall = TilePrimitiveCall(
        A[0:64], B[0:64], op=Op.get("tirx.tile.copy"), workspace={}, config={}
    )

    # Force predicate rejection to trigger formatted error with opcall IR
    sctx = _DummySctx(target_kind="trn", exec_scope="warp")
    with pytest.raises(RuntimeError) as e:
        run_dispatch(real_opcall, sctx)

    out = str(e.value)
    print(out)
    # Verify header and that the opcall IR is included in the table
    assert "TIRx schedule dispatch failed: op=tirx.tile.copy target=trn" in out
    assert "Variant" in out
    assert "opcall:" in out
    # IR should mention the operator name
    assert "tirx.tile.copy" in out
