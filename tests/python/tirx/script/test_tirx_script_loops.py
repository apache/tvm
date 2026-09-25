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

"""TIRx script loops."""

import pytest

import tvm
import tvm.script
import tvm.testing
from tvm.ir import PrimType, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


def test_loop_control_validation_preserves_valid_and_unchecked_ir():
    # Invalid loop placement must be rejected, while disabled checks preserve the original IR.
    from tvm import ir, tirx

    @T.prim_func(check_well_formed=False)
    def invalid():
        T.evaluate(tirx.break_loop())

    # This is exactly the native node emitted by source `break`; constructing
    # the intrinsic explicitly keeps the surrounding Python definition valid.
    ir.assert_structural_equal(invalid.body, tirx.Evaluate(tirx.break_loop()))
    with pytest.raises(ValueError, match="requires an enclosing loop"):

        @T.prim_func
        def rejected():
            T.evaluate(tirx.break_loop())

    @T.prim_func
    def valid():
        for i in range(2):
            break

    assert isinstance(valid.body, tirx.For)
    ir.assert_structural_equal(valid.body.body, invalid.body)

    @I.ir_module(check_well_formed=False, extra_vars={"invalid": invalid})
    class Unchecked:
        bad = invalid

    assert Unchecked["bad"].same_as(invalid)
    with pytest.raises(ValueError, match="requires an enclosing loop"):

        @I.ir_module(extra_vars={"invalid": invalid})
        class Rejected:
            bad = invalid


def test_roundtrip_break_for():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((10,), 'int32')):

        T.device_entry()
        for i in T.serial(10):
            if i > 5:
                break
            A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_roundtrip_break_while():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((10,), 'int32')):

        T.device_entry()
        i = T.alloc_buffer((1,), "int32", scope="local")
        i[0] = 0
        while i[0] < 10:
            A[i[0]] = i[0] * 2
            if A[i[0]] > 10:
                break
            i[0] = i[0] + 1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_nested():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((9,), 'int32')):

        T.device_entry()
        idx = T.alloc_buffer((1,), "int32", scope="local")
        idx[0] = 0
        for i in T.serial(3):
            for j in T.serial(3):
                A[idx[0]] = i * 10 + j
                idx[0] += 1
                if j == 1:
                    break
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_for():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((10,), 'int32')):

        T.device_entry()
        for i in T.serial(10):
            if (i % 2) == 0:
                continue
            A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_while():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((10,), 'int32')):

        T.device_entry()
        i = T.alloc_buffer((1,), "int32", scope="local")
        i[0] = 0
        while i[0] < 10:
            if (i[0] % 2) == 1:
                i[0] += 1
                continue
            A[i[0]] = i[0]
            i[0] += 1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_nested():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((9,), 'int32')):

        T.device_entry()
        idx = T.alloc_buffer((1,), dtype="int32", scope="local")
        idx[0] = 0
        for i in T.serial(3):
            for j in T.serial(3):
                if j == 1:
                    continue
                A[idx[0]] = i * 10 + j
                idx[0] += 1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_and_continue():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((10,), 'int32')):

        T.device_entry()
        for i in T.serial(10):
            if i == 2:
                continue
            if i == 7:
                break
            A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unreachable_after_break():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((5,), 'int32')):

        T.device_entry()
        for i in T.serial(5):
            A[i] = i
            break
                    # This line is never reached
            A[i] = -1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_false():
    """T.serial(N, unroll=False) should round-trip."""

    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, unroll=False):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=False" in code, f"printer should emit unroll=False, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_true():
    """T.serial(N, unroll=True) should round-trip as a pragma-unroll request."""

    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, unroll=True):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=True" in code, f"printer should emit unroll=True, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_count():
    """T.serial(N, unroll=2) should preserve the requested unroll count."""

    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, unroll=2):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=2" in code, f"printer should emit unroll=2, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_false_with_other_annotations():
    """When other annotations exist alongside disable_unroll, fall back to full dict."""

    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, annotations={"disable_unroll": True, "custom": 42}):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "annotations=" in code, "printer should emit full annotations when multiple keys exist"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_loop_var_dtype_uint32():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128,), 'float32')):

        for i in T.serial(128, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    _assert_roundtrip(func)


def _assert_roundtrip(func):
    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_loop_var_dtype_uint32_with_step():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128,), 'float32')):

        for i in T.serial(4, 128, step=2, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    assert loop.step.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("for_kind", ["serial", "parallel", "vectorized", "unroll"])
def test_loop_var_dtype_uint32_all_for_kinds(for_kind):
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((4,), 'float32')):

        for i in getattr(T, for_kind)(4, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body.loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_grid_loop_var_dtype_uint32():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((8, 16), 'float32')):

        for i, j in T.grid(8, 16, dtype="uint32"):
            A[i, j] = T.float32(1)
    # fmt: on

    outer = func.body
    assert outer.loop_var.ty == PrimType("uint32")
    assert outer.body.loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_defaults_to_int32():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128,), 'float32')):

        for i in range(128):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body.loop_var.ty == PrimType("int32")
    _assert_roundtrip(func)


def test_loop_var_dtype_inferred_from_unsigned_extent():
    """A uint32 extent makes the loop var uint32 without an explicit dtype."""

    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128,), 'float32'), n: T.uint32):

        for i in range(n):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body.loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_casts_mismatched_bound():
    """A non-literal bound of another dtype is cast to the requested loop dtype."""

    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128,), 'float32'), n: T.int32):

        for i in T.serial(n, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "int16", "float32"])
def test_loop_var_dtype_rejects_unsupported(dtype):
    with pytest.raises(Exception, match='must be "int32" or "uint32"'):
        T.serial(4, dtype=dtype)


def test_thread_binding_has_no_dtype_parameter():
    with pytest.raises(TypeError):
        T.thread_binding(0, 128, "threadIdx.x", dtype="uint32")


def test_hand_built_for_promotes_int_literal_bounds_to_uint32():
    """The For constructor retypes literal bounds to the loop var's dtype."""
    loop_var = tvm.tirx.Var("i", "uint32")
    loop = tvm.tirx.For(loop_var, 0, 128, tvm.tirx.ForKind.SERIAL, tvm.tirx.Evaluate(0))
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")


def test_hand_built_for_rejects_negative_literal_for_uint32():
    loop_var = tvm.tirx.Var("i", "uint32")
    with pytest.raises(Exception, match="not representable"):
        tvm.tirx.For(loop_var, -1, 128, tvm.tirx.ForKind.SERIAL, tvm.tirx.Evaluate(0))
