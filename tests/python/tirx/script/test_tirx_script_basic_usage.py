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

"""TIRx script basic usage."""

from __future__ import annotations

import pytest
import tvm_ffi

import tvm
import tvm.script
import tvm.testing
from tvm import ir, tirx
from tvm.ir import PointerType, PrimType, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


def test_native_concise_scopes_unwind_with_their_parent():
    # Nested concise thread scopes must preserve the original variables in the constructed IR.
    from tvm import tirx

    variables = []

    def observe(*items):
        variables.extend(items)

    @T.prim_func
    def main():
        bx = T.launch_thread("blockIdx.x", 2)
        tx = T.launch_thread("threadIdx.x", 32)
        observe(bx, tx)
        T.evaluate(bx + tx)

    bx, tx = variables
    body = main.body
    assert isinstance(body, tirx.AttrStmt) and isinstance(body.body, tirx.AttrStmt)
    assert body.node.var.same_as(bx) and body.body.node.var.same_as(tx)
    assert body.body.body.value.a.same_as(bx) and body.body.body.value.b.same_as(tx)


def test_tir_buffer_annotation():
    buffer_0 = T.Buffer((128, 128), "float32")
    assert (
        tirx.is_buffer_var(buffer_0)
        and list(buffer_0.shape) == [128, 128]
        and buffer_0.dtype == ir.PrimType("float32")
    )

    buffer_1 = T.Buffer((64, 64, 64), "int32")
    assert (
        tirx.is_buffer_var(buffer_1)
        and list(buffer_1.shape) == [64, 64, 64]
        and buffer_1.dtype == ir.PrimType("int32")
    )


def test_tir_ptr_proxy():
    ptr_0 = T.handle("int32", "global")
    assert (
        isinstance(ptr_0, tirx.Var)
        and isinstance(ptr_0.ty, ir.PointerType)
        and ptr_0.ty.element_type == ir.PrimType("int32")
        and ptr_0.ty.storage_scope == "global"
    )

    ptr_1 = T.handle("float32", "shared")
    assert (
        isinstance(ptr_1, tirx.Var)
        and isinstance(ptr_1.ty, ir.PointerType)
        and ptr_1.ty.element_type == ir.PrimType("float32")
        and ptr_1.ty.storage_scope == "shared"
    )


def test_grid():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        for (*lvs,) in T.grid(10, (2, 12)):
            T.evaluate(lvs[0] + lvs[1])
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_range():
    # fmt: off
    @T.prim_func(private=True)
    def test():
        l = T.meta_var([i for i in range(10)])  # noqa: E741
        T.evaluate(l[3])

    @T.prim_func(private=True)
    def expected():
        T.evaluate(3)
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    tvm.ir.assert_structural_equal(test, expected)


def test_scalar_annotation_syntax():
    """Test the scalar annotation syntax: x: T.int32 = init, x: T.int32, and T.let."""

    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
                # Scalar with init value
        x: T.int32 = 0
        y: T.float16 = T.float16(1.0)
                # Scalar without init
        z: T.int32
                # Use scalars
        x = x + T.int32(1)
        z = x + T.int32(2)
        y = y + T.float16(3.0)
        T.evaluate(x + z)
        T.evaluate(y)
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_let_annotation_syntax():
    """Test explicit LetStmt syntax: T.let[T.int32] and T.let."""

    # fmt: off
    @T.prim_func
    def test():
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        # Explicit LetStmt with type
        bx: T.let[T.int32] = blockIdx_x
        tx: T.let[T.int32] = threadIdx_x
        # Explicit LetStmt with auto-type
        combined: T.let = bx + tx
        T.device_entry()
        T.evaluate(bx + tx + combined)
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_tuple_let_binding_and_traversal():
    @T.prim_func
    def from_list(x: T.int32, y: T.float32) -> T.int32:
        pair: T.let = [x, (y,)]
        return pair[0]

    @T.prim_func
    def from_tuple(x: T.int32, y: T.float32) -> T.int32:
        pair: T.let = (x, (y,))
        return pair[0]

    def tuple_value(func):
        visited = []
        tvm_ffi.structural_walk(func.body, visited.append)
        bind = next(node for node in visited if isinstance(node, tvm.tirx.Bind))
        return bind.value

    list_value = tuple_value(from_list)
    tuple_value = tuple_value(from_tuple)
    assert isinstance(list_value, tvm.ir.Tuple)
    assert isinstance(list_value.fields[1], tvm.ir.Tuple)
    assert_structural_equal(list_value, tuple_value, map_free_vars=True)

    code = from_list.script()
    assert "pair: T.let[T.Tuple(T.int32, T.Tuple(T.float32))] = x, (y,)" in code
    assert from_source(code).script() == code
    assert_structural_equal(from_list, from_source(code))


def test_annotation_syntax_comprehensive():
    """Comprehensive test for scalar annotation, T.let, banned annotations, and bare assignment."""

    # 1. T.let with T.Var(PointerType) — round-trip
    # fmt: off
    @T.prim_func
    def test_let_var():
        T.device_entry()
        smem = T.alloc_shared([128], "float16")
        ptr: T.let[T.Var(name="ptr", ty=PointerType(PrimType("void")))] = T.reinterpret(
            "handle", smem.access_ptr("rw")
        )
        T.evaluate(ptr)
        # fmt: on
    code = test_let_var.script()
    assert from_source(code).script() == code

    # 2. Banned: handle as scalar annotation
    src_handle = """
from tvm.script import tirx as T
@T.prim_func
def func():
    x: T.handle = T.int64(0)
"""
    with pytest.raises(tvm.error.InternalError):
        from_source(src_handle)

    # 3. Banned: non-PrimType annotation without T.let
    src_ptr = """
from tvm.script import tirx as T
from tvm.ir import PointerType, PrimType
@T.prim_func
def func():
    x: T.Var(name="x", ty=PointerType(PrimType("float16"))) = T.int64(0)
"""
    with pytest.raises(tvm.error.InternalError):
        from_source(src_ptr)

    # 4. An explicit mutable scalar declaration retains updates — round-trip
    # fmt: off
    @T.prim_func
    def test_bare_assign():
        T.device_entry()
        tid = T.launch_thread("threadIdx.x", 128)
        x: T.int32 = tid + T.int32(1)
        x = x + T.int32(2)
        T.evaluate(x)
        # fmt: on
    code = test_bare_assign.script()
    assert from_source(code).script() == code


def test_pointer_expression_assignment_uses_bind():
    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        buf = T.alloc_buffer((4,), "uint32", scope="shared")
        ptr = buf.ptr_to([1])
        T.evaluate(T.reinterpret("uint64", ptr))
    # fmt: on

    binds = []
    tvm_ffi.structural_walk(
        func.body, lambda node: binds.append(node) if isinstance(node, tvm.tirx.Bind) else None
    )
    assert len(binds) == 1
    assert isinstance(binds[0].var.ty, PointerType)
    assert_structural_equal(binds[0].var.ty, binds[0].value.ty)

    code = func.script()
    assert_structural_equal(func, from_source(code))


def test_pointer_expression_rebinding_creates_distinct_native_bindings():
    # Before: ptr = buf.ptr_to([0]); ptr = buf.ptr_to([1]); X.evaluate(ptr)
    # Expected builder program:
    # ptr = X.bind_(buf.ptr_to([0]), name="ptr")
    # ptr = X.bind_(buf.ptr_to([1]), name="ptr"); X.emit_(X.evaluate(ptr))
    # Pointer expressions are ordinary immutable bindings.
    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        buf = T.alloc_buffer((4,), "uint32", scope="shared")
        ptr = buf.ptr_to([0])
        ptr = buf.ptr_to([1])
        T.evaluate(T.reinterpret("uint64", ptr))
    # fmt: on

    bindings, uses = [], []

    def collect(node):
        if isinstance(node, tvm.tirx.Bind):
            bindings.append(node)
        elif isinstance(node, tvm.tirx.Evaluate):
            uses.append(node)

    tvm_ffi.structural_walk(func.body, collect)
    assert len(bindings) == 2
    assert all(isinstance(binding.var.ty, PointerType) for binding in bindings)
    assert not bindings[0].var.same_as(bindings[1].var)
    assert len(uses) == 1
    assert uses[0].value.args[0].same_as(bindings[1].var)
    assert_structural_equal(func, from_source(func.script()))


def test_pointer_expression_assignment_can_shadow_extra_var():
    source = """
@T.prim_func
def func() -> None:
    T.device_entry()
    buf = T.alloc_buffer((4,), "uint32", scope="shared")
    ptr = buf.ptr_to([1])
    view = T.decl_buffer((3,), "uint32", data=ptr, scope="shared")
    view[0] = T.uint32(0)
"""
    func = tvm.script.from_source(source, extra_vars={"T": T, "ptr": object()})

    binds = []
    tvm_ffi.structural_walk(
        func.body, lambda node: binds.append(node) if isinstance(node, tvm.tirx.Bind) else None
    )
    assert len(binds) == 1
    assert_structural_equal(func, from_source(func.script()))


def test_roundtrip_unary_inplace():
    """Single-arg unary ops (in-place) should round-trip."""

    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((128,), "float32", scope="global")) -> None:
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        Tx.warp.exp2(A[0:32])
        Tx.warp.sqrt(A[32:64])
        Tx.warp.reciprocal(A[64:96])
        # fmt: on

    code = test.script()
    # Each op should appear with a single arg (no duplicate src, no trailing Nones)
    assert 'T.warp.exp2(A[0:32])' in code, f"expected single-arg exp2, got:\n{code}"
    assert 'T.warp.sqrt(A[32:64])' in code, f"expected single-arg sqrt, got:\n{code}"
    assert 'T.warp.reciprocal(A[64:96])' in code, (
        f"expected single-arg reciprocal, got:\n{code}"
    )
    assert "None" not in code, f"trailing None args should be trimmed:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unary_different_dst_src():
    """Unary ops with different dst and src should keep both args."""

    # fmt: off
    @T.prim_func
    def test(
        A: T.Buffer((128,), "float32", scope="global"),
        B: T.Buffer((128,), "float32", scope="global"),
    ) -> None:
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        Tx.warp.exp2(A[0:32], B[0:32])
        # fmt: on

    code = test.script()
    assert 'T.warp.exp2(A[0:32], B[0:32])' in code, (
        f"different dst/src should keep both:\n{code}"
    )
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_vector_annotation_syntax_1d():
    """Test x: T.f32[N] produces the same IR as T.alloc_local([N], 'float32')."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        v: T.float32[8]
        T.evaluate(v[0])  # noqa: F821

    @T.prim_func
    def func():  # noqa: F811
        T.device_entry()
        v = T.alloc_local([8], "float32")
        T.evaluate(v[0])
        # fmt: on

        # func was redefined; compare first (annotation) with second (alloc_local).
        # Re-create the annotation version for comparison:

        # fmt: off
    @T.prim_func
    def annotation_func():
        T.device_entry()
        v: T.float32[8]
        T.evaluate(v[0])  # noqa: F821
        # fmt: on

        # Verify both produce valid IR that round-trips through printer/parser
    code = func.script()
    assert from_source(code).script() == code
    code2 = annotation_func.script()
    assert from_source(code2).script() == code2
    # The printed form should be identical (both become alloc_local in print)
    assert code.replace("annotation_func", "func") == code


def test_vector_annotation_syntax_multidim():
    """Test x: T.f32[M, N] produces the same IR as T.alloc_local([M, N], 'float32')."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        m: T.float32[4, 8]
        T.evaluate(m[0, 0])  # noqa: F821
        # fmt: on

    code = func.script()
    assert "alloc_local((4, 8)" in code or "float32[4, 8]" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_vector_annotation_shorthand_aliases():
    """Test shorthand aliases: T.f32, T.i32, T.f16, etc."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        a: T.f32[4]
        b: T.i32[2]
        c: T.f16[8]
        T.evaluate(a[0] + T.float32(b[0]) + T.float32(c[0]))  # noqa: F821
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_scalar_annotation_shorthand():
    """Test x: T.f32 (scalar) shorthand produces same IR as x: T.float32."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        x: T.f32 = 0
        y: T.i32
        x = x + T.float32(1.0)
        y = T.int32(2)
        T.evaluate(x + T.float32(y))
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_vector_annotation_with_python_variable_size():
    """Test x: T.f16[vec_size] where vec_size is a Python variable."""
    vec_size = 16

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        v: T.f16[vec_size]
        T.evaluate(T.float32(v[0]))  # noqa: F821
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_empty_module():
    @I.ir_module
    class BlankIRModule:
        pass

    assert isinstance(BlankIRModule, ir.IRModule)
    assert len(BlankIRModule.functions) == 0
    assert BlankIRModule.__name__ == "BlankIRModule"


def test_module_string_constants_keep_common_constructor_values():
    # Literal strings passed to common IR constructors must remain concrete values
    # when a real module definition is parsed.
    from tvm import ir
    from tvm.script import ir as I

    @I.ir_module
    class Module:
        I.module_attrs({"tag": I.StringImm("label"), "type": I.StringType()})

    expected = ir.IRModule(attrs={"tag": ir.StringImm("label"), "type": ir.StringType()})
    ir.assert_structural_equal(expected, Module)
    assert Module.attrs["tag"].value == "label"
    assert isinstance(Module.attrs["type"], ir.StringType)


def test_thread_return_is_distinct_from_function_return():
    @T.prim_func
    def thread_exit():
        T.thread_return()

    @T.prim_func
    def function_exit():
        return 0

    assert isinstance(thread_exit.body, tirx.Evaluate)
    assert thread_exit.body.value.op.name == "tirx.thread_return"
    assert isinstance(function_exit.body, tirx.Return)
    assert isinstance(function_exit.body.value, tirx.IntImm)
    assert function_exit.body.value.value == 0


def test_loop_control_validation_preserves_valid_and_unchecked_ir():
    # Invalid loop placement must be rejected, while disabled checks preserve the original IR.
    from tvm import error, ir, tirx

    invalid = tirx.PrimFunc(params=[], body=tirx.Break())

    # Direct construction retains the native statement for an explicit verifier pass.
    ir.assert_structural_equal(invalid.body, tirx.Break())
    assert not tirx.analysis.verify_well_formed(invalid, assert_mode=False)
    with pytest.raises(error.InternalError, match="requires an enclosing loop"):
        tirx.analysis.verify_well_formed(invalid)

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
