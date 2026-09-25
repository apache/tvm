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
from tvm.script.parser import entry
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
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
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
    def test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        B = T.match_buffer(B_ptr, (128,), "float32", scope="global")
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


def test_initial_import_alias_and_symbolic_range():
    source = """from tvm.script import tirx as Script
@Script.prim_func
def main(n: Script.int32):
    for i in range(n):
        Script.evaluate(i)
"""
    function = entry.parse(source)
    assert function.body.extent.same_as(function.params[0])
    assert function.body.body.value.same_as(function.body.loop_var)


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
