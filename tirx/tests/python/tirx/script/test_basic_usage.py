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

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.script
import tvm_tirx.testing
from tvm_tirx import ir, tirx
from tvm_tirx.ir import PrimType, PtrType, assert_structural_equal
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def test_native_concise_scopes_unwind_with_their_parent():
    # Nested concise thread scopes must preserve the original variables in the constructed IR.

    variables = []

    def observe(*items):
        variables.extend(items)

    @T.function
    def main():
        bx = T.launch_thread("blockIdx.x", 2)
        tx = T.launch_thread("threadIdx.x", 32)
        observe(bx, tx)
        T.evaluate(bx + tx)

    bx, tx = variables
    body = main.body[0]
    assert isinstance(body, tvm_tirx.ir.RegionStmt) and isinstance(
        body.body[0], tvm_tirx.ir.RegionStmt
    )
    assert body.body_params[0].same_as(bx) and body.body[0].body_params[0].same_as(tx)
    assert body.body[0].body[0].value.a.same_as(bx) and body.body[0].body[0].value.b.same_as(tx)


def test_tir_tensor_annotation():
    tensor_0 = T.Tensor((128, 128), "float32")
    assert (
        isinstance(tensor_0, tirx.TensorType)
        and list(tensor_0.shape) == [128, 128]
        and tensor_0.dtype == ir.PrimType("float32")
    )

    tensor_1 = T.Tensor((64, 64, 64), "int32")
    assert (
        isinstance(tensor_1, tirx.TensorType)
        and list(tensor_1.shape) == [64, 64, 64]
        and tensor_1.dtype == ir.PrimType("int32")
    )


def test_tir_ptr_proxy():
    ptr_0 = T.handle("int32", "global")
    assert (
        isinstance(ptr_0, tirx.Var)
        and isinstance(ptr_0.ty, ir.PtrType)
        and ptr_0.ty.element_type == ir.PrimType("int32")
        and ptr_0.ty.scope == tvm_tirx.runtime.MemoryScope("global")
    )

    ptr_1 = T.handle("float32", "shared")
    assert (
        isinstance(ptr_1, tirx.Var)
        and isinstance(ptr_1.ty, ir.PtrType)
        and ptr_1.ty.element_type == ir.PrimType("float32")
        and ptr_1.ty.scope == tvm_tirx.runtime.MemoryScope("shared")
    )


def test_grid():
    # fmt: off
    @T.function
    def test():
        T.device_entry()
        for (*lvs,) in T.grid(10, (2, 12)):
            T.evaluate(lvs[0] + lvs[1])
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def from_source(code):
    return tvm_tirx.script.from_source(
        code, extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx}
    )


def test_range():
    # fmt: off
    @T.function(private=True)
    def test():
        l = T.meta_var([i for i in range(10)])  # noqa: E741
        T.evaluate(l[3])

    @T.function(private=True)
    def expected():
        T.evaluate(3)
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    tvm_tirx.ir.assert_structural_equal(test, expected)


def test_scalar_annotation_syntax():
    """Test the scalar annotation syntax: x: T.int32 = init, x: T.int32, and T.let."""

    # fmt: off
    @T.function
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
    @T.function
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
    @T.function
    def from_list(x: T.int32, y: T.float32) -> T.int32:
        pair: T.let = [x, (y,)]
        return pair[0]

    @T.function
    def from_tuple(x: T.int32, y: T.float32) -> T.int32:
        pair: T.let = (x, (y,))
        return pair[0]

    def tuple_value(func):
        visited = []
        tvm_ffi.structural_walk(func.body, visited.append)
        bind = next(node for node in visited if isinstance(node, tvm_tirx.ir.Bind))
        return bind.value

    list_value = tuple_value(from_list)
    tuple_value = tuple_value(from_tuple)
    assert isinstance(list_value, tvm_tirx.ir.Tuple)
    assert isinstance(list_value.fields[1], tvm_tirx.ir.Tuple)
    assert_structural_equal(list_value, tuple_value, map_free_vars=True)

    code = from_list.script()
    assert "pair: T.let[T.Tuple(T.int32, T.Tuple(T.float32))] = x, (y,)" in code
    assert from_source(code).script() == code
    assert_structural_equal(from_list, from_source(code))


def test_annotation_syntax_comprehensive():
    """Comprehensive test for scalar annotation, T.let, banned annotations, and bare assignment."""

    # 1. T.let with T.Var(PtrType) — round-trip
    # fmt: off
    @T.function
    def test_let_var():
        T.device_entry()
        smem = T.alloc_shared([128], "float16")
        ptr: T.let[T.Var(name="ptr", ty=PtrType(PrimType("void")))] = T.reinterpret(
             smem.ptr_to([0] * len(smem.shape))
        , ty="handle")
        T.evaluate(ptr)
        # fmt: on
    code = test_let_var.script()
    assert from_source(code).script() == code

    # 2. Banned: handle as scalar annotation
    src_handle = """
from tvm_tirx.script import tirx as T
@T.function
def func():
    x: T.handle = T.int64(0)
"""
    with pytest.raises(RuntimeError):
        from_source(src_handle)

    # 3. Banned: non-PrimType annotation without T.let
    src_ptr = """
from tvm_tirx.script import tirx as T
from tvm_tirx.ir import PtrType, PrimType
@T.function
def func():
    x: T.Var(name="x", ty=PtrType(PrimType("float16"))) = T.int64(0)
"""
    with pytest.raises(RuntimeError):
        from_source(src_ptr)

    # 4. An explicit mutable scalar declaration retains updates — round-trip
    # fmt: off
    @T.function
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
    @T.function
    def func() -> None:
        T.device_entry()
        buf = T.alloc_tensor(ty_args=[T.Tensor((4,), "uint32", scope="shared")])
        ptr = buf.ptr_to([1])
        T.evaluate(T.reinterpret( ptr, ty="uint64"))
    # fmt: on

    binds = []

    def collect_pointer_bind(node):
        if isinstance(node, tvm_tirx.ir.Bind) and isinstance(node.var.ty, PtrType):
            binds.append(node)

    tvm_ffi.structural_walk(func.body, collect_pointer_bind)
    assert len(binds) == 1
    assert isinstance(binds[0].var.ty, PtrType)
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
    @T.function
    def func() -> None:
        T.device_entry()
        buf = T.alloc_tensor(ty_args=[T.Tensor((4,), "uint32", scope="shared")])
        ptr = buf.ptr_to([0])
        ptr = buf.ptr_to([1])
        T.evaluate(T.reinterpret( ptr, ty="uint64"))
    # fmt: on

    bindings, uses = [], []

    def collect(node):
        if isinstance(node, tvm_tirx.ir.Bind) and isinstance(node.var.ty, PtrType):
            bindings.append(node)
        elif isinstance(node, tvm_tirx.ir.Evaluate):
            uses.append(node)

    tvm_ffi.structural_walk(func.body, collect)
    assert len(bindings) == 2
    assert all(isinstance(binding.var.ty, PtrType) for binding in bindings)
    assert not bindings[0].var.same_as(bindings[1].var)
    assert len(uses) == 1
    assert uses[0].value.args[0].same_as(bindings[1].var)
    assert_structural_equal(func, from_source(func.script()))


def test_pointer_expression_assignment_can_shadow_extra_var():
    source = """
@T.function
def func() -> None:
    T.device_entry()
    buf = T.alloc_tensor(ty_args=[T.Tensor((4,), T.uint32, scope="shared")])
    ptr = buf.ptr_to([1])
    view = T.decl_tensor(ptr, ty_args=[T.Tensor((3,), T.uint32, scope="shared")])
    view[0] = T.uint32(0)
"""
    func = tvm_tirx.script.from_source(source, extra_vars={"T": T, "ptr": object()})

    binds = []

    def collect_pointer_bind(node):
        if isinstance(node, tvm_tirx.ir.Bind) and isinstance(node.var.ty, PtrType):
            binds.append(node)

    tvm_ffi.structural_walk(func.body, collect_pointer_bind)
    assert len(binds) == 1
    assert_structural_equal(func, from_source(func.script()))


def test_tensor_region_roundtrip():
    @T.function
    def func(A: T.Tensor((4, 4), "float32")):
        T.cuda.tile.apply(T.ptx["sqrt.rn.f32"], A[0, 1:2], A[1, 2:3])

    assert_structural_equal(func, from_source(func.script()))


def test_roundtrip_unary_inplace():
    """Explicit unary input/output aliases should round-trip."""

    # fmt: off
    @T.function
    def test(A: T.Tensor((128,), "float32", scope="global")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        T.cuda.tile.apply(T.ptx['ex2.approx.ftz.f32'], A[0:32], A[0:32], scope='warp')
        T.cuda.tile.apply(T.ptx['sqrt.rn.f32'], A[32:64], A[32:64], scope='warp')
        T.cuda.tile.apply(
            T.ptx['div.rn.f32'], A[64:96], T.cast(1, ty=A.dtype), A[64:96], scope='warp'
        )
        # fmt: on

    code = test.script()
    # Opcode and explicit operand aliases survive printing.
    expected = (
        'T.cuda.tile.apply("ex2.approx.ftz.f32", (A[0:32], A[0:32]), '
        '(), scope=T.ThreadScope("cuda.warp"))'
    )
    assert expected in code, (
        f"expected explicit exp2, got:\n{code}"
    )
    expected = (
        'T.cuda.tile.apply("sqrt.rn.f32", (A[32:64], A[32:64]), '
        '(), scope=T.ThreadScope("cuda.warp"))'
    )
    assert expected in code, (
        f"expected explicit sqrt, got:\n{code}"
    )
    expected = (
        'T.cuda.tile.apply("div.rn.f32", (A[64:96], T.float32(1.0), A[64:96]), '
        '(), scope=T.ThreadScope("cuda.warp"))'
    )
    assert expected in code, (
        f"expected explicit reciprocal, got:\n{code}"
    )
    assert "None" not in code, f"trailing None args should be trimmed:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unary_different_dst_src():
    """Unary ops with different dst and src should keep both args."""

    # fmt: off
    @T.function
    def test(
        A: T.Tensor((128,), "float32", scope="global"),
        B: T.Tensor((128,), "float32", scope="global"),
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        T.cuda.tile.apply(T.ptx['ex2.approx.ftz.f32'], A[0:32], B[0:32], scope='warp')
        # fmt: on

    code = test.script()
    expected = (
        'T.cuda.tile.apply("ex2.approx.ftz.f32", (A[0:32], B[0:32]), '
        '(), scope=T.ThreadScope("cuda.warp"))'
    )
    assert expected in code, (
        f"different dst/src should keep both:\n{code}"
    )
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_vector_annotation_syntax_1d():
    """Test x: T.f32[N] produces the same IR as T.alloc_local([N], 'float32')."""

    # fmt: off
    @T.function
    def func():
        T.device_entry()
        v: T.float32[8]
        T.evaluate(v[0])  # noqa: F821

    @T.function
    def func():  # noqa: F811
        T.device_entry()
        v = T.alloc_local([8], "float32")
        T.evaluate(v[0])
        # fmt: on

        # func was redefined; compare first (annotation) with second (alloc_local).
        # Re-create the annotation version for comparison:

        # fmt: off
    @T.function
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
    @T.function
    def func():
        T.device_entry()
        m: T.float32[4, 8]
        T.evaluate(m[0, 0])  # noqa: F821
        # fmt: on

    code = func.script()
    assert ('alloc_tensor(ty_args=[T.Tensor((4, 8), T.float32, scope="local"') in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_vector_annotation_shorthand_aliases():
    """Test shorthand aliases: T.f32, T.i32, T.f16, etc."""

    # fmt: off
    @T.function
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
    @T.function
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
    @T.function
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
    from tvm_tirx import ir
    from tvm_tirx.script import ir as I

    @I.ir_module
    class Module:
        I.module_attrs({"tag": I.StringImm("label"), "type": I.StringType()})

    expected = ir.IRModule(attrs={"tag": ir.StringImm("label"), "type": ir.StringType()})
    ir.assert_structural_equal(expected, Module)
    assert Module.attrs["tag"].value == "label"
    assert isinstance(Module.attrs["type"], ir.StringType)


def test_thread_return_is_distinct_from_function_return():
    @T.function
    def thread_exit():
        T.gpu_thread_return()

    @T.function
    def function_exit():
        return 0

    assert isinstance(thread_exit.body[0], tvm_tirx.ir.Evaluate)
    assert thread_exit.body[0].value.op.name == "tirx.gpu_thread_return"
    assert isinstance(function_exit.body[0], tvm_tirx.ir.Return)
    assert isinstance(function_exit.body[0].value, tvm_tirx.ir.IntImm)
    assert function_exit.body[0].value.value == 0


def test_loop_control_validation_preserves_valid_and_unchecked_ir():
    # Invalid loop placement must be rejected, while disabled checks preserve the original IR.
    from tvm_tirx import ir, tirx

    invalid = tirx.Function(params=[], body=tvm_tirx.ir.Break())

    # Direct construction retains the native statement for an explicit verifier pass.
    ir.assert_structural_equal(invalid.body, tvm_tirx.ir.SeqStmt([tvm_tirx.ir.Break()]))
    assert not tirx.analysis.verify_well_formed(invalid, assert_mode=False)
    with pytest.raises(RuntimeError, match="requires an enclosing loop"):
        tirx.analysis.verify_well_formed(invalid)

    @T.function
    def valid():
        for i in range(2):
            break

    assert isinstance(valid.body[0], tvm_tirx.ir.For)
    ir.assert_structural_equal(valid.body[0].body, invalid.body)

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
    @T.function
    def test(A: T.Tensor((10,), 'int32')):

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
    @T.function
    def test(A: T.Tensor((10,), 'int32')):

        T.device_entry()
        i = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
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
    @T.function
    def test(A: T.Tensor((9,), 'int32')):

        T.device_entry()
        idx = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
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
    @T.function
    def test(A: T.Tensor((10,), 'int32')):

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
    @T.function
    def test(A: T.Tensor((10,), 'int32')):

        T.device_entry()
        i = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
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
    @T.function
    def test(A: T.Tensor((9,), 'int32')):

        T.device_entry()
        idx = T.alloc_tensor(ty_args=[T.Tensor((1,), "int32", scope="local")])
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
    @T.function
    def test(A: T.Tensor((10,), 'int32')):

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
    @T.function
    def test(A: T.Tensor((5,), 'int32')):

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
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        for _ in T.serial(10, unroll=False):
            T.cuda.tile.apply(T.ptx['mov.b32'], A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    assert "unroll=False" in code, f"printer should emit unroll=False, got:\n{code}"
    assert "annotations=" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_true():
    """T.serial(N, unroll=True) should round-trip as a pragma-unroll request."""

    # fmt: off
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        for _ in T.serial(10, unroll=True):
            T.cuda.tile.apply(T.ptx['mov.b32'], A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    assert "unroll=True" in code, f"printer should emit unroll=True, got:\n{code}"
    assert "annotations=" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_count():
    """T.serial(N, unroll=2) should preserve the requested unroll count."""

    # fmt: off
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        for _ in T.serial(10, unroll=2):
            T.cuda.tile.apply(T.ptx['mov.b32'], A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    assert "unroll=2" in code, f"printer should emit unroll=2, got:\n{code}"
    assert "annotations=" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_false_with_other_annotations():
    """When other annotations exist alongside disable_unroll, fall back to full dict."""

    # fmt: off
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        for _ in T.serial(10, annotations={"tirx.disable_unroll": True, "custom": 42}):
            T.cuda.tile.apply(T.ptx['mov.b32'], A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    assert "annotations=" in code, "printer should emit full annotations when multiple keys exist"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_loop_var_dtype_uint32():
    # fmt: off
    @T.function
    def func(A: T.Tensor((128,), 'float32')):

        for i in T.serial(128, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body[0]
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
    @T.function
    def func(A: T.Tensor((128,), 'float32')):

        for i in T.serial(4, 128, step=2, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body[0]
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    assert loop.step.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("for_kind", ["serial", "parallel", "vectorized", "unroll"])
def test_loop_var_dtype_uint32_all_for_kinds(for_kind):
    # fmt: off
    @T.function
    def func(A: T.Tensor((4,), 'float32')):

        for i in getattr(T, for_kind)(4, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body[0].loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_grid_loop_var_dtype_uint32():
    # fmt: off
    @T.function
    def func(A: T.Tensor((8, 16), 'float32')):

        for i, j in T.grid(8, 16, dtype="uint32"):
            A[i, j] = T.float32(1)
    # fmt: on

    outer = func.body[0]
    assert outer.loop_var.ty == PrimType("uint32")
    assert outer.body[0].loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_defaults_to_int32():
    # fmt: off
    @T.function
    def func(A: T.Tensor((128,), 'float32')):

        for i in range(128):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body[0].loop_var.ty == PrimType("int32")
    _assert_roundtrip(func)


def test_loop_var_dtype_inferred_from_unsigned_extent():
    """A uint32 extent makes the loop var uint32 without an explicit dtype."""

    # fmt: off
    @T.function
    def func(A: T.Tensor((128,), 'float32'), n: T.uint32):

        for i in range(n):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body[0].loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_casts_mismatched_bound():
    """A non-literal bound of another dtype is cast to the requested loop dtype."""

    # fmt: off
    @T.function
    def func(A: T.Tensor((128,), 'float32'), n: T.int32):

        for i in T.serial(n, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body[0]
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "int16", "float32"])
def test_loop_var_dtype_rejects_unsupported(dtype):
    with pytest.raises(Exception, match='must be "int32" or "uint32"'):
        T.serial(4, dtype=dtype)


def test_hand_built_for_promotes_int_literal_bounds_to_uint32():
    """The For constructor retypes literal bounds to the loop var's dtype."""
    loop_var = tvm_tirx.tirx.Var("i", "uint32")
    loop = tvm_tirx.ir.For(loop_var, 0, 128, tvm_tirx.ir.ForKind.DEFAULT, tvm_tirx.ir.Evaluate(0))
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")


def test_hand_built_for_rejects_negative_literal_for_uint32():
    loop_var = tvm_tirx.tirx.Var("i", "uint32")
    with pytest.raises(Exception, match="not representable"):
        tvm_tirx.ir.For(loop_var, -1, 128, tvm_tirx.ir.ForKind.DEFAULT, tvm_tirx.ir.Evaluate(0))


def _assert_equal_functions(lhs, rhs, *args, **kwargs):
    tvm_tirx.ir.assert_structural_equal(
        lhs.with_attr("global_symbol", "main"),
        rhs.with_attr("global_symbol", "main"),
        *args,
        **kwargs,
    )


@T.function
def ceildiv_test(A: T.Tensor(16, "int32")):
    for i in range(16):
        A[i] = T.ceildiv(A[i], 4)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_ceildiv():
    f = tvm_tirx.tirx.compile(ceildiv_test, "llvm")
    a = tvm_tirx.runtime.tensor(np.arange(16).astype("int32"))
    f(a)
    ref = (np.arange(16) + 3) // 4
    tvm_tirx.testing.assert_allclose(a.numpy(), ref)


def test_tir_loop_steps():
    N = T.Var("N", "int32")

    @T.function(private=True)
    def loop_with_steps(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
        tid: T.int32,
        v: T.int32,
    ):
        for i in T.serial(tid, N, step=2):
            C[i] = A[i] + B[i]
        for i in T.unroll(tid, N, step=3):
            C[i] = A[i] + B[i]
        for i in T.vectorized(tid, N, step=4):
            C[i] = A[i] + B[i]
        for i in T.parallel(tid, N, step=5):
            C[i] = A[i] + B[i]
        for i in T.serial(tid, N, step=v):
            C[i] = A[i] + B[i]

    stmts = loop_with_steps.body.seq
    assert stmts[0].step == 2
    assert stmts[1].step == 3
    assert stmts[2].step == 4
    assert stmts[3].step == 5
    assert stmts[4].step.name == "v"


def test_tir_empty_tuple_index():
    @T.inline
    def bar(val):
        T.evaluate(val)

    @T.function(private=True)
    def func_with_empty_tuple(A: T.Tensor((), "int32"), B: T.Tensor((), "int32")):
        bar(val=A[()])

    @T.function(private=True)
    def expected(A: T.Tensor((), "int32"), B: T.Tensor((), "int32")):
        T.evaluate(A[()])

    tvm_tirx.ir.assert_structural_equal(func_with_empty_tuple, expected)


def test_thread_binding_dtype():
    @T.function(private=True)
    def func(A: T.Tensor((128, 128), "float32"), B: T.Tensor((128, 128), "float32")):
        for i in T.parallel(T.int64(128), annotations={"thread_binding": "threadIdx.x"}):
            for j in T.parallel(128, annotations={"thread_binding": "threadIdx.y"}):
                B[i, j] = A[i, j]

    loop_i = func.body[0]
    loop_j = loop_i.body[0]
    assert loop_i.loop_var.ty.dtype == "int64"
    assert loop_j.loop_var.ty.dtype == "int32"


def test_inferred_ty_with_prim_args():
    """A Function may have inferred Type"""

    @T.function
    def func(M: T.int32, N: T.int32) -> T.int32:
        return M * N

    expected = tvm_tirx.ir.FuncType(
        [
            tvm_tirx.ir.PrimType("int32"),
            tvm_tirx.ir.PrimType("int32"),
        ],
        tvm_tirx.ir.PrimType("int32"),
    )
    tvm_tirx.ir.assert_structural_equal(func.ty, expected)


def test_inferred_ty_with_tensor_args():
    """Function tensor arguments retain native tensor types."""

    @T.function
    def func(A: T.Tensor([16, 16], "float32"), B: T.Tensor([256], "int32")) -> T.float32:
        return T.float32(42.0)

    expected = tvm_tirx.ir.FuncType(
        [
            tvm_tirx.tirx.Var(
                "tensor", tvm_tirx.tirx.TensorType([16, 16], "float32", layout=None)
            ).ty,
            tvm_tirx.tirx.Var("tensor", tvm_tirx.tirx.TensorType([256], "int32", layout=None)).ty,
        ],
        tvm_tirx.ir.PrimType("float32"),
    )
    tvm_tirx.ir.assert_structural_equal(func.ty, expected)


def test_inferred_ty_with_internal_allocation():
    """Internal allocations do not change the native signature."""

    @T.function
    def func(A: T.Tensor([16, 16], "float32")) -> T.float32:
        Sum = T.alloc_tensor(ty_args=[T.Tensor([], "float32")])
        Sum[()] = 0.0
        for i, j in T.grid(16, 16):
            Sum[()] = Sum[()] + A[i, j]

        return Sum[()]

    expected = tvm_tirx.ir.FuncType(
        [
            tvm_tirx.tirx.Var(
                "tensor", tvm_tirx.tirx.TensorType([16, 16], "float32", layout=None)
            ).ty,
        ],
        tvm_tirx.ir.PrimType("float32"),
    )
    tvm_tirx.ir.assert_structural_equal(func.ty, expected)


def test_inferred_ty_with_output_tensor():
    """Output tensors remain native parameters with a void result."""

    @T.function
    def func(A: T.Tensor(16, "float32"), B: T.Tensor(16, "float32")):
        for i in range(16):
            B[i] = A[i]

    expected = tvm_tirx.ir.FuncType(
        [
            tvm_tirx.tirx.Var("tensor", tvm_tirx.tirx.TensorType([16], "float32", layout=None)).ty,
            tvm_tirx.tirx.Var("tensor", tvm_tirx.tirx.TensorType([16], "float32", layout=None)).ty,
        ],
        tvm_tirx.ir.TupleType([]),
    )
    tvm_tirx.ir.assert_structural_equal(func.ty, expected)


def test_launch_thread_i64():
    """Test launching thread with int64"""

    @T.function
    def func() -> None:
        blockIdx_x = T.launch_thread("blockIdx.x", T.int64(1))
        if blockIdx_x == T.int64(0):
            T.evaluate(T.int64(0))
        else:
            T.evaluate(T.int64(1))

    assert func.body[0].body_params[0].ty.dtype == "int64"
    assert func.body[0].args[1].ty.dtype == "int64"


def test_ifexp():
    @T.function(private=True)
    def func(A: T.Tensor((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            A[i, j] = i if i < j else j

    @T.function(private=True)
    def expected(A: T.Tensor((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            A[i, j] = T.if_then_else(i < j, i, j)

    tvm_tirx.ir.assert_structural_equal(func, expected)


def test_sequence_compare():
    @T.function(private=True)
    def tir_func(A: T.Tensor((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            if 0 < i < 128 and 0 < j < 128:
                A[i, j] = 1
            else:
                A[i, j] = 0

    @T.function(private=True)
    def expected(A: T.Tensor((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            if (0 < i and i < 128) and (0 < j and j < 128):
                A[i, j] = 1
            else:
                A[i, j] = 0

    tvm_tirx.ir.assert_structural_equal(tir_func, expected)


def launch_env_thread():
    @T.function
    def main(inputs: T.Tensor((64, 2, 4), "float32")) -> None:
        bx = T.launch_thread("blockIdx.x", 64)
        for i, j in T.grid(2, 4):
            T.evaluate(inputs[bx, i, j])

    return main


def vthread_func():
    @T.function
    def vthread_func(A: T.Tensor([256], "float32"), C: T.Tensor([256], "float32")) -> None:
        i0 = T.launch_thread("blockIdx.x", 4)
        i1 = T.launch_thread("threadIdx.x", 2)
        i2 = T.launch_thread("vthread", 2)
        B = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32", scope="local")])
        for j in range(16):
            B[j] = A[i0 * 64 + i1 * 32 + i2 * 16 + j] + T.float32(1)
        for j in range(16):
            C[i0 * 64 + i1 * 32 + i2 * 16 + j] = B[j] * T.float32(2)

    return vthread_func


def for_thread_binding():
    @T.function
    def for_thread_binding(
        A: T.Tensor((16, 16), "float32"), B: T.Tensor((16, 16), "float32")
    ) -> None:
        for i in T.parallel(0, 16, annotations={"thread_binding": "threadIdx.x"}):
            for j in T.parallel(
                0, 16, annotations={"thread_binding": "threadIdx.y", "attr_key": "attr_value"}
            ):
                A[i, j] = B[i, j] + T.float32(1)

    return for_thread_binding


def test_for_thread_binding():
    func = for_thread_binding()
    rt_func = tvm_tirx.script.from_source(
        func.script(),
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, rt_func)

    assert isinstance(rt_func.body[0], tvm_tirx.ir.For)
    assert rt_func.body[0].kind == tvm_tirx.ir.ForKind.PARALLEL
    assert rt_func.body[0].annotations.get("thread_binding") == "threadIdx.x"
    assert isinstance(rt_func.body[0].body[0], tvm_tirx.ir.For)
    assert rt_func.body[0].body[0].kind == tvm_tirx.ir.ForKind.PARALLEL
    assert rt_func.body[0].body[0].annotations.get("thread_binding") == "threadIdx.y"
    assert rt_func.body[0].body[0].annotations["attr_key"] == "attr_value"


def boolean_argument():
    @T.function
    def func(a: T.boolean) -> None:
        T.evaluate(a)

    return func


def bool_argument():
    @T.function
    def func(a: T.bool) -> None:
        T.evaluate(a)

    return func


def bool_variable_annotation():
    @T.function
    def func() -> None:
        a: T.let[T.bool] = T.call_extern("dummy", ty="bool")
        T.evaluate(0)

    return func


def return_none():
    @T.function
    def func():
        T.evaluate(0)

    return func


def implicit_evaluate():
    @T.function
    def func(A: T.Tensor(1, "int32")):
        T.evaluate(T.assume(A[0] == 5))
        A[0] = 10

    return func


def if_true_else():
    @T.function
    def func() -> None:
        if True:
            T.evaluate(0)
        else:
            T.evaluate(1)

    return func


def elif_chain_without_else():
    @T.function
    def func(i: T.int32) -> None:
        if i == 0:
            T.evaluate(0)
        elif i == 1:
            T.evaluate(1)
        elif i == 2:
            T.evaluate(2)

    return func


def elif_chain_with_else():
    @T.function
    def func(i: T.int32) -> None:
        if i == 0:
            T.evaluate(0)
        elif i == 1:
            T.evaluate(1)
        elif i == 2:
            T.evaluate(2)
        else:
            T.evaluate(3)

    return func


def bind_var():
    @T.function
    def func():
        x = T.bind(0)
        y = T.bind(0)
        T.evaluate(0)
        T.evaluate(0)

    return func


def if_then_else_var():
    @T.function
    def main(n: T.int32):
        if n == 0:
            x = 5
            T.evaluate(x)
        else:
            x = 10
            T.evaluate(x)

    return main


def ir_module_with_attrs():
    @I.ir_module
    class Module:
        I.module_attrs({"attr": 10})

        @T.function
        def tir_func(A: T.Tensor(16, "int32"), B: T.Tensor(16, "int32")):
            for i in range(16):
                B[i] = A[i]

    return Module


def subroutine_call():
    """A GlobalVar may reference other functions in the module"""

    @I.ir_module
    class mod:
        @T.function
        def main(A: T.Tensor(16, "float32")):
            mod.subroutine(A.data, T.int32(16))

        @T.function
        def subroutine(A_data: T.handle("float32"), n: T.int32):
            T.evaluate(0)

    return mod


def subroutine_call_returning_int():
    """An internal function call may return non-void"""

    @I.ir_module
    class mod:
        @T.function
        def main(A: T.Tensor(2, "float32")):
            mod.subroutine(A[0]) + mod.subroutine(A[1])

        @T.function
        def subroutine(x: T.float32) -> T.float32:
            return x * x

    return mod


def subroutine_call_without_arguments():
    @I.ir_module
    class mod:
        @T.function
        def main():
            mod.subroutine()

        @T.function
        def subroutine():
            T.evaluate(0)

    return mod


def return_zero():
    @T.function
    def func() -> T.int32:
        return 0

    return func


def return_zero_private():
    @T.function(private=True)
    def func() -> T.int32:
        return 0

    return func


def return_zero_private_with_attr():
    @T.function(private=True)
    def func() -> T.int32:
        T.func_attr({"greeting": "hello"})
        return 0

    return func


def func_with_loop_jumps():
    @T.function
    def func(In: T.Tensor((1,), "int32"), Out: T.Tensor((2,), "int32")):
        Out[0] = 0
        Out[1] = 0
        for i in range(1000):
            if i % 13 == 0:
                Out[1] = Out[1] + 1
                continue
            Out[0] = Out[0] + 1
            if Out[0] >= In[0]:
                break

    return func


def func_with_loop_steps():
    @T.function
    def func(
        A: T.Tensor((1024,), "float32"),
        B: T.Tensor((1024,), "float32"),
        C: T.Tensor((1024,), "float32"),
        tid: T.int32,
        v: T.int32,
    ):
        for i in T.serial(tid, 1024, step=2):
            C[i] = A[i] + B[i]
        for i in T.unroll(tid, 1024, step=3):
            C[i] = A[i] + B[i]
        for i in T.vectorized(tid, 1024, step=4):
            C[i] = A[i] + B[i]
        for i in T.parallel(tid, 1024, step=5):
            C[i] = A[i] + B[i]
        for i in range(tid, 1024, 6):
            C[i] = A[i] + B[i]

    return func


def test_return_none_no_trailing_type():
    func = return_none()
    script = func.script()
    assert "-> None" not in script


@pytest.mark.parametrize(
    "error_type,message_parts",
    [
        ("RuntimeError", ["x must be positive"]),
        ("ValueError", ["Shape mismatch"]),
        ("TypeError", ["Expected Tensor but got int"]),
        ("TypeError", ["Expected ", "Tensor", " but got ", "int"]),
    ],
    ids=["runtime_error", "value_error", "type_error", "multi_parts"],
)
def test_assert_stmt_roundtrip(error_type, message_parts):
    """Exception types and split messages survive a TIRx script roundtrip."""

    @T.function
    def func(x: T.int32):
        assert x > 0, (error_type, message_parts)

    roundtrip = tvm_tirx.script.from_source(
        func.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(func, roundtrip, map_free_vars=True)


@T.function
def loop_no_syntax_sugar(A: T.Tensor((128, 128, 128, 128), "float32")) -> None:
    for i in T.serial(0, 128):
        for j in T.parallel(0, 128):
            for k in T.vectorized(0, 128):
                for x in T.unroll(0, 128):
                    for y in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
                        for z in T.parallel(0, 128, annotations={"thread_binding": "threadIdx.x"}):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


@T.function
def loop_syntax_sugar(A: T.Tensor((128, 128, 128, 128), "float32")) -> None:
    for i in T.serial(128):
        for j in T.parallel(128):
            for k in T.vectorized(128):
                for x in T.unroll(128):
                    for y in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                        for z in T.parallel(128, annotations={"thread_binding": "threadIdx.x"}):
                            A[i, j, k, x] = A[i, j, k, x] * 2.0


def test_loop_syntax_sugar():
    _assert_equal_functions(loop_no_syntax_sugar, loop_syntax_sugar)


def test_tensor_1d():
    @T.function
    def func_no_sugar(A: T.Tensor(shape=(16,), dtype="float32")):
        for i in T.serial(16):
            A[i] = 0.0

    @T.function
    def func_with_sugar(A: T.Tensor(16, "float32")):
        for i in T.serial(16):
            A[i] = 0.0

    _assert_equal_functions(func_no_sugar, func_with_sugar)


def test_bind_tensorload_without_type_annotation():
    # Variable assignment of Expr types uses the dtype of the
    # Expr to determine the variable's dtype.  Parsing of
    # buf[indices] is done by generating a TensorSlice object, which
    # handles both store and load cases.  TensorSlice is not a
    # Expr, and implements TensorSlice.dtype explicitly.

    # Failure occurred during parsing of the tvmscript.
    @T.function
    def func_without_type_annotation(A: T.Tensor((1,), "int32")):
        x = A[0]
        T.evaluate(x)


def test_implicit_evaluate_assume():
    @T.function
    def explicit(A: T.Tensor(1, "int32")):
        T.evaluate(T.assume(A[0] == 5))
        A[0] = 10

    @T.function
    def implicit(A: T.Tensor(1, "int32")):
        T.assume(A[0] == 5)
        A[0] = 10

    _assert_equal_functions(implicit, explicit)


def test_implicit_evaluate_call_extern():
    @T.function
    def explicit(A: T.Tensor(1, "int32")):
        T.evaluate(T.call_extern("extern_func", A.data, ty="int32"))

    @T.function
    def implicit(A: T.Tensor(1, "int32")):
        T.call_extern("extern_func", A.data, ty="int32")

    _assert_equal_functions(implicit, explicit)


def test_preserve_trivial_let_binding():
    """Trivial `T.let[...]` annotations survive the parser as LetStmt and are not inlined.

    In fork, bare `j = i` lowers to a local_scalar (AllocTensor + TensorStore); the
    LetStmt form is opt-in via `T.let[T.dtype]`. Both the explicit `T.bind(..., var=j)`
    builder API and the `j: T.let[T.dtype]` annotation produce the same LetStmt IR.
    """

    j = T.dynamic("j", "int32")

    @T.function
    def explicit(i: T.int32):
        T.bind(i, var=j)
        T.evaluate(j)

    @T.function
    def implicit(i: T.int32):
        j: T.let[T.int32] = i
        T.evaluate(j)

    _assert_equal_functions(implicit, explicit)


def test_preserve_trivial_let_binding_of_value():
    """Same as test_preserve_trivial_let_binding but with a constant RHS."""

    j = T.dynamic("j", "int32")

    @T.function
    def explicit(i: T.int32):
        T.bind(42, var=j)
        T.evaluate(j)

    @T.function
    def implicit(i: T.int32):
        j: T.let[T.int32] = 42
        T.evaluate(j)

    _assert_equal_functions(implicit, explicit)


def test_preserve_parameter_name():
    @T.function
    def func(i: T.int32):
        j = i
        T.evaluate(j)

    param_name = func.params[0].name
    assert param_name == "i"


@pytest.mark.parametrize("mutable", [False, True])
def test_preserve_variable_name(mutable):
    """Use variable name when generating tirx::Bind / AllocTensor"""

    # Bare bindings name the immutable Var; explicit declarations name scalar storage.
    annotation = ": T.int32" if mutable else ""
    func = tvm_tirx.script.from_source(
        f"""@T.function
def func():
    for i in T.serial(16):
        j{annotation} = i // 4
        T.evaluate(j)
""",
        extra_vars={"I": tvm_tirx.script.ir, "T": tvm_tirx.script.tirx},
    )
    binding = func.body[0].body.seq[0]
    if mutable:
        assert isinstance(binding, tvm_tirx.ir.Bind) and isinstance(binding.value, tvm_tirx.ir.Call)
        assert binding.value.op.name == "tirx.alloc_tensor"
        var_name = binding.var.name
    else:
        assert isinstance(binding, tvm_tirx.ir.Bind)
        var_name = binding.var.name
    assert var_name == "j"


def test_boolean_constant():
    """Python booleans should become T.Bool objects"""

    @T.function
    def explicit():
        T.evaluate(T.bool(True))

    @T.function
    def implicit():
        T.evaluate(True)

    _assert_equal_functions(implicit, explicit)


def test_foldable_boolean_in_assert():
    """Foldable booleans T.Bool objects

    The condition of an assert statement should be a boolean
    expression.  Previously, this test failed because the FFI does not
    distinguish between integer primitives and boolean primitives.
    """

    @T.function
    def explicit():
        assert T.bool(False), "Message"
        T.evaluate(0)

    @T.function
    def implicit():
        assert 0 == 1, "Message"
        T.evaluate(0)

    _assert_equal_functions(implicit, explicit)


def test_return_statement():
    """A Python `return` statement creates a first-class Return node."""

    @T.function
    def explicit():
        T.return_(T.int32(5))

    @T.function
    def implicit():
        return 5

    _assert_equal_functions(implicit, explicit)


def test_loop_jump_statement():
    """`break` and `continue` emit native TIR statements."""

    @T.function
    def implicit():
        for i in range(16):
            if i % 2 == 0:
                continue
            if i < 15:
                break

    jumps = []
    tvm_ffi.structural_walk(
        implicit.body,
        lambda node: jumps.append(node)
        if isinstance(node, tvm_tirx.ir.Break | tvm_tirx.ir.Continue)
        else None,
    )
    assert sum(isinstance(node, tvm_tirx.ir.Break) for node in jumps) == 1
    assert sum(isinstance(node, tvm_tirx.ir.Continue) for node in jumps) == 1


@pytest.mark.parametrize(
    "ir_generator",
    [
        launch_env_thread,
        vthread_func,
        boolean_argument,
        bool_argument,
        bool_variable_annotation,
        return_none,
        implicit_evaluate,
        if_true_else,
        elif_chain_without_else,
        elif_chain_with_else,
        bind_var,
        if_then_else_var,
        ir_module_with_attrs,
        subroutine_call,
        subroutine_call_returning_int,
        subroutine_call_without_arguments,
        return_zero,
        return_zero_private,
        return_zero_private_with_attr,
        func_with_loop_jumps,
        func_with_loop_steps,
    ],
    ids=lambda factory: factory.__name__,
)
def test_roundtrip_basic_usage(ir_generator):
    original = ir_generator()
    after_roundtrip = tvm_tirx.script.from_source(
        original.script(show_meta=True),
        check_well_formed=False,
        extra_vars={
            "I": tvm_tirx.script.ir,
            "T": tvm_tirx.script.tirx,
        },
    )
    tvm_tirx.ir.assert_structural_equal(original, after_roundtrip, map_free_vars=True)
