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
# pylint: disable=missing-function-docstring, missing-module-docstring

import pytest

import tvm
from tvm.script import tir as T
from tvm.tir.schedule.testing import assert_structural_equal_ignore_global_symbol


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, n: T.int32) -> None:
    m = T.int32()
    A = T.match_buffer(a, [m, n])
    B = T.match_buffer(b, [m, n])
    C = T.match_buffer(c, [m, m])

    for i, j, k in T.grid(m, m, n):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_128(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_m_128(a: T.handle, b: T.handle, c: T.handle) -> None:
    m = T.int32()
    A = T.match_buffer(a, [m, 128])
    B = T.match_buffer(b, [m, 128])
    C = T.match_buffer(c, [m, m])

    for i, j, k in T.grid(m, m, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


# x is considered undefined because it appears as part of x*8,
# but not on its own
@T.prim_func(check_well_formed=False)
def matmul_m_8x(a: T.handle, b: T.handle, c: T.handle) -> None:
    x = T.int32()
    m = T.int32()
    A = T.match_buffer(a, [m, x * 8])
    B = T.match_buffer(b, [m, x * 8])
    C = T.match_buffer(c, [m, m])

    for i, j, k in T.grid(m, m, x * 8):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def element_wise(a: T.handle, c: T.handle) -> None:
    m = T.int32()
    n = T.int32()
    A = T.match_buffer(a, (m, n), "float32")
    C = T.match_buffer(c, (m, n), "float32")

    B = T.alloc_buffer((m, n), "float32")

    for i, j in T.grid(m, n):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

    for i, j in T.grid(m, n):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_128_64(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 64), "float32")
    C = T.match_buffer(c, (128, 64), "float32")
    B = T.alloc_buffer((128, 64), "float32")

    for i, j in T.grid(128, 64):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

    for i, j in T.grid(128, 64):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_128_n(a: T.handle, c: T.handle) -> None:
    n = T.int32()
    A = T.match_buffer(a, (128, n), "float32")
    C = T.match_buffer(c, (128, n), "float32")
    B = T.alloc_buffer((128, n), "float32")

    for i, j in T.grid(128, n):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0

    for i, j in T.grid(128, n):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32, p: T.int32, q: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float32", strides=[p, 1], elem_offset=q)
    B = T.match_buffer(b, (m, n), "float32", strides=[p, 1], elem_offset=q)

    for i, j in T.grid(m, n):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


@T.prim_func
def mem_copy_16_16_8_4(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", strides=[8, 1], elem_offset=4)
    B = T.match_buffer(b, (16, 16), "float32", strides=[8, 1], elem_offset=4)

    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


@T.prim_func
def mem_copy_m_n_p_n(a: T.handle, b: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float32", strides=[p, 1], elem_offset=n)
    B = T.match_buffer(b, (m, n), "float32", strides=[p, 1], elem_offset=n)

    for i, j in T.grid(m, n):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj]


def test_specialize_nothing():
    func = matmul.specialize({})
    assert func.same_as(matmul)  # Pointer the same


def test_specialize_matmul():
    a, _, _, n = matmul.params
    # fully specialized
    func = matmul.specialize({a: tvm.tir.decl_buffer((128, 128))})
    assert_structural_equal_ignore_global_symbol(func, matmul_128)
    # partially specialized
    func = matmul.specialize({n: 128})
    assert_structural_equal_ignore_global_symbol(func, matmul_m_128)
    # symbolic specialized
    func = matmul.specialize({n: tvm.tir.Var("x", "int32") * 8})
    assert_structural_equal_ignore_global_symbol(func, matmul_m_8x)


def test_specialize_elemwise():
    a, c = element_wise.params
    C = element_wise.buffer_map[c]
    # fully specialized
    func = element_wise.specialize({a: tvm.tir.decl_buffer((128, 64))})
    assert_structural_equal_ignore_global_symbol(func, element_wise_128_64)
    # partially specialized
    func = element_wise.specialize({c: tvm.tir.decl_buffer((128, C.shape[1]))})
    assert_structural_equal_ignore_global_symbol(func, element_wise_128_n)


def test_specialize_mem_copy():
    a, _, m, n, p, q = mem_copy.params
    # fully specialized
    func = mem_copy.specialize({a: tvm.tir.decl_buffer((16, 16), strides=[8, 1], elem_offset=4)})
    assert_structural_equal_ignore_global_symbol(func, mem_copy_16_16_8_4)
    func = mem_copy.specialize({n: 16, m: 16, p: 8, q: 4})
    assert_structural_equal_ignore_global_symbol(func, mem_copy_16_16_8_4)
    # partially specialized
    func = mem_copy.specialize({q: n})
    assert_structural_equal_ignore_global_symbol(func, mem_copy_m_n_p_n)


def test_specialize_recursive_load():
    # TODO(Siyuan): add recursive Load testcase, e.g. A[C[i]]
    pass


def test_specialize_with_const_folding():
    @T.prim_func
    def before(a: T.handle, b: T.handle):
        n = T.int32()
        A = T.match_buffer(a, [n // 8, 8], "int32")
        B = T.match_buffer(b, [n], "int32")
        for i in range(n - 1):
            with T.block():
                vi = T.axis.S(n - 1, i)
                B[vi] = A[vi // 8, vi % 8] + (n + 1) * 42

    @T.prim_func
    def expected(a: T.handle, b: T.handle):
        A = T.match_buffer(a, [2, 8], "int32")
        B = T.match_buffer(b, [16], "int32")
        for i in range(15):
            with T.block():
                vi = T.axis.S(15, i)
                B[vi] = A[vi // 8, vi % 8] + 714

    b = before.params[1]
    after = before.specialize({b: tvm.tir.decl_buffer([16], dtype="int32")})
    assert_structural_equal_ignore_global_symbol(expected, after)


def test_specialize_decl_buffer():
    """Buffers occurring in a DeclBuffer statement should be updated"""

    @T.prim_func(private=True)
    def before(A_data: T.handle("float32"), A_size: T.int32):
        A_buf = T.decl_buffer(A_size, "float32", data=A_data)
        for i in range(A_size):
            A_buf[i] = A_buf[i] * 2.0

    @T.prim_func(private=True)
    def expected(A_data: T.handle("float32")):
        A_buf = T.decl_buffer(16, "float32", data=A_data)
        for i in range(16):
            A_buf[i] = A_buf[i] * 2.0

    param_map = {before.params[1]: T.int32(16)}
    after = before.specialize(param_map)

    tvm.ir.assert_structural_equal(expected, after)


def test_specialize_buffer_var_to_var():
    """A buffer var may be remapped by specialization

    If a buffer variable is replaced by a specialization, then other
    buffers using the same buffer var should also be updated.
    """

    @T.prim_func(private=True)
    def before(A: T.Buffer([16, 16], "float32"), B: T.Buffer([16, 16], "float32")):
        A_flat = T.decl_buffer([256], "float32", data=A.data)
        B_flat = T.decl_buffer([256], "float32", data=B.data)
        for i in range(256):
            B_flat[i] = A_flat[i] * 2.0

    # well-formed checker complains about multiple nested definitions of B_flat
    # since it appears in the buffer map twice
    @T.prim_func(private=True, check_well_formed=False)
    def expected(A: T.Buffer([16, 16], "float32"), B_handle: T.handle):
        B = T.match_buffer(B_handle, [16, 16], "float32", data=A.data)
        A_flat = T.decl_buffer([256], "float32", data=A.data)
        B_flat = T.decl_buffer([256], "float32", data=A.data)
        for i in range(256):
            B_flat[i] = A_flat[i] * 2.0

    A = before.buffer_map[before.params[0]]
    B_handle = before.params[1]
    param_map = {B_handle: A}
    after = before.specialize(param_map)

    tvm.ir.assert_structural_equal(expected, after)


def test_specialize_buffer_var_to_expr():
    """Handle specialization of buffer var

    The `tir::Buffer::data` field must be an explicit `tir::Var`, and
    cannot be replaced with a `tir::PrimExpr` of type
    `DataType::Handle()`.  However, these substitutions are useful
    when lowering.  If these occur, a binding of the `tir::Var` is
    included in the specialized function.
    """

    @T.prim_func(private=True)
    def before(A_data: T.handle("float32"), B_data: T.handle("float32")):
        A_buf = T.decl_buffer(32, "float32", data=A_data)
        B_buf = T.decl_buffer(16, "float32", data=B_data)
        for i in range(16):
            B_buf[i] = A_buf[i] * 2.0

    @T.prim_func(private=True)
    def expected(A_data: T.handle("float32")):
        A_buf = T.decl_buffer(32, "float32", data=A_data)
        B_data: T.Ptr[T.float32] = T.address_of(A_buf[16])
        B_buf = T.decl_buffer(16, "float32", data=B_data)
        for i in range(16):
            B_buf[i] = A_buf[i] * 2.0

    B_data = before.params[1]
    A_buf = before.body.buffer
    param_map = {B_data: tvm.tir.address_of(A_buf[16])}
    after = before.specialize(param_map)

    tvm.ir.assert_structural_equal(expected, after)


def test_specialization_updates_struct_info():
    """Update struct info in specialization

    A PrimFunc may have a `relax.StructInfo`.  If that PrimFunc is
    specialized, the struct info should be updated.
    """

    @T.prim_func(private=True)
    def before(n: T.int32) -> T.int32:
        T.ret(n * 10)

    @T.prim_func(private=True)
    def expected() -> T.int32:
        T.ret(50)

    sinfo_before = tvm.relax.FuncStructInfo(
        [tvm.relax.PrimStructInfo("int32")], tvm.relax.PrimStructInfo("int32")
    )
    tvm.ir.assert_structural_equal(before.struct_info, sinfo_before)

    sinfo_expected = tvm.relax.FuncStructInfo([], tvm.relax.PrimStructInfo("int32"))
    tvm.ir.assert_structural_equal(expected.struct_info, sinfo_expected)

    n = before.params[0]
    param_map = {n: 5}
    after = before.specialize(param_map)

    tvm.ir.assert_structural_equal(after, expected)
    tvm.ir.assert_structural_equal(after.struct_info, sinfo_expected)


if __name__ == "__main__":
    tvm.testing.main()
