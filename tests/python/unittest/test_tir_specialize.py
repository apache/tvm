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

import tvm
from tvm.script import tir as T


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle, n: T.int32) -> None:
    m = T.var("int32")
    A = T.match_buffer(a, [m, n])
    B = T.match_buffer(b, [m, n])
    C = T.match_buffer(c, [m, m])

    with T.block([m, m, T.reduce_axis(0, n)], "update") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_128(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block([128, 128, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_m_128(a: T.handle, b: T.handle, c: T.handle) -> None:
    m = T.var("int32")
    A = T.match_buffer(a, [m, 128])
    B = T.match_buffer(b, [m, 128])
    C = T.match_buffer(c, [m, m])

    with T.block([m, m, T.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_m_8x(a: T.handle, b: T.handle, c: T.handle) -> None:
    x = T.var("int32")
    m = T.var("int32")
    A = T.match_buffer(a, [m, x * 8])
    B = T.match_buffer(b, [m, x * 8])
    C = T.match_buffer(c, [m, m])

    with T.block([m, m, T.reduce_axis(0, x * 8)], "update") as [vi, vj, vk]:
        with T.init():
            C[vi, vj] = 0.0
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def element_wise(a: T.handle, c: T.handle) -> None:
    m = T.var("int32")
    n = T.var("int32")
    A = T.match_buffer(a, (m, n), "float32")
    C = T.match_buffer(c, (m, n), "float32")

    B = T.alloc_buffer((m, n), "float32")

    with T.block([m, n], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with T.block([m, n], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_128_64(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 64), "float32")
    C = T.match_buffer(c, (128, 64), "float32")
    B = T.alloc_buffer((128, 64), "float32")

    with T.block([128, 64], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with T.block([128, 64], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def element_wise_128_n(a: T.handle, c: T.handle) -> None:
    n = T.var("int32")
    A = T.match_buffer(a, (128, n), "float32")
    C = T.match_buffer(c, (128, n), "float32")
    B = T.alloc_buffer((128, n), "float32")

    with T.block([128, n], "B") as [vi, vj]:
        B[vi, vj] = A[vi, vj] * 2.0

    with T.block([128, n], "C") as [vi, vj]:
        C[vi, vj] = B[vi, vj] + 1.0


@T.prim_func
def mem_copy(a: T.handle, b: T.handle, m: T.int32, n: T.int32, p: T.int32, q: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float32", strides=[p, 1], elem_offset=q)
    B = T.match_buffer(b, (m, n), "float32", strides=[p, 1], elem_offset=q)

    with T.block([m, n], "") as [vi, vj]:
        B[vi, vj] = A[vi, vj]


@T.prim_func
def mem_copy_16_16_8_4(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", strides=[8, 1], elem_offset=4)
    B = T.match_buffer(b, (16, 16), "float32", strides=[8, 1], elem_offset=4)

    with T.block([16, 16], "") as [vi, vj]:
        B[vi, vj] = A[vi, vj]


@T.prim_func
def mem_copy_m_n_p_n(a: T.handle, b: T.handle, m: T.int32, n: T.int32, p: T.int32) -> None:
    A = T.match_buffer(a, (m, n), "float32", strides=[p, 1], elem_offset=n)
    B = T.match_buffer(b, (m, n), "float32", strides=[p, 1], elem_offset=n)

    with T.block([m, n], "") as [vi, vj]:
        B[vi, vj] = A[vi, vj]


@T.prim_func
def param_in_arith_exprs(a: T.handle, b: T.handle) -> None:
    n = T.var("int32")
    A = T.match_buffer(a, [n // 8, 8], "int32")
    B = T.match_buffer(b, [n], "int32")
    with T.block([n - 1], "") as [vi]:
        B[vi] = A[vi // 8, vi % 8] + (n + 1) * 42


@T.prim_func
def param_in_arith_exprs_n_16(a: T.handle, b: T.handle) -> None:
    n = T.var("int32")
    A = T.match_buffer(a, [2, 8], "int32")
    B = T.match_buffer(b, [16], "int32")
    with T.block([15], "") as [vi]:
        B[vi] = A[vi // 8, vi % 8] + 714


def test_specialize_nothing():
    func = matmul.specialize({})
    assert func.same_as(matmul)  # Pointer the same


def test_specialize_matmul():
    a, _, _, n = matmul.params
    # fully specialized
    func = matmul.specialize({a: tvm.tir.decl_buffer((128, 128))})
    tvm.ir.assert_structural_equal(func, matmul_128)
    # partially specialized
    func = matmul.specialize({n: 128})
    tvm.ir.assert_structural_equal(func, matmul_m_128)
    # symbolic specialized
    func = matmul.specialize({n: tvm.tir.Var("x", "int32") * 8})
    tvm.ir.assert_structural_equal(func, matmul_m_8x)


def test_specialize_elemwise():
    a, c = element_wise.params
    C = element_wise.buffer_map[c]
    # fully specialized
    func = element_wise.specialize({a: tvm.tir.decl_buffer((128, 64))})
    tvm.ir.assert_structural_equal(func, element_wise_128_64)
    # partially specialized
    func = element_wise.specialize({c: tvm.tir.decl_buffer((128, C.shape[1]))})
    tvm.ir.assert_structural_equal(func, element_wise_128_n)


def test_specialize_mem_copy():
    a, _, m, n, p, q = mem_copy.params
    # fully specialized
    func = mem_copy.specialize({a: tvm.tir.decl_buffer((16, 16), strides=[8, 1], elem_offset=4)})
    tvm.ir.assert_structural_equal(func, mem_copy_16_16_8_4)
    func = mem_copy.specialize({n: 16, m: 16, p: 8, q: 4})
    tvm.ir.assert_structural_equal(func, mem_copy_16_16_8_4)
    # partially specialized
    func = mem_copy.specialize({q: n})
    tvm.ir.assert_structural_equal(func, mem_copy_m_n_p_n)


def test_specialize_recursive_load():
    # TODO(Siyuan): add recursive Load testcase, e.g. A[C[i]]
    pass


def test_specialize_with_const_folding():
    b = param_in_arith_exprs.params[1]
    func = param_in_arith_exprs.specialize({b: tvm.tir.decl_buffer([16])})
    tvm.ir.assert_structural_equal(func, param_in_arith_exprs_n_16)


if __name__ == "__main__":
    test_specialize_nothing()
    test_specialize_matmul()
    test_specialize_elemwise()
    test_specialize_mem_copy()
    test_specialize_recursive_load()
    test_specialize_with_const_folding()
