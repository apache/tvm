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

import tvm
from tvm import tir
from tvm.script import ty


def _check(original, transformed):
    mod = tvm.IRModule.from_expr(original)
    mod = tvm.tir.transform.LowerMatchBuffer()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed)


def _check_fail(original):
    mod = tvm.IRModule.from_expr(original)
    with pytest.raises(tvm.TVMError):
        mod = tvm.tir.transform.LowerMatchBuffer()(mod)


@tvm.script.tir
def buffer_load_store(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16))
    C = tir.match_buffer(c, (16, 16))
    for i, j, k in tir.grid(4, 16, 8):
        with tir.block([]):
            tir.reads(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            tir.writes(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            sub_A = tir.match_buffer(
                A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2], (4, 1, 2), offset_factor=1
            )
            sub_C = tir.match_buffer(
                C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2], (4, 2), offset_factor=1
            )
            for ii, kk in tir.grid(4, 2):
                sub_A[ii, 0, kk] += sub_C[ii, kk]


@tvm.script.tir
def transformed_buffer_load_store(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16, 16))
    C = tir.match_buffer(c, (16, 16))
    for i, j, k in tir.grid(4, 16, 8):
        with tir.block([]):
            tir.reads(C[i * 4 : i * 4 + 4, k * 2 : k * 2 + 2])
            tir.writes(A[i * 4 : i * 4 + 4, j, k * 2 : k * 2 + 2])
            for ii, kk in tir.grid(4, 2):
                A[i * 4 + ii, j, k * 2 + kk] += C[i * 4 + ii, k * 2 + kk]


@tvm.ir.register_op_attr("tir.intrin_test", "")
def intrin_test(data, elem_offset, stride_0, stride_1, shape_0, shape_1):
    return 0


@tvm.script.tir
def opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (32, 64, 128))
    B = tir.match_buffer(b, (64, 64, 64))
    for i, j, k in tir.grid(2, 64, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            sub_A = tir.match_buffer(
                A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16],
                (16, 1, 16),
                strides=[8192, 128, 1],
                offset_factor=1,
            )
            tir.evaluate(
                tir.intrin_test(
                    sub_A.data,
                    sub_A.elem_offset,
                    sub_A.strides[0],
                    sub_A.strides[1],
                    sub_A.shape[0],
                    sub_A.shape[1],
                    dtype="handle",
                )
            )
    for i, j, k in tir.grid(64, 2, 8):
        with tir.block([]):
            Bs_0 = tir.var("int32")
            Bs_1 = tir.var("int32")
            tir.reads([])
            tir.writes(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            sub_B = tir.match_buffer(
                B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8],
                (32, 8),
                strides=[Bs_0, Bs_1],
                offset_factor=1,
            )
            tir.evaluate(
                tir.intrin_test(
                    sub_B.data,
                    sub_B.elem_offset,
                    sub_B.strides[0],
                    sub_B.strides[1],
                    sub_B.shape[0],
                    sub_B.shape[1],
                    dtype="handle",
                )
            )


@tvm.script.tir
def transformed_opaque_access(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (32, 64, 128))
    B = tir.match_buffer(b, (64, 64, 64))
    for i, j, k in tir.grid(2, 64, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(A[i * 16 : i * 16 + 16, j, k * 16 : k * 16 + 16])
            tir.evaluate(
                tir.intrin_test(
                    A.data,
                    i * 131072 + j * 128 + k * 16,
                    8192,
                    128,
                    16,
                    1,
                    dtype="handle",
                )
            )
    for i, j, k in tir.grid(64, 2, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(B[i, j * 32 : j * 32 + 32, k * 8 : k * 8 + 8])
            tir.evaluate(
                tir.intrin_test(
                    B.data,
                    i * 4096 + j * 2048 + k * 8,
                    64,
                    1,
                    32,
                    8,
                    dtype="handle",
                )
            )


@tvm.script.tir
def high_dim_opaque_access(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 32, 64))
    for i, j, k in tir.grid(16, 2, 4):
        with tir.block([]):
            As_0 = tir.var("int32")
            As_1 = tir.var("int32")
            tir.reads([])
            tir.writes(A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
            sub_A = tir.match_buffer(
                A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                strides=[As_0, As_1],
                offset_factor=1,
            )
            tir.evaluate(
                tir.intrin_test(
                    sub_A.data,
                    sub_A.elem_offset,
                    sub_A.strides[0],
                    sub_A.strides[1],
                    sub_A.shape[0],
                    sub_A.shape[1],
                    dtype="handle",
                )
            )


@tvm.script.tir
def transformed_high_dim_opaque_access(a: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 32, 64))
    for i, j, k in tir.grid(16, 2, 4):
        with tir.block([]):
            tir.reads([])
            tir.writes(A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
            tir.evaluate(
                tir.intrin_test(
                    A.data,
                    i * 2048 + j * 1024 + k * 16,
                    64,
                    1,
                    16,
                    16,
                    dtype="handle",
                )
            )


@tvm.script.tir
def recursive_match(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (64, 64, 64))
    B = tir.match_buffer(b, (64, 64, 64))
    for i, j, k in tir.grid(64, 4, 4):
        with tir.block([]):
            tir.reads([])
            tir.writes(
                [
                    A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                    B[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                ]
            )
            As_0 = tir.var("int32")
            As_1 = tir.var("int32")
            sub_A = tir.match_buffer(
                A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                strides=[As_0, As_1],
                offset_factor=1,
            )
            sub_B = tir.match_buffer(
                B[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                (16, 16),
                offset_factor=1,
            )
            for jj, kk in tir.grid(4, 4):
                with tir.block([]):
                    tir.reads([])
                    tir.writes(
                        [
                            sub_A[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                            sub_B[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                        ]
                    )
                    Ass_0 = tir.var("int32")
                    Ass_1 = tir.var("int32")
                    sub_sub_A = tir.match_buffer(
                        sub_A[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                        (4, 4),
                        strides=[Ass_0, Ass_1],
                        offset_factor=1,
                    )
                    sub_sub_B = tir.match_buffer(
                        sub_B[jj * 4 : jj * 4 + 4, kk * 4 : kk * 4 + 4],
                        (4, 4),
                        offset_factor=1,
                    )
                    tir.evaluate(
                        tir.intrin_test(
                            sub_sub_A.data,
                            sub_sub_A.elem_offset,
                            sub_sub_A.strides[0],
                            sub_sub_A.strides[1],
                            sub_sub_A.shape[0],
                            sub_sub_A.shape[1],
                            dtype="handle",
                        )
                    )
                    for jjj, kkk in tir.grid(4, 4):
                        sub_sub_B[jjj, kkk] = 1


@tvm.script.tir
def transformed_recursive_match(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (64, 64, 64))
    B = tir.match_buffer(b, (64, 64, 64))
    for i, j, k in tir.grid(64, 4, 4):
        with tir.block([]):
            tir.reads([])
            tir.writes(
                [
                    A[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                    B[i, j * 16 : j * 16 + 16, k * 16 : k * 16 + 16],
                ]
            )
            for jj, kk in tir.grid(4, 4):
                with tir.block([]):
                    tir.reads([])
                    tir.writes(
                        [
                            A[
                                i,
                                j * 16 + jj * 4 : j * 16 + jj * 4 + 4,
                                k * 16 + kk * 4 : k * 16 + kk * 4 + 4,
                            ],
                            B[
                                i,
                                j * 16 + jj * 4 : j * 16 + jj * 4 + 4,
                                k * 16 + kk * 4 : k * 16 + kk * 4 + 4,
                            ],
                        ]
                    )
                    tir.evaluate(
                        tir.intrin_test(
                            A.data,
                            i * 4096 + j * 1024 + jj * 256 + k * 16 + kk * 4,
                            64,
                            1,
                            4,
                            4,
                            dtype="handle",
                        )
                    )
                    for jjj, kkk in tir.grid(4, 4):
                        B[i, j * 16 + jj * 4 + jjj, k * 16 + kk * 4 + kkk] = 1


@tvm.script.tir
def symbolic_match(a: ty.handle, b: ty.handle, n: ty.int32, m: ty.int32) -> None:
    A = tir.match_buffer(a, (n * m, m))
    B = tir.match_buffer(b, (n * 2, m * 4))
    for i in range(0, n):
        with tir.block([]):
            tir.reads([])
            tir.writes([A[i * m : i * m + n, 0:m], B[i * n : i * n + 2, 0 : m * 4]])
            Bs_0 = tir.var("int32")
            Bs_1 = tir.var("int32")
            sub_A = tir.match_buffer(A[i * m : i * m + m, 0:m], (m, m), offset_factor=1)
            sub_B = tir.match_buffer(
                B[i * n : i * n + 2, 0 : m * 4], (2, m * 4), strides=[Bs_0, Bs_1], offset_factor=1
            )
            for ii, jj in tir.grid(m, m):
                sub_A[ii, jj] = 1
            for j in range(0, 4):
                tir.evaluate(
                    tir.intrin_test(
                        sub_B.data,
                        sub_B.elem_offset,
                        sub_B.strides[0],
                        sub_B.strides[1],
                        sub_B.shape[0],
                        sub_B.shape[1],
                        dtype="handle",
                    )
                )


@tvm.script.tir
def transformed_symbolic_match(a: ty.handle, b: ty.handle, n: ty.int32, m: ty.int32) -> None:
    A = tir.match_buffer(a, (n * m, m))
    B = tir.match_buffer(b, (n * 2, m * 4))
    for i in range(0, n):
        with tir.block([]):
            tir.reads([])
            tir.writes([A[i * m : i * m + n, 0:m], B[i * n : i * n + 2, 0 : m * 4]])
            for ii, jj in tir.grid(m, m):
                A[i * m + ii, jj] = 1
            for j in range(0, 4):
                tir.evaluate(
                    tir.intrin_test(
                        B.data,
                        i * n * (m * 4),
                        m * 4,
                        1,
                        2,
                        m * 4,
                        dtype="handle",
                    )
                )


@tvm.script.tir
def rank0_buffer(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (8, 8))
    B = tir.match_buffer(b, (8, 8))
    for i, j in tir.grid(8, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes([A[i, j], B[i, j]])
            sub_A = tir.match_buffer(A[i, j], (), offset_factor=1)
            sub_B = tir.match_buffer(B[i, j], (), offset_factor=1)
            sub_A[()] = 1
            tir.evaluate(
                tir.intrin_test(
                    sub_B.data,
                    sub_B.elem_offset,
                    0,
                    0,
                    0,
                    0,
                    dtype="handle",
                )
            )


@tvm.script.tir
def transformed_rank0_buffer(a: ty.handle, b: ty.handle) -> None:
    A = tir.match_buffer(a, (8, 8))
    B = tir.match_buffer(b, (8, 8))
    for i, j in tir.grid(8, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes([A[i, j], B[i, j]])
            A[i, j] = 1
            tir.evaluate(
                tir.intrin_test(
                    B.data,
                    i * 8 + j,
                    0,
                    0,
                    0,
                    0,
                    dtype="handle",
                )
            )


@tvm.script.tir
def fail_match_load(a: ty.handle) -> None:
    A = tir.match_buffer(a, (8, 8))
    for i, j in tir.grid(8, 8):
        with tir.block([]):
            tir.reads(A[i, j])
            tir.writes([])
            sub_A = tir.match_buffer(A[i, j], ())
            tir.evaluate(tir.load("float32", sub_A.data, 0))


@tvm.script.tir
def fail_match_store(a: ty.handle) -> None:
    A = tir.match_buffer(a, (8, 8))
    for i, j in tir.grid(8, 8):
        with tir.block([]):
            tir.reads([])
            tir.writes(A[i, j])
            sub_A = tir.match_buffer(A[i, j], ())
            sub_A.data[0] = 1


@tvm.script.tir
def fail_buffer_bind(a: ty.handle) -> None:
    A = tir.match_buffer(a, (8, 8))
    for i, j in tir.grid(8, 2):
        with tir.block([]):
            stride = tir.var("int32")
            sub_A = tir.match_buffer(
                A[i, j * 4 : j * 4 + 4], (1, 4), strides=[stride, stride], offset_factor=1
            )
            for jj in range(0, 4):
                sub_A[i, j * 4 + jj] = 1


@tvm.script.tir
def fail_match_func_param(a: ty.handle, m: ty.handle, n: ty.handle) -> None:
    A = tir.match_buffer(a, (8, 8))
    for i, j in tir.grid(8, 2):
        with tir.block([]):
            sub_A = tir.match_buffer(
                A[i, j * 4 : j * 4 + 4], (1, 4), strides=[m, n], offset_factor=1
            )
            for jj in range(0, 4):
                sub_A[i, j * 4 + jj] = 1


def test_buffer_load_store():
    _check(buffer_load_store, transformed_buffer_load_store)


def test_opaque_access():
    _check(opaque_access, transformed_opaque_access)


def test_high_dim_opaque_access():
    _check(high_dim_opaque_access, transformed_high_dim_opaque_access)


def test_recursive_match():
    _check(recursive_match, transformed_recursive_match)


def test_symbolic_match():
    _check(symbolic_match, transformed_symbolic_match)


def test_rank0_buffer():
    _check(rank0_buffer, transformed_rank0_buffer)


def test_fail_load_store():
    _check_fail(fail_match_load)
    _check_fail(fail_match_store)


def test_fail_buffer_bind():
    _check_fail(fail_buffer_bind)


def test_fail_match_func_param():
    _check_fail(fail_match_func_param)


if __name__ == "__main__":
    test_buffer_load_store()
    test_opaque_access()
    test_high_dim_opaque_access()
    test_recursive_match()
    test_symbolic_match()
    test_rank0_buffer()
    test_fail_load_store()
    test_fail_buffer_bind()
    test_fail_match_func_param()
