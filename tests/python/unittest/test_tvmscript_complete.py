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

import tvm
from tvm.ir import Range
from tvm.script import tir as T


@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j, k in T.grid(128, 128, 128):
        with T.block("update"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@T.prim_func
def matmul_original(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    for i, j in T.grid(32, 32):
        with T.block("init"):
            vi, vj = T.axis.remap("SS", [i, j])
            for ii, jj in T.grid(4, 4):
                C[vi * 4 + ii, vj * 4 + jj] = T.float32(0)

        for k in range(0, 32):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                for ii, jj, kk in T.grid(4, 4, 4):
                    C[vi * 4 + ii, vj * 4 + jj] = (
                        C[vi * 4 + ii, vj * 4 + jj]
                        + A[vi * 4 + ii, vk * 4 + kk] * B[vj * 4 + jj, vk * 4 + kk]
                    )


@T.prim_func
def elementwise_with_root(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block():
        for i, j in T.grid(128, 128):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)
        for i, j in T.grid(128, 128):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)


def func_with_opaque_block(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block():
        with T.block():
            B[0, 0] = A[0, 0] + T.float32(1)
        for i, j in T.grid(128, 128):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)


@T.prim_func
def func_with_part_access_region(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, [128, 128])
    B = T.match_buffer(b, [128, 128])
    C = T.match_buffer(c, [128, 128])

    with T.block():
        for i, j in T.grid(128, 128):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(A[vi, vj])
                B[vi, vj] = A[vi, vj] + T.float32(1)

        for i, j in T.grid(128, 128):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                T.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)


def test_complete_matmul():
    func = matmul
    A, B, C = [func.buffer_map[x] for x in func.params]

    block = func.body.block.body.body.body.body.block
    assert isinstance(block, tvm.tir.Block)
    vi, vj, vk = [x.var for x in block.iter_vars]
    access_A = tvm.tir.BufferRegion(A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vk, 1)])
    access_B = tvm.tir.BufferRegion(B, [Range.from_min_extent(vj, 1), Range.from_min_extent(vk, 1)])
    access_C = tvm.tir.BufferRegion(C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])
    tvm.ir.assert_structural_equal(block.reads, [access_A, access_B])
    tvm.ir.assert_structural_equal(block.writes, [access_C])


def test_complete_matmul_original():
    func = matmul_original
    A, B, C = [func.buffer_map[x] for x in func.params]

    block1 = func.body.block.body.body.body[0].block
    assert isinstance(block1, tvm.tir.Block)
    vi, vj = [x.var for x in block1.iter_vars]
    access_C = tvm.tir.BufferRegion(
        C, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)]
    )
    tvm.ir.assert_structural_equal(block1.reads, [])
    tvm.ir.assert_structural_equal(block1.writes, [access_C])

    block2 = func.body.block.body.body.body[1].body.block
    assert isinstance(block2, tvm.tir.Block)
    vi, vj, vk = [x.var for x in block2.iter_vars]
    access_A = tvm.tir.BufferRegion(
        A, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vk * 4, 4)]
    )
    access_B = tvm.tir.BufferRegion(
        B, [Range.from_min_extent(vj * 4, 4), Range.from_min_extent(vk * 4, 4)]
    )
    access_C = tvm.tir.BufferRegion(
        C, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)]
    )
    tvm.ir.assert_structural_equal(block2.reads, [access_C, access_A, access_B])
    tvm.ir.assert_structural_equal(block2.writes, [access_C])


def _check_elementwise(func):
    A, B, C = [func.buffer_map[x] for x in func.params]

    root_block = func.body.block
    assert len(root_block.reads) == 0
    assert len(root_block.writes) == 0

    block1 = func.body.block.body[0].body.body.block
    assert isinstance(block1, tvm.tir.Block)
    vi, vj = [x.var for x in block1.iter_vars]

    tvm.ir.assert_structural_equal(
        block1.reads,
        [tvm.tir.BufferRegion(A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )
    tvm.ir.assert_structural_equal(
        block1.writes,
        [tvm.tir.BufferRegion(B, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )

    block2 = func.body.block.body[1].body.body.block
    assert isinstance(block2, tvm.tir.Block)
    vi, vj = [x.var for x in block2.iter_vars]
    tvm.ir.assert_structural_equal(
        block2.reads,
        [tvm.tir.BufferRegion(B, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )
    tvm.ir.assert_structural_equal(
        block2.writes,
        [tvm.tir.BufferRegion(C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )


def test_complete_with_root():
    _check_elementwise(elementwise_with_root)


def test_complete_part_region():
    _check_elementwise(func_with_part_access_region)


@T.prim_func
def func_with_bufferslice_indices(data: T.handle, index: T.handle) -> None:
    data_buf = T.match_buffer(data, (16, 16), "float32")
    index_buf = T.match_buffer(index, (1,), "int32")
    out_buf = T.alloc_buffer((16, 16), "float32")

    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@T.prim_func
def expected_bufferslice_indices(data: T.handle, index: T.handle) -> None:
    index_buf = T.match_buffer(index, [1], dtype="int32", elem_offset=0, align=64, offset_factor=1)
    data_buf = T.match_buffer(data, [16, 16], elem_offset=0, align=64, offset_factor=1)
    with T.block("root"):
        T.reads([])
        T.writes([])
        out_buf = T.alloc_buffer([16, 16], elem_offset=0, align=64, offset_factor=1)
        for i0, i1 in T.grid(16, 16):
            with T.block():
                vi, vj = T.axis.remap("SS", [i0, i1])
                T.reads([data_buf[vi, index_buf[0]], index_buf[0]])
                T.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@T.prim_func
def func_with_recursive_bufferslice_indices(data: T.handle, index: T.handle) -> None:
    data_buf = T.match_buffer(data, (16, 16), "float32")
    index_buf = T.match_buffer(index, (1,), "int32")
    out_buf = T.alloc_buffer((16, 16), "float32")

    for i, j in T.grid(16, 16):
        with T.block():
            vi, vj = T.axis.remap("SS", [i, j])
            out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


@T.prim_func
def expected_recursive_bufferslice_indices(data: T.handle, index: T.handle) -> None:
    index_buf = T.match_buffer(index, [1], dtype="int32", elem_offset=0, align=64, offset_factor=1)
    data_buf = T.match_buffer(data, [16, 16], elem_offset=0, align=64, offset_factor=1)
    with T.block("root"):
        T.reads([])
        T.writes([])
        out_buf = T.alloc_buffer([16, 16], elem_offset=0, align=64, offset_factor=1)
        for i0, i1 in T.grid(16, 16):
            with T.block():
                vi, vj = T.axis.remap("SS", [i0, i1])
                T.reads(
                    [
                        data_buf[index_buf[index_buf[0]], index_buf[0]],
                        index_buf[T.min(index_buf[0], 0) : T.max(index_buf[0], 0) + 1],
                    ]
                )
                T.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


def test_complete_buffer_indices():
    new_func = tvm.script.from_source(func_with_bufferslice_indices.script()).with_attr(
        "global_symbol", "main"
    )
    tvm.ir.assert_structural_equal(
        new_func, expected_bufferslice_indices.with_attr("global_symbol", "main")
    )
    new_func = tvm.script.from_source(func_with_recursive_bufferslice_indices.script()).with_attr(
        "global_symbol", "main"
    )
    tvm.ir.assert_structural_equal(
        new_func, expected_recursive_bufferslice_indices.with_attr("global_symbol", "main")
    )


@T.prim_func
def match_buffer_func(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    for i in range(0, 16):
        with T.block():
            A0 = T.match_buffer(A[i, 0:16], (16))
            with T.block():
                for j in range(0, 16):
                    with T.block():
                        A1 = T.match_buffer(A0[j], ())
                        A1[()] = 1.0


@T.prim_func
def expected_match_buffer_func(a: T.handle) -> None:
    A = T.match_buffer(a, (16, 16))
    for i in range(0, 16):
        with T.block():
            T.reads([])
            T.writes(A[i, 0:16])
            A0 = T.match_buffer(A[i, 0:16], (16))
            with T.block():
                T.reads([])
                T.writes(A0[0:16])
                for j in range(0, 16):
                    with T.block():
                        T.reads([])
                        T.writes(A0[j])
                        A1 = T.match_buffer(A0[j], ())
                        A1[()] = 1.0


def test_complete_match_buffer():
    tvm.ir.assert_structural_equal(
        match_buffer_func.with_attr("global_symbol", "main"),
        expected_match_buffer_func.with_attr("global_symbol", "main"),
    )


@T.prim_func
def alloc_buffer_func(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [2, 2], dtype="float32")
    B = T.match_buffer(b, [2, 2], dtype="float32")
    C = T.alloc_buffer([2, 2], dtype="float32")
    A[(0, 0)] = T.float32(2)
    C[(0, 0)] = A[(0, 0)] + B[(0, 0)]
    B[(0, 0)] = C[(0, 0)]


@T.prim_func
def expect_alloc_buffer_func(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, [2, 2], dtype="float32", elem_offset=0, align=64, offset_factor=1)
    B = T.match_buffer(b, [2, 2], dtype="float32", elem_offset=0, align=64, offset_factor=1)
    with T.block("root"):
        T.reads([])
        T.writes([])
        C = T.alloc_buffer([2, 2], dtype="float32", elem_offset=0, align=64, offset_factor=1)
        A[(0, 0)] = T.float32(2)
        C[(0, 0)] = A[(0, 0)] + B[(0, 0)]
        B[(0, 0)] = C[(0, 0)]


def test_complete_alloc_buffer():
    rt_func = tvm.script.from_source(alloc_buffer_func.script()).with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(
        rt_func, expect_alloc_buffer_func.with_attr("global_symbol", "main")
    )


if __name__ == "__main__":
    test_complete_matmul()
    test_complete_matmul_original()
    test_complete_with_root()
    test_complete_part_region()
    test_complete_buffer_indices()
    test_complete_match_buffer()
    test_complete_alloc_buffer()
