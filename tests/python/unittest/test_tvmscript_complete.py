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
from tvm import tir
from tvm.ir import Range
from tvm.script import ty, from_source
from tvm.ir.diagnostics import override_renderer


@tvm.script.tir
def matmul(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
        with tir.init():
            C[vi, vj] = tir.float32(0)
        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@tvm.script.tir
def matmul_original(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    for i, j in tir.grid(32, 32):
        with tir.block([32, 32], "init") as [vi, vj]:
            for ii, jj in tir.grid(4, 4):
                C[vi * 4 + ii, vj * 4 + jj] = tir.float32(0)

        for k in range(0, 32):
            with tir.block([128, 128, tir.reduce_axis(0, 128)], "update") as [vi, vj, vk]:
                for ii, jj, kk in tir.grid(4, 4, 4):
                    C[vi * 4 + ii, vj * 4 + jj] = (
                        C[vi * 4 + ii, vj * 4 + jj]
                        + A[vi * 4 + ii, vk * 4 + kk] * B[vj * 4 + jj, vk * 4 + kk]
                    )


@tvm.script.tir
def elementwise_with_root(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([]) as []:
        with tir.block([128, 128]) as [vi, vj]:
            B[vi, vj] = A[vi, vj] + tir.float32(1)

        with tir.block([128, 128]) as [vi, vj]:
            C[vi, vj] = B[vi, vj] + tir.float32(1)


def func_with_opaque_block(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([]) as []:
        with tir.block([]) as []:
            B[0, 0] = A[0, 0] + tir.float32(1)

        with tir.block([128, 128]) as [vi, vj]:
            C[vi, vj] = B[vi, vj] + tir.float32(1)


@tvm.script.tir
def func_with_part_access_region(a: ty.handle, b: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    with tir.block([]) as []:
        with tir.block([128, 128]) as [vi, vj]:
            tir.reads(A[vi, vj])
            B[vi, vj] = A[vi, vj] + tir.float32(1)

        with tir.block([128, 128]) as [vi, vj]:
            tir.writes(C[vi, vj])
            C[vi, vj] = B[vi, vj] + tir.float32(1)


def test_complete_matmul():
    func = matmul
    A, B, C = [func.buffer_map[x] for x in func.params]

    block = func.body.block.body.body.body.body.block
    assert isinstance(block, tvm.tir.Block)
    vi, vj, vk = [x.var for x in block.iter_vars]
    access_A = tir.BufferRegion(A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vk, 1)])
    access_B = tir.BufferRegion(B, [Range.from_min_extent(vj, 1), Range.from_min_extent(vk, 1)])
    access_C = tir.BufferRegion(C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])
    tvm.ir.assert_structural_equal(block.reads, [access_C, access_A, access_B])
    tvm.ir.assert_structural_equal(block.writes, [access_C])


def test_complete_matmul_original():
    func = matmul_original
    A, B, C = [func.buffer_map[x] for x in func.params]

    block1 = func.body.block.body.body.body[0].block
    assert isinstance(block1, tvm.tir.Block)
    vi, vj = [x.var for x in block1.iter_vars]
    access_C = tir.BufferRegion(
        C, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)]
    )
    tvm.ir.assert_structural_equal(block1.reads, [])
    tvm.ir.assert_structural_equal(block1.writes, [access_C])

    block2 = func.body.block.body.body.body[1].body.block
    assert isinstance(block2, tvm.tir.Block)
    vi, vj, vk = [x.var for x in block2.iter_vars]
    access_A = tir.BufferRegion(
        A, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vk * 4, 4)]
    )
    access_B = tir.BufferRegion(
        B, [Range.from_min_extent(vj * 4, 4), Range.from_min_extent(vk * 4, 4)]
    )
    access_C = tir.BufferRegion(
        C, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)]
    )
    tvm.ir.assert_structural_equal(block2.reads, [access_C, access_A, access_B])
    tvm.ir.assert_structural_equal(block2.writes, [access_C])


def _check_elementwise(func):
    A, B, C = [func.buffer_map[x] for x in func.params]

    block1 = func.body.block.body[0].body.body.block
    assert isinstance(block1, tvm.tir.Block)
    vi, vj = [x.var for x in block1.iter_vars]

    tvm.ir.assert_structural_equal(
        block1.reads,
        [tir.BufferRegion(A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )
    tvm.ir.assert_structural_equal(
        block1.writes,
        [tir.BufferRegion(B, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )

    block2 = func.body.block.body[1].body.body.block
    assert isinstance(block2, tvm.tir.Block)
    vi, vj = [x.var for x in block2.iter_vars]
    tvm.ir.assert_structural_equal(
        block2.reads,
        [tir.BufferRegion(B, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )
    tvm.ir.assert_structural_equal(
        block2.writes,
        [tir.BufferRegion(C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )


def test_complete_with_root():
    _check_elementwise(elementwise_with_root)


def test_complete_part_region():
    _check_elementwise(func_with_part_access_region)


def test_complete_opaque_block_error():
    def render(e):
        pass

    override_renderer(render)

    try:
        from_source(func_with_opaque_block)
    except tvm.error.DiagnosticError:
        return
    assert False


@tvm.script.tir
def func_with_bufferslice_indices(data: ty.handle, index: ty.handle) -> None:
    data_buf = tir.match_buffer(data, (16, 16), "float32")
    index_buf = tir.match_buffer(index, (1,), "int32")
    out_buf = tir.alloc_buffer((16, 16), "float32")

    with tir.block([16, 16]) as [vi, vj]:
        out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@tvm.script.tir
def expected_bufferslice_indices(data: ty.handle, index: ty.handle) -> None:
    index_buf = tir.match_buffer(
        index, [1], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    data_buf = tir.match_buffer(data, [16, 16], elem_offset=0, align=128, offset_factor=1)
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        out_buf = tir.alloc_buffer([16, 16], elem_offset=0, align=128, offset_factor=1)
        for i0, i1 in tir.grid(16, 16):
            with tir.block([16, 16], "") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, i1)
                tir.reads([data_buf[vi, 0:16], index_buf[0]])
                tir.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@tvm.script.tir
def func_with_recursive_bufferslice_indices(data: ty.handle, index: ty.handle) -> None:
    data_buf = tir.match_buffer(data, (16, 16), "float32")
    index_buf = tir.match_buffer(index, (1,), "int32")
    out_buf = tir.alloc_buffer((16, 16), "float32")

    with tir.block([16, 16]) as [vi, vj]:
        out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


@tvm.script.tir
def expected_recursive_bufferslice_indices(data: ty.handle, index: ty.handle) -> None:
    index_buf = tir.match_buffer(
        index, [1], dtype="int32", elem_offset=0, align=128, offset_factor=1
    )
    data_buf = tir.match_buffer(data, [16, 16], elem_offset=0, align=128, offset_factor=1)
    with tir.block([], "root"):
        tir.reads([])
        tir.writes([])
        out_buf = tir.alloc_buffer([16, 16], elem_offset=0, align=128, offset_factor=1)
        for i0, i1 in tir.grid(16, 16):
            with tir.block([16, 16], "") as [vi, vj]:
                tir.bind(vi, i0)
                tir.bind(vj, i1)
                tir.reads([data_buf[0:16, 0:16], index_buf[0]])
                tir.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


def test_complete_buffer_indices():
    new_func = tvm.script.from_source(tvm.script.asscript(func_with_bufferslice_indices))
    tvm.ir.assert_structural_equal(new_func, expected_bufferslice_indices)
    new_func = tvm.script.from_source(tvm.script.asscript(func_with_recursive_bufferslice_indices))
    tvm.ir.assert_structural_equal(new_func, expected_recursive_bufferslice_indices)


if __name__ == "__main__":
    test_complete_matmul()
    test_complete_matmul_original()
    test_complete_with_root()
    test_complete_opaque_block_error()
    test_complete_part_region()
    test_complete_buffer_indices()
