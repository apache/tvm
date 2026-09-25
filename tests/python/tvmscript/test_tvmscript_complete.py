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

import tvm.testing
from tvm.ir import Range
from tvm.script import s_tir as Ts
from tvm.script import tirx as T


@Ts.prim_func
def matmul(A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])) -> None:
    for i, j, k in T.grid(128, 128, 128):
        with Ts.sblock("update"):
            vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
            with Ts.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]


@Ts.prim_func
def matmul_original(
    A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
) -> None:
    for i, j in T.grid(32, 32):
        with Ts.sblock("init"):
            vi, vj = Ts.axis.remap("SS", [i, j])
            for ii, jj in T.grid(4, 4):
                C[vi * 4 + ii, vj * 4 + jj] = T.float32(0)

        for k in range(0, 32):
            with Ts.sblock("update"):
                vi, vj, vk = Ts.axis.remap("SSR", [i, j, k])
                for ii, jj, kk in T.grid(4, 4, 4):
                    C[vi * 4 + ii, vj * 4 + jj] = (
                        C[vi * 4 + ii, vj * 4 + jj]
                        + A[vi * 4 + ii, vk * 4 + kk] * B[vj * 4 + jj, vk * 4 + kk]
                    )


@Ts.prim_func
def elementwise_with_root(
    A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
) -> None:
    with Ts.sblock():
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)


def func_with_opaque_block(
    A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
) -> None:
    with Ts.sblock():
        with Ts.sblock():
            B[0, 0] = A[0, 0] + T.float32(1)
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                C[vi, vj] = B[vi, vj] + T.float32(1)


@Ts.prim_func
def func_with_part_access_region(
    A: T.Buffer([128, 128]), B: T.Buffer([128, 128]), C: T.Buffer([128, 128])
) -> None:
    with Ts.sblock():
        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.reads(A[vi, vj])
                B[vi, vj] = A[vi, vj] + T.float32(1)

        for i, j in T.grid(128, 128):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i, j])
                Ts.writes(C[vi, vj])
                C[vi, vj] = B[vi, vj] + T.float32(1)


def test_complete_matmul():
    func = matmul
    A, B, C = [x for x in func.params if tvm.tirx.is_buffer_var(x)]

    block = func.body.block.body.body.body.body.block
    assert isinstance(block, tvm.s_tir.SBlock)
    vi, vj, vk = [x.var for x in block.iter_vars]
    access_A = tvm.tirx.BufferRegion(
        A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vk, 1)]
    )
    access_B = tvm.tirx.BufferRegion(
        B, [Range.from_min_extent(vj, 1), Range.from_min_extent(vk, 1)]
    )
    access_C = tvm.tirx.BufferRegion(
        C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)]
    )
    tvm.ir.assert_structural_equal(block.reads, [access_A, access_B])
    tvm.ir.assert_structural_equal(block.writes, [access_C])


def test_complete_matmul_original():
    func = matmul_original
    A, B, C = [x for x in func.params if tvm.tirx.is_buffer_var(x)]

    block1 = func.body.block.body.body.body[0].block
    assert isinstance(block1, tvm.s_tir.SBlock)
    vi, vj = [x.var for x in block1.iter_vars]
    access_C = tvm.tirx.BufferRegion(
        C, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)]
    )
    tvm.ir.assert_structural_equal(block1.reads, [])
    tvm.ir.assert_structural_equal(block1.writes, [access_C])

    block2 = func.body.block.body.body.body[1].body.block
    assert isinstance(block2, tvm.s_tir.SBlock)
    vi, vj, vk = [x.var for x in block2.iter_vars]
    access_A = tvm.tirx.BufferRegion(
        A, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vk * 4, 4)]
    )
    access_B = tvm.tirx.BufferRegion(
        B, [Range.from_min_extent(vj * 4, 4), Range.from_min_extent(vk * 4, 4)]
    )
    access_C = tvm.tirx.BufferRegion(
        C, [Range.from_min_extent(vi * 4, 4), Range.from_min_extent(vj * 4, 4)]
    )
    tvm.ir.assert_structural_equal(block2.reads, [access_C, access_A, access_B])
    tvm.ir.assert_structural_equal(block2.writes, [access_C])


def _check_elementwise(func):
    A, B, C = [x for x in func.params if tvm.tirx.is_buffer_var(x)]

    root_block = func.body.block
    assert len(root_block.reads) == 0
    assert len(root_block.writes) == 0

    block1 = func.body.block.body[0].body.body.block
    assert isinstance(block1, tvm.s_tir.SBlock)
    vi, vj = [x.var for x in block1.iter_vars]

    tvm.ir.assert_structural_equal(
        block1.reads,
        [tvm.tirx.BufferRegion(A, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )
    tvm.ir.assert_structural_equal(
        block1.writes,
        [tvm.tirx.BufferRegion(B, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )

    block2 = func.body.block.body[1].body.body.block
    assert isinstance(block2, tvm.s_tir.SBlock)
    vi, vj = [x.var for x in block2.iter_vars]
    tvm.ir.assert_structural_equal(
        block2.reads,
        [tvm.tirx.BufferRegion(B, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )
    tvm.ir.assert_structural_equal(
        block2.writes,
        [tvm.tirx.BufferRegion(C, [Range.from_min_extent(vi, 1), Range.from_min_extent(vj, 1)])],
    )


def test_complete_with_root():
    _check_elementwise(elementwise_with_root)


def test_complete_part_region():
    _check_elementwise(func_with_part_access_region)


@Ts.prim_func
def func_with_bufferslice_indices(
    data_buf: T.Buffer((16, 16), "float32"), index_buf: T.Buffer((1,), "int32")
) -> None:
    out_buf = Ts.sblock_alloc_buffer((16, 16), "float32")

    for i, j in T.grid(16, 16):
        with Ts.sblock():
            vi, vj = Ts.axis.remap("SS", [i, j])
            out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@Ts.prim_func
def expected_bufferslice_indices(
    data_buf: T.Buffer([16, 16], elem_offset=0, align=64, offset_factor=1),
    index_buf: T.Buffer([1], dtype="int32", elem_offset=0, align=64, offset_factor=1),
) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        out_buf = Ts.sblock_alloc_buffer([16, 16], elem_offset=0, align=64, offset_factor=1)
        for i0, i1 in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i0, i1])
                Ts.reads([data_buf[vi, index_buf[0]], index_buf[0]])
                Ts.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[vi, index_buf[0]]


@Ts.prim_func
def func_with_recursive_bufferslice_indices(
    data_buf: T.Buffer((16, 16), "float32"), index_buf: T.Buffer((1,), "int32")
) -> None:
    out_buf = Ts.sblock_alloc_buffer((16, 16), "float32")

    for i, j in T.grid(16, 16):
        with Ts.sblock():
            vi, vj = Ts.axis.remap("SS", [i, j])
            out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


@Ts.prim_func
def expected_recursive_bufferslice_indices(
    data_buf: T.Buffer([16, 16], elem_offset=0, align=64, offset_factor=1),
    index_buf: T.Buffer([1], dtype="int32", elem_offset=0, align=64, offset_factor=1),
) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        out_buf = Ts.sblock_alloc_buffer([16, 16], elem_offset=0, align=64, offset_factor=1)
        for i0, i1 in T.grid(16, 16):
            with Ts.sblock():
                vi, vj = Ts.axis.remap("SS", [i0, i1])
                Ts.reads(
                    [
                        data_buf[index_buf[index_buf[0]], index_buf[0]],
                        index_buf[T.min(index_buf[0], 0) : T.max(index_buf[0], 0) + 1],
                    ]
                )
                Ts.writes([out_buf[vi, vj]])
                out_buf[vi, vj] = data_buf[index_buf[index_buf[0]], index_buf[0]]


def test_complete_buffer_indices():
    new_func = tvm.script.from_source(
        func_with_bufferslice_indices.script(),
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
    ).with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(
        new_func, expected_bufferslice_indices.with_attr("global_symbol", "main")
    )
    new_func = tvm.script.from_source(
        func_with_recursive_bufferslice_indices.script(),
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
    ).with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(
        new_func, expected_recursive_bufferslice_indices.with_attr("global_symbol", "main")
    )


@Ts.prim_func
def match_buffer_func(A: T.Buffer((16, 16))) -> None:
    for i in range(0, 16):
        with Ts.sblock():
            A0 = Ts.match_buffer(A[i, 0:16], (16))
            with Ts.sblock():
                for j in range(0, 16):
                    with Ts.sblock():
                        A1 = Ts.match_buffer(A0[j], ())
                        A1[()] = 1.0


@Ts.prim_func
def expected_match_buffer_func(A: T.Buffer((16, 16))) -> None:
    for i in range(0, 16):
        with Ts.sblock():
            Ts.reads([])
            Ts.writes(A[i, 0:16])
            A0 = Ts.match_buffer(A[i, 0:16], (16))
            with Ts.sblock():
                Ts.reads([])
                Ts.writes(A0[0:16])
                for j in range(0, 16):
                    with Ts.sblock():
                        Ts.reads([])
                        Ts.writes(A0[j])
                        A1 = Ts.match_buffer(A0[j], ())
                        A1[()] = 1.0


def test_complete_match_buffer():
    tvm.ir.assert_structural_equal(
        match_buffer_func.with_attr("global_symbol", "main"),
        expected_match_buffer_func.with_attr("global_symbol", "main"),
    )


@Ts.prim_func
def alloc_buffer_func(
    A: T.Buffer([2, 2], dtype="float32"), B: T.Buffer([2, 2], dtype="float32")
) -> None:
    C = Ts.sblock_alloc_buffer([2, 2], dtype="float32")
    A[(0, 0)] = T.float32(2)
    C[(0, 0)] = A[(0, 0)] + B[(0, 0)]
    B[(0, 0)] = C[(0, 0)]


@Ts.prim_func
def expect_alloc_buffer_func(
    A: T.Buffer([2, 2], dtype="float32", elem_offset=0, align=64, offset_factor=1),
    B: T.Buffer([2, 2], dtype="float32", elem_offset=0, align=64, offset_factor=1),
) -> None:
    with Ts.sblock("root"):
        Ts.reads([])
        Ts.writes([])
        C = Ts.sblock_alloc_buffer(
            [2, 2], dtype="float32", elem_offset=0, align=64, offset_factor=1
        )
        A[(0, 0)] = T.float32(2)
        C[(0, 0)] = A[(0, 0)] + B[(0, 0)]
        B[(0, 0)] = C[(0, 0)]


def test_complete_alloc_buffer():
    rt_func = tvm.script.from_source(
        alloc_buffer_func.script(),
        extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx, "Ts": tvm.script.s_tir},
    ).with_attr("global_symbol", "main")
    tvm.ir.assert_structural_equal(
        rt_func, expect_alloc_buffer_func.with_attr("global_symbol", "main")
    )


if __name__ == "__main__":
    tvm.testing.main()
