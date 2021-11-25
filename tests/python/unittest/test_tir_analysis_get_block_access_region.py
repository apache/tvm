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
from tvm.script import tir as T
from tvm.ir import Range


@T.prim_func
def func() -> None:
    A = T.alloc_buffer((128, 128), "float32")
    B = T.alloc_buffer((128, 128), "float32")
    C = T.alloc_buffer((128, 128), "float32")
    D = T.alloc_buffer((128, 128), "float32")
    with T.block():
        # Need add read/write region manually to avoid triggering block access region detector
        T.reads([B[0, 0], C[0:16, 0:16], A[4:12, 4:12]])
        T.writes([A[0:12, 0:12]])
        for i, j in T.grid(8, 8):
            A[i, j] = B[0, 0] + C[0, 0]
        for i, j in T.grid(2, 2):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads([A[vi * 4 + 4 : vi * 4 + 8, vj * 4 + 4 : vj * 4 + 8], C[12:16, 12:16]])
                T.writes([A[vi * 4 + 4 : vi * 4 + 8, vj * 4 + 4 : vj * 4 + 8]])
                for i, j in T.grid(4, 4):
                    A[vi * 4 + 4 + i, vj * 4 + 4 + j] += C[i + 12, j + 12]
        T.evaluate(D.data)


@T.prim_func
def match_buffer_func() -> None:
    with T.block("root"):
        A = T.alloc_buffer((128, 128), "float32")
        B = T.alloc_buffer((128, 128), "float32")
        T.reads([])
        T.writes([])
        # Need add read/write region manually to avoid triggering block access region detector
        for i, j in T.grid(8, 8):
            with T.block("block"):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(B[vi * 16 + 2 : vi * 16 + 12, vj * 16 + 2 : vj * 16 + 16])
                T.writes(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
                AA = T.match_buffer(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16], (16, 16))
                B0 = T.match_buffer(B[vi * 16 + 2 : vi * 16 + 6, vj * 16 + 2 : vj * 16 + 6], (4, 4))
                B1 = T.match_buffer(
                    B[vi * 16 + 8 : vi * 16 + 12, vj * 16 + 8 : vj * 16 + 16], (4, 8)
                )
                for ii, jj in T.grid(16, 16):
                    with T.block("AAA"):
                        vii, vjj = T.axis.remap("SS", [ii, jj])
                        T.reads([])
                        T.writes(AA[vii, vjj])
                        AAA = T.match_buffer(AA[vii, vjj], ())
                        AAA[()] = 1.0
                T.evaluate(B0.data)
                T.evaluate(B1.data)


@T.prim_func
def opaque_block_func() -> None:
    with T.block("root"):
        A = T.alloc_buffer((16, 16), "float32")
        B = T.alloc_buffer((16, 16), "float32")
        T.reads([])
        T.writes([])
        # Need add read/write region manually to avoid triggering block access region detector
        for i in range(0, 16):
            with T.block():
                T.reads(A[i, 0:16])
                T.writes([B[i, 0:16]])
                for j in range(0, 16):
                    with T.block():
                        T.reads(A[i, j])
                        T.writes(B[i, j])
                        B[i, j] = A[i, j] + 1.0


@T.prim_func
def opaque_access_func() -> None:
    A = T.alloc_buffer([1024])
    B = T.alloc_buffer([1024])
    for i in T.serial(0, 8):
        with T.block():
            v = T.axis.S(8, i)
            T.reads([A[v * 128 : v * 128 + 128]])
            T.writes([B[v * 128 : v * 128 + 128]])
            T.evaluate(
                T.call_extern("test", B.data, v * 128, 128, A.data, v * 128, 128, dtype="float32")
            )


def test_block_access_region_detector():
    block = func.body.block.body.block
    alloc_buffers = func.body.block.alloc_buffers
    buffer_var_map = {buf.data: buf for buf in alloc_buffers}
    ret = tir.analysis.get_block_access_region(block, buffer_var_map)

    tvm.ir.assert_structural_equal(block.reads, ret[0])
    tvm.ir.assert_structural_equal(block.writes, ret[1])
    D = alloc_buffers[-1]
    tvm.ir.assert_structural_equal(
        [tvm.tir.BufferRegion(D, [Range(0, 128), Range(0, 128)])], ret[2]
    )


def test_opaque_block():
    alloc_buffers = opaque_block_func.body.block.alloc_buffers
    buffer_var_map = {buf.data: buf for buf in alloc_buffers}

    block0 = opaque_block_func.body.block.body.body.block
    ret = tir.analysis.get_block_access_region(block0, buffer_var_map)
    tvm.ir.assert_structural_equal(block0.reads, ret[0])
    tvm.ir.assert_structural_equal(block0.writes, ret[1])

    block1 = block0.body.body.block
    ret = tir.analysis.get_block_access_region(block1, buffer_var_map)
    tvm.ir.assert_structural_equal(block1.reads, ret[0])
    tvm.ir.assert_structural_equal(block1.writes, ret[1])


def test_opaque_access():
    block = opaque_access_func.body.block.body.body.block
    alloc_buffers = opaque_access_func.body.block.alloc_buffers
    buffer_var_map = {buf.data: buf for buf in alloc_buffers}

    ret0 = tir.analysis.get_block_read_write_region(block, buffer_var_map)
    ret1 = tir.analysis.get_block_access_region(block, buffer_var_map)
    with pytest.raises(ValueError):
        tvm.ir.assert_structural_equal(ret0[0], ret1[0])
    with pytest.raises(ValueError):
        tvm.ir.assert_structural_equal(ret0[1], ret1[1])


def test_match_buffer():
    root_block = match_buffer_func.body.block
    block = root_block.body.body.body.block
    block_inner = block.body[0].body.body.block
    alloc_buffers = match_buffer_func.body.block.alloc_buffers
    buffer_var_map = {buf.data: buf for buf in alloc_buffers}

    # Check block
    ret = tir.analysis.get_block_access_region(block, buffer_var_map)
    tvm.ir.assert_structural_equal(block.writes, ret[1])
    # B is opaque access
    tvm.ir.assert_structural_equal(block.reads, ret[2])

    # Check inner block AAA without updating buffer_var_map
    ret = tir.analysis.get_block_access_region(block_inner, buffer_var_map)
    # Since AA is not in the buffer_var_map, region of AA will not be collected.
    tvm.ir.assert_structural_equal([], ret[1])

    # Check inner block AAA
    for match_buffer in block.match_buffers:
        target_buffer = match_buffer.buffer
        buffer_var_map[target_buffer.data] = target_buffer

    ret = tir.analysis.get_block_access_region(block_inner, buffer_var_map)
    tvm.ir.assert_structural_equal(block_inner.reads, ret[0])
    tvm.ir.assert_structural_equal(block_inner.writes, ret[1])


if __name__ == "__main__":
    test_block_access_region_detector()
    test_opaque_block()
    test_opaque_access()
    test_match_buffer()
