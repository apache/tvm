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
from tvm import tir, script
from tvm.ir import Range


@tvm.script.tir
def func() -> None:
    A = tir.alloc_buffer((128, 128), "float32")
    B = tir.alloc_buffer((128, 128), "float32")
    C = tir.alloc_buffer((128, 128), "float32")
    D = tir.alloc_buffer((128, 128), "float32")
    with tir.block([]):
        # Need add read/write region manually to avoid triggering block access region detector
        tir.reads([B[0, 0], C[0:16, 0:16], A[4:12, 4:12]])
        tir.writes([A[0:12, 0:12]])
        for i, j in tir.grid(8, 8):
            A[i, j] = B[0, 0] + C[0, 0]
        with tir.block([2, 2]) as [vi, vj]:
            tir.reads([A[vi * 4 + 4 : vi * 4 + 8, vj * 4 + 4 : vj * 4 + 8], C[12:16, 12:16]])
            tir.writes([A[vi * 4 + 4 : vi * 4 + 8, vj * 4 + 4 : vj * 4 + 8]])
            for i, j in tir.grid(4, 4):
                A[vi * 4 + 4 + i, vj * 4 + 4 + j] += C[i + 12, j + 12]
        tir.evaluate(D.data)


@tvm.script.tir
def match_buffer_func() -> None:
    with tir.block([], "root"):
        A = tir.alloc_buffer((128, 128), "float32")
        B = tir.alloc_buffer((128, 128), "float32")
        tir.reads([])
        tir.writes([])
        # Need add read/write region manually to avoid triggering block access region detector
        with tir.block([8, 8], "block") as [vi, vj]:
            tir.reads(B[vi * 16 + 2 : vi * 16 + 12, vj * 16 + 2 : vj * 16 + 16])
            tir.writes(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
            AA = tir.match_buffer(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16], (16, 16))
            B0 = tir.match_buffer(B[vi * 16 + 2 : vi * 16 + 6, vj * 16 + 2 : vj * 16 + 6], (4, 4))
            B1 = tir.match_buffer(B[vi * 16 + 8 : vi * 16 + 12, vj * 16 + 8 : vj * 16 + 16], (4, 8))
            with tir.block([16, 16], "AAA") as [i, j]:
                tir.reads([])
                tir.writes(AA[i, j])
                AAA = tir.match_buffer(AA[i, j], ())
                AAA[()] = 1.0
            tir.evaluate(B0.data)
            tir.evaluate(B1.data)


@tvm.script.tir
def opaque_block_func() -> None:
    with tir.block([], "root"):
        A = tir.alloc_buffer((16, 16), "float32")
        B = tir.alloc_buffer((16, 16), "float32")
        tir.reads([])
        tir.writes([])
        # Need add read/write region manually to avoid triggering block access region detector
        for i in range(0, 16):
            with tir.block([]):
                tir.reads(A[i, 0:16])
                tir.writes([B[i, 0:16]])
                for j in range(0, 16):
                    with tir.block([]):
                        tir.reads(A[i, j])
                        tir.writes(B[i, j])
                        B[i, j] = A[i, j] + 1.0


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


def test_match_buffer():
    root_block = match_buffer_func.body.block
    block = root_block.body.body.body.block
    block_inner = block.body[0].body.body.block
    alloc_buffers = func.body.block.alloc_buffers
    buffer_var_map = {buf.data: buf for buf in alloc_buffers}

    # Check inner block AAA
    ret = tir.analysis.get_block_access_region(block_inner, buffer_var_map)
    tvm.ir.assert_structural_equal(block_inner.reads, ret[0])
    tvm.ir.assert_structural_equal(block_inner.writes, ret[1])

    # Check block
    ret = tir.analysis.get_block_access_region(block, buffer_var_map)
    tvm.ir.assert_structural_equal(block.writes, ret[1])
    # B is opaque access
    tvm.ir.assert_structural_equal(block.reads, ret[2])


if __name__ == "__main__":
    test_block_access_region_detector()
    test_opaque_block()
    test_match_buffer()
