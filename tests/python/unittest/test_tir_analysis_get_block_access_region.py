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


if __name__ == "__main__":
    test_block_access_region_detector()
