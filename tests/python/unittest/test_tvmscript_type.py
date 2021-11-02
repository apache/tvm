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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys
import pytest
import tvm
from tvm import tir
from tvm.script import tir as T

"""
This module tests the type of
T.prim_func, T.handle, T.match_buffer, T.block
T.reads, T.writes, T.alloc_buffer, T.serial
T.block_attr, T.float32
"""


@pytest.mark.mypy_testing
@tvm.script.ir_module
class Module:
    @T.prim_func
    def element_wise_storage_align(a: T.handle, c: T.handle) -> None:
        C = T.match_buffer(c, [128, 128], elem_offset=0, align=128, offset_factor=1)
        A = T.match_buffer(a, [128, 128], elem_offset=0, align=128, offset_factor=1)
        # body
        with T.block("root"):
            T.reads([])
            T.writes([])
            B = T.alloc_buffer([128, 128], elem_offset=0, align=128, offset_factor=1)
            for i0 in T.serial(0, 128):
                for ax1 in T.serial(0, 128):
                    with T.block("B"):
                        vi, vj = T.axis.remap("SS", [i0, ax1])
                        T.reads([A[vi, vj]])
                        T.writes([B[vi, vj]])
                        T.block_attr({"buffer_dim_align": [[0, 0, 128, 127]]})
                        B[vi, vj] = A[vi, vj] * T.float32(2)
                for i1 in T.serial(0, 128):
                    with T.block("C"):
                        vi_1, vj_1 = T.axis.remap("SS", [i0, i1])
                        T.reads([B[vi_1, vj_1]])
                        T.writes([C[vi_1, vj_1]])
                        C[vi_1, vj_1] = B[vi_1, vj_1] + T.float32(1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
