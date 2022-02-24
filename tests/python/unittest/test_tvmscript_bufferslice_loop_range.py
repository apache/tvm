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
from tvm.script import tir as T


@T.prim_func
def segment_sum(A_ptr: T.handle, B_ptr: T.handle, indptr_ptr: T.handle, n: T.int32, m: T.int32) -> None:
    A = T.match_buffer(A_ptr, [m], dtype="float32")
    B = T.match_buffer(B_ptr, [n], dtype="float32")
    indptr = T.match_buffer(indptr_ptr, [n + 1], dtype="int32")
    # body
    # with T.block("root")
    for i in T.serial(n):
        with T.block("outer"):
            vi = T.axis.spatial(n, i)
            T.reads(indptr[i : i + 2], B[vi], A[indptr[i] : indptr[i + 1]])
            T.writes(B[vi])
            for j in T.serial(indptr[i], indptr[i + 1]):
                with T.block("inner"):
                    vj = T.axis.reduce(m, j)
                    T.reads(B[vi], A[vj])
                    T.writes(B[vi])
                    with T.init():
                        B[vi] = T.float32(0)
                    B[vi] = B[vi] + A[vj]


def test_parse_segment_sum():
    tvm.ir.assert_structural_equal(segment_sum, tvm.script.from_source(segment_sum.script()))


if __name__ == "__main__":
    test_parse_segment_sum()
