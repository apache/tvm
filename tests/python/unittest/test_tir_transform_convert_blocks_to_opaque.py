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
from tvm.script import ty


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func)
    mod = tvm.tir.transform.ConvertBlocksToOpaque()(mod)
    mod = tvm.tir.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed)


@tvm.script.tir
def elementwise_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with tir.block([]):
            tir.reads(A[i, 0:16])
            tir.writes(C[i, 0:16])
            B = tir.alloc_buffer((16, 16), "float32")
            for j in range(0, 16):
                with tir.block([16, 16]) as [vi, vj]:
                    tir.bind(vi, i)
                    tir.bind(vj, j)
                    B[vi, vj] = A[vi, vj] + 1.0
            for j in range(0, 16):
                with tir.block([16, 16]) as [vi, vj]:
                    tir.bind(vi, i)
                    tir.bind(vj, j)
                    C[vi, vj] = B[vi, vj] * 2.0


@tvm.script.tir
def substituted_elementwise_func(a: ty.handle, c: ty.handle) -> None:
    A = tir.match_buffer(a, (16, 16), "float32")
    C = tir.match_buffer(c, (16, 16), "float32")
    for i in range(0, 16):
        with tir.block([]):
            tir.reads(A[i, 0:16])
            tir.writes(C[i, 0:16])
            B = tir.alloc_buffer([16, 16], "float32")
            for j in range(0, 16):
                with tir.block() as []:
                    tir.reads(A[i, j])
                    tir.writes(B[i, j])
                    B[i, j] = A[i, j] + 1.0
            for j in range(0, 16):
                with tir.block() as []:
                    tir.reads(B[i, j])
                    tir.writes(C[i, j])
                    C[i, j] = B[i, j] * 2.0


def test_elementwise():
    _check(elementwise_func, substituted_elementwise_func)


if __name__ == "__main__":
    test_elementwise()
