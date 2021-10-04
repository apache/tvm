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
from tvm.script import tir as T


@T.prim_func
def buffer_load_store_func(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.match_buffer(b, (128, 128), "float32")
    C = T.alloc_buffer((128, 128), "float32")
    D = T.alloc_buffer((128, 128), "float32")
    with T.block([128, 128]) as [i, j]:
        A[i, j] = T.float32(0)
    with T.block([32, 32, T.reduce_axis(0, 32)]) as [i, j, k]:
        with T.init():
            for ii, jj in T.grid(4, 4):
                B[i * 4 + ii, j * 4 + jj] = A[i * 4 + ii, j * 4 + jj]
        for ii, jj in T.grid(4, 4):
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += C[i * 4 + ii, k * 4 + kk]
            for kk in range(0, 4):
                B[i * 4 + ii, j * 4 + jj] += D[j * 4 + jj, k * 4 + kk] * C[i * 4 + ii, k * 4 + kk]


@T.prim_func
def buffer_opaque_access(b: T.handle, c: T.handle) -> None:
    B = T.match_buffer(b, [16, 16], "float32")
    C = T.match_buffer(c, [16, 16], "float32")

    with T.block([]):
        T.reads([])
        T.writes(B[0:16, 0:16])
        A = T.allocate(256, "float32", "global")
        for i, j in T.grid(16, 16):
            T.store(A, i * 16 + j, 1)
        for i in range(0, 16):
            for j in range(0, 16):
                T.evaluate(T.load("float32", A, i * 16 + j))
            for j in range(0, 16):
                T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, T.float32(0), dtype="handle"))

    for i, j in T.grid(16, 16):
        with T.block([16, 16]) as [vi, vj]:
            T.bind(vi, i)
            T.bind(vj, j)
            C[vi, vj] = B[vi, vj]


@T.prim_func
def lca_is_func_root(a: T.handle) -> None:
    A = T.match_buffer(a, [0, 0], "float32")
    A.data[0] = 1.0


@T.prim_func
def match_buffer_func(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128), "float32")
    B = T.match_buffer(b, (128, 128), "float32")
    with T.block([8, 8], "block") as [vi, vj]:
        T.reads(B[vi * 16 + 2 : vi * 16 + 12, vj * 16 + 2 : vj * 16 + 16])
        T.writes(A[vi * 16 : vi * 16 + 16, vj * 16 : vj * 16 + 16])
        B0 = T.match_buffer(B[vi * 16 + 2 : vi * 16 + 6, vj * 16 + 2 : vj * 16 + 6], (4, 4))
        B1 = T.match_buffer(B[vi * 16 + 8 : vi * 16 + 12, vj * 16 + 8 : vj * 16 + 16], (4, 8))
        with T.block([16, 16], "AAA") as [i, j]:
            AA = T.match_buffer(A[i, j], ())
            AA[()] = 1.0
        T.evaluate(B0.data)
        T.evaluate(B1.data)


def test_buffer_load_store():
    func = buffer_load_store_func
    A, B = [func.buffer_map[x] for x in func.params]
    C, D = func.body.block.alloc_buffers
    lca = tir.analysis.detect_buffer_access_lca(func)

    # LCA of Buffer A is root
    root_block = func.body.block
    assert lca[A] == func.body.block

    # LCA of Buffer B is reduction block
    reduce_block = root_block.body[1].body.body.body.block
    assert lca[B] == reduce_block

    # LCA of Buffer C is the second loop kk
    loop_jj = reduce_block.body.body
    assert lca[C] == loop_jj

    # LCA of Buffer D is loop jj
    loop_kk = loop_jj.body[1]
    assert lca[D] == loop_kk


def test_opaque_access():
    func = buffer_opaque_access
    B, C = [func.buffer_map[x] for x in func.params]
    lca = tir.analysis.detect_buffer_access_lca(func)

    # Cannot detect buffer A since it is define by low-level Allocate

    # LCA of Buffer B is root
    root_block = func.body.block
    assert lca[B] == func.body.block

    # LCA of Buffer C is the correspond block
    assert lca[C] == root_block.body[1].body.body.block


def test_lca_func_root():
    func = lca_is_func_root
    (A,) = [func.buffer_map[x] for x in func.params]
    lca = tir.analysis.detect_buffer_access_lca(func)
    assert lca[A] is None


def test_match_buffer():
    func = match_buffer_func
    A, B = [func.buffer_map[x] for x in func.params]
    lca = tir.analysis.detect_buffer_access_lca(func)

    root_block = func.body.block
    block = root_block.body.body.body.block
    block_inner = block.body[0].body.body.block

    # LCA of Buffer C is the inner block
    assert lca[A] == block_inner

    # LCA of Buffer C is the main block
    assert lca[B] == block


if __name__ == "__main__":
    test_buffer_load_store()
    test_opaque_access()
    test_lca_func_root()
    test_match_buffer()
