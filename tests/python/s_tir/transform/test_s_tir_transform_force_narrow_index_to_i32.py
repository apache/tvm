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
import tvm.testing
from tvm.s_tir import dlight as dl
from tvm.script import tirx as T
from tvm.testing import env


def _narrow(func):
    mod = tvm.IRModule.from_expr(func)
    return tvm.s_tir.transform.ForceNarrowIndexToInt32()(mod)["main"]


def test_block():
    @T.prim_func(private=True, s_tir=True)
    def before(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int64(16)):
            for j in T.serial(0, T.int64(8)):
                with T.sblock():
                    vi = T.axis.spatial(T.int64(128), i * T.int64(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.serial(0, T.int32(16)):
            for j in T.serial(0, T.int32(8)):
                with T.sblock():
                    vi = T.axis.spatial(T.int32(128), i * T.int32(8) + j)
                    B[vi] = A[vi] + T.float32(1)

    tvm.ir.assert_structural_equal(_narrow(before), expected)


def test_block_iters_used_only_in_regions():
    """Blockized blocks use their iterators only in access and match_buffer regions."""

    @T.prim_func(private=True, s_tir=True)
    def before(
        A: T.Buffer((T.int64(16), T.int64(16)), "float32"),
        B: T.Buffer((T.int64(16), T.int64(16)), "float32"),
    ):
        for i_o, j_o in T.grid(T.int64(2), T.int64(2)):
            with T.sblock("tile_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_o, j_o])
                T.reads(
                    A[
                        vi_o * T.int64(8) : vi_o * T.int64(8) + T.int64(8),
                        vj_o * T.int64(8) : vj_o * T.int64(8) + T.int64(8),
                    ]
                )
                T.writes(
                    B[
                        vi_o * T.int64(8) : vi_o * T.int64(8) + T.int64(8),
                        vj_o * T.int64(8) : vj_o * T.int64(8) + T.int64(8),
                    ]
                )
                A_tile = T.match_buffer(
                    A[
                        vi_o * T.int64(8) : vi_o * T.int64(8) + T.int64(8),
                        vj_o * T.int64(8) : vj_o * T.int64(8) + T.int64(8),
                    ],
                    (T.int64(8), T.int64(8)),
                    offset_factor=1,
                )
                B_tile = T.match_buffer(
                    B[
                        vi_o * T.int64(8) : vi_o * T.int64(8) + T.int64(8),
                        vj_o * T.int64(8) : vj_o * T.int64(8) + T.int64(8),
                    ],
                    (T.int64(8), T.int64(8)),
                    offset_factor=1,
                )
                for i_i, j_i in T.grid(T.int64(8), T.int64(8)):
                    with T.sblock("tile"):
                        vi_i, vj_i = T.axis.remap("SS", [i_i, j_i])
                        B_tile[vi_i, vj_i] = A_tile[vi_i, vj_i] + T.float32(1)

    @T.prim_func(private=True, s_tir=True)
    def expected(A: T.Buffer((16, 16), "float32"), B: T.Buffer((16, 16), "float32")):
        for i_o, j_o in T.grid(2, 2):
            with T.sblock("tile_o"):
                vi_o, vj_o = T.axis.remap("SS", [i_o, j_o])
                T.reads(A[vi_o * 8 : vi_o * 8 + 8, vj_o * 8 : vj_o * 8 + 8])
                T.writes(B[vi_o * 8 : vi_o * 8 + 8, vj_o * 8 : vj_o * 8 + 8])
                A_tile = T.match_buffer(
                    A[vi_o * 8 : vi_o * 8 + 8, vj_o * 8 : vj_o * 8 + 8], (8, 8), offset_factor=1
                )
                B_tile = T.match_buffer(
                    B[vi_o * 8 : vi_o * 8 + 8, vj_o * 8 : vj_o * 8 + 8], (8, 8), offset_factor=1
                )
                for i_i, j_i in T.grid(8, 8):
                    with T.sblock("tile"):
                        vi_i, vj_i = T.axis.remap("SS", [i_i, j_i])
                        B_tile[vi_i, vj_i] = A_tile[vi_i, vj_i] + T.float32(1)

    tvm.ir.assert_structural_equal(_narrow(before), expected)


def test_fail_on_buffer_param():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((128,), "int64"), B: T.Buffer((128,), "int64")):
        for i in T.serial(0, 16):
            for j in T.serial(0, 8):
                with T.sblock():
                    vi = T.axis.spatial(128, i * 8 + j)
                    B[vi] = A[vi] + T.int64(1)

    with pytest.raises(RuntimeError):
        _narrow(func)


def test_fail_on_block_alloc_buffer():
    @T.prim_func(private=True, s_tir=True)
    def func(A: T.Buffer((128,), "int32"), B: T.Buffer((128,), "int32")):
        C = T.sblock_alloc_buffer((128,), "int64")
        for i in T.serial(0, 16):
            for j in T.serial(0, 8):
                with T.sblock():
                    vi = T.axis.spatial(128, i * 8 + j)
                    C[vi] = T.cast(A[vi], "int64") + T.int64(1)
        for i in T.serial(0, 16):
            for j in T.serial(0, 8):
                with T.sblock():
                    vi = T.axis.spatial(128, i * 8 + j)
                    B[vi] = T.cast(C[vi] + T.int64(1), "int32")

    with pytest.raises(RuntimeError):
        _narrow(func)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm for the host module")
def test_metal_simdgroup_matmul_builds():
    """Narrowing a DLight-scheduled Metal matmul keeps its tensorized blocks consistent."""

    @T.prim_func(s_tir=True)
    def main(
        var_A: T.handle, B: T.Buffer((T.int64(256), T.int64(256)), "float16"), var_C: T.handle
    ):
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(256)), "float16")
        C = T.match_buffer(var_C, (T.int64(1), n, T.int64(256)), "float16")
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(256), T.int64(256)):
            with T.sblock("NT_matmul"):
                v0, v1, v2, vk = T.axis.remap("SSSR", [i0, i1, i2, k])
                with T.init():
                    C[v0, v1, v2] = T.float16(0)
                C[v0, v1, v2] = C[v0, v1, v2] + A[v0, v1, vk] * B[v2, vk]

    target = tvm.target.Target("metal", host="llvm")
    with target:
        mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(tvm.IRModule({"main": main}))
    assert "metal.simdgroup" in mod.script()
    mod = tvm.s_tir.transform.ForceNarrowIndexToInt32()(mod)
    tvm.tirx.build(mod, target=target)


if __name__ == "__main__":
    tvm.testing.main()
