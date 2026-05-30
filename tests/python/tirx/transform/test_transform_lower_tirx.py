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
from tvm.script import tirx as Tx
from tvm.tirx.function import PrimFunc
from tvm.tirx.layout import laneid, warpid, wg_local_layout
from tvm.tirx.stmt import ExecScopeStmt
from tvm.tirx.stmt_functor import post_order_visit
from tvm.tirx.transform import LowerTIRx, StmtSimplify


def _contains_exec_scope(mod):
    found = [False]

    def _visit(node):
        if isinstance(node, ExecScopeStmt):
            found[0] = True

    for _gv, base_func in mod.functions.items():
        if isinstance(base_func, PrimFunc):
            post_order_visit(base_func.body, _visit)
    return found[0]


def compare(before, after, transform):
    """Compare lowered output against expected ``after`` IR."""
    if isinstance(before, PrimFunc):
        before = tvm.IRModule({"main": before})
    if isinstance(after, PrimFunc):
        after = tvm.IRModule({"main": after})
    assert isinstance(before, tvm.IRModule)
    assert isinstance(after, tvm.IRModule)
    with tvm.target.Target("cuda"):
        lowered = transform()(before)
        lowered.show()
        assert not _contains_exec_scope(lowered)
        tvm.ir.assert_structural_equal(lowered, after, map_free_vars=False)


def _int_pair(side, axis):
    return tuple(int(x) for x in side[axis])


def _int_triple(side, axis):
    return tuple(int(x) for x in side[axis])


L_LANE = Tx.TileLayout(Tx.S[32 : 1 @ laneid])


def test_lower_view_get():
    @Tx.prim_func(private=True)
    def before1(in_buf: Tx.Buffer(64, "float32"), out: Tx.Buffer(64, "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            Tx.warp_id([1])
            lane_id = Tx.lane_id([32])
            with Tx.thread():
                A = Tx.alloc_buffer(
                    [2], dtype="float16", scope="local", layout=Tx.TileLayout(Tx.S[2:1])
                )
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                with Tx.warp():
                    B = A.view(64, layout=B_layout)
                    with Tx.thread():
                        A_local = B.local(2)
                        for i in Tx.vectorized(2):
                            A_local[i] = Tx.float32(in_buf[lane_id * 2 + i])
                with Tx.warp():
                    B = A.view(64, layout=B_layout)
                    with Tx.thread():
                        A_local = B.local(2)
                        for i in Tx.vectorized(2):
                            out[lane_id * 2 + i] = Tx.float32(A_local[i])

    @Tx.prim_func(private=True)
    def after1(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (64,), layout=None)
        out = Tx.match_buffer(out_handle, (64,), layout=None)
        out_1 = Tx.decl_buffer((64,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((64,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        v: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(v)
        A = Tx.alloc_local((2,), "float16", layout=None)
        B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        for i in Tx.vectorized(2):
            A_local[i] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2 + i])
        B_1 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
        A_local_1 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        for i in Tx.vectorized(2):
            out_1[threadIdx_x * 2 + i] = Tx.Cast("float32", A_local_1[i])

    compare(before1, after1, LowerTIRx)

    @Tx.prim_func(private=True)
    def before2(
        in_buf: Tx.Buffer((16, 16), "float32"), out: Tx.Buffer((16, 16), "float32")
    ) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            Tx.warp_id([1])
            lane_id = Tx.lane_id([32])
            with Tx.thread():
                atom = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
                tile = Tx.TileLayout(Tx.S[(2, 2) : (2, 1)])
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
                A = Tx.alloc_buffer(
                    [4, 2], dtype="float32", scope="local", layout=atom.tile(tile, (2, 2), (1, 2))
                )
                B_layout = warp_atom.tile(tile, (2, 2), (8, 8))
                with Tx.warp():
                    B = A.view(16, 16, layout=B_layout)
                    with Tx.thread():
                        A_local = B.local(2, 2, 2)
                        for i in Tx.unroll(4):
                            for j in Tx.vectorized(2):
                                A_local[i // 2, i % 2, j] = in_buf[
                                    i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j
                                ]
                with Tx.warp():
                    B = A.view(16, 16, layout=B_layout)
                    with Tx.thread():
                        A_local = B.local(8)
                        for i in Tx.vectorized(2):
                            out[
                                lane_id // 4 * 8 + i // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i % 2
                            ] = A_local[i]

    @Tx.prim_func(private=True)
    def after2(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (16, 16), layout=None)
        out = Tx.match_buffer(out_handle, (16, 16), layout=None)
        out_1 = Tx.decl_buffer((256,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((256,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        v: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(v)
        A = Tx.alloc_local((8,), layout=None)
        B = Tx.decl_buffer((256,), data=A.data, scope="local", layout=None)
        A_local = Tx.decl_buffer((8,), data=A.data, scope="local", layout=None)
        for i in Tx.unroll(4):
            for j in Tx.vectorized(2):
                A_local[i * 2 + j] = in_buf_1[
                    i // 2 * 128 + threadIdx_x // 4 * 16 + i % 2 * 8 + j + threadIdx_x % 4
                ]
        B_1 = Tx.decl_buffer((256,), data=A.data, scope="local", layout=None)
        A_local_1 = Tx.decl_buffer((8,), data=A.data, scope="local", layout=None)
        for i in Tx.vectorized(2):
            out_1[threadIdx_x // 4 * 128 + threadIdx_x % 4 * 18 + i] = A_local_1[i]

    compare(before2, after2, LowerTIRx)

    @Tx.prim_func(private=True)
    def before3_wgmma_layout(
        in_buf: Tx.Buffer((128, 128), "float32"), out: Tx.Buffer((128, 128), "float32")
    ) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            wg_id = Tx.warpgroup_id([2])
            warp_id_in_wg = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
            with Tx.thread():
                atom = Tx.TileLayout(Tx.S[1, 2])
                warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
                tile = Tx.TileLayout(Tx.S[(2, 128 // 8) : (1, 2)])
                warp_layout = warp_atom.tile(tile, (2, 128 // 8), (8, 8))
                L_warp = Tx.TileLayout(Tx.S[8 : 1 @ warpid])
                layout = warp_layout.tile(L_warp, (8, 1), (16, 128))
                acc = Tx.alloc_buffer(
                    [64],
                    dtype="float32",
                    scope="local",
                    layout=atom.tile(tile, (2, 128 // 8), (1, 2)),
                )
                with Tx.cta():
                    A = acc.view(128, 128, layout=layout)
                    with Tx.thread():
                        acc_local = A.local(16, 2, 2, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in Tx.serial(128 // 8):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    acc_local[i, j, vec] = in_buf[
                                        wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                        i * 8 + lane_id % 4 * 2 + vec,
                                    ]
                with Tx.cta():
                    A = acc.view(128, 128, layout=layout)
                    with Tx.thread():
                        acc_local = A.local(64, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
                        for i in Tx.serial(128 // 8):
                            for j in Tx.unroll(2):
                                for vec in Tx.vectorized(2):
                                    out[
                                        wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                        i * 8 + lane_id % 4 * 2 + vec,
                                    ] = acc_local[i * 4 + j * 2 + vec]

    @Tx.prim_func(private=True)
    def after3_wgmma_layout(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (128, 128), layout=None)
        out = Tx.match_buffer(out_handle, (128, 128), layout=None)
        out_1 = Tx.decl_buffer((16384,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((16384,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        wg_id: Tx.let[Tx.int32] = warp_id_in_cta // 4
        warp_id_in_wg: Tx.let[Tx.int32] = warp_id_in_cta % 4
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        acc = Tx.alloc_local((64,), layout=None)
        B = Tx.decl_buffer((16384,), data=acc.data, scope="local", layout=None)
        acc_local = Tx.decl_buffer((64,), data=acc.data, scope="local", layout=None)
        for i in range(16):
            for j in Tx.unroll(2):
                for vec in Tx.vectorized(2):
                    acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec] = in_buf_1[
                        warp_id_in_cta * 2048
                        + j * 1024
                        + threadIdx_x % 32 // 4 * 128
                        + i * 8
                        + threadIdx_x % 4 * 2
                        + vec
                    ]
        B_1 = Tx.decl_buffer((16384,), data=acc.data, scope="local", layout=None)
        acc_local_1 = Tx.decl_buffer((64,), data=acc.data, scope="local", layout=None)
        for i in range(16):
            for j in Tx.unroll(2):
                for vec in Tx.vectorized(2):
                    out_1[
                        warp_id_in_cta * 2048
                        + j * 1024
                        + threadIdx_x % 32 // 4 * 128
                        + i * 8
                        + threadIdx_x % 4 * 2
                        + vec
                    ] = acc_local_1[i % 8 * 8 + j * 4 + i // 8 * 2 + vec]

    compare(before3_wgmma_layout, after3_wgmma_layout, LowerTIRx)

    @Tx.prim_func(private=True)
    def before4_multi_view_get(
        in_buf: Tx.Buffer(64, "float32"), out: Tx.Buffer(64, "float32")
    ) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            Tx.warp_id([1])
            lane_id = Tx.lane_id([32])
            with Tx.thread():
                A = Tx.alloc_buffer(
                    [2], dtype="float16", scope="local", layout=Tx.TileLayout(Tx.S[2:1])
                )
                B_layout = A.layout.tile(L_LANE, (32,), (2,))
                with Tx.warp():
                    B = A.view(64, layout=B_layout)
                    B_1 = A.view(64, layout=B_layout)
                    with Tx.thread():
                        A_local = B.local(2)
                        A_local[0] = Tx.float32(in_buf[lane_id * 2])
                        A_local_1 = B_1.local(2)
                        A_local_1[1] = Tx.float32(in_buf[lane_id * 2 + 1])
                "\n                write A into out\n                "
                with Tx.warp():
                    B = A.view(64, layout=B_layout)
                    B_1 = A.view(64, layout=B_layout)
                    with Tx.thread():
                        A_local = B.local(2)
                        out[lane_id * 2] = Tx.float32(A_local[0])
                        A_local_1 = B_1.local(2)
                        out[lane_id * 2 + 1] = Tx.float32(A_local_1[1])

    @Tx.prim_func(private=True)
    def after4_multi_view_get(in_buf_handle: Tx.handle, out_handle: Tx.handle):
        in_buf = Tx.match_buffer(in_buf_handle, (64,), layout=None)
        out = Tx.match_buffer(out_handle, (64,), layout=None)
        out_1 = Tx.decl_buffer((64,), data=out.data, layout=None)
        in_buf_1 = Tx.decl_buffer((64,), data=in_buf.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        v: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(v)
        A = Tx.alloc_local((2,), "float16", layout=None)
        B = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
        B_1 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
        A_local = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        A_local[0] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2])
        A_local_1 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        A_local_1[1] = Tx.Cast("float16", in_buf_1[threadIdx_x * 2 + 1])
        B_2 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
        B_3 = Tx.decl_buffer((64,), "float16", data=A.data, scope="local", layout=None)
        A_local_2 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        out_1[threadIdx_x * 2] = Tx.Cast("float32", A_local_2[0])
        A_local_3 = Tx.decl_buffer((2,), "float16", data=A.data, scope="local", layout=None)
        out_1[threadIdx_x * 2 + 1] = Tx.Cast("float32", A_local_3[1])

    compare(before4_multi_view_get, after4_multi_view_get, LowerTIRx)


def test_lower_scope_id():
    @Tx.prim_func(private=True)
    def before1() -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5])
            tx = Tx.thread_id([32])
            with Tx.thread():
                Tx.evaluate(bx + by + bz + tx)

    @Tx.prim_func(private=True)
    def after1() -> None:
        blockIdx_x = Tx.launch_thread("blockIdx.x", 3)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 32)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(bx + by + bz + tx)

    compare(before1, after1, LowerTIRx)

    @Tx.prim_func(private=True)
    def before2() -> None:
        with Tx.kernel():
            cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 2])
            bx, by, bz = Tx.cta_id([8, 8, 8])
            warp_id = Tx.warp_id([4])
            lane_id = Tx.lane_id([32])
            with Tx.thread():
                Tx.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    @Tx.prim_func(private=True)
    def after2() -> None:
        clusterCtaIdx_x = Tx.launch_thread("clusterCtaIdx.x", 2)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 8)
        clusterCtaIdx_y = Tx.launch_thread("clusterCtaIdx.y", 2)
        clusterCtaIdx_z = Tx.launch_thread("clusterCtaIdx.z", 2)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 8)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 8)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        cbx: Tx.let[Tx.int32] = clusterCtaIdx_x
        cby: Tx.let[Tx.int32] = clusterCtaIdx_y
        cbz: Tx.let[Tx.int32] = clusterCtaIdx_z
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    compare(before2, after2, LowerTIRx)

    @Tx.prim_func(private=True)
    def before3() -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12])
            cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 1])
            clx, cly, clz = Tx.cluster_id([4, 5, 12])
            wg_id = Tx.warpgroup_id([3])
            warp_id_in_wg = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
            tid_in_wg = Tx.thread_id_in_wg([128])
            with Tx.cta():
                with Tx.warpgroup():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)
                        Tx.evaluate(wg_id + warp_id_in_wg + lane_id + tid_in_wg)

    @Tx.prim_func(private=True)
    def after3() -> None:
        clusterCtaIdx_x = Tx.launch_thread("clusterCtaIdx.x", 2)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 12)
        clusterCtaIdx_y = Tx.launch_thread("clusterCtaIdx.y", 2)
        clusterCtaIdx_z = Tx.launch_thread("clusterCtaIdx.z", 1)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 8)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 384)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 10)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        cbx: Tx.let[Tx.int32] = clusterCtaIdx_x
        cby: Tx.let[Tx.int32] = clusterCtaIdx_y
        cbz: Tx.let[Tx.int32] = clusterCtaIdx_z
        clx: Tx.let[Tx.int32] = Tx.ptx.fetch_register(32, "clusterid.x")
        cly: Tx.let[Tx.int32] = Tx.ptx.fetch_register(32, "clusterid.y")
        clz: Tx.let[Tx.int32] = Tx.ptx.fetch_register(32, "clusterid.z")
        wg_id: Tx.let[Tx.int32] = warp_id_in_cta // 4
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta % 4
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        tid_in_wg: Tx.let[Tx.int32] = threadIdx_x % 128
        Tx.evaluate(bx + by + bz)
        Tx.evaluate(cbx + cby + cbz)
        Tx.evaluate(clx + cly + clz)
        Tx.evaluate(wg_id + warp_id + lane_id + tid_in_wg)

    compare(before3, after3, LowerTIRx)


def test_lower_scope_id2():
    @Tx.inline
    def func(warp_id, tx):
        with Tx.cta():
            wg_id = Tx.warpgroup_id([2])
            with Tx.thread():
                Tx.evaluate(wg_id + warp_id + tx)

    @Tx.prim_func(private=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5])
            warp_id = Tx.warp_id([8])
            tx = Tx.thread_id([256])
            func(warp_id, tx)

    @Tx.prim_func(private=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 3)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        wg_id: Tx.let[Tx.int32] = warp_id_in_cta // 4
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(wg_id + warp_id + tx)

    compare(before, after, LowerTIRx)


def test_lower_scope_id3():
    @Tx.prim_func(private=True)
    def before():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([3, 4, 5])
            warp_id = Tx.warp_id([4])
            tx = Tx.thread_id([128])
            with Tx.cta():
                with Tx.thread():
                    Tx.evaluate(bx + by + bz + warp_id + tx)
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([6, 7, 8])
            warp_id = Tx.warp_id([8])
            tx = Tx.thread_id([256])
            with Tx.cta():
                with Tx.thread():
                    Tx.evaluate(bx + by + bz + warp_id + tx)

    @Tx.prim_func(private=True)
    def after():
        with Tx.launch_thread("blockIdx.x", 3) as blockIdx_x:
            threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
            blockIdx_y = Tx.launch_thread("blockIdx.y", 4)
            blockIdx_z = Tx.launch_thread("blockIdx.z", 5)
            warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
                Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
            )
            bx: Tx.let[Tx.int32] = blockIdx_x
            by: Tx.let[Tx.int32] = blockIdx_y
            bz: Tx.let[Tx.int32] = blockIdx_z
            warp_id: Tx.let[Tx.int32] = warp_id_in_cta
            tx: Tx.let[Tx.int32] = threadIdx_x
            Tx.evaluate(bx + by + bz + warp_id + tx)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 6)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 256)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 7)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 8)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta
        tx: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(bx + by + bz + warp_id + tx)

    compare(before, after, LowerTIRx)


def test_lower_layout():
    @Tx.prim_func(private=True)
    def before(A: Tx.Buffer((128, 32), "float16")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            Tx.warp_id([4])
            Tx.lane_id([32])
            tid = Tx.thread_id([128])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(
                    [128, 32], dtype="float16", scope="shared", layout=Tx.SwizzleLayout(3, 3, 3)
                )
                with Tx.thread():
                    thread_col = Tx.meta_var(4)
                    thread_row = Tx.meta_var(32)
                    for tile in Tx.serial(128 // thread_row):
                        row = Tx.meta_var(tile * thread_row + tid // thread_col)
                        col = Tx.meta_var(tid % thread_col * 8)
                        for vec in Tx.vectorized(8):
                            A_smem[row, col + vec] = A[bx * 128 + row, col + vec]

    @Tx.prim_func(private=True)
    def after(A_handle: Tx.handle) -> None:
        A = Tx.match_buffer(A_handle, (128, 32), "float16", layout=None)
        A_1 = Tx.decl_buffer((4096,), "float16", data=A.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        v: Tx.let[Tx.int32] = warp_id_in_cta
        v_1: Tx.let[Tx.int32] = threadIdx_x % 32
        tid: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(v)
        Tx.evaluate(v_1)
        A_smem = Tx.alloc_shared((4096,), "float16", layout=None)
        for tile in range(4):
            for vec in Tx.vectorized(8):
                A_smem[
                    Tx.shift_left(
                        Tx.bitwise_xor(
                            tile * 128 + threadIdx_x,
                            Tx.shift_right(Tx.bitwise_and(tile * 128 + threadIdx_x, 56), 3),
                        ),
                        3,
                    )
                    + vec
                ] = A_1[tile * 1024 + threadIdx_x * 8 + vec]

    compare(before, after, LowerTIRx)


def test_lower_opcall_fail():
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            Tx.warp_id([1])
            Tx.lane_id([32])
            with Tx.cta():
                A_smem = Tx.alloc_buffer([64], dtype="float32", scope="shared")
                Tx.copy(A[0:64], A_smem[0:64])
                for i in range(10):
                    Tx.fill(A_smem[0:64], Tx.float32(0))
                    Tx.gemm(A_smem, A_smem, A_smem, A_smem)
                Tx.copy(A_smem[0:64], A[0:64])

    with pytest.raises(Exception):
        LowerTIRx()(tvm.IRModule({"main": test}))


def test_lower_decl_buffer_access_ptr():
    @Tx.prim_func(private=True)
    def before():
        with Tx.kernel():
            Tx.cta_id([1])
            Tx.thread_id([128])
            with Tx.cta():
                buf = Tx.alloc_buffer([1024], "uint8", scope="shared.dyn")
                A = Tx.decl_buffer([128], "float16", buf.data, elem_offset=32)
                with Tx.thread():
                    Tx.evaluate(A.access_ptr("rw", offset=A.elem_offset_of([64])))

    @Tx.prim_func(private=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        v: Tx.let[Tx.int32] = blockIdx_x
        v_1: Tx.let[Tx.int32] = threadIdx_x
        Tx.evaluate(v)
        Tx.evaluate(v_1)
        buf = Tx.alloc_buffer((1024,), "uint8", scope="shared.dyn", layout=None)
        A = Tx.decl_buffer(
            (128,), "float16", data=buf.data, elem_offset=32, scope="shared.dyn", layout=None
        )
        Tx.tvm_access_ptr(
            Tx.type_annotation("float16"), buf.data, Tx.Add(32, 64), Tx.Sub(128, 64), 3
        )

    compare(before, after, LowerTIRx)


def test_lower_separate_scope_id_def():
    @Tx.prim_func(private=True)
    def before():
        with Tx.kernel():
            Tx.cta_id([1])
            with Tx.cta():
                tx = Tx.thread_id([128])
                if Tx.filter(tx, tx == 0):
                    with Tx.thread():
                        Tx.evaluate(tx)

    @Tx.prim_func(private=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        tx: Tx.let[Tx.int32] = threadIdx_x
        v: Tx.let[Tx.int32] = blockIdx_x
        Tx.evaluate(v)
        if tx == 0:
            Tx.evaluate(tx)

    compare(before, after, LowerTIRx)


def test_lower_exec_context_infers_plain_predicate_for_dispatch():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_plain_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            warp_id = Tx.warp_id([4])
            lane_id = Tx.lane_id([32])
            with Tx.cta():
                if (warp_id == 0) & (lane_id == 0):
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "thread"
    assert _int_pair(seen[0]["inter"], "laneid") == (1, 0)
    assert _int_pair(seen[0]["inter"], "warpid") == (1, 0)
    assert _int_pair(seen[0]["inter"], "cta_id") == (1, 0)
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_infers_warpgroup_range_predicate_for_dispatch():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_warpgroup_range_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([2])
            Tx.warp_id_in_wg([4])
            Tx.lane_id([32])
            with Tx.cta():
                if wg_id == 0:
                    with Tx.warpgroup():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)
                if (0 <= wg_id) & (wg_id < 1):
                    with Tx.warpgroup():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)
                with Tx.warpgroup((0 <= wg_id) & (wg_id < 1)):
                    Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 3
    for item in seen:
        assert item["scope_kind"] == "warpgroup"
        assert _int_pair(item["inter"], "wgid") == (1, 0)
        assert _int_pair(item["inter"], "cta_id") == (1, 0)
        assert _int_pair(item["intra"], "laneid") == (32, 0)
        assert _int_pair(item["intra"], "wid_in_wg") == (4, 0)


def test_lower_exec_context_tracks_cta_thread_range_predicate_for_dispatch():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_cta_thread_range_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            tid = Tx.thread_id([256])
            with Tx.cta():
                if (0 <= tid) & (tid < 128):
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "thread"
    assert _int_pair(seen[0]["inter"], "laneid") == (32, 0)
    assert _int_pair(seen[0]["inter"], "warpid") == (4, 0)
    assert _int_pair(seen[0]["inter"], "cta_id") == (1, 0)
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_cta_thread_single_warp_range_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_cta_thread_single_warp_range_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            tid = Tx.thread_id([256])
            with Tx.cta():
                with Tx.thread((34 <= tid) & (tid < 40)):
                    Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "thread"
    assert _int_pair(seen[0]["inter"], "laneid") == (6, 2)
    assert _int_pair(seen[0]["inter"], "warpid") == (1, 1)
    assert _int_pair(seen[0]["inter"], "cta_id") == (1, 0)
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_warpgroup_thread_range_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_warpgroup_thread_range_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([2])
            tid_in_wg = Tx.thread_id_in_wg([128])
            with Tx.cta():
                if wg_id == 1:
                    with Tx.warpgroup():
                        if (32 <= tid_in_wg) & (tid_in_wg < 64):
                            Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "warpgroup"
    assert _int_pair(seen[0]["inter"], "wgid") == (1, 1)
    assert _int_pair(seen[0]["inter"], "cta_id") == (1, 0)
    assert _int_pair(seen[0]["intra"], "laneid") == (32, 0)
    assert _int_pair(seen[0]["intra"], "wid_in_wg") == (1, 1)


def test_lower_exec_context_tracks_dependent_conjunctive_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_dependent_conjunctive_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([2])
            tid_in_wg = Tx.thread_id_in_wg([128])
            with Tx.cta():
                if ((32 <= tid_in_wg) & (tid_in_wg < 64)) & (wg_id == 1):
                    with Tx.warpgroup():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "warpgroup"
    assert _int_pair(seen[0]["inter"], "wgid") == (1, 1)
    assert _int_pair(seen[0]["inter"], "cta_id") == (1, 0)
    assert _int_pair(seen[0]["intra"], "laneid") == (32, 0)
    assert _int_pair(seen[0]["intra"], "wid_in_wg") == (1, 1)


def test_lower_exec_context_keeps_plain_predicate_condition():
    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([2])
            Tx.warp_id_in_wg([4])
            Tx.lane_id([32])
            with Tx.cta():
                if wg_id == 0:
                    Tx.evaluate(A[0])

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(tvm.IRModule({"main": before}))

    script = lowered.script(extra_config={"tirx.prefix": "Tx"})
    assert "if wg_id == 0:" in script
    assert "0 <= wg_id" not in script
    assert "wg_id < 1" not in script


def test_lower_exec_context_keeps_plain_scope_predicate_condition():
    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([2])
            Tx.warp_id_in_wg([4])
            Tx.lane_id([32])
            with Tx.cta():
                if wg_id == 0:
                    with Tx.warpgroup():
                        with Tx.thread():
                            A[0] = Tx.float32(1)

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(tvm.IRModule({"main": before}))

    script = lowered.script(extra_config={"tirx.prefix": "Tx"})
    assert "if wg_id == 0:" in script
    assert "0 <= wg_id" not in script
    assert "wg_id < 1" not in script


def test_simplify_uses_floor_div_scope_predicate_as_context_fact():
    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (16,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            wg_id = Tx.warpgroup_id([2])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
            with Tx.cta():
                if wg_id == 0:
                    with Tx.warpgroup():
                        with Tx.thread():
                            A[warp_id] = Tx.float32(lane_id)

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(tvm.IRModule({"main": before}))
        simplified = StmtSimplify()(lowered)

    script = simplified.script(extra_config={"tirx.prefix": "Tx"})
    assert "if warp_id_in_cta // 4 == 0:" in script
    assert "if 0 <= warp_id_in_cta" not in script
    assert "A_1[warp_id_in_cta] = Tx.Cast" in script
    assert "A_1[warp_id_in_cta % 4]" not in script


def test_lower_exec_context_selector_filter_for_elect_sync():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_elect_selector__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append(sctx.inter["laneid"][1].script(extra_config={"tirx.prefix": "Tx"}))

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            Tx.warp_id([1])
            lane_id = Tx.lane_id([32])
            with Tx.warp():
                if Tx.filter(lane_id, Tx.ptx.elect_sync()):
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)
                if Tx.filter(lane_id, Tx.ptx.elect_sync() != 0):
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)
                with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                    Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 3
    assert any("Tx.selector(lane_id, Tx.ptx.elect_sync())" in item for item in seen)
    assert any("Tx.selector(lane_id, Tx.ptx.elect_sync() != Tx.uint32(0))" in item for item in seen)


def test_lower_exec_context_scope_guard_mixes_structural_and_selector():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_scope_guard_mixed__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append({"inter": sctx.inter, "intra": sctx.intra})

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id([1])
            warp_id = Tx.warp_id([4])
            lane_id = Tx.lane_id([32])
            with Tx.cta():
                with Tx.thread((warp_id == 0) & Tx.filter(lane_id, Tx.ptx.elect_sync())):
                    Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert _int_pair(seen[0]["inter"], "warpid") == (1, 0)
    assert int(seen[0]["inter"]["laneid"][0]) == 1
    assert (
        seen[0]["inter"]["laneid"][1].script(extra_config={"tirx.prefix": "Tx"})
        == "Tx.selector(lane_id, Tx.ptx.elect_sync())"
    )
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_factorized_cta_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_cbx_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            cbx, cby = Tx.cta_id_in_cluster([2, 3])
            Tx.thread_id([32])
            with Tx.cta():
                if cbx == 0:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert _int_pair(seen[0], "cbx") == (1, 0)
    assert _int_pair(seen[0], "cby") == (3, 0)


def test_lower_exec_context_keeps_kernel_cta_predicate_out_of_cluster_active_set():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = {}
    kernel_variant = "__probe_exec_context_kernel_cta_in_cluster__"
    cluster_variant = "__probe_exec_context_cluster_cta_in_cluster__"

    @register_dispatch("copy", "cuda", variant=kernel_variant, priority=10_000)
    def _probe_kernel(op_call, sctx):
        seen["kernel"] = sctx.inter

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @register_dispatch("copy", "cuda", variant=cluster_variant, priority=10_000)
    def _probe_cluster(op_call, sctx):
        seen["cluster"] = sctx.inter

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            bx = Tx.cta_id([8])
            cbx = Tx.cta_id_in_cluster([2])
            Tx.thread_id([32])
            with Tx.cta():
                if bx == 0:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=kernel_variant)
                if cbx == 0:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=cluster_variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert set(seen) == {"kernel", "cluster"}
    assert _int_pair(seen["kernel"], "cta_id") == (2, 0)
    assert _int_pair(seen["cluster"], "cta_id") == (1, 0)


def test_lower_exec_context_tracks_cta_axis_modulo_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_cbx_modulo_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            cbx, cby = Tx.cta_id_in_cluster([4, 2])
            Tx.thread_id([32])
            with Tx.cta():
                if cbx % 2 == 0:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert _int_triple(seen[0], "cbx") == (2, 0, 2)
    assert _int_pair(seen[0], "cby") == (2, 0)


def test_lower_exec_context_tracks_cta_id_in_pair_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_cta_pair_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            cbx, cby = Tx.cta_id_in_cluster([4, 2])
            cta_id_in_pair = Tx.cta_id_in_pair()
            Tx.thread_id([32])
            with Tx.cta():
                if cta_id_in_pair == 0:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert _int_triple(seen[0], "cbx") == (2, 0, 2)
    assert _int_pair(seen[0], "cby") == (2, 0)


def test_lower_exec_context_tracks_two_cta_pair_predicates():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = {}
    zero_variant = "__probe_exec_context_cta_pair_two_cta_zero__"
    one_variant = "__probe_exec_context_cta_pair_two_cta_one__"

    @register_dispatch("copy", "cuda", variant=zero_variant, priority=10_000)
    def _probe_zero(op_call, sctx):
        seen["zero"] = sctx.inter

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @register_dispatch("copy", "cuda", variant=one_variant, priority=10_000)
    def _probe_one(op_call, sctx):
        seen["one"] = sctx.inter

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            Tx.cta_id_in_cluster([2])
            cta_id_in_pair = Tx.cta_id_in_pair()
            Tx.thread_id([32])
            with Tx.cta():
                if cta_id_in_pair == 0:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=zero_variant)
                if cta_id_in_pair == 1:
                    with Tx.thread():
                        Tx.copy(B[0:1], A[0:1], dispatch=one_variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert set(seen) == {"zero", "one"}
    assert _int_triple(seen["zero"], "cta_id") == (1, 0, 2)
    assert _int_triple(seen["one"], "cta_id") == (1, 1, 2)


def test_lower_exec_context_tracks_cta_id_in_pair_after_axis_predicate():
    import tvm.tirx.operator.tile_primitive as _  # noqa: F401
    from tvm.tirx.operator.tile_primitive.dispatcher import register_dispatch

    seen = []
    variant = "__probe_exec_context_cta_pair_after_axis_predicate__"

    @register_dispatch("copy", "cuda", variant=variant, priority=10_000)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        @Tx.prim_func(private=True)
        def impl():
            Tx.evaluate(0)

        return impl

    @Tx.prim_func(private=True)
    def before(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (1,), "float32", scope="global")
        with Tx.kernel():
            cbx, cby = Tx.cta_id_in_cluster([3, 2])
            cta_id_in_pair = Tx.cta_id_in_pair()
            Tx.thread_id([32])
            with Tx.cta():
                if cbx == 0:
                    if cta_id_in_pair == 1:
                        with Tx.thread():
                            Tx.copy(B[0:1], A[0:1], dispatch=variant)

    with tvm.target.Target("cuda"):
        LowerTIRx()(tvm.IRModule({"main": before}))

    assert len(seen) == 1
    assert _int_pair(seen[0], "cbx") == (1, 0)
    assert _int_triple(seen[0], "cby") == (1, 1, 2)


def test_lower_buffer_offset():
    @Tx.prim_func(private=True)
    def before():
        with Tx.kernel():
            Tx.cta_id([1])
            with Tx.cta():
                Tx.thread_id([128])
                with Tx.thread():
                    A = Tx.alloc_buffer([64, 64], "float16", scope="local")
                    A0 = Tx.decl_buffer(
                        [64], "float16", A.data, elem_offset=A.elem_offset_of([32, 32])
                    )
                    with Tx.thread():
                        Tx.evaluate(Tx.address_of(A0[32]))

    @Tx.prim_func(private=True)
    def after():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        v: Tx.let[Tx.int32] = threadIdx_x
        v_1: Tx.let[Tx.int32] = blockIdx_x
        Tx.evaluate(v_1)
        Tx.evaluate(v)
        A = Tx.alloc_local((4096,), "float16", layout=None)
        A0 = Tx.decl_buffer(
            (64,), "float16", data=A.data, elem_offset=2080, scope="local", layout=None
        )
        Tx.address_of(A0[32])

    compare(before, after, LowerTIRx)


def test_lower_alloc_decl_buffer_outside_of_parser():
    @Tx.meta_class
    class State:
        def __init__(self, smem):
            self.A = Tx.alloc_local([1], "float16")
            self.B = Tx.alloc_local([1], "float16")
            self.C = Tx.decl_buffer([1], "float16", smem, elem_offset=0, scope="shared.dyn")

    def int_var1(val):
        buf = Tx.local_scalar("int32")
        if val is not None:
            Tx.buffer_store(buf.buffer, val, 0)
        return buf

    def int_var2(val):
        buf = Tx.alloc_local([1], "int32")
        if val is not None:
            Tx.buffer_store(buf, val, 0)
        return buf

    @Tx.prim_func(private=True)
    def before():
        with Tx.kernel():
            with Tx.thread():
                smem = Tx.alloc_buffer([100], "uint8", scope="shared.dyn")
                state = State(smem.data)
                state.A[0] = Tx.float16(1)
                state.B[0] = Tx.float16(2)
                state.C[0] = Tx.float16(3)
                D = int_var1(1)
                D = D + 1
                E = int_var1(2)
                E = E + 2
                F = int_var2(3)
                F[0] = F[0] + 3
                G = int_var2(4)
                G[0] = G[0] + 4

    @Tx.prim_func(private=True)
    def after():
        smem = Tx.alloc_buffer([100], "uint8", scope="shared.dyn", layout=None)
        A = Tx.alloc_local((1,), "float16", layout=None)
        B = Tx.alloc_local((1,), "float16", layout=None)
        C = Tx.decl_buffer(
            (1,), "float16", data=smem.data, elem_offset=0, scope="shared.dyn", layout=None
        )
        A[0] = Tx.float16(1)
        B[0] = Tx.float16(2)
        C[0] = Tx.float16(3)
        D = Tx.alloc_local((1,), "int32", layout=None)
        D = 1
        D = D[0] + 1
        E = Tx.alloc_local((1,), "int32", layout=None)
        E = 2
        E = E[0] + 2
        F = Tx.alloc_local((1,), "int32", layout=None)
        F = 3
        F = F[0] + 3
        G = Tx.alloc_local((1,), "int32", layout=None)
        G = 4
        G = G[0] + 4

    compare(before, after, LowerTIRx)


def test_alloc_buffer_with_thread_axis_layout():
    """alloc_buffer with thread-axis layout should lower to 1D physical buffer with memory-axis span."""  # noqa: E501

    @Tx.prim_func(private=True)
    def before(out: Tx.Buffer((128, 4), "float32")) -> None:
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([1, 1, 1])
            Tx.warpgroup_id([1])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])
            with Tx.warpgroup():
                with Tx.thread():
                    reg_wg = Tx.alloc_buffer(
                        (128, 4), "float32", scope="local", layout=wg_local_layout(4)
                    )
                    reg = reg_wg.local(4)
                    for i in Tx.serial(4):
                        reg[i] = out[lane_id + warp_id * 32, i]

    @Tx.prim_func(private=True)
    def after(out_handle: Tx.handle):
        out = Tx.match_buffer(out_handle, (128, 4), layout=None)
        out_1 = Tx.decl_buffer((512,), data=out.data, layout=None)
        blockIdx_x = Tx.launch_thread("blockIdx.x", 1)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        blockIdx_y = Tx.launch_thread("blockIdx.y", 1)
        blockIdx_z = Tx.launch_thread("blockIdx.z", 1)
        warp_id_in_cta: Tx.let[Tx.int32] = Tx.tvm_warp_shuffle(
            Tx.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
        )
        bx: Tx.let[Tx.int32] = blockIdx_x
        by: Tx.let[Tx.int32] = blockIdx_y
        bz: Tx.let[Tx.int32] = blockIdx_z
        v: Tx.let[Tx.int32] = warp_id_in_cta // 4
        warp_id: Tx.let[Tx.int32] = warp_id_in_cta % 4
        lane_id: Tx.let[Tx.int32] = threadIdx_x % 32
        Tx.evaluate(v)
        reg_wg = Tx.alloc_local((4,), layout=None)
        reg = Tx.decl_buffer((4,), data=reg_wg.data, scope="local", layout=None)
        for i in range(4):
            reg[i] = out_1[warp_id_in_cta % 4 * 128 + threadIdx_x % 32 * 4 + i]

    compare(before, after, LowerTIRx)


def test_scope_id_compliment_no_div_by_zero():
    """Regression test: Compliment must not divide by zero when kernel extent < cluster extent.

    Before the fix, defining cluster cta_id with extent > kernel cta_id extent would crash
    with a divide-by-zero in the Compliment function during ScopeIdDef verification.
    After the fix, it raises a validation error instead of crashing.
    """
    with pytest.raises(Exception):

        @Tx.prim_func
        def func(A: Tx.Buffer((1,))):
            with Tx.kernel():
                cb_m, cb_n = Tx.cta_id_in_cluster([2, 2])
                bx = Tx.cta_id([1])
                tx = Tx.thread_id([128])
                with Tx.thread():
                    Tx.evaluate(bx + cb_m + cb_n + tx)


def test_scope_id_compliment_non_divisible():
    """Regression test: Compliment must error on provably non-divisible extents.

    cta->thread=100 and cta->warp=3 would produce warp->thread = floordiv(100, 3) = 33,
    which is semantically wrong. The fix detects this and raises an error.
    """
    with pytest.raises(Exception):

        @Tx.prim_func
        def func():
            with Tx.kernel():
                bx = Tx.cta_id([1])
                wid = Tx.warp_id([3])
                tx = Tx.thread_id([100])
                with Tx.thread():
                    Tx.evaluate(bx + wid + tx)


def test_empty_kernel_no_thread_id():
    """Regression test: kernel with ScopeIdDefs but no thread launch params must error early.

    Before the fix, this would crash late in codegen with poor diagnostics.
    """

    @Tx.prim_func
    def func():
        with Tx.kernel():
            bx = Tx.cta_id([32])
            with Tx.cta():
                with Tx.thread():
                    Tx.evaluate(bx)

    with pytest.raises(Exception, match="kernel has no thread launch parameters"):
        with tvm.target.Target("cuda"):
            LowerTIRx()(tvm.IRModule({"main": func}))


def test_lower_preferred_cluster():
    @Tx.prim_func(private=True)
    def before() -> None:
        with Tx.kernel():
            bx = Tx.cta_id([8])
            cbx, cby = Tx.cta_id_in_cluster([2, 1], preferred=[2, 2])
            tx = Tx.thread_id([128])
            with Tx.thread():
                Tx.evaluate(bx + cbx + cby + tx)

    with tvm.target.Target("cuda"):
        after_mod = LowerTIRx()(tvm.IRModule({"main": before}))
    assert not _contains_exec_scope(after_mod)
    after_str = str(after_mod["main"])
    assert 'launch_thread("clusterCtaIdx.x", 2)' in after_str
    assert 'launch_thread("clusterCtaIdx.y", 1)' in after_str
    assert 'launch_thread("preferredClusterCtaIdx.x", 2)' in after_str
    assert 'launch_thread("preferredClusterCtaIdx.y", 2)' in after_str
    assert "clusterCtaIdx_x" in after_str
    assert "clusterCtaIdx_y" in after_str
