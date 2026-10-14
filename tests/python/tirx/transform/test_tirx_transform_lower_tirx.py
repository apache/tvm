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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.backend.cuda.axis import Axis as CUDAAxis
from tvm.backend.cuda.axis import laneid, warpid
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.parser import register_namespace
from tvm.script.parser.protocol_registry import register_mutable_decl
from tvm.tirx.layout import S, TileLayout
from tvm.tirx.transform import LowerTIRx, StmtSimplify

_CTA_LAUNCH_ATTRS = {
    "cuda.kernel_attrs": {},
    "cuda.launch_fields": ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"],
}

_CLUSTER_LAUNCH_ATTRS = {
    "cuda.kernel_attrs": {},
    "cuda.launch_fields": [
        "grid.x",
        "grid.y",
        "grid.z",
        "block.x",
        "block.y",
        "block.z",
        "cluster.x",
        "cluster.y",
        "cluster.z",
    ],
}

_PREFERRED_CLUSTER_LAUNCH_ATTRS = {
    "cuda.kernel_attrs": {},
    "cuda.launch_fields": [
        "grid.x",
        "grid.y",
        "grid.z",
        "block.x",
        "block.y",
        "block.z",
        "cluster.x",
        "cluster.y",
        "cluster.z",
        "preferred_cluster.x",
        "preferred_cluster.y",
        "preferred_cluster.z",
    ],
}


def compare(Before, Expected, transform):
    with tvm.target.Target("cuda"):
        After = transform()(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def _int_range(side, axis):
    return tuple(int(x) for x in side[axis])


@T.function(private=True)
def _probe_impl():
    T.evaluate(0)


L_LANE = T.TileLayout(T.S[32 : 1 @ laneid])


def test_unroll_loop_optional_annotations():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor(8, "float32", layout="default"), B: T.Tensor(8, "float32", layout="default")
        ):
            for i in T.serial(8, annotations={"pragma_unroll": None}):
                B[i] = A[i] + 1.0
            for i in T.serial(8, annotations={"unroll_explicit": None}):
                B[i] = A[i] + 2.0
            for i in T.serial(8, annotations={"pragma_unroll": False}):
                B[i] = A[i] + 3.0
            for i in T.serial(8, annotations={"auto_unroll_max_step": 16, "unroll_explicit": 0}):
                B[i] = A[i] + 4.0

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(
            A: T.Tensor(8, "float32", layout="default"), B: T.Tensor(8, "float32", layout="default")
        ):
            for i in T.serial(8, annotations={"pragma_unroll": None}):
                B[i] = A[i] + 1.0
            for i in T.serial(8):
                B[i] = A[i] + 2.0
            for i in T.serial(8, annotations={"pragma_unroll": False}):
                B[i] = A[i] + 3.0
            for i in T.unroll(8):
                B[i] = A[i] + 4.0

    lowered = tvm.tirx.transform.UnrollLoop()(Before)
    tvm.ir.assert_structural_equal(lowered, Expected)


def test_lower_view_get():
    @I.ir_module
    class Before1:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor(64, "float32", layout="default"),
            out: T.Tensor(64, "float32", layout="default"),
        ) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=1 * 32))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            lane_id = T.cuda.lane_id()
            A = T.alloc_tensor(
                ty_args=[T.Tensor([2], "float16", scope="local", layout=T.TileLayout(T.S[2:1]))]
            )
            B_layout = A.layout.tile(L_LANE, (32,), (2,))
            B = A.view(64, layout=B_layout)
            A_local = B.view(2, layout="default")
            for i in T.vectorized(2):
                A_local[i] = T.float32(in_buf[lane_id * 2 + i])
            B_1 = A.view(64, layout=B_layout)
            A_local_1 = B_1.view(2, layout="default")
            for i in T.vectorized(2):
                out[lane_id * 2 + i] = T.float32(A_local_1[i])

    @I.ir_module
    class Expected1:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor((64,), "float32", layout=None),
            out: T.Tensor((64,), "float32", layout=None),
        ):
            out_1 = T.decl_tensor(
                out.data, ty_args=[T.Tensor((64,), "float32", layout=None, data=out.data)]
            )
            in_buf_1 = T.decl_tensor(
                in_buf.data, ty_args=[T.Tensor((64,), "float32", layout=None, data=in_buf.data)]
            )
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 32, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                A = T.alloc_local((2,), "float16", layout=None)
                B = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)]
                )
                A_local = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                for i in T.vectorized(2):
                    A_local[i] = T.Cast("float16", in_buf_1[T.cuda.thread_idx("x") * 2 + i])
                B_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)]
                )
                A_local_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                for i in T.vectorized(2):
                    out_1[T.cuda.thread_idx("x") * 2 + i] = T.Cast("float32", A_local_1[i])

    compare(Before1, Expected1, LowerTIRx)

    @I.ir_module
    class Before2:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor((16, 16), "float32", layout="default"),
            out: T.Tensor((16, 16), "float32", layout="default"),
        ) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=1 * 32))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            lane_id = T.cuda.lane_id()
            atom = T.TileLayout(T.S[(1, 2) : (2, 1)])
            tile = T.TileLayout(T.S[(2, 2) : (2, 1)])
            warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
            A = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [4, 2], "float32", scope="local", layout=atom.tile(tile, (2, 2), (1, 2))
                    )
                ]
            )
            B_layout = warp_atom.tile(tile, (2, 2), (8, 8))
            B = A.view(16, 16, layout=B_layout)
            A_local = B.view(2, 2, 2, layout="default")
            for i in T.unroll(4):
                for j in T.vectorized(2):
                    A_local[i // 2, i % 2, j] = in_buf[
                        i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j
                    ]
            B_1 = A.view(16, 16, layout=B_layout)
            A_local_1 = B_1.view(8, layout="default")
            for i in T.vectorized(2):
                out[lane_id // 4 * 8 + i // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i % 2] = (
                    A_local_1[i]
                )

    @I.ir_module
    class Expected2:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor((16, 16), "float32", layout=None),
            out: T.Tensor((16, 16), "float32", layout=None),
        ):
            out_1 = T.decl_tensor(
                out.data, ty_args=[T.Tensor((256,), "float32", layout=None, data=out.data)]
            )
            in_buf_1 = T.decl_tensor(
                in_buf.data, ty_args=[T.Tensor((256,), "float32", layout=None, data=in_buf.data)]
            )
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 32, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                A = T.alloc_local((8,), layout=None)
                B = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((256,), "float32", scope="local", layout=None)]
                )
                A_local = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((8,), "float32", scope="local", layout=None)]
                )
                for i in T.unroll(4):
                    for j in T.vectorized(2):
                        A_local[i * 2 + j] = in_buf_1[
                            i // 2 * 128
                            + T.cuda.thread_idx("x") // 4 * 16
                            + i % 2 * 8
                            + j
                            + T.cuda.thread_idx("x") % 4
                        ]
                B_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((256,), "float32", scope="local", layout=None)]
                )
                A_local_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((8,), "float32", scope="local", layout=None)]
                )
                for i in T.vectorized(2):
                    out_1[
                        T.cuda.thread_idx("x") // 4 * 128 + T.cuda.thread_idx("x") % 4 * 18 + i
                    ] = A_local_1[i]

    compare(Before2, Expected2, LowerTIRx)

    @I.ir_module
    class Before3_wgmma_layout:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor((128, 128), "float32", layout="default"),
            out: T.Tensor((128, 128), "float32", layout="default"),
        ) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=2 * 128))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            wg_id = T.cuda.warpgroup_id()
            warp_id_in_wg = T.cuda.warp_in_warpgroup()
            lane_id = T.cuda.lane_id()
            atom = T.TileLayout(T.S[1, 2])
            warp_atom = atom.tile(L_LANE, (8, 4), (1, 2))
            tile = T.TileLayout(T.S[(2, 128 // 8) : (1, 2)])
            warp_layout = warp_atom.tile(tile, (2, 128 // 8), (8, 8))
            L_warp = T.TileLayout(T.S[8 : 1 @ warpid])
            layout = warp_layout.tile(L_warp, (8, 1), (16, 128))
            acc = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [64],
                        "float32",
                        scope="local",
                        layout=atom.tile(tile, (2, 128 // 8), (1, 2)),
                    )
                ]
            )
            A = acc.view(128, 128, layout=layout)
            acc_local = A.view(16, 2, 2, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
            for i in T.serial(128 // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        acc_local[i, j, vec] = in_buf[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ]
            A_1 = acc.view(128, 128, layout=layout)
            acc_local_1 = A_1.view(64, layout=atom.tile(tile, (2, 128 // 8), (1, 2)))
            for i in T.serial(128 // 8):
                for j in T.unroll(2):
                    for vec in T.vectorized(2):
                        out[
                            wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                            i * 8 + lane_id % 4 * 2 + vec,
                        ] = acc_local_1[i * 4 + j * 2 + vec]

    @I.ir_module
    class Expected3_wgmma_layout:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor((128, 128), "float32", layout=None),
            out: T.Tensor((128, 128), "float32", layout=None),
        ):
            out_1 = T.decl_tensor(
                out.data, ty_args=[T.Tensor((16384,), "float32", layout=None, data=out.data)]
            )
            in_buf_1 = T.decl_tensor(
                in_buf.data, ty_args=[T.Tensor((16384,), "float32", layout=None, data=in_buf.data)]
            )
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 256, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                warp_id_in_wg: T.let[T.int32] = warp_id_in_cta % 4
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                acc = T.alloc_local((64,), layout=None)
                B = T.decl_tensor(
                    acc.data, ty_args=[T.Tensor((16384,), "float32", scope="local", layout=None)]
                )
                acc_local = T.decl_tensor(
                    acc.data, ty_args=[T.Tensor((64,), "float32", scope="local", layout=None)]
                )
                for i in range(16):
                    for j in T.unroll(2):
                        for vec in T.vectorized(2):
                            acc_local[i % 8 * 8 + j * 4 + i // 8 * 2 + vec] = in_buf_1[
                                warp_id_in_cta * 2048
                                + j * 1024
                                + T.cuda.thread_idx("x") % 32 // 4 * 128
                                + i * 8
                                + T.cuda.thread_idx("x") % 4 * 2
                                + vec
                            ]
                B_1 = T.decl_tensor(
                    acc.data, ty_args=[T.Tensor((16384,), "float32", scope="local", layout=None)]
                )
                acc_local_1 = T.decl_tensor(
                    acc.data, ty_args=[T.Tensor((64,), "float32", scope="local", layout=None)]
                )
                for i in range(16):
                    for j in T.unroll(2):
                        for vec in T.vectorized(2):
                            out_1[
                                warp_id_in_cta * 2048
                                + j * 1024
                                + T.cuda.thread_idx("x") % 32 // 4 * 128
                                + i * 8
                                + T.cuda.thread_idx("x") % 4 * 2
                                + vec
                            ] = acc_local_1[i % 8 * 8 + j * 4 + i // 8 * 2 + vec]

    compare(Before3_wgmma_layout, Expected3_wgmma_layout, LowerTIRx)

    @I.ir_module
    class Before4_multi_view_get:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor(64, "float32", layout="default"),
            out: T.Tensor(64, "float32", layout="default"),
        ) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=1 * 32))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            lane_id = T.cuda.lane_id()
            A = T.alloc_tensor(
                ty_args=[T.Tensor([2], "float16", scope="local", layout=T.TileLayout(T.S[2:1]))]
            )
            B_layout = A.layout.tile(L_LANE, (32,), (2,))
            B = A.view(64, layout=B_layout)
            B_1 = A.view(64, layout=B_layout)
            A_local = B.view(2, layout="default")
            A_local[0] = T.float32(in_buf[lane_id * 2])
            A_local_1 = B_1.view(2, layout="default")
            A_local_1[1] = T.float32(in_buf[lane_id * 2 + 1])
            "\n                write A into out\n                "
            B_2 = A.view(64, layout=B_layout)
            B_3 = A.view(64, layout=B_layout)
            A_local_2 = B_2.view(2, layout="default")
            out[lane_id * 2] = T.float32(A_local_2[0])
            A_local_3 = B_3.view(2, layout="default")
            out[lane_id * 2 + 1] = T.float32(A_local_3[1])

    @I.ir_module
    class Expected4_multi_view_get:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor((64,), "float32", layout=None),
            out: T.Tensor((64,), "float32", layout=None),
        ):
            out_1 = T.decl_tensor(
                out.data, ty_args=[T.Tensor((64,), "float32", layout=None, data=out.data)]
            )
            in_buf_1 = T.decl_tensor(
                in_buf.data, ty_args=[T.Tensor((64,), "float32", layout=None, data=in_buf.data)]
            )
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 32, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                A = T.alloc_local((2,), "float16", layout=None)
                B = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)]
                )
                B_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)]
                )
                A_local = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                A_local[0] = T.Cast("float16", in_buf_1[T.cuda.thread_idx("x") * 2])
                A_local_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                A_local_1[1] = T.Cast("float16", in_buf_1[T.cuda.thread_idx("x") * 2 + 1])
                B_2 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)]
                )
                B_3 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)]
                )
                A_local_2 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                out_1[T.cuda.thread_idx("x") * 2] = T.Cast("float32", A_local_2[0])
                A_local_3 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                out_1[T.cuda.thread_idx("x") * 2 + 1] = T.Cast("float32", A_local_3[1])

    compare(Before4_multi_view_get, Expected4_multi_view_get, LowerTIRx)


def test_lower_scope_id():
    @I.ir_module
    class Before1:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(3, 4, 5), block=(32,)))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            tx = T.cuda.thread_idx("x")
            T.evaluate(bx + by + bz + tx)

    @I.ir_module
    class Expected1:
        @T.function(private=True)
        def main() -> None:
            with T.region(
                "tirx.device_scope",
                [3, 4, 5, 32, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = T.cuda.block_idx("x")
                by: T.let[T.int32] = T.cuda.block_idx("y")
                bz: T.let[T.int32] = T.cuda.block_idx("z")
                tx: T.let[T.int32] = T.cuda.thread_idx("x")
                T.evaluate(bx + by + bz + tx)

    compare(Before1, Expected1, LowerTIRx)

    @I.ir_module
    class Before2:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(
                launch=T.cuda.LaunchConfig(grid=(8, 8, 8), block=4 * 32, cluster=(2, 2, 2))
            )
            cbx, cby, cbz = (
                T.cuda.cluster_cta_id("x"),
                T.cuda.cluster_cta_id("y"),
                T.cuda.cluster_cta_id("z"),
            )
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            warp_id = T.cuda.warp_id()
            lane_id = T.cuda.lane_id()
            T.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    @I.ir_module
    class Expected2:
        @T.function(private=True)
        def main() -> None:
            with T.region(
                "tirx.device_scope",
                [8, 8, 8, 128, 1, 1, 2, 2, 2],
                attrs=_CLUSTER_LAUNCH_ATTRS,
            ):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                cbz: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.z")
                bx: T.let[T.int32] = T.cuda.block_idx("x")
                by: T.let[T.int32] = T.cuda.block_idx("y")
                bz: T.let[T.int32] = T.cuda.block_idx("z")
                warp_id: T.let[T.int32] = warp_id_in_cta
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                T.evaluate(bx + by + bz + warp_id + lane_id + cbx + cby + cbz)

    compare(Before2, Expected2, LowerTIRx)

    @I.ir_module
    class Before3:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(
                launch=T.cuda.LaunchConfig(grid=(8, 10, 12), block=3 * 128, cluster=(2, 2, 1))
            )
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            cbx, cby, cbz = (
                T.cuda.cluster_cta_id("x"),
                T.cuda.cluster_cta_id("y"),
                T.cuda.cluster_cta_id("z"),
            )
            clx, cly, clz = (T.cuda.cluster_id("x"), T.cuda.cluster_id("y"), T.cuda.cluster_id("z"))
            wg_id = T.cuda.warpgroup_id()
            warp_id_in_wg = T.cuda.warp_in_warpgroup()
            lane_id = T.cuda.lane_id()
            tid_in_wg = T.cuda.thread_in_warpgroup()
            T.evaluate(bx + by + bz)
            T.evaluate(cbx + cby + cbz)
            T.evaluate(clx + cly + clz)
            T.evaluate(wg_id + warp_id_in_wg + lane_id + tid_in_wg)

    @I.ir_module
    class Expected3:
        @T.function(private=True)
        def main() -> None:
            with T.region(
                "tirx.device_scope",
                [8, 10, 12, 384, 1, 1, 2, 2, 1],
                attrs=_CLUSTER_LAUNCH_ATTRS,
            ):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                bx: T.let[T.int32] = T.cuda.block_idx("x")
                by: T.let[T.int32] = T.cuda.block_idx("y")
                bz: T.let[T.int32] = T.cuda.block_idx("z")
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                cbz: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.z")
                clx: T.let[T.int32] = T.cuda.mov_sreg(32, "clusterid.x")
                cly: T.let[T.int32] = T.cuda.mov_sreg(32, "clusterid.y")
                clz: T.let[T.int32] = T.cuda.mov_sreg(32, "clusterid.z")
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                warp_id_in_wg: T.let[T.int32] = warp_id_in_cta % 4
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                tid_in_wg: T.let[T.int32] = T.cuda.thread_idx("x") % 128
                T.evaluate(bx + by + bz)
                T.evaluate(cbx + cby + cbz)
                T.evaluate(clx + cly + clz)
                T.evaluate(wg_id + warp_id_in_wg + lane_id + tid_in_wg)

    compare(Before3, Expected3, LowerTIRx)


def test_lower_ordinary_cta_has_no_cluster_launch_tags():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.device_scope(1, 1, 1, 32, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                pass

    with tvm.target.Target("cuda"):
        after = LowerTIRx()(Before)["main"]
    tvm.ir.assert_structural_equal(after, Expected["main"])


def test_lower_explicit_singleton_cluster_launch_tags_survive_when_unused():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def cluster_2d() -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1), block=(32,), cluster=(1, 1)))
            unused_cbx, unused_cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            unused_bx, unused_by = (T.cuda.block_idx("x"), T.cuda.block_idx("y"))

        @T.function(private=True)
        def cluster_3d() -> None:
            T.device_entry(
                launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=(32,), cluster=(1, 1, 1))
            )
            unused_cbx, unused_cby, unused_cbz = (
                T.cuda.cluster_cta_id("x"),
                T.cuda.cluster_cta_id("y"),
                T.cuda.cluster_cta_id("z"),
            )
            unused_bx, unused_by, unused_bz = (
                T.cuda.block_idx("x"),
                T.cuda.block_idx("y"),
                T.cuda.block_idx("z"),
            )

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def cluster_2d():
            with T.device_scope(1, 1, 1, 32, 1, 1, 1, 1, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                unused_cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                unused_cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                unused_bx: T.let[T.int32] = 0
                unused_by: T.let[T.int32] = 0

        @T.function(private=True)
        def cluster_3d():
            with T.device_scope(1, 1, 1, 32, 1, 1, 1, 1, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                unused_cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                unused_cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                unused_cbz: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.z")
                unused_bx: T.let[T.int32] = 0
                unused_by: T.let[T.int32] = 0
                unused_bz: T.let[T.int32] = 0

    with tvm.target.Target("cuda"):
        after = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_lower_multi_cta_cluster_launch_tags_remain_unchanged():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 1), block=(32,), cluster=(2, 1)))
            unused_cbx, unused_cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            unused_bx, unused_by = (T.cuda.block_idx("x"), T.cuda.block_idx("y"))

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.device_scope(2, 1, 1, 32, 1, 1, 2, 1, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                unused_cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                unused_cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                unused_bx: T.let[T.int32] = T.cuda.block_idx("x")
                unused_by: T.let[T.int32] = 0

    with tvm.target.Target("cuda"):
        after = LowerTIRx()(Before)["main"]
    tvm.ir.assert_structural_equal(after, Expected["main"])


def test_lower_singleton_cluster_preserves_preferred_cluster_tags():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(
                launch=T.cuda.LaunchConfig(
                    grid=(2, 2), block=(32,), cluster=(1, 1), preferred_cluster=[2, 2]
                )
            )
            unused_cbx, unused_cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            unused_bx, unused_by = (T.cuda.block_idx("x"), T.cuda.block_idx("y"))

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.device_scope(
                2, 2, 1, 32, 1, 1, 1, 1, 1, 2, 2, 1, attrs=_PREFERRED_CLUSTER_LAUNCH_ATTRS
            ):
                unused_cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                unused_cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                unused_bx: T.let[T.int32] = T.cuda.block_idx("x")
                unused_by: T.let[T.int32] = T.cuda.block_idx("y")

    with tvm.target.Target("cuda"):
        after = LowerTIRx()(Before)["main"]
    tvm.ir.assert_structural_equal(after, Expected["main"])


def test_lower_scope_id2():
    @T.inline
    def func(warp_id, tx):
        wg_id = T.cuda.warpgroup_id()
        T.evaluate(wg_id + warp_id + tx)

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(3, 4, 5), block=(256,)))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            warp_id = T.cuda.warp_id()
            tx = T.cuda.thread_idx("x")
            func(warp_id, tx)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.region(
                "tirx.device_scope",
                [3, 4, 5, 256, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                bx: T.let[T.int32] = T.cuda.block_idx("x")
                by: T.let[T.int32] = T.cuda.block_idx("y")
                bz: T.let[T.int32] = T.cuda.block_idx("z")
                warp_id: T.let[T.int32] = warp_id_in_cta
                tx: T.let[T.int32] = T.cuda.thread_idx("x")
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                T.evaluate(wg_id + warp_id + tx)

    compare(Before, Expected, LowerTIRx)


@pytest.mark.skip(
    reason=(
        "Tested multi-kernel-per-Function behavior where a second sibling "
        "`with T.thread():` would redefine scope-ids and produce a second "
        "launch. The T.device_entry() refactor allows only one device-region "
        "marker per Function; this case is out of scope."
    )
)
def test_lower_scope_id3():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(3, 4, 5), block=(128,)))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            warp_id = T.cuda.warp_id()
            tx = T.cuda.thread_idx("x")
            T.evaluate(bx + by + bz + warp_id + tx)
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            warp_id = T.cuda.warp_id()
            tx = T.cuda.thread_idx("x")
            T.evaluate(bx + by + bz + warp_id + tx)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.launch_thread("blockIdx.x", 3) as blockIdx_x:
                threadIdx_x = T.launch_thread("threadIdx.x", 128)
                blockIdx_y = T.launch_thread("blockIdx.y", 4)
                blockIdx_z = T.launch_thread("blockIdx.z", 5)
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
                )
                bx: T.let[T.int32] = blockIdx_x
                by: T.let[T.int32] = blockIdx_y
                bz: T.let[T.int32] = blockIdx_z
                warp_id: T.let[T.int32] = warp_id_in_cta
                tx: T.let[T.int32] = threadIdx_x
                T.evaluate(bx + by + bz + warp_id + tx)
            blockIdx_x = T.launch_thread("blockIdx.x", 6)
            threadIdx_x = T.launch_thread("threadIdx.x", 256)
            blockIdx_y = T.launch_thread("blockIdx.y", 7)
            blockIdx_z = T.launch_thread("blockIdx.z", 8)
            warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                T.uint32(4294967295), threadIdx_x // 32, 0, 32, 32
            )
            bx: T.let[T.int32] = blockIdx_x
            by: T.let[T.int32] = blockIdx_y
            bz: T.let[T.int32] = blockIdx_z
            warp_id: T.let[T.int32] = warp_id_in_cta
            tx: T.let[T.int32] = threadIdx_x
            T.evaluate(bx + by + bz + warp_id + tx)

    compare(Before, Expected, LowerTIRx)


def test_lower_layout():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(A: T.Tensor((128, 32), "float16", layout="default")) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=(128,)))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            _lane = T.cuda.lane_id()
            tid = T.cuda.thread_idx("x")
            A_smem = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        [128, 32],
                        "float16",
                        scope="shared",
                        layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(512,)])),
                    )
                ]
            )
            thread_col = T.meta_var(4)
            thread_row = T.meta_var(32)
            for tile in T.serial(128 // thread_row):
                row = T.meta_var(tile * thread_row + tid // thread_col)
                col = T.meta_var(tid % thread_col * 8)
                for vec in T.vectorized(8):
                    A_smem[row, col + vec] = A[bx * 128 + row, col + vec]

    compose_m = T.dynamic("compose_m", "int32")
    compose_q = T.dynamic("compose_q", "int32")

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((128, 32), "float16", layout=None)) -> None:
            A_1 = T.decl_tensor(
                A.data, ty_args=[T.Tensor((4096,), "float16", layout=None, data=A.data)]
            )
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 128, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                _lane: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                tid: T.let[T.int32] = T.cuda.thread_idx("x")
                A_smem = T.alloc_shared((4096,), "float16", layout=None)
                thread_col = 4
                thread_row = 32
                for tile in range(128 // thread_row):
                    row = tile * thread_row + tid // thread_col
                    col = tid % thread_col * 8
                    for vec in T.vectorized(8):
                        # The swizzle lowers to its composition bindings rather than a
                        # folded closed form: compose_m is the flat element index, so
                        # compose_m // 8 is the row and compose_m % 8 the lane, which
                        # substituted back gives the same address.
                        A_smem[
                            T.Let(
                                T.Let(
                                    T.shift_left(
                                        T.bitwise_xor(
                                            compose_q,
                                            T.shift_right(T.bitwise_and(compose_q, 56), 3),
                                        ),
                                        3,
                                    )
                                    + compose_m % 8,
                                    where={compose_q: compose_m // 8},
                                ),
                                where={compose_m: tile * 1024 + T.cuda.thread_idx("x") * 8 + vec},
                            )
                        ] = A_1[tile * 1024 + T.cuda.thread_idx("x") * 8 + vec]

    compare(Before, Expected, LowerTIRx)


@pytest.mark.parametrize(
    "target,error,message",
    [
        (None, tvm.error.InternalError, "Target context required"),
        ("cuda", RuntimeError, "An explicit layout is required for shared tensors"),
    ],
    ids=["missing-target", "missing-shared-layout"],
)
def test_lower_opcall_fail(target, error, message):
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
            shared = T.alloc_shared((64,), "float32")
            local = T.alloc_local(
                (1, 32, 1), "float32", layout=T.TileLayout(T.S[(1, 32, 1) : (32, 1 @ laneid, 1)])
            )
            T.cuda.tile.apply(T.ptx["ld.shared.b32"], local, shared, scope="cta")

    with (
        tvm.target.Target(target) if target else nullcontext(),
        pytest.raises(error, match=message),
    ):
        LowerTIRx()(Before)


def test_lower_decl_tensor_pointer():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
            buf = T.alloc_tensor(ty_args=[T.Tensor([1024], "uint8", scope="shared.dyn")])
            A = T.decl_tensor(
                buf.data, ty_args=[T.Tensor([128], "float16", elem_offset=32, data=buf.data)]
            )
            T.evaluate(A.ptr_to([64]))

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 128, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                buf = T.alloc_tensor(
                    ty_args=[T.Tensor((1024,), "uint8", scope="shared.dyn", layout=None)]
                )
                A = T.decl_tensor(
                    buf.data,
                    ty_args=[
                        T.Tensor((128,), "float16", elem_offset=32, scope="shared.dyn", layout=None)
                    ],
                )
                T.address_of(A[64])

    compare(Before, Expected, LowerTIRx)


def test_lower_separate_scope_id_def():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
            tx = T.cuda.thread_idx("x")
            if tx == 0:
                T.evaluate(tx)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 128, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                tx: T.let[T.int32] = T.cuda.thread_idx("x")
                if tx == 0:
                    T.evaluate(tx)

    compare(Before, Expected, LowerTIRx)


def test_lower_uint32_scope_id_casts_at_bind():
    """uint32 scope ids get a Cast at the bind; launch params stay int32."""

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
            tx = T.cast(T.cuda.thread_idx("x"), "uint32")
            for k in T.serial(4, dtype="uint32"):
                T.evaluate(tx + k)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 128, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                tx: T.let[T.uint32] = T.Cast("uint32", T.cuda.thread_idx("x"))
                for k in T.serial(T.uint32(4)):
                    T.evaluate(tx + k)

    compare(Before, Expected, LowerTIRx)


def test_lower_exec_context_infers_plain_predicate_for_dispatch():

    seen = []
    variant = "__probe_exec_context_plain_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
            warp_id = T.cuda.warp_id()
            lane_id = T.cuda.lane_id()
            if (warp_id == 0) & (lane_id == 0):
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 128, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                warp_id: T.let[T.int32] = warp_id_in_cta
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if (warp_id == 0) & (lane_id == 0):
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "thread"
    assert _int_range(seen[0]["inter"], "laneid") == (1, 0)
    assert _int_range(seen[0]["inter"], "warpid") == (1, 0)
    assert _int_range(seen[0]["inter"], "cta_id") == (1, 0)
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_let_aliases_and_hoists_warp_shuffle():
    seen = []
    variant = "__probe_let_alias_scope__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", layout="default"),
            B: T.Tensor((1,), "float32", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
            lane: T.let[T.int32] = T.cuda.lane_id()
            alias_lane: T.let[T.int32] = lane
            if alias_lane == 0:
                warp: T.let[T.int32] = T.cuda.warp_id()
                alias_warp: T.let[T.int32] = warp
                if alias_warp == 1:
                    getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 128, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                lane: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                alias_lane: T.let[T.int32] = lane
                if alias_lane == 0:
                    warp: T.let[T.int32] = warp_id_in_cta
                    alias_warp: T.let[T.int32] = warp
                    if alias_warp == 1:
                        pass

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(lowered, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0], "laneid") == (1, 0)
    assert _int_range(seen[0], "warpid") == (1, 1)
    code = lowered.script()
    assert code.index("gpu_warp_shuffle") < code.index("if ")


def test_lower_exec_context_respects_preferred_cluster_bounds():
    seen = []
    variant = "__probe_preferred_cluster_bounds__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", layout="default"),
            B: T.Tensor((1,), "float32", layout="default"),
        ):
            T.device_entry(
                launch=T.cuda.LaunchConfig(
                    grid=(2, 2), block=32, cluster=1, preferred_cluster=(2, 2)
                )
            )
            cby = T.cuda.cluster_cta_id("y")
            if cby == 1:
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(
                2, 2, 1, 32, 1, 1, 1, 1, 1, 2, 2, 1, attrs=_PREFERRED_CLUSTER_LAUNCH_ATTRS
            ):
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                if cby == 1:
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0], "cbx") == (2, 0)
    assert _int_range(seen[0], "cby") == (1, 1)

    @I.ir_module
    class fixed_shape_instruction:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", layout="default"),
            B: T.Tensor((1,), "float32", layout="default"),
        ):
            T.device_entry(
                launch=T.cuda.LaunchConfig(
                    grid=(2, 2), block=32, cluster=1, preferred_cluster=(2, 2)
                )
            )
            getattr(T.cuda.tile, variant)(B[0:1], A[0:1], scope="cluster")

    with tvm.target.Target("cuda"), pytest.raises(ValueError, match="one static cluster shape"):
        LowerTIRx()(fixed_shape_instruction)


def test_lower_exec_context_infers_warpgroup_range_predicate_for_dispatch():

    seen = []
    variant = "__probe_exec_context_warpgroup_range_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 128))
            wg_id = T.cuda.warpgroup_id()
            _lane = T.cuda.lane_id()
            if wg_id == 0:
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1], scope="warpgroup")
            if (0 <= wg_id) & (wg_id < 1):
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1], scope="warpgroup")
            if (0 <= wg_id) & (wg_id < 1):
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1], scope="warpgroup")

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                _lane: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if wg_id == 0:
                    pass
                if (0 <= wg_id) & (wg_id < 1):
                    pass
                if (0 <= wg_id) & (wg_id < 1):
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 3
    for item in seen:
        assert item["scope_kind"] == "warpgroup"
        assert _int_range(item["inter"], "wgid") == (1, 0)
        assert _int_range(item["inter"], "cta_id") == (1, 0)
        assert _int_range(item["intra"], "laneid") == (32, 0)
        assert _int_range(item["intra"], "wid_in_wg") == (4, 0)


def test_lower_exec_context_tracks_cta_thread_range_predicate_for_dispatch():

    seen = []
    variant = "__probe_exec_context_cta_thread_range_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(256,)))
            tid = T.cuda.thread_idx("x")
            if (0 <= tid) & (tid < 128):
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                tid: T.let[T.int32] = T.cuda.thread_idx("x")
                if (0 <= tid) & (tid < 128):
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "thread"
    assert _int_range(seen[0]["inter"], "laneid") == (32, 0)
    assert _int_range(seen[0]["inter"], "warpid") == (4, 0)
    assert _int_range(seen[0]["inter"], "cta_id") == (1, 0)
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_cta_thread_single_warp_range_predicate():

    seen = []
    variant = "__probe_exec_context_cta_thread_single_warp_range_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(256,)))
            tid = T.cuda.thread_idx("x")
            if (34 <= tid) & (tid < 40):
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                tid: T.let[T.int32] = T.cuda.thread_idx("x")
                if (34 <= tid) & (tid < 40):
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "thread"
    assert _int_range(seen[0]["inter"], "laneid") == (6, 2)
    assert _int_range(seen[0]["inter"], "warpid") == (1, 1)
    assert _int_range(seen[0]["inter"], "cta_id") == (1, 0)
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_warpgroup_thread_range_predicate():

    seen = []
    variant = "__probe_exec_context_warpgroup_thread_range_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 128))
            wg_id = T.cuda.warpgroup_id()
            tid_in_wg = T.cuda.thread_in_warpgroup()
            if wg_id == 1:
                if (32 <= tid_in_wg) & (tid_in_wg < 64):
                    getattr(T.cuda.tile, variant)(B[0:1], A[0:1], scope="warpgroup")

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                tid_in_wg: T.let[T.int32] = T.cuda.thread_idx("x") % 128
                if wg_id == 1:
                    if (32 <= tid_in_wg) & (tid_in_wg < 64):
                        pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "warpgroup"
    assert _int_range(seen[0]["inter"], "wgid") == (1, 1)
    assert _int_range(seen[0]["inter"], "cta_id") == (1, 0)
    assert _int_range(seen[0]["intra"], "laneid") == (32, 0)
    assert _int_range(seen[0]["intra"], "wid_in_wg") == (1, 1)


def test_lower_exec_context_tracks_dependent_conjunctive_predicate():

    seen = []
    variant = "__probe_exec_context_dependent_conjunctive_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"scope_kind": sctx.scope_kind, "inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 128))
            wg_id = T.cuda.warpgroup_id()
            tid_in_wg = T.cuda.thread_in_warpgroup()
            if ((32 <= tid_in_wg) & (tid_in_wg < 64)) & (wg_id == 1):
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1], scope="warpgroup")

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                tid_in_wg: T.let[T.int32] = T.cuda.thread_idx("x") % 128
                if (32 <= tid_in_wg) & (tid_in_wg < 64) & (wg_id == 1):
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert seen[0]["scope_kind"] == "warpgroup"
    assert _int_range(seen[0]["inter"], "wgid") == (1, 1)
    assert _int_range(seen[0]["inter"], "cta_id") == (1, 0)
    assert _int_range(seen[0]["intra"], "laneid") == (32, 0)
    assert _int_range(seen[0]["intra"], "wid_in_wg") == (1, 1)


def test_lower_exec_context_keeps_plain_predicate_condition():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(A: T.Tensor((1,), "float32", scope="global", layout="default")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 128))
            wg_id = T.cuda.warpgroup_id()
            _lane = T.cuda.lane_id()
            if wg_id == 0:
                T.evaluate(A[0])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32)):
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                _lane: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if wg_id == 0:
                    T.evaluate(A_1[0])

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(lowered, Expected)

    script = lowered.script(extra_config={"tirx.prefix": "T"})
    assert "if wg_id == 0:" in script
    assert "0 <= v_1" not in script
    assert "v_1 < 1" not in script


def test_lower_exec_context_keeps_plain_scope_predicate_condition():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(A: T.Tensor((1,), "float32", scope="global", layout="default")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 128))
            wg_id = T.cuda.warpgroup_id()
            _lane = T.cuda.lane_id()
            if wg_id == 0:
                A[0] = T.float32(1)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32)):
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                _lane: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if wg_id == 0:
                    A_1[0] = T.float32(1.0)

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(lowered, Expected)

    script = lowered.script(extra_config={"tirx.prefix": "T"})
    assert "if wg_id == 0:" in script
    assert "0 <= v_1" not in script
    assert "v_1 < 1" not in script


def test_simplify_uses_floor_div_scope_predicate_as_context_fact():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(A: T.Tensor((16,), "float32", scope="global", layout="default")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=2 * 128))
            wg_id = T.cuda.warpgroup_id()
            warp_id = T.cuda.warp_in_warpgroup()
            lane_id = T.cuda.lane_id()
            if wg_id == 0:
                A[warp_id] = T.float32(lane_id)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((16,), T.float32)):
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                warp_id: T.let[T.int32] = warp_id_in_cta % 4
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if wg_id == 0:
                    A_1[warp_id_in_cta % 4] = T.Cast("float32", lane_id)

    @I.ir_module
    class ExpectedSimplified:
        @T.function(private=True)
        def main(A: T.Tensor((16,), T.float32)):
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((16,), T.float32)])
            with T.device_scope(1, 1, 1, 256, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                wg_id: T.let[T.int32] = warp_id_in_cta // 4
                warp_id: T.let[T.int32] = warp_id_in_cta % 4
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if warp_id_in_cta // 4 == 0:
                    A_1[warp_id_in_cta] = T.Cast("float32", T.cuda.thread_idx("x") % 32)

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)
        simplified = StmtSimplify()(lowered)
    tvm.ir.assert_structural_equal(lowered, Expected)
    tvm.ir.assert_structural_equal(simplified, ExpectedSimplified)

    script = simplified.script(extra_config={"tirx.prefix": "T"})
    assert "if warp_id_in_cta // 4 == 0:" in script
    assert "if 0 <= warp_id_in_cta" not in script
    assert "A_1[warp_id_in_cta] = T.Cast" in script
    assert "A_1[warp_id_in_cta % 4]" not in script


def test_lower_exec_context_selector_filter_for_elect_sync():

    seen = []
    variant = "__probe_exec_context_elect_selector__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter["laneid"][1].script(extra_config={"tirx.prefix": "T"}))

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
            lane_id = T.cuda.lane_id()
            if T.cuda.elect_sync():
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])
            if T.cuda.elect_sync() != 0:
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])
            if T.cuda.elect_sync():
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 32, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if T.cuda.elect_sync():
                    pass
                if T.cuda.elect_sync() != T.uint32(0):
                    pass
                if T.cuda.elect_sync():
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 3
    assert any(
        "T.gpu_active_thread_selector(lane_id, T.cuda.elect_sync())" in item for item in seen
    )
    assert any(
        "T.gpu_active_thread_selector(lane_id, T.cuda.elect_sync() != T.uint32(0))" in item
        for item in seen
    )


def test_lower_cleanup_accepts_bool_elect_sync_else_path():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(A: T.Tensor((32,), "int32", scope="global", layout="default")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
            lane_id = T.cuda.lane_id()
            if T.cuda.elect_sync() != T.uint32(0):
                A[lane_id] = 1
            else:
                A[lane_id] = 0

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((32,), T.int32)):
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((32,), T.int32)])
            with T.device_scope(1, 1, 1, 32, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if T.cuda.elect_sync() != T.uint32(0):
                    A_1[T.cuda.thread_idx("x")] = 1
                else:
                    A_1[T.cuda.thread_idx("x")] = 0

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(lowered, Expected)

    script = lowered.script(extra_config={"tirx.prefix": "T"})
    assert "T.cuda.elect_sync() != T.uint32(0)" in script
    assert "else:" in script


def test_lower_exec_context_scope_guard_mixes_structural_and_selector():

    seen = []
    variant = "__probe_exec_context_scope_guard_mixed__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append({"inter": sctx.inter, "intra": sctx.intra})

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
            warp_id = T.cuda.warp_id()
            lane_id = T.cuda.lane_id()
            if (warp_id == 0) & T.cuda.elect_sync():
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(1, 1, 1, 128, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                warp_id: T.let[T.int32] = warp_id_in_cta
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                if T.Cast("uint32", warp_id == 0) & T.cuda.elect_sync():
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0]["inter"], "warpid") == (1, 0)
    assert int(seen[0]["inter"]["laneid"][0]) == 1
    assert (
        seen[0]["inter"]["laneid"][1].script(extra_config={"tirx.prefix": "T"})
        == "T.gpu_active_thread_selector(lane_id, T.cuda.elect_sync())"
    )
    assert len(seen[0]["intra"]) == 0


def test_lower_exec_context_tracks_factorized_cta_predicate():

    seen = []
    variant = "__probe_exec_context_cbx_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 3), block=(32,), cluster=(2, 3)))
            cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            if cbx == 0:
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(2, 3, 1, 32, 1, 1, 2, 3, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                if cbx == 0:
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0], "cbx") == (1, 0)
    assert _int_range(seen[0], "cby") == (3, 0)


def test_lower_exec_context_keeps_kernel_cta_predicate_out_of_cluster_active_set():

    seen = {}
    kernel_variant = "__probe_exec_context_kernel_cta_in_cluster__"
    cluster_variant = "__probe_exec_context_cluster_cta_in_cluster__"

    @_register_probe(kernel_variant)
    def _probe_kernel(op_call, sctx):
        seen["kernel"] = sctx.inter

        return _probe_impl

    @_register_probe(cluster_variant)
    def _probe_cluster(op_call, sctx):
        seen["cluster"] = sctx.inter

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(8,), block=(32,), cluster=(2,)))
            bx = T.cuda.block_idx("x")
            cbx = T.cuda.cluster_cta_id("x")
            if bx == 0:
                getattr(T.cuda.tile, kernel_variant)(B[0:1], A[0:1])
            if cbx == 0:
                getattr(T.cuda.tile, cluster_variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(8, 1, 1, 32, 1, 1, 2, 1, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                bx: T.let[T.int32] = T.cuda.block_idx("x")
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                if bx == 0:
                    pass
                if cbx == 0:
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert set(seen) == {"kernel", "cluster"}
    assert _int_range(seen["kernel"], "cta_id") == (2, 0)
    assert _int_range(seen["cluster"], "cta_id") == (1, 0)


def test_lower_exec_context_tracks_cta_axis_modulo_predicate():

    seen = []
    variant = "__probe_exec_context_cbx_modulo_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(4, 2), block=(32,), cluster=(4, 2)))
            cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            if cbx % 2 == 0:
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(4, 2, 1, 32, 1, 1, 4, 2, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                if cbx % 2 == 0:
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0], "cbx") == (2, 0, 2)
    assert _int_range(seen[0], "cby") == (2, 0)


def test_lower_exec_context_tracks_cta_id_in_pair_predicate():

    seen = []
    variant = "__probe_exec_context_cta_pair_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(4, 2), block=(32,), cluster=(4, 2)))
            cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            cta_id_in_pair = T.cuda.cta_pair_id()
            if cta_id_in_pair == 0:
                getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(4, 2, 1, 32, 1, 1, 4, 2, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                cta_id_in_pair: T.let[T.int32] = (
                    T.cuda.mov_sreg(32, "cluster_ctaid.x")
                    + T.cuda.mov_sreg(32, "cluster_nctaid.x")
                    * (
                        T.cuda.mov_sreg(32, "cluster_ctaid.y")
                        + T.cuda.mov_sreg(32, "cluster_nctaid.y")
                        * T.cuda.mov_sreg(32, "cluster_ctaid.z")
                    )
                ) % 2
                if cta_id_in_pair == 0:
                    pass

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(lowered, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0], "cbx") == (2, 0, 2)
    assert _int_range(seen[0], "cby") == (2, 0)


def test_lower_exec_context_tracks_two_cta_pair_predicates():

    seen = {}
    zero_variant = "__probe_exec_context_cta_pair_two_cta_zero__"
    one_variant = "__probe_exec_context_cta_pair_two_cta_one__"

    @_register_probe(zero_variant)
    def _probe_zero(op_call, sctx):
        seen["zero"] = sctx.inter

        return _probe_impl

    @_register_probe(one_variant)
    def _probe_one(op_call, sctx):
        seen["one"] = sctx.inter

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=(32,), cluster=(2,)))
            cta_id_in_pair = T.cuda.cta_pair_id()
            if cta_id_in_pair == 0:
                getattr(T.cuda.tile, zero_variant)(B[0:1], A[0:1])
            if cta_id_in_pair == 1:
                getattr(T.cuda.tile, one_variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(2, 1, 1, 32, 1, 1, 2, 1, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                cta_id_in_pair: T.let[T.int32] = (
                    T.cuda.mov_sreg(32, "cluster_ctaid.x")
                    + T.cuda.mov_sreg(32, "cluster_nctaid.x")
                    * (
                        T.cuda.mov_sreg(32, "cluster_ctaid.y")
                        + T.cuda.mov_sreg(32, "cluster_nctaid.y")
                        * T.cuda.mov_sreg(32, "cluster_ctaid.z")
                    )
                ) % 2
                if cta_id_in_pair == 0:
                    pass
                if cta_id_in_pair == 1:
                    pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert set(seen) == {"zero", "one"}
    assert _int_range(seen["zero"], "cta_id") == (1, 0, 2)
    assert _int_range(seen["one"], "cta_id") == (1, 1, 2)


def test_lower_exec_context_tracks_cta_id_in_pair_after_axis_predicate():

    seen = []
    variant = "__probe_exec_context_cta_pair_after_axis_predicate__"

    @_register_probe(variant)
    def _probe(op_call, sctx):
        seen.append(sctx.inter)

        return _probe_impl

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(
            A: T.Tensor((1,), "float32", scope="global", layout="default"),
            B: T.Tensor((1,), "float32", scope="global", layout="default"),
        ):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(3, 2), block=(32,), cluster=(3, 2)))
            cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            cta_id_in_pair = T.cuda.cta_pair_id()
            if cbx == 0:
                if cta_id_in_pair == 1:
                    getattr(T.cuda.tile, variant)(B[0:1], A[0:1])

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            B_1 = T.decl_tensor(B.data, ty_args=[T.Tensor((1,), T.float32)])
            A_1 = T.decl_tensor(A.data, ty_args=[T.Tensor((1,), T.float32)])
            with T.device_scope(3, 2, 1, 32, 1, 1, 3, 2, 1, attrs=_CLUSTER_LAUNCH_ATTRS):
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                cta_id_in_pair: T.let[T.int32] = (
                    T.cuda.mov_sreg(32, "cluster_ctaid.x")
                    + T.cuda.mov_sreg(32, "cluster_nctaid.x")
                    * (
                        T.cuda.mov_sreg(32, "cluster_ctaid.y")
                        + T.cuda.mov_sreg(32, "cluster_nctaid.y")
                        * T.cuda.mov_sreg(32, "cluster_ctaid.z")
                    )
                ) % 2
                if cbx == 0:
                    if cta_id_in_pair == 1:
                        pass

    with tvm.target.Target("cuda"):
        After = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(After, Expected)

    assert len(seen) == 1
    assert _int_range(seen[0], "cbx") == (1, 0)
    assert _int_range(seen[0], "cby") == (1, 1, 2)


@pytest.mark.parametrize("offset", [0, 16])
def test_lower_remaps_tensor_memory_address_metadata(offset):
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry()
            addr = T.alloc_shared((1,), "uint32")
            for i in T.serial(2):
                tensor = T.cuda.decl_tmem(
                    addr[0] + T.uint32(offset) + T.cast(i, "uint32"),
                    ty_args=[T.Tensor((64,), "float32", scope="tmem", layout="default")],
                )
                T.evaluate(tensor.data)

    def expected_address(addr, i):
        if offset:
            return addr[0] + T.uint32(offset) + T.Cast("uint32", i)
        return addr[0] + T.Cast("uint32", i)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            addr = T.alloc_tensor(ty_args=[T.Tensor((1,), T.uint32, scope="shared")])
            for i in range(2):
                tensor = T.decl_tensor(
                    T.reinterpret(
                        expected_address(addr, i),
                        ty=T.Ptr(T.PrimType("float32"), scope="tmem"),
                    ),
                    ty_args=[T.Tensor((64,), T.float32, scope="tmem")],
                )
                T.evaluate(tensor.data)

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)["main"]
    tvm.ir.assert_structural_equal(lowered, Expected["main"])
    assert not tvm.tirx.analysis.undefined_vars(lowered.body, lowered.params)
    tvm.tirx.analysis.verify_well_formed(lowered)
    bindings = []
    tvm_ffi.structural_walk(lowered.body, (tvm.ir.Bind, bindings.append))
    address = next(
        node.var for node in bindings if node.var.scope() == tvm.runtime.MemoryScope("shared")
    )
    tensor = next(node for node in bindings if node.var.scope() == tvm.runtime.MemoryScope("tmem"))
    loads = []
    tvm_ffi.structural_walk(tensor.value.args[0], (tvm.ir.TensorLoad, loads.append))
    assert len(loads) == 1
    assert loads[0].source.same_as(address)
    tvm.ir.assert_structural_equal(tensor.var.ty, tensor.value.ty)


def test_lower_tensor_offset():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
            A = T.alloc_tensor(ty_args=[T.Tensor([64, 64], "float16", scope="local")])
            A0 = T.decl_tensor(
                A.ptr_to([32, 32]),
                ty_args=[T.Tensor([64], "float16", elem_offset=0, data=A.ptr_to([32, 32]))],
            )
            T.evaluate(T.address_of(A0[32]))

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 128, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                A = T.alloc_local((4096,), "float16", layout=None)
                A0 = T.decl_tensor(
                    T.address_of(A[2080]),
                    ty_args=[T.Tensor((64,), "float16", elem_offset=0, scope="local", layout=None)],
                )
                T.address_of(A0[32])

    compare(Before, Expected, LowerTIRx)


def test_lower_alloc_decl_tensor_outside_of_parser():
    @T.meta_class
    class State:
        def __init__(self, smem):
            self.A = T.alloc_local([1], "float16")
            self.B = T.alloc_local([1], "float16")
            self.C = T.bind(
                T.decl_tensor(
                    smem, ty_args=[T.Tensor([1], "float16", elem_offset=0, scope="shared.dyn")]
                )
            )

    @register_mutable_decl("TestMutableCells.int_var1")
    def int_var1(val):
        buf = T.local_scalar("int32")
        if val is not None:
            T.tensor_store(buf.source, 0, val)
        return buf

    TestMutableCells = SimpleNamespace(int_var1=int_var1)
    register_namespace("TestMutableCells", TestMutableCells)

    def int_var2(val):
        buf = T.alloc_local([1], "int32")
        if val is not None:
            T.tensor_store(buf, 0, val)
        return buf

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry()
            smem = T.alloc_tensor(ty_args=[T.Tensor([100], "uint8", scope="shared.dyn")])
            state = State(smem.data)
            state.A[0] = T.float16(1)
            state.B[0] = T.float16(2)
            state.C[0] = T.float16(3)
            D = TestMutableCells.int_var1(1)
            D = D + 1
            E = TestMutableCells.int_var1(2)
            E = E + 2
            F = int_var2(3)
            F[0] = F[0] + 3
            G = int_var2(4)
            G[0] = G[0] + 4

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            smem = T.alloc_tensor(
                ty_args=[T.Tensor([100], "uint8", scope="shared.dyn", layout=None)]
            )
            A = T.alloc_local((1,), "float16", layout=None)
            B = T.alloc_local((1,), "float16", layout=None)
            C = T.decl_tensor(
                smem.data,
                ty_args=[T.Tensor((1,), "float16", elem_offset=0, scope="shared.dyn", layout=None)],
            )
            A[0] = T.float16(1)
            B[0] = T.float16(2)
            C[0] = T.float16(3)
            D = T.alloc_local((1,), "int32", layout=None)
            D = 1
            D = D[0] + 1
            E = T.alloc_local((1,), "int32", layout=None)
            E = 2
            E = E[0] + 2
            F = T.alloc_local((1,), "int32", layout=None)
            F = 3
            F = F[0] + 3
            G = T.alloc_local((1,), "int32", layout=None)
            G = 4
            G = G[0] + 4

    compare(Before, Expected, LowerTIRx)


def test_alloc_tensor_with_thread_axis_layout():
    """alloc_tensor with thread-axis layout should lower to 1D physical tensor with memory-axis span."""  # noqa: E501

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main(out: T.Tensor((128, 4), "float32", layout="default")) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=1 * 128))
            bx, by, bz = (T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z"))
            warp_id = T.cuda.warp_in_warpgroup()
            lane_id = T.cuda.lane_id()
            reg_wg = T.alloc_tensor(
                ty_args=[
                    T.Tensor(
                        (128, 4),
                        "float32",
                        scope="local",
                        layout=TileLayout(S[(128, 4) : (1 @ CUDAAxis.tid_in_wg, 1)]),
                    )
                ]
            )
            reg = reg_wg.view(4, layout="default")
            for i in T.serial(4):
                reg[i] = out[lane_id + warp_id * 32, i]

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main(out: T.Tensor((128, 4), "float32", layout=None)):
            out_1 = T.decl_tensor(
                out.data, ty_args=[T.Tensor((512,), "float32", layout=None, data=out.data)]
            )
            with T.region(
                "tirx.device_scope",
                [1, 1, 1, 128, 1, 1],
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                warp_id_in_cta: T.let[T.int32] = T.gpu_warp_shuffle(
                    T.uint32(4294967295), T.cuda.thread_idx("x") // 32, 0, 32, 32
                )
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                warp_id: T.let[T.int32] = warp_id_in_cta % 4
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                reg_wg = T.alloc_local((4,), layout=None)
                reg = T.decl_tensor(
                    reg_wg.data, ty_args=[T.Tensor((4,), "float32", scope="local", layout=None)]
                )
                for i in range(4):
                    reg[i] = out_1[warp_id_in_cta % 4 * 128 + T.cuda.thread_idx("x") % 32 * 4 + i]

    compare(Before, Expected, LowerTIRx)


def test_cluster_larger_than_grid_is_rejected():
    with pytest.raises(ValueError, match="divisible"):

        @I.ir_module
        class Before:
            @T.function(private=True)
            def main():
                T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128, cluster=(2, 2)))


def test_warp_indices_require_complete_warps():
    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"global_symbol": "kernel"})
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=100))
            T.evaluate(T.cuda.warp_id())

    with pytest.raises(ValueError, match="divisible by 32"):
        with tvm.target.Target("cuda"):
            LowerTIRx()(Before)


def test_launch_configuration_does_not_require_index_uses():
    @I.ir_module
    class Before:
        @T.function
        def main():
            T.func_attr({"global_symbol": "kernel"})
            T.device_entry(launch=T.cuda.LaunchConfig(grid=32, block=128))
            T.evaluate(0)

    @I.ir_module
    class Expected:
        @T.function
        def main():
            T.func_attr({"global_symbol": "kernel"})
            with T.device_scope(32, 1, 1, 128, 1, 1, attrs=_CTA_LAUNCH_ATTRS):
                pass

    with tvm.target.Target("cuda"):
        lowered = LowerTIRx()(Before)["main"]
    tvm.ir.assert_structural_equal(lowered, Expected["main"])


def test_lower_preferred_cluster():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main() -> None:
            T.device_entry(
                launch=T.cuda.LaunchConfig(
                    grid=(8, 2), block=(128,), cluster=(2, 1), preferred_cluster=[2, 2]
                )
            )
            bx = T.cuda.block_idx("x")
            cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
            tx = T.cuda.thread_idx("x")
            T.evaluate(bx + cbx + cby + tx)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            with T.device_scope(
                8, 2, 1, 128, 1, 1, 2, 1, 1, 2, 2, 1, attrs=_PREFERRED_CLUSTER_LAUNCH_ATTRS
            ):
                bx: T.let[T.int32] = T.cuda.block_idx("x")
                cbx: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.x")
                cby: T.let[T.int32] = T.cuda.mov_sreg(32, "cluster_ctaid.y")
                tx: T.let[T.int32] = T.cuda.thread_idx("x")
                T.evaluate(bx + cbx + cby + tx)

    with tvm.target.Target("cuda"):
        after_mod = LowerTIRx()(Before)
    tvm.ir.assert_structural_equal(after_mod, Expected)
    assert 'T.cuda.mov_sreg(32, "cluster_ctaid.y")' in after_mod.script()


def _register_probe(name):
    """Register a test instruction whose lowerer inspects its execution context."""
    from tvm.tirx.tile.instruction import Instruction, Operand

    def decorate(lower):
        spec = Instruction(
            "tirx.cuda.tile." + name,
            "copy",
            (Operand("dst", "region"), Operand("src", "region")),
            scopes=("cluster", "cta", "warpgroup", "warp", "thread"),
            lower=lower,
        )
        factory = spec.register()
        setattr(T.cuda.tile, name, factory)
        return lower

    return decorate


@pytest.mark.parametrize("scope", ["local", "shared"])
def test_alloc_tensor_layout_span_without_threads(scope):
    """A layout's holes consume storage even when it has no thread axes."""
    layout = T.TileLayout(T.S[(3, 4) : (16, 2)])

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            A = T.alloc_tensor(ty_args=[T.Tensor((3, 4), "uint32", scope=scope, layout=layout)])
            A[2, 3] = T.uint32(7)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            A = T.alloc_tensor(ty_args=[T.Tensor((39,), "uint32", scope=scope, layout=None)])
            A[38] = T.uint32(7)

    compare(Before, Expected, LowerTIRx)


@pytest.mark.parametrize("scope", ["local", "shared"])
def test_alloc_tensor_swizzle_storage_span(scope):
    """Swizzling a gapped physical view must not underallocate its memory."""
    layout = T.ComposeLayout(2, 1, 3, T.TileLayout(T.S[(128, 4) : (8, 1)]))

    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            A = T.alloc_tensor(ty_args=[T.Tensor((128, 4), "uint32", scope=scope, layout=layout)])
            A[127, 3] = T.uint32(7)

    @I.ir_module
    class Expected:
        @T.function(private=True)
        def main():
            A = T.alloc_tensor(ty_args=[T.Tensor((1024,), "uint32", scope=scope, layout=None)])
            A[1023] = T.uint32(7)

    compare(Before, Expected, LowerTIRx)
