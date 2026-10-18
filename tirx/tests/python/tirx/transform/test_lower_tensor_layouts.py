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

from types import SimpleNamespace

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.axis import Axis as CUDAAxis
from tvm_tirx.backend.cuda.axis import laneid, warpid
from tvm_tirx.script import ir as I
from tvm_tirx.script import tirx as T
from tvm_tirx.script.parser import register_namespace
from tvm_tirx.script.parser.protocol_registry import register_mutable_decl
from tvm_tirx.tirx.layout import S, TileLayout
from tvm_tirx.tirx.transform import LowerTensorLayouts, TileDispatch

_CTA_LAUNCH_ATTRS = {
    "cuda.kernel_attrs": {},
    "cuda.launch_fields": ["grid.x", "grid.y", "grid.z", "block.x", "block.y", "block.z"],
}


L_LANE = T.TileLayout(T.S[32 : 1 @ laneid])


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
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                A_local = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                for i in T.vectorized(2):
                    A_local[i] = T.Cast("float16", in_buf_1[T.cuda.thread_idx("x") * 2 + i])
                B_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                A_local_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                for i in T.vectorized(2):
                    out_1[T.cuda.thread_idx("x") * 2 + i] = T.Cast("float32", A_local_1[i])

    @I.ir_module
    class Dispatched:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor(
                (64,),
                T.float32,
                layout="default",
            ),
            out: T.Tensor(
                (64,),
                T.float32,
                layout="default",
            ),
        ):
            with T.device_scope(
                1,
                1,
                1,
                32,
                1,
                1,
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                A = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ]
                )
                v = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float16,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(32, 1, "laneid"), T.Iter(2, 1, "m")], replica=[]
                            ),
                        )
                    ],
                )
                v_1 = T.decl_tensor(
                    v.data,
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                for i in T.vectorized(0, 2):
                    v_1[i] = T.Cast("float16", in_buf[lane_id * 2 + i])
                v_2 = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float16,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(32, 1, "laneid"), T.Iter(2, 1, "m")], replica=[]
                            ),
                        )
                    ],
                )
                v_3 = T.decl_tensor(
                    v_2.data,
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                for i_1 in T.vectorized(0, 2):
                    out[lane_id * 2 + i_1] = T.Cast("float32", v_3[i_1])

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before1)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        after = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(after, Expected1)

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
                    A.data, ty_args=[T.Tensor((8,), "float32", scope="local", layout=None)]
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
                    A.data, ty_args=[T.Tensor((8,), "float32", scope="local", layout=None)]
                )
                A_local_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((8,), "float32", scope="local", layout=None)]
                )
                for i in T.vectorized(2):
                    out_1[
                        T.cuda.thread_idx("x") // 4 * 128 + T.cuda.thread_idx("x") % 4 * 18 + i
                    ] = A_local_1[i]

    @I.ir_module
    class Dispatched2:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor(
                (16, 16),
                T.float32,
                layout="default",
            ),
            out: T.Tensor(
                (16, 16),
                T.float32,
                layout="default",
            ),
        ):
            with T.device_scope(
                1,
                1,
                1,
                32,
                1,
                1,
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                A = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (4, 2),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(shard=[T.Iter(8, 1, "m")], replica=[]),
                        )
                    ]
                )
                v = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (16, 16),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[
                                    T.Iter(2, 4, "m"),
                                    T.Iter(8, 4, "laneid"),
                                    T.Iter(2, 2, "m"),
                                    T.Iter(4, 1, "laneid"),
                                    T.Iter(2, 1, "m"),
                                ],
                                replica=[],
                            ),
                        )
                    ],
                )
                v_1 = T.decl_tensor(
                    v.data,
                    ty_args=[
                        T.Tensor(
                            (2, 2, 2),
                            T.float32,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                for i in T.unroll(0, 4):
                    for j in T.vectorized(0, 2):
                        v_1[i // 2, i % 2, j] = in_buf[
                            i // 2 * 8 + lane_id // 4, i % 2 * 8 + lane_id % 4 + j
                        ]
                v_2 = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (16, 16),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[
                                    T.Iter(2, 4, "m"),
                                    T.Iter(8, 4, "laneid"),
                                    T.Iter(2, 2, "m"),
                                    T.Iter(4, 1, "laneid"),
                                    T.Iter(2, 1, "m"),
                                ],
                                replica=[],
                            ),
                        )
                    ],
                )
                v_3 = T.decl_tensor(
                    v_2.data,
                    ty_args=[
                        T.Tensor(
                            (8,),
                            T.float32,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                for i_1 in T.vectorized(0, 2):
                    out[
                        lane_id // 4 * 8 + i_1 // 2 * 8 + lane_id % 4, lane_id % 4 * 2 + i_1 % 2
                    ] = v_3[i_1]

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before2)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched2)
        after = LowerTensorLayouts()(Dispatched2)
    tvm_tirx.ir.assert_structural_equal(after, Expected2)

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
                    acc.data, ty_args=[T.Tensor((64,), "float32", scope="local", layout=None)]
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
                    acc.data, ty_args=[T.Tensor((64,), "float32", scope="local", layout=None)]
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

    @I.ir_module
    class Dispatched3:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor(
                (128, 128),
                T.float32,
                layout="default",
            ),
            out: T.Tensor(
                (128, 128),
                T.float32,
                layout="default",
            ),
        ):
            with T.device_scope(
                1,
                1,
                1,
                256,
                1,
                1,
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
                acc = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(2, 2, "m"), T.Iter(16, 4, "m"), T.Iter(2, 1, "m")],
                                replica=[],
                            ),
                        )
                    ]
                )
                v = T.decl_tensor(
                    acc.data,
                    ty_args=[
                        T.Tensor(
                            (128, 128),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[
                                    T.Iter(8, 1, "warpid"),
                                    T.Iter(2, 2, "m"),
                                    T.Iter(8, 4, "laneid"),
                                    T.Iter(16, 4, "m"),
                                    T.Iter(4, 1, "laneid"),
                                    T.Iter(2, 1, "m"),
                                ],
                                replica=[],
                            ),
                        )
                    ],
                )
                v_1 = T.decl_tensor(
                    v.data,
                    ty_args=[
                        T.Tensor(
                            (16, 2, 2),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(2, 2, "m"), T.Iter(16, 4, "m"), T.Iter(2, 1, "m")],
                                replica=[],
                            ),
                        )
                    ],
                )
                for i in range(16):
                    for j in T.unroll(0, 2):
                        for vec in T.vectorized(0, 2):
                            v_1[i, j, vec] = in_buf[
                                wg_id * 64 + warp_id_in_wg * 16 + j * 8 + lane_id // 4,
                                i * 8 + lane_id % 4 * 2 + vec,
                            ]
                v_2 = T.decl_tensor(
                    acc.data,
                    ty_args=[
                        T.Tensor(
                            (128, 128),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[
                                    T.Iter(8, 1, "warpid"),
                                    T.Iter(2, 2, "m"),
                                    T.Iter(8, 4, "laneid"),
                                    T.Iter(16, 4, "m"),
                                    T.Iter(4, 1, "laneid"),
                                    T.Iter(2, 1, "m"),
                                ],
                                replica=[],
                            ),
                        )
                    ],
                )
                v_3 = T.decl_tensor(
                    v_2.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(2, 2, "m"), T.Iter(16, 4, "m"), T.Iter(2, 1, "m")],
                                replica=[],
                            ),
                        )
                    ],
                )
                for i_1 in range(16):
                    for j_1 in T.unroll(0, 2):
                        for vec_1 in T.vectorized(0, 2):
                            out[
                                wg_id * 64 + warp_id_in_wg * 16 + j_1 * 8 + lane_id // 4,
                                i_1 * 8 + lane_id % 4 * 2 + vec_1,
                            ] = v_3[i_1 * 4 + j_1 * 2 + vec_1]

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before3_wgmma_layout)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched3)
        after = LowerTensorLayouts()(Dispatched3)
    tvm_tirx.ir.assert_structural_equal(after, Expected3_wgmma_layout)

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
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                B_1 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
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
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                B_3 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                A_local_2 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                out_1[T.cuda.thread_idx("x") * 2] = T.Cast("float32", A_local_2[0])
                A_local_3 = T.decl_tensor(
                    A.data, ty_args=[T.Tensor((2,), "float16", scope="local", layout=None)]
                )
                out_1[T.cuda.thread_idx("x") * 2 + 1] = T.Cast("float32", A_local_3[1])

    @I.ir_module
    class Dispatched4:
        @T.function(private=True)
        def main(
            in_buf: T.Tensor(
                (64,),
                T.float32,
                layout="default",
            ),
            out: T.Tensor(
                (64,),
                T.float32,
                layout="default",
            ),
        ):
            with T.device_scope(
                1,
                1,
                1,
                32,
                1,
                1,
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                A = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ]
                )
                v = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float16,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(32, 1, "laneid"), T.Iter(2, 1, "m")], replica=[]
                            ),
                        )
                    ],
                )
                v_1 = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float16,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(32, 1, "laneid"), T.Iter(2, 1, "m")], replica=[]
                            ),
                        )
                    ],
                )
                v_2 = T.decl_tensor(
                    v.data,
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                v_2[0] = T.Cast("float16", in_buf[lane_id * 2])
                v_3 = T.decl_tensor(
                    v_1.data,
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                v_3[1] = T.Cast("float16", in_buf[lane_id * 2 + 1])
                v_4 = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float16,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(32, 1, "laneid"), T.Iter(2, 1, "m")], replica=[]
                            ),
                        )
                    ],
                )
                v_5 = T.decl_tensor(
                    A.data,
                    ty_args=[
                        T.Tensor(
                            (64,),
                            T.float16,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(32, 1, "laneid"), T.Iter(2, 1, "m")], replica=[]
                            ),
                        )
                    ],
                )
                v_6 = T.decl_tensor(
                    v_4.data,
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                out[lane_id * 2] = T.Cast("float32", v_6[0])
                v_7 = T.decl_tensor(
                    v_5.data,
                    ty_args=[
                        T.Tensor(
                            (2,),
                            T.float16,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                out[lane_id * 2 + 1] = T.Cast("float32", v_7[1])

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before4_multi_view_get)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched4)
        after = LowerTensorLayouts()(Dispatched4)
    tvm_tirx.ir.assert_structural_equal(after, Expected4_multi_view_get)


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

    @I.ir_module
    class Dispatched:
        @T.function(private=True)
        def main(
            A: T.Tensor(
                (128, 32),
                T.float16,
                layout="default",
            ),
        ):
            with T.device_scope(
                1,
                1,
                1,
                128,
                1,
                1,
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                bx: T.let[T.int32] = 0
                by: T.let[T.int32] = 0
                bz: T.let[T.int32] = 0
                _lane: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                tid: T.let[T.int32] = T.cuda.thread_idx("x")
                A_smem = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (128, 32),
                            T.float16,
                            scope="shared",
                            layout=T.ComposeLayout(
                                3,
                                3,
                                3,
                                T.TileLayout.from_iters(shard=[T.Iter(512, 1, "m")], replica=[]),
                            ),
                        )
                    ]
                )
                thread_col: T.let[T.int32] = 4
                thread_row: T.let[T.int32] = 32
                for tile in range(128 // thread_row):
                    row: T.let[T.int32] = tile * thread_row + tid // thread_col
                    col: T.let[T.int32] = tid % thread_col * 8
                    for vec in T.vectorized(0, 8):
                        A_smem[row, col + vec] = A[bx * 128 + row, col + vec]

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        after = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_lower_decl_tensor_pointer():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
            buf = T.alloc_tensor(ty_args=[T.Tensor([1024], "uint8", scope="shared.dyn")])
            A = T.decl_tensor(
                buf.data,
                ty_args=[
                    T.Tensor(
                        [128], "float16", layout=T.TileLayout(T.S[[128]] + (32)), data=buf.data
                    )
                ],
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
                        T.Tensor(
                            (128,),
                            "float16",
                            scope="shared.dyn",
                            layout=T.TileLayout(T.S[(128,)] + 32),
                        )
                    ],
                )
                T.address_of(A[64])

    @I.ir_module
    class Dispatched:
        @T.function(private=True)
        def main():
            with T.device_scope(
                1,
                1,
                1,
                128,
                1,
                1,
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                buf = T.alloc_tensor(ty_args=[T.Tensor((1024,), T.uint8, scope="shared.dyn")])
                A = T.decl_tensor(
                    buf.data,
                    ty_args=[
                        T.Tensor(
                            (128,),
                            T.float16,
                            scope="shared.dyn",
                            layout=T.TileLayout(T.S[(128,)] + 32),
                        )
                    ],
                )
                T.address_of(A[64])

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        after = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


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

    with tvm_tirx.target.Target("cuda"):

        @I.ir_module
        class Dispatched:
            @T.function(private=True)
            def main(
                A: T.Tensor(
                    (32,),
                    T.int32,
                    layout="default",
                ),
            ):
                with T.device_scope(
                    1,
                    1,
                    1,
                    32,
                    1,
                    1,
                    attrs=_CTA_LAUNCH_ATTRS,
                ):
                    lane_id: T.let[T.int32] = T.cuda.thread_idx("x") % 32
                    if T.cuda.elect_sync() != T.uint32(0):
                        A[lane_id] = 1
                    else:
                        A[lane_id] = 0

        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        lowered = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(lowered, Expected)

    script = lowered.script(extra_config={"tirx.prefix": "T"})
    assert "T.cuda.elect_sync() != T.uint32(0)" in script
    assert "else:" in script


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
                    addr[0] + T.uint32(offset) + T.cast(i, ty="uint32"),
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
                        expected_address(addr, i), ty=T.Ptr(T.PrimType("float32"), scope="tmem")
                    ),
                    ty_args=[T.Tensor((64,), T.float32, scope="tmem")],
                )
                T.evaluate(tensor.data)

    with tvm_tirx.target.Target("cuda"):

        @I.ir_module
        class Dispatched:
            @T.function(private=True)
            def main():
                addr = T.alloc_tensor(ty_args=[T.Tensor((1,), T.uint32, scope="shared")])
                for i in range(2):
                    tensor = T.cuda.decl_tmem(
                        expected_address(addr, i),
                        ty_args=[
                            T.Tensor(
                                (64,),
                                T.float32,
                                scope="tmem",
                                layout="default",
                            )
                        ],
                    )
                    T.evaluate(tensor.data)

        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        lowered = LowerTensorLayouts()(Dispatched)["main"]
    tvm_tirx.ir.assert_structural_equal(lowered, Expected["main"])
    assert not tvm_tirx.tirx.analysis.undefined_vars(lowered.body, lowered.params)
    tvm_tirx.tirx.analysis.verify_well_formed(lowered)
    bindings = []
    tvm_ffi.structural_walk(lowered.body, (tvm_tirx.ir.Bind, bindings.append))
    address = next(
        node.var for node in bindings if node.var.scope() == tvm_tirx.runtime.MemoryScope("shared")
    )
    tensor = next(
        node for node in bindings if node.var.scope() == tvm_tirx.runtime.MemoryScope("tmem")
    )
    loads = []
    tvm_ffi.structural_walk(tensor.value.args[0], (tvm_tirx.ir.TensorLoad, loads.append))
    assert len(loads) == 1
    assert loads[0].source.same_as(address)
    tvm_tirx.ir.assert_structural_equal(tensor.var.ty, tensor.value.ty)


def test_lower_tensor_offset():
    @I.ir_module
    class Before:
        @T.function(private=True)
        def main():
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
            A = T.alloc_tensor(ty_args=[T.Tensor([64, 64], "float16", scope="local")])
            A0 = T.decl_tensor(
                A.ptr_to([32, 32]),
                ty_args=[T.Tensor([64], "float16", data=A.ptr_to([32, 32]))],
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
                    ty_args=[T.Tensor((64,), "float16", scope="local", layout=None)],
                )
                T.address_of(A0[32])

    @I.ir_module
    class Dispatched:
        @T.function(private=True)
        def main():
            with T.device_scope(
                1,
                1,
                1,
                128,
                1,
                1,
                attrs=_CTA_LAUNCH_ATTRS,
            ):
                A = T.alloc_tensor(ty_args=[T.Tensor((64, 64), T.float16, scope="local")])
                A0 = T.decl_tensor(
                    T.address_of(A[32, 32]), ty_args=[T.Tensor((64,), T.float16, scope="local")]
                )
                T.address_of(A0[32])

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        after = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


def test_lower_alloc_decl_tensor_outside_of_parser():
    @T.meta_class
    class State:
        def __init__(self, smem):
            self.A = T.alloc_local([1], "float16")
            self.B = T.alloc_local([1], "float16")
            self.C = T.bind(
                T.decl_tensor(smem, ty_args=[T.Tensor([1], "float16", scope="shared.dyn")])
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
                ty_args=[T.Tensor((1,), "float16", scope="shared.dyn", layout=None)],
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

    @I.ir_module
    class Dispatched:
        @T.function(private=True)
        def main():
            smem = T.alloc_tensor(ty_args=[T.Tensor((100,), T.uint8, scope="shared.dyn")])
            v = T.alloc_tensor(ty_args=[T.Tensor((1,), T.float16, scope="local")])
            v_1 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.float16, scope="local")])
            v_2 = T.decl_tensor(smem.data, ty_args=[T.Tensor((1,), T.float16, scope="shared.dyn")])
            v[0] = T.float16(1.0)
            v_1[0] = T.float16(2.0)
            v_2[0] = T.float16(3.0)
            D = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
            D[0] = 1
            D[0] = D[0] + 1
            E = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
            E[0] = 2
            E[0] = E[0] + 2
            v_3 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
            v_3[0] = 3
            v_3[0] = v_3[0] + 3
            v_4 = T.alloc_tensor(ty_args=[T.Tensor((1,), T.int32, scope="local")])
            v_4[0] = 4
            v_4[0] = v_4[0] + 4

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        after = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


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

    @I.ir_module
    class Dispatched:
        @T.function(private=True)
        def main(
            out: T.Tensor(
                (128, 4),
                T.float32,
                layout="default",
            ),
        ):
            with T.device_scope(
                1,
                1,
                1,
                128,
                1,
                1,
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
                reg_wg = T.alloc_tensor(
                    ty_args=[
                        T.Tensor(
                            (128, 4),
                            T.float32,
                            scope="local",
                            layout=T.TileLayout.from_iters(
                                shard=[T.Iter(128, 1, "tid_in_wg"), T.Iter(4, 1, "m")], replica=[]
                            ),
                        )
                    ]
                )
                v = T.decl_tensor(
                    reg_wg.data,
                    ty_args=[
                        T.Tensor(
                            (4,),
                            T.float32,
                            scope="local",
                            layout="default",
                        )
                    ],
                )
                for i in range(4):
                    v[i] = out[lane_id + warp_id * 32, i]

    with tvm_tirx.target.Target("cuda"):
        dispatched = TileDispatch()(Before)
        tvm_tirx.ir.assert_structural_equal(dispatched, Dispatched)
        after = LowerTensorLayouts()(Dispatched)
    tvm_tirx.ir.assert_structural_equal(after, Expected)


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

    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        After = LowerTensorLayouts()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)


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

    with tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_100a"}):
        After = LowerTensorLayouts()(Before)
    tvm_tirx.ir.assert_structural_equal(After, Expected)
