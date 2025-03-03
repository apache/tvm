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
import numpy as np
import pytest
import tvm
import tvm.script
from tvm import te, topi
from tvm.script import tir as T


@tvm.script.ir_module
class PreRollingBuffer:
    @T.prim_func
    def main(
        A: T.handle,
        tensor: T.handle,
        tensor_2: T.Buffer(
            [1, 10, 12, 16],
            dtype="int8",
            elem_offset=0,
            align=64,
            offset_factor=1,
        ),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A_1 = T.match_buffer(
            A, [1, 12, 14, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1
        )
        tensor_1 = T.match_buffer(
            tensor, [1, 8, 8, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1
        )
        # body
        T.realize(tensor_1[0:1, 0:8, 0:8, 0:16], "")
        for ax1_outer in T.serial(0, 2):
            T.realize(tensor_2[0:1, (ax1_outer * 4) : ((ax1_outer * 4) + 6), 0:12, 0:16], "")
            T.attr(tensor_2, "rolling_buffer_scope", True)
            for ax1 in T.serial(0, 6):
                for ax2 in T.serial(0, 12):
                    for ax3 in T.serial(0, 16):
                        tensor_2[0, (ax1 + (ax1_outer * 4)), ax2, ax3] = T.int8(0)
                        for dh in T.serial(0, 3):
                            for dw in T.serial(0, 3):
                                tensor_2[0, (ax1 + (ax1_outer * 4)), ax2, ax3] = T.max(
                                    tensor_2[0, (ax1 + (ax1_outer * 4)), ax2, ax3],
                                    A_1[0, ((ax1 + (ax1_outer * 4)) + dh), (ax2 + dw), ax3],
                                )
            for ax1_inner in T.serial(0, 4):
                for ax2_inner in T.serial(0, 8):
                    for ax3_inner in T.serial(0, 16):
                        tensor_1[
                            0,
                            (ax1_inner + (ax1_outer * 4)),
                            ax2_inner,
                            ax3_inner,
                        ] = T.int8(0)
                        for dh_1 in T.serial(0, 3):
                            for dw_1 in T.serial(0, 5):
                                tensor_1[
                                    0,
                                    (ax1_inner + (ax1_outer * 4)),
                                    ax2_inner,
                                    ax3_inner,
                                ] = T.max(
                                    tensor_1[
                                        0,
                                        (ax1_inner + (ax1_outer * 4)),
                                        ax2_inner,
                                        ax3_inner,
                                    ],
                                    tensor_2[
                                        0,
                                        ((ax1_inner + (ax1_outer * 4)) + dh_1),
                                        (ax2_inner + dw_1),
                                        ax3_inner,
                                    ],
                                )


@tvm.script.ir_module
class PostRollingBuffer:
    @T.prim_func
    def main(
        A: T.handle,
        tensor: T.handle,
        tensor_2: T.Buffer(
            [1, 10, 12, 16],
            dtype="int8",
            elem_offset=0,
            align=64,
            offset_factor=1,
        ),
    ) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A_1 = T.match_buffer(
            A, [1, 12, 14, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1
        )
        tensor_1 = T.match_buffer(
            tensor, [1, 8, 8, 16], dtype="int8", elem_offset=0, align=64, offset_factor=1
        )
        # body
        T.realize(tensor_1[0:1, 0:8, 0:8, 0:16], "")
        T.realize(tensor_2[0:1, 0:6, 0:12, 0:16], "")
        for ax1_outer in T.serial(0, 2):
            for ax1 in T.serial(0, 6):
                for ax2 in T.serial(0, 12):
                    for ax3 in T.serial(0, 16):
                        if T.likely(((ax1_outer < 1) or (ax1 >= 2)), dtype="bool"):
                            tensor_2[
                                0,
                                T.floormod((ax1 + (ax1_outer * 4)), 6),
                                ax2,
                                ax3,
                            ] = T.int8(0)
                        for dh in T.serial(0, 3):
                            for dw in T.serial(0, 3):
                                if T.likely(((ax1_outer < 1) or (ax1 >= 2)), dtype="bool"):
                                    tensor_2[
                                        0, T.floormod((ax1 + (ax1_outer * 4)), 6), ax2, ax3
                                    ] = T.max(
                                        tensor_2[
                                            0, T.floormod((ax1 + (ax1_outer * 4)), 6), ax2, ax3
                                        ],
                                        A_1[0, ((ax1 + (ax1_outer * 4)) + dh), (ax2 + dw), ax3],
                                    )
            for ax1_inner in T.serial(0, 4):
                for ax2_inner in T.serial(0, 8):
                    for ax3_inner in T.serial(0, 16):
                        tensor_1[
                            0,
                            (ax1_inner + (ax1_outer * 4)),
                            ax2_inner,
                            ax3_inner,
                        ] = T.int8(0)
                        for dh_1 in T.serial(0, 3):
                            for dw_1 in T.serial(0, 5):
                                tensor_1[
                                    0,
                                    (ax1_inner + (ax1_outer * 4)),
                                    ax2_inner,
                                    ax3_inner,
                                ] = T.max(
                                    tensor_1[
                                        0, (ax1_inner + (ax1_outer * 4)), ax2_inner, ax3_inner
                                    ],
                                    tensor_2[
                                        0,
                                        T.floormod(((ax1_inner + (ax1_outer * 4)) + dh_1), 6),
                                        (ax2_inner + dw_1),
                                        ax3_inner,
                                    ],
                                )


def test_rolling_buffer_ir_transform():
    mod = PreRollingBuffer
    mod = tvm.tir.transform.InjectRollingBuffer()(mod)
    script = mod.script()
    mod = tvm.script.from_source(script)
    tvm.ir.assert_structural_equal(mod["main"], PostRollingBuffer["main"], True)


if __name__ == "__main__":
    tvm.testing.main()
