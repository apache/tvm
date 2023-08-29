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
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T


def test_reshape_expand_dims():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def reshape(
            rxplaceholder: T.Buffer((T.int64(8), T.int64(3)), "float32"),
            T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"),
        ):
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        rxplaceholder[
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                    ]

        @T.prim_func
        def expand_dims(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"),
            expand_dims: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)), "float32"
            ),
        ):
            for i0, i1, i2, i3, i4 in T.grid(
                T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)
            ):
                with T.block("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(rxplaceholder[i0_1, i2_1, i4_1])
                    T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i2_1, i4_1]

        @R.function
        def main(
            x: R.Tensor((8, 3), dtype="float32")
        ) -> R.Tensor((2, 1, 4, 1, 3), dtype="float32"):
            cls = Module
            with R.dataflow():
                y = R.call_tir(cls.reshape, (x,), out_sinfo=R.Tensor((2, 4, 3), dtype="float32"))
                z = R.call_tir(
                    cls.expand_dims, (y,), out_sinfo=R.Tensor((2, 1, 4, 1, 3), "float32")
                )
                R.output(z)
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def reshape(
            rxplaceholder: T.Buffer((T.int64(8), T.int64(3)), "float32"),
            T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"),
        ):
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        rxplaceholder[
                            (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3),
                            (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3),
                        (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) % T.int64(3),
                    ]

        @T.prim_func
        def expand_dims(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"),
            expand_dims: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)), "float32"
            ),
        ):
            for i0, i1, i2, i3, i4 in T.grid(
                T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)
            ):
                with T.block("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(rxplaceholder[i0_1, i2_1, i4_1])
                    T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i2_1, i4_1]

        @R.function
        def main(
            x: R.Tensor((8, 3), dtype="float32")
        ) -> R.Tensor((2, 1, 4, 1, 3), dtype="float32"):
            with R.dataflow():
                cls = Expected
                y: R.Tensor((2, 4, 3), "float32") = R.reshape(x, (2, 4, 3))
                # Note: `z` is the output var of the dataflow block, and is thus
                # not expected to be rewritten.
                z = R.call_tir(
                    cls.expand_dims, (y,), out_sinfo=R.Tensor((2, 1, 4, 1, 3), dtype="float32")
                )
                R.output(z)
            return z

    assert relax.analysis.has_reshape_pattern(Module["expand_dims"])
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_reshape_pattern_detect():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32")):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(10)):
                        with T.block("T_reshape"):
                            v_ax0 = T.axis.spatial(T.int64(2), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(1310720))
                            v_ax1 = T.axis.spatial(T.int64(4096), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(1310720) // T.int64(320))
                            v_ax2 = T.axis.spatial(T.int64(5), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(320) // T.int64(64))
                            v_ax3 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                            T.reads(rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)])
                            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)]

        @T.prim_func
        def expand_dims(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32"),
            expand_dims: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)),
                "float32",
            ),
        ):
            for i0, i1, i2, i3, i4, i5 in T.grid(
                T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)
            ):
                with T.block("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[i0_1, i2_1, i4_1, i5_1])
                    T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1, i5_1]

        @R.function
        def main(
            x: R.Tensor((2, 4096, 320), dtype="float32")
        ) -> R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"):
            cls = Module
            with R.dataflow():
                y = R.call_tir(cls.reshape, (x,), out_sinfo=R.Tensor((2, 4096, 5, 64), dtype="float32"))
                z = R.call_tir(
                    cls.expand_dims, (y,), out_sinfo=R.Tensor((2, 1, 4096, 1, 5, 64), "float32")
                )
                R.output(z)
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def expand_dims(rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32"), expand_dims_1: T.Buffer((T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)), "float32")):
            # with T.block("root"):
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(1), T.int64(4096), T.int64(1), T.int64(5), T.int64(64)):
                with T.block("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[i0_1, i2_1, i4_1, i5_1])
                    T.writes(expand_dims_1[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    expand_dims_1[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1, i5_1]

        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float32")):
            # with T.block("root"):
            for ax0_ax1_ax2_ax3_fused_1 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
                for ax0_ax1_ax2_ax3_fused_2 in T.thread_binding(T.int64(1024), thread="threadIdx.x"):
                    for ax0_ax1_ax2_ax3_fused_0 in range(T.int64(10)):
                        with T.block("T_reshape"):
                            v_ax0 = T.axis.spatial(T.int64(2), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) // T.int64(1310720))
                            v_ax1 = T.axis.spatial(T.int64(4096), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(1310720) // T.int64(320))
                            v_ax2 = T.axis.spatial(T.int64(5), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(320) // T.int64(64))
                            v_ax3 = T.axis.spatial(T.int64(64), (ax0_ax1_ax2_ax3_fused_0 * T.int64(262144) + ax0_ax1_ax2_ax3_fused_1 * T.int64(1024) + ax0_ax1_ax2_ax3_fused_2) % T.int64(64))
                            T.reads(rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)])
                            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(320)]

        @R.function
        def main(x: R.Tensor((2, 4096, 320), dtype="float32")) -> R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"):
            cls = Expected
            with R.dataflow():
                y: R.Tensor((2, 4096, 5, 64), dtype="float32") = R.reshape(x, R.shape([2, 4096, 5, 64]))
                z = R.call_tir(cls.expand_dims, (y,), out_sinfo=R.Tensor((2, 1, 4096, 1, 5, 64), dtype="float32"))
                R.output(z)
            return z
    # fmt: on

    assert relax.analysis.has_reshape_pattern(Module["reshape"])
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_reshape_non_dataflow():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def reshape(
            rxplaceholder: T.Buffer((T.int64(8), T.int64(3)), "float32"),
            T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"),
        ):
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(
                        rxplaceholder[
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                            (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) % T.int64(3),
                    ]

        @R.function
        def main(x: R.Tensor((8, 3), dtype="float32")) -> R.Tensor((2, 4, 3), dtype="float32"):
            cls = Module
            y = R.call_tir(cls.reshape, (x,), out_sinfo=R.Tensor((2, 4, 3), dtype="float32"))
            return y

    assert relax.analysis.has_reshape_pattern(Module["reshape"])
    # The binding var of the call_tir is not a DataflowVar. So the pass does no change.
    mod = relax.transform.RewriteDataflowReshape()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_tuple_get_reshape():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def fused_reshape5(
            lv2_0: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_1: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_2: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            T_reshape_handle_intermediate: T.Buffer(
                (T.int64(2), T.int64(4096), T.int64(8), T.int64(40)), "float16"
            ),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(8), T.int64(40)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(
                        lv2_0[
                            (
                                ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1)
                                // T.int64(4096)
                                + v_ax0
                            )
                            % T.int64(2),
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                            (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                        ]
                    )
                    T.writes(T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2_0[
                        (
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096)
                            + v_ax0
                        )
                        % T.int64(2),
                        ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                        (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                    ]

        @R.function
        def main(
            lv41_1: R.Tuple(
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
            )
        ) -> R.Tensor((2, 4096, 8, 40), dtype="float16"):
            cls = Module
            with R.dataflow():
                lv: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[0]
                lv1: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[1]
                lv2: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[2]
                lv645 = R.call_tir(
                    cls.fused_reshape5,
                    (lv, lv1, lv2),
                    out_sinfo=R.Tensor((2, 4096, 8, 40), dtype="float16"),
                )
                out: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.add(lv645, lv645)
                R.output(out)
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def fused_reshape5(
            lv2_0: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_1: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            lv2_2: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"),
            T_reshape_handle_intermediate: T.Buffer(
                (T.int64(2), T.int64(4096), T.int64(8), T.int64(40)), "float16"
            ),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(8), T.int64(40)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(
                        lv2_0[
                            (
                                ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1)
                                // T.int64(4096)
                                + v_ax0
                            )
                            % T.int64(2),
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                            (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                        ]
                    )
                    T.writes(T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_handle_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv2_0[
                        (
                            ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) // T.int64(4096)
                            + v_ax0
                        )
                        % T.int64(2),
                        ((v_ax2 * T.int64(40) + v_ax3) // T.int64(320) + v_ax1) % T.int64(4096),
                        (v_ax2 * T.int64(40) + v_ax3) % T.int64(320),
                    ]

        @R.function
        def main(
            lv41_1: R.Tuple(
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
                R.Tensor((2, 4096, 320), dtype="float16"),
            )
        ) -> R.Tensor((2, 4096, 8, 40), dtype="float16"):
            with R.dataflow():
                lv: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[0]
                lv1: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[1]
                lv2: R.Tensor((2, 4096, 320), dtype="float16") = lv41_1[2]
                lv645: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.reshape(
                    lv, R.shape([2, 4096, 8, 40])
                )
                out: R.Tensor((2, 4096, 8, 40), dtype="float16") = R.add(lv645, lv645)
                R.output(out)
            return out

    rewritten = relax.transform.RewriteDataflowReshape()(Module)
    tvm.ir.assert_structural_equal(rewritten, Expected)


def test_invalid_reshape():
    @tvm.script.ir_module
    class Module:
        # The strided_slice op has the reshape pattern, but it can take only a part of the input.
        # It can't be replaced with the reshape op because reshape expects to preserve the "volume"
        # of the input.
        @T.prim_func
        def strided_slice(
            A: T.Buffer((T.int64(1), T.int64(1024)), "int32"),
            T_strided_slice: T.Buffer((T.int64(1), T.int64(1000)), "int32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1000)):
                with T.block("T_strided_slice"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_strided_slice[v_ax0, v_ax1])
                    T_strided_slice[v_ax0, v_ax1] = A[v_ax0, v_ax1]

        @T.prim_func
        def add_one(
            A: T.Buffer((T.int64(1), T.int64(1000)), "int32"),
            T_add_one: T.buffer((T.int64(1), T.int64(1000)), "int32"),
        ):
            for ax0, ax1 in T.grid(T.int64(1), T.int64(1000)):
                with T.block("T_add_one"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_add_one[v_ax0, v_ax1])
                    T_add_one[v_ax0, v_ax1] = A[v_ax0, v_ax1] + 1

        @R.function
        def main(A: R.Tensor((1, 1024), dtype="int32")) -> R.Tensor((1, 1000), dtype="int32"):
            with R.dataflow():
                cls = Module
                S = R.call_tir(
                    cls.strided_slice, (A,), out_sinfo=R.Tensor((1, 1000), dtype="int32")
                )
                A = R.call_tir(cls.add_one, (S,), out_sinfo=R.Tensor((1, 1000), dtype="int32"))
                R.output(A)
            return A

    assert relax.analysis.has_reshape_pattern(Module["strided_slice"])
    rewritten = relax.transform.RewriteDataflowReshape()(Module)
    tvm.ir.assert_structural_equal(rewritten, Module)


def test_reshape_detect_nop():
    @tvm.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((8, 8), dtype="float16")) -> R.Tensor((8, 8), dtype="float16"):
            with R.dataflow():
                gv = R.call_pure_packed(
                    "foo", x, x, sinfo_args=(R.Tensor((8, 8), dtype="float16"),)
                )
                out = R.call_pure_packed(
                    "foo", gv, gv, sinfo_args=(R.Tensor((8, 8), dtype="float16"),)
                )
                R.output(out)
            return out

    rewritten = relax.transform.RewriteDataflowReshape()(Module)
    tvm.ir.assert_structural_equal(rewritten, Module)


def test_reshape_scalar():
    @tvm.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
            with R.dataflow():
                lv1: R.Tensor((1,), dtype="float32") = R.reshape(x, [1])
                lv2: R.Tensor((1,), dtype="float32") = R.add(lv1, lv1)
                R.output(lv2)
            return lv2

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def add(
            A: T.Buffer((T.int64(1),), "float32"),
            B: T.Buffer((T.int64(1),), "float32"),
            T_add: T.Buffer((T.int64(1),), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0 in range(T.int64(1)):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                    T.reads(A[v_ax0], B[v_ax0])
                    T.writes(T_add[v_ax0])
                    T_add[v_ax0] = A[v_ax0] + B[v_ax0]

        @T.prim_func(private=True)
        def reshape(A: T.Buffer((), "float32"), T_reshape: T.Buffer((T.int64(1),), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for ax0 in range(T.int64(1)):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(1), ax0)
                    T.reads(A[()])
                    T.writes(T_reshape[v_ax0])
                    T_reshape[v_ax0] = A[()]

        @R.function
        def main(x: R.Tensor((), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv1: R.Tensor((1,), dtype="float32") = R.reshape(x, R.shape([1]))
                lv2 = R.call_tir(cls.add, (lv1, lv1), out_sinfo=R.Tensor((1,), dtype="float32"))
                R.output(lv2)
            return lv2

    mod = Module
    mod = relax.transform.LegalizeOps()(mod)
    rewritten = relax.transform.RewriteDataflowReshape()(mod)
    tvm.ir.assert_structural_equal(rewritten, Expected)


if __name__ == "__main__":
    tvm.testing.main()
