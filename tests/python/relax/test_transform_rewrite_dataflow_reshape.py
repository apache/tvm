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
                            (v_ax1 * 12 + v_ax2 * 3 + v_ax2) % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                        (v_ax1 * 12 + v_ax2 * 3 + v_ax2) % T.int64(3),
                    ]

        @T.prim_func
        def expand_dims(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(3)), "float32"),
            expand_dims: T.Buffer(
                (T.int64(2), T.int64(1), T.int64(4), T.int64(1), T.int64(3)),
                "float32",
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
                            (v_ax1 * T.int64(12) + v_ax2 * T.int64(3) + v_ax2) % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * T.int64(12) + v_ax1 * T.int64(3) + v_ax2) // T.int64(3),
                        (v_ax1 * T.int64(12) + v_ax2 * T.int64(3) + v_ax2) % T.int64(3),
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
        def reshape5(rxplaceholder: T.Buffer((T.int64(2), T.int64(4096), T.int64(320)), "float16"), T_reshape: T.Buffer((T.int64(2), T.int64(4096), T.int64(5), T.int64(64)), "float16")):
            T.func_attr({"global_symbol": "main", "op_pattern": 2, "tir.noalias": True})
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
    # fmt: on

    assert relax.analysis.has_reshape_pattern(Module["reshape5"])


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
                            (v_ax1 * 12 + v_ax2 * 3 + v_ax2) % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = rxplaceholder[
                        (v_ax0 * 12 + v_ax1 * 3 + v_ax2) // T.int64(3),
                        (v_ax1 * 12 + v_ax2 * 3 + v_ax2) % T.int64(3),
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


if __name__ == "__main__":
    tvm.testing.main()
