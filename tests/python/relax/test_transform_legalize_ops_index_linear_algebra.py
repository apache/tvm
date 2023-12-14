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
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T, ir as I
import tvm.testing


##################### Indexing #####################


def test_take():
    # fmt: off
    @tvm.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32"), indices: R.Tensor((4,), "int64")) -> R.Tensor((2, 4, 4), "float32"):
            gv: R.Tensor((2, 4, 4), "float32") = R.take(x, indices, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32"), indices: R.Tensor((4,), "int64")) -> R.Tensor((2, 4, 4), "float32"):
            gv = R.call_tir(Expected.take, (x, indices), R.Tensor((2, 4, 4), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def take(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), rxplaceholder_1: T.Buffer(T.int64(4), "int64"), T_take: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
                with T.block("T_take"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, rxplaceholder_1[ax1], ax2], rxplaceholder_1[ax1])
                    T.writes(T_take[ax0, ax1, ax2])
                    T_take[ax0, ax1, ax2] = rxplaceholder[ax0, rxplaceholder_1[ax1], ax2]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_take_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Take:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), indices: R.Tensor(("i",), "int64")) -> R.Tensor(("m", "i"), "float32"):
            m = T.int64()
            i = T.int64()
            gv: R.Tensor((m, i), "float32") = R.take(x, indices, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32"), indices: R.Tensor(("i",), "int64")) -> R.Tensor(("m", "i"), "float32"):
            m = T.int64()
            i = T.int64()
            gv = R.call_tir(Expected.take, (x, indices), R.Tensor((m, i), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def take(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_T_take: T.handle):
            T.func_attr({"tir.noalias": True})
            i = T.int64()
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [i], dtype="int64")
            T_take = T.match_buffer(var_T_take, [m, i], dtype="float32")
            for i0, i1 in T.grid(m, i):
                with T.block("T_take"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, rxplaceholder_1[ax1]], rxplaceholder_1[ax1])
                    T.writes(T_take[ax0, ax1])
                    T_take[ax0, ax1] = rxplaceholder[ax0, rxplaceholder_1[ax1]]
    # fmt: on

    mod = LegalizeOps()(Take)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_strided_slice():
    # fmt: off
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) -> R.Tensor((4, 9, 10, 3), "float32"):
            gv: R.Tensor((4, 9, 10, 3), "float32") = R.strided_slice(x, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), dtype="float32")) -> R.Tensor((4, 9, 10, 3), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), R.Tensor((4, 9, 10, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def strided_slice(rxplaceholder: T.Buffer((T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(4), T.int64(9), T.int64(10), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(9), T.int64(10), T.int64(3)):
                with T.block("T_strided_slice_with_axes"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)])
                    T.writes(T_strided_slice_with_axes[ax0, ax1, ax2, ax3])
                    T_strided_slice_with_axes[ax0, ax1, ax2, ax3] = rxplaceholder[ax0 * T.int64(2) + T.int64(1), ax1, ax2, T.int64(8) - ax3 * T.int64(3)]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_no_strides():
    # fmt: off
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) :
            gv: R.Tensor((4, 9, 10, 3), "float32") = R.strided_slice(x, axes=[0, 1, 3], begin=[1, 0, 2], end=[8, 9, 4])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), dtype="float32")):
            gv = R.call_tir(Expected.strided_slice, (x,), out_sinfo=R.Tensor((7, 9, 10, 2), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def strided_slice(rxplaceholder: T.Buffer((T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"), T_strided_slice_with_axes: T.Buffer((T.int64(7), T.int64(9), T.int64(10), T.int64(2)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(7), T.int64(9), T.int64(10), T.int64(2)):
                with T.block("T_strided_slice_with_axes"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0 + T.int64(1), v_ax1, v_ax2, v_ax3 + T.int64(2)])
                    T.writes(T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_strided_slice_with_axes[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0 + T.int64(1), v_ax1, v_ax2, v_ax3 + T.int64(2)]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_symbolic_sliced_axis():
    # fmt: off
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor((2, "n"), "float32"):
            n = T.int64()
            gv: R.Tensor((2, n), "float32") = R.strided_slice(x, axes=[0], begin=[1], end=[8], strides=[3])
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def strided_slice(var_A: T.handle, var_T_dynamic_strided_slice_with_axes: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            m, n = T.int64(), T.int64()
            A = T.match_buffer(var_A, (m, n))
            T_dynamic_strided_slice_with_axes = T.match_buffer(var_T_dynamic_strided_slice_with_axes, (T.int64(3), n))
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(3), n):
                with T.block("T_dynamic_strided_slice_with_axes"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0 * T.int64(3) + T.int64(1), v_ax1])
                    T.writes(T_dynamic_strided_slice_with_axes[v_ax0, v_ax1])
                    T_dynamic_strided_slice_with_axes[v_ax0, v_ax1] = A[v_ax0 * T.int64(3) + T.int64(1), v_ax1]

        @R.function
        def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor((3, "n"), dtype="float32"):
            n = T.int64()
            m = T.int64()
            cls = Expected
            gv = R.call_tir(cls.strided_slice, (x,), out_sinfo=R.Tensor((3, n), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((10, "n"), "float32")) -> R.Tensor((3, "n"), "float32"):
            n = T.int64()
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0], begin=[1], end=[8], strides=[3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, "n"), dtype="float32")) -> R.Tensor((3, "n"), dtype="float32"):
            n = T.int64()
            gv = R.call_tir(Expected.strided_slice, (x,), R.Tensor((3, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def strided_slice(var_rxplaceholder: T.handle, var_T_strided_slice_with_axes: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(10), n], dtype="float32")
            T_strided_slice_with_axes = T.match_buffer(var_T_strided_slice_with_axes, [T.int64(3), n], dtype="float32")
            for i0, i1 in T.grid(T.int64(3), n):
                with T.block("T_strided_slice_with_axes"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1])
                    T.writes(T_strided_slice_with_axes[ax0, ax1])
                    T_strided_slice_with_axes[ax0, ax1] = rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1]
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_strided_slice_symbolic_bound():
    # fmt: off
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((10, "n"), "float32")) -> R.Tensor((3, "n"), "float32"):
            n = T.int64(is_size_var=True)
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0, 1], begin=[1, 0], end=[8, n], strides=[3, 1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, "n"), dtype="float32")) -> R.Tensor((3, "n"), dtype="float32"):
            n = T.int64(is_size_var=True)
            gv = R.call_tir(Expected.strided_slice, (x,), R.Tensor((3, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def strided_slice(var_rxplaceholder: T.handle, var_T_strided_slice_with_axes: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(10), n], dtype="float32")
            T_strided_slice_with_axes = T.match_buffer(var_T_strided_slice_with_axes, [T.int64(3), n], dtype="float32")
            for i0, i1 in T.grid(T.int64(3), n):
                with T.block("T_strided_slice_with_axes"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1])
                    T.writes(T_strided_slice_with_axes[ax0, ax1])
                    T_strided_slice_with_axes[ax0, ax1] = rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1]


def test_strided_slice_non_unit_stride():
    # fmt: off
    @tvm.script.ir_module
    class StridedSlice:
        @R.function
        def main(x: R.Tensor((10, "n"), "float32")) -> R.Tensor((3, "n"), "float32"):
            n = T.int64(is_size_var=True)
            gv: R.Tensor((3, n), "float32") = R.strided_slice(x, axes=[0, 1], begin=[1, 0], end=[8, n], strides=[3, 1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((10, "n"), dtype="float32")) -> R.Tensor((3, "n"), dtype="float32"):
            n = T.int64(is_size_var=True)
            gv = R.call_tir(Expected.strided_slice, (x,), R.Tensor((3, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def strided_slice(var_rxplaceholder: T.handle, var_T_strided_slice_with_axes: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [T.int64(10), n], dtype="float32")
            T_strided_slice_with_axes = T.match_buffer(var_T_strided_slice_with_axes, [T.int64(3), n], dtype="float32")
            for i0, i1 in T.grid(T.int64(3), n):
                with T.block("T_strided_slice_with_axes"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1])
                    T.writes(T_strided_slice_with_axes[ax0, ax1])
                    T_strided_slice_with_axes[ax0, ax1] = rxplaceholder[ax0 * T.int64(3) + T.int64(1), ax1]


def test_dynamic_strided_slice():
    # fmt: off
    @tvm.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), "float32"), begin: R.Tensor((4,),"int64"), end: R.Tensor((4,),"int64"), strides: R.Tensor((4,),"int64")) -> R.Tensor("float32", ndim=4):
            gv: R.Tensor("float32", ndim=4) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dynamic_strided_slice(
            rxplaceholder: T.Buffer(
                (T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"
            ),
            rxplaceholder_1: T.Buffer((T.int64(4),), "int64"),
            rxplaceholder_2: T.Buffer((T.int64(4),), "int64"),
            rxplaceholder_3: T.Buffer((T.int64(4),), "int64"),
            var_T_strided_slice_dynamic: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            s, s_1, s_2, s_3 = T.int64(), T.int64(), T.int64(), T.int64()
            T_strided_slice_dynamic = T.match_buffer(
                var_T_strided_slice_dynamic, (s, s_1, s_2, s_3)
            )
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(s, s_1, s_2, s_3):
                with T.block("T_strided_slice_dynamic"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(
                        rxplaceholder[
                            T.min(rxplaceholder_1[T.int64(0)], T.int64(7))
                            + v_ax0 * rxplaceholder_3[T.int64(0)],
                            T.min(rxplaceholder_1[T.int64(1)], T.int64(8))
                            + v_ax1 * rxplaceholder_3[T.int64(1)],
                            T.min(rxplaceholder_1[T.int64(2)], T.int64(9))
                            + v_ax2 * rxplaceholder_3[T.int64(2)],
                            T.min(rxplaceholder_1[T.int64(3)], T.int64(9))
                            + v_ax3 * rxplaceholder_3[T.int64(3)],
                        ],
                        rxplaceholder_1[T.int64(0) : T.int64(4)],
                        rxplaceholder_3[T.int64(0) : T.int64(4)],
                    )
                    T.writes(T_strided_slice_dynamic[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_strided_slice_dynamic[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[
                        T.min(rxplaceholder_1[T.int64(0)], T.int64(7))
                        + v_ax0 * rxplaceholder_3[T.int64(0)],
                        T.min(rxplaceholder_1[T.int64(1)], T.int64(8))
                        + v_ax1 * rxplaceholder_3[T.int64(1)],
                        T.min(rxplaceholder_1[T.int64(2)], T.int64(9))
                        + v_ax2 * rxplaceholder_3[T.int64(2)],
                        T.min(rxplaceholder_1[T.int64(3)], T.int64(9))
                        + v_ax3 * rxplaceholder_3[T.int64(3)],
                    ]

        @T.prim_func(private=True)
        def shape_func(
            rxplaceholder: T.Buffer(
                (T.int64(8), T.int64(9), T.int64(10), T.int64(10)), "float32"
            ),
            rxplaceholder_1: T.Buffer((T.int64(4),), "int64"),
            rxplaceholder_2: T.Buffer((T.int64(4),), "int64"),
            rxplaceholder_3: T.Buffer((T.int64(4),), "int64"),
            T_shape_func_strided_slice_dynamic: T.Buffer((T.int64(4),), "int64"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i in range(T.int64(4)):
                with T.block("T_shape_func_strided_slice_dynamic"):
                    v_i = T.axis.spatial(T.int64(4), i)
                    T.reads(
                        rxplaceholder_3[v_i], rxplaceholder_1[v_i], rxplaceholder_2[v_i]
                    )
                    T.writes(T_shape_func_strided_slice_dynamic[v_i])
                    T_shape_func_strided_slice_dynamic[v_i] = T.Select(
                        rxplaceholder_3[v_i] < T.int64(0),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                )
                                - T.int64(1),
                            )
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_2[v_i] < T.int64(0),
                                        rxplaceholder_2[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_2[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                )
                                - T.int64(1),
                            )
                            - rxplaceholder_3[v_i]
                            - T.int64(1)
                        )
                        // (rxplaceholder_3[v_i] * T.int64(-1)),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_2[v_i] < T.int64(0),
                                        rxplaceholder_2[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_2[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            + rxplaceholder_3[v_i]
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(3),
                                            T.int64(10),
                                            T.Select(
                                                v_i == T.int64(2),
                                                T.int64(10),
                                                T.Select(
                                                    v_i == T.int64(1),
                                                    T.int64(9),
                                                    T.Select(
                                                        v_i == T.int64(0),
                                                        T.int64(8),
                                                        T.int64(-1),
                                                    ),
                                                ),
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(3),
                                    T.int64(10),
                                    T.Select(
                                        v_i == T.int64(2),
                                        T.int64(10),
                                        T.Select(
                                            v_i == T.int64(1),
                                            T.int64(9),
                                            T.Select(
                                                v_i == T.int64(0), T.int64(8), T.int64(-1)
                                            ),
                                        ),
                                    ),
                                ),
                            )
                            - T.int64(1)
                        )
                        // rxplaceholder_3[v_i],
                    )

        @R.function
        def main(
            x: R.Tensor((8, 9, 10, 10), dtype="float32"),
            begin: R.Tensor((4,), dtype="int64"),
            end: R.Tensor((4,), dtype="int64"),
            strides: R.Tensor((4,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=4):
            s = T.int64()
            s_1 = T.int64()
            s_2 = T.int64()
            s_3 = T.int64()
            gv = R.call_tir(
                Expected.shape_func,
                (x, begin, end, strides),
                out_sinfo=R.Tensor((4,), dtype="int64"),
            )
            gv1: R.Shape(ndim=4) = R.call_pure_packed(
                "vm.builtin.tensor_to_shape", gv, sinfo_args=(R.Shape(ndim=4),)
            )
            gv2: R.Shape([s, s_1, s_2, s_3]) = R.match_cast(
                gv1, R.Shape([s, s_1, s_2, s_3])
            )
            gv_1 = R.call_tir(
                Expected.dynamic_strided_slice,
                (x, begin, end, strides),
                out_sinfo=R.Tensor((s, s_1, s_2, s_3), dtype="float32"),
            )
            return gv_1
    # fmt: on
    mod = LegalizeOps()(DynamicStridedSlice)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dynamic_strided_slice_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class DynamicStridedSlice:
        @R.function
        def main(x: R.Tensor((10, "n"), "float32"), begin:R.Tensor((2,), "int64"), end:R.Tensor((2,), "int64"), strides:R.Tensor((2,), "int64")) -> R.Tensor("float32", ndim=2):
            n = T.int64()
            gv: R.Tensor("float32", ndim=2) = R.dynamic_strided_slice(x, begin, end, strides)
            return gv
    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dynamic_strided_slice(
            var_rxplaceholder: T.handle,
            rxplaceholder: T.Buffer((T.int64(2),), "int64"),
            rxplaceholder_1: T.Buffer((T.int64(2),), "int64"),
            rxplaceholder_2: T.Buffer((T.int64(2),), "int64"),
            var_T_strided_slice_dynamic: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            rxplaceholder_3 = T.match_buffer(var_rxplaceholder, (T.int64(10), n))
            s, s_1 = T.int64(), T.int64()
            T_strided_slice_dynamic = T.match_buffer(var_T_strided_slice_dynamic, (s, s_1))
            # with T.block("root"):
            for ax0, ax1 in T.grid(s, s_1):
                with T.block("T_strided_slice_dynamic"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(
                        rxplaceholder_3[
                            T.min(rxplaceholder[T.int64(0)], T.int64(9))
                            + v_ax0 * rxplaceholder_2[T.int64(0)],
                            T.min(rxplaceholder[T.int64(1)], n - T.int64(1))
                            + v_ax1 * rxplaceholder_2[T.int64(1)],
                        ],
                        rxplaceholder[T.int64(0) : T.int64(2)],
                        rxplaceholder_2[T.int64(0) : T.int64(2)],
                    )
                    T.writes(T_strided_slice_dynamic[v_ax0, v_ax1])
                    T_strided_slice_dynamic[v_ax0, v_ax1] = rxplaceholder_3[
                        T.min(rxplaceholder[T.int64(0)], T.int64(9))
                        + v_ax0 * rxplaceholder_2[T.int64(0)],
                        T.min(rxplaceholder[T.int64(1)], n - T.int64(1))
                        + v_ax1 * rxplaceholder_2[T.int64(1)],
                    ]

        @T.prim_func(private=True)
        def shape_func(
            var_rxplaceholder: T.handle,
            rxplaceholder: T.Buffer((T.int64(2),), "int64"),
            rxplaceholder_1: T.Buffer((T.int64(2),), "int64"),
            rxplaceholder_2: T.Buffer((T.int64(2),), "int64"),
            T_shape_func_strided_slice_dynamic: T.Buffer((T.int64(2),), "int64"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            rxplaceholder_3 = T.match_buffer(var_rxplaceholder, (T.int64(10), n))
            # with T.block("root"):
            for i in range(T.int64(2)):
                with T.block("T_shape_func_strided_slice_dynamic"):
                    v_i = T.axis.spatial(T.int64(2), i)
                    T.reads(rxplaceholder_2[v_i], rxplaceholder[v_i], rxplaceholder_1[v_i])
                    T.writes(T_shape_func_strided_slice_dynamic[v_i])
                    T_shape_func_strided_slice_dynamic[v_i] = T.Select(
                        rxplaceholder_2[v_i] < T.int64(0),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder[v_i] < T.int64(0),
                                        rxplaceholder[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                )
                                - T.int64(1),
                            )
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(-1),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                )
                                - T.int64(1),
                            )
                            - rxplaceholder_2[v_i]
                            - T.int64(1)
                        )
                        // (rxplaceholder_2[v_i] * T.int64(-1)),
                        (
                            T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder_1[v_i] < T.int64(0),
                                        rxplaceholder_1[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder_1[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                ),
                            )
                            + rxplaceholder_2[v_i]
                            - T.min(
                                T.max(
                                    T.Select(
                                        rxplaceholder[v_i] < T.int64(0),
                                        rxplaceholder[v_i]
                                        + T.Select(
                                            v_i == T.int64(1),
                                            n,
                                            T.Select(
                                                v_i == T.int64(0), T.int64(10), T.int64(-1)
                                            ),
                                        ),
                                        rxplaceholder[v_i],
                                    ),
                                    T.int64(0),
                                ),
                                T.Select(
                                    v_i == T.int64(1),
                                    n,
                                    T.Select(v_i == T.int64(0), T.int64(10), T.int64(-1)),
                                ),
                            )
                            - T.int64(1)
                        )
                        // rxplaceholder_2[v_i],
                    )

        @R.function
        def main(
            x: R.Tensor((10, "n"), dtype="float32"),
            begin: R.Tensor((2,), dtype="int64"),
            end: R.Tensor((2,), dtype="int64"),
            strides: R.Tensor((2,), dtype="int64"),
        ) -> R.Tensor(dtype="float32", ndim=2):
            n = T.int64()
            s = T.int64()
            s_1 = T.int64()
            gv = R.call_tir(
                Expected.shape_func,
                (x, begin, end, strides),
                out_sinfo=R.Tensor((2,), dtype="int64"),
            )
            gv1: R.Shape(ndim=2) = R.call_pure_packed(
                "vm.builtin.tensor_to_shape", gv, sinfo_args=(R.Shape(ndim=2),)
            )
            gv2: R.Shape([s, s_1]) = R.match_cast(gv1, R.Shape([s, s_1]))
            gv_1 = R.call_tir(
                Expected.dynamic_strided_slice,
                (x, begin, end, strides),
                out_sinfo=R.Tensor((s, s_1), dtype="float32"),
            )
            return gv_1
    # fmt: on

    mod = LegalizeOps()(DynamicStridedSlice)
    tvm.ir.assert_structural_equal(mod, Expected)


##################### Linear algebra #####################


def test_matmul_1_4():
    # fmt: off
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 3, 5), "float32"):
            gv: R.Tensor((2, 3, 5), "float32") = R.matmul(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((2, 3, 4, 5), "float32")) -> R.Tensor((2, 3, 5), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), R.Tensor((2, 3, 5), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def matmul(rxplaceholder: T.Buffer(T.int64(4), "float32"), rxplaceholder_1: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(3), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(5), T.int64(4)):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[k], rxplaceholder_1[i0_1, i1_1, k, i2_1])
                    T.writes(matmul[i0_1, i1_1, i2_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = matmul[i0_1, i1_1, i2_1] + rxplaceholder[k] * rxplaceholder_1[i0_1, i1_1, k, i2_1]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_1():
    # fmt: off
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv: R.Tensor((2, 3, 4), "float32") = R.matmul(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor((2, 3, 4), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), R.Tensor((2, 3, 4), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def matmul(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Buffer(T.int64(5), "float32"), matmul: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, k], rxplaceholder_1[k])
                    T.writes(matmul[i0_1, i1_1, i2_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1] = matmul[i0_1, i1_1, i2_1] + rxplaceholder[i0_1, i1_1, i2_1, k] * rxplaceholder_1[k]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_1_1():
    # fmt: off
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.matmul(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4,), "float32"), y: R.Tensor((4,), "float32")) -> R.Tensor((), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def matmul(rxplaceholder: T.Buffer(T.int64(4), "float32"), rxplaceholder_1: T.Buffer(T.int64(4), "float32"), matmul: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0 in T.serial(T.int64(4)):
                with T.block("matmul"):
                    k = T.axis.reduce(T.int64(4), i0)
                    T.reads(rxplaceholder[k], rxplaceholder_1[k])
                    T.writes(matmul[()])
                    with T.init():
                        matmul[()] = T.float32(0)
                    matmul[()] = matmul[()] + rxplaceholder[k] * rxplaceholder_1[k]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_5():
    # fmt: off
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), y: R.Tensor((6, 2, 3, 5, 7), "float16")) -> R.Tensor((6, 2, 3, 4, 7), "float32"):
            gv: R.Tensor((6, 2, 3, 4, 7), "float32") = R.matmul(x, y, out_dtype="float32")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), y: R.Tensor((6, 2, 3, 5, 7), "float16")) -> R.Tensor((6, 2, 3, 4, 7), "float32"):
            gv = R.call_tir(Expected.matmul, (x, y), R.Tensor((6, 2, 3, 4, 7), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def matmul(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16"), rxplaceholder_1: T.Buffer((T.int64(6), T.int64(2), T.int64(3), T.int64(5), T.int64(7)), "float16"), matmul: T.Buffer((T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(6), T.int64(2), T.int64(3), T.int64(4), T.int64(7), T.int64(5)):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, k = T.axis.remap("SSSSSR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[i1_1, i2_1, i3_1, k], rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1])
                    T.writes(matmul[i0_1, i1_1, i2_1, i3_1, i4_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = matmul[i0_1, i1_1, i2_1, i3_1, i4_1] + T.Cast("float32", rxplaceholder[i1_1, i2_1, i3_1, k]) * T.Cast("float32", rxplaceholder_1[i0_1, i1_1, i2_1, k, i4_1])
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_4_5_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor(("b", 1, "m", "k"), "float32"), y: R.Tensor(("a", 1, "c", "k", "n"), "float32")) -> R.Tensor(("a", "b", "c", "m", "n"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            m = T.int64()
            n = T.int64()
            gv: R.Tensor((a, b, c, m, n), "float32") = R.matmul(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("b", 1, "m", "k"), "float32"), y: R.Tensor(("a", 1, "c", "k", "n"), "float32")) -> R.Tensor(("a", "b", "c", "m", "n"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.matmul, (x, y), R.Tensor((a, b, c, m, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def matmul(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_matmul: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            k = T.int64()
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [b, T.int64(1), m, k], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, T.int64(1), c, k, n], dtype="float32")
            matmul = T.match_buffer(var_matmul, [a, b, c, m, n], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(a, b, c, m, n, k):
                with T.block("matmul"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, k_1 = T.axis.remap("SSSSSR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[i1_1, T.int64(0), i3_1, k_1], rxplaceholder_1[i0_1, T.int64(0), i2_1, k_1, i4_1])
                    T.writes(matmul[i0_1, i1_1, i2_1, i3_1, i4_1])
                    with T.init():
                        matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = T.float32(0)
                    matmul[i0_1, i1_1, i2_1, i3_1, i4_1] = matmul[i0_1, i1_1, i2_1, i3_1, i4_1] + rxplaceholder[i1_1, T.int64(0), i3_1, k_1] * rxplaceholder_1[i0_1, T.int64(0), i2_1, k_1, i4_1]
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_matmul_batching_dim_1():
    # fmt: off
    @tvm.script.ir_module
    class Matmul:
        @R.function
        def main(x: R.Tensor((1, 1, 4, 5), "float32"), y: R.Tensor((1, 1, 5, 7), "float32")) -> R.Tensor((1, 1, 4, 7), "float32"):
            gv: R.Tensor((1, 1, 4, 7), "float32") = R.matmul(x, y, out_dtype="float32")
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def matmul(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4), T.int64(5)), "float32"), B: T.Buffer((T.int64(1), T.int64(1), T.int64(5), T.int64(7)), "float32"), matmul_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4), T.int64(7)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(1), T.int64(4), T.int64(7), T.int64(5)):
                with T.block("matmul"):
                    v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
                    T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
                    T.writes(matmul_1[v_i0, v_i1, v_i2, v_i3])
                    with T.init():
                        matmul_1[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
                    matmul_1[v_i0, v_i1, v_i2, v_i3] = matmul_1[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

        @R.function
        def main(x: R.Tensor((1, 1, 4, 5), dtype="float32"), y: R.Tensor((1, 1, 5, 7), dtype="float32")) -> R.Tensor((1, 1, 4, 7), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.matmul, (x, y), out_sinfo=R.Tensor((1, 1, 4, 7), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Matmul)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_einsum():
    # fmt: off
    @I.ir_module
    class Einsum:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((3, 4), "float32")):
            gv = R.einsum((x, y), subscripts="ij,jk->ik")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((3, 4), dtype="float32")
        ) -> R.Tensor((2, 4), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.einsum, (x, y), out_sinfo=R.Tensor((2, 4), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def einsum(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            rxplaceholder_1: T.Buffer((T.int64(3), T.int64(4)), "float32"),
            T_einsum: T.Buffer((T.int64(2), T.int64(4)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1, j in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with T.block("T_einsum"):
                    v_ax0, v_ax1, v_j = T.axis.remap("SSR", [ax0, ax1, j])
                    T.reads(rxplaceholder[v_ax0, v_j], rxplaceholder_1[v_j, v_ax1])
                    T.writes(T_einsum[v_ax0, v_ax1])
                    with T.init():
                        T_einsum[v_ax0, v_ax1] = T.float32(0)
                    T_einsum[v_ax0, v_ax1] = (
                        T_einsum[v_ax0, v_ax1]
                        + rxplaceholder[v_ax0, v_j] * rxplaceholder_1[v_j, v_ax1]
                    )
    # fmt: on

    mod = LegalizeOps()(Einsum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_einsum_symbolic():
    # fmt: off
    @I.ir_module
    class Einsum:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32"), y: R.Tensor(("b", "c"), "float32")):
            gv = R.einsum((x, y), subscripts="ij,jk->ik")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(("a", "b"), dtype="float32"),
            y: R.Tensor(("b", "c"), dtype="float32"),
        ) -> R.Tensor(("a", "c"), dtype="float32"):
            a = T.int64()
            c = T.int64()
            b = T.int64()
            cls = Expected
            gv = R.call_tir(cls.einsum, (x, y), out_sinfo=R.Tensor((a, c), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def einsum(
            var_rxplaceholder: T.handle,
            var_rxplaceholder_1: T.handle,
            var_T_einsum: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            a, b = T.int64(), T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b))
            c = T.int64()
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (b, c))
            T_einsum = T.match_buffer(var_T_einsum, (a, c))
            for ax0, ax1, j in T.grid(a, c, b):
                with T.block("T_einsum"):
                    v_ax0, v_ax1, v_j = T.axis.remap("SSR", [ax0, ax1, j])
                    T.reads(rxplaceholder[v_ax0, v_j], rxplaceholder_1[v_j, v_ax1])
                    T.writes(T_einsum[v_ax0, v_ax1])
                    with T.init():
                        T_einsum[v_ax0, v_ax1] = T.float32(0)
                    T_einsum[v_ax0, v_ax1] = (
                        T_einsum[v_ax0, v_ax1]
                        + rxplaceholder[v_ax0, v_j] * rxplaceholder_1[v_j, v_ax1]
                    )
    # fmt: on

    mod = LegalizeOps()(Einsum)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
