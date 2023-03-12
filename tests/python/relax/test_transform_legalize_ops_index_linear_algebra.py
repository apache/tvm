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
from tvm.script import relax as R, tir as T
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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
        def main(x: R.Tensor((8, 9, 10, 10), "float32")) -> R.Tensor((4, 9, 10, 3), "float32"):
            gv: R.Tensor((4, 9, 10, 3), "float32") = R.strided_slice(x, axes=[0, 1, 3], begin=[1, 0, 2], end=[8, 9, 4])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((8, 9, 10, 10), dtype="float32")) -> R.Tensor((4, 9, 10, 3), dtype="float32"):
            gv = R.call_tir(Expected.strided_slice, (x,), out_sinfo=R.Tensor((7, 9, 10, 2), dtype="float32"))
            return gv

        @T.prim_func
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
    # fmt: on

    mod = LegalizeOps()(StridedSlice)
    tvm.ir.assert_structural_equal(mod, StridedSlice)


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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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


if __name__ == "__main__":
    tvm.testing.main()
