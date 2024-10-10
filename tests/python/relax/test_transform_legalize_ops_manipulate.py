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
from tvm import relax
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T, ir as I
import tvm.testing


##################### Manipulation #####################


def test_broadcast_to():
    # fmt: off
    @tvm.script.ir_module
    class BroadcastTo:
        @R.function
        def main(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor((4, 2, 5, 3), "float32"):
            gv: R.Tensor((4, 2, 5, 3), "float32") = R.broadcast_to(x, (4, 2, 5, 3))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3), "float32")) -> R.Tensor((4, 2, 5, 3), "float32"):
            gv = R.call_tir(Expected.broadcast_to, (x,), R.Tensor((4, 2, 5, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def broadcast_to(rxplaceholder: T.Buffer((T.int64(2), T.int64(1), T.int64(3)), "float32"), T_broadcast_to: T.Buffer((T.int64(4), T.int64(2), T.int64(5), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(2), T.int64(5), T.int64(3)):
                with T.block("T_broadcast_to"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax1, T.int64(0), ax3])
                    T.writes(T_broadcast_to[ax0, ax1, ax2, ax3])
                    T_broadcast_to[ax0, ax1, ax2, ax3] = rxplaceholder[ax1, T.int64(0), ax3]
    # fmt: on

    mod = LegalizeOps()(BroadcastTo)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_broadcast_to_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class BroadcastTo:
        @R.function
        def main(dumb_param: R.Tensor(("a", "c")), x: R.Tensor(("b", 1, "d"), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((a, b, c, d), "float32") = R.broadcast_to(x, (a, b, c, d))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor(("a", "c")), x: R.Tensor(("b", 1, "d"), "float32")) -> R.Tensor(("a", "b", "c", "d"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv = R.call_tir(Expected.broadcast_to, (x,), R.Tensor((a, b, c, d), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def broadcast_to(var_rxplaceholder: T.handle, var_T_broadcast_to: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [b, T.int64(1), d], dtype="float32")
            T_broadcast_to = T.match_buffer(var_T_broadcast_to, [a, b, c, d], dtype="float32")
            for i0, i1, i2, i3 in T.grid(a, b, c, d):
                with T.block("T_broadcast_to"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax1, T.int64(0), ax3])
                    T.writes(T_broadcast_to[ax0, ax1, ax2, ax3])
                    T_broadcast_to[ax0, ax1, ax2, ax3] = rxplaceholder[ax1, T.int64(0), ax3]
    # fmt: on

    mod = LegalizeOps()(BroadcastTo)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_concat():
    # fmt: off
    @tvm.script.ir_module
    class Concat:
        @R.function
        def main(x1: R.Tensor((1, 2, 3), "float32"), x2: R.Tensor((1, 3, 3), "float32"), x3: R.Tensor((1, 4, 3), "float32")) -> R.Tensor((1, 9, 3), "float32"):
            gv: R.Tensor((1, 9, 3), "float32") = R.concat((x1, x2, x3), axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x1: R.Tensor((1, 2, 3), "float32"), x2: R.Tensor((1, 3, 3), "float32"), x3: R.Tensor((1, 4, 3), "float32")) -> R.Tensor((1, 9, 3), "float32"):
            gv = R.call_tir(Expected.concatenate, (x1, x2, x3), R.Tensor((1, 9, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def concatenate(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(1), T.int64(3), T.int64(3)), "float32"), rxplaceholder_2: T.Buffer((T.int64(1), T.int64(4), T.int64(3)), "float32"), T_concat: T.Buffer((T.int64(1), T.int64(9), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(9), T.int64(3)):
                with T.block("T_concat"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder_2[ax0, ax1 - T.int64(5), ax2], rxplaceholder_1[ax0, ax1 - T.int64(2), ax2], rxplaceholder[ax0, ax1, ax2])
                    T.writes(T_concat[ax0, ax1, ax2])
                    T_concat[ax0, ax1, ax2] = T.if_then_else(T.int64(5) <= ax1, rxplaceholder_2[ax0, ax1 - T.int64(5), ax2], T.if_then_else(T.int64(2) <= ax1, rxplaceholder_1[ax0, ax1 - T.int64(2), ax2], rxplaceholder[ax0, ax1, ax2]))
    # fmt: on

    mod = LegalizeOps()(Concat)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_concat_input_tuple_var():
    # fmt: off
    @tvm.script.ir_module
    class Concat:
        @R.function
        def main(t: R.Tuple(R.Tensor((3, 4), "float32"), R.Tensor((3, 5), "float32"))) -> R.Tensor((3, 9), "float32"):
            gv: R.Tensor((3, 9), "float32") = R.concat(t, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(t: R.Tuple(R.Tensor((3, 4), "float32"), R.Tensor((3, 5), "float32"))) -> R.Tensor((3, 9), "float32"):
            gv: R.Tensor((3, 4), dtype="float32") = t[0]
            gv1: R.Tensor((3, 5), dtype="float32") = t[1]
            gv2 = R.call_tir(Expected.concatenate, (gv, gv1), R.Tensor((3, 9), dtype="float32"))
            return gv2

        @T.prim_func(private=True)
        def concatenate(rxplaceholder: T.Buffer((T.int64(3), T.int64(4)), "float32"), rxplaceholder_1: T.Buffer((T.int64(3), T.int64(5)), "float32"), T_concat: T.Buffer((T.int64(3), T.int64(9)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(3), T.int64(9)):
                with T.block("T_concat"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_1[ax0, ax1 - T.int64(4)], rxplaceholder[ax0, ax1])
                    T.writes(T_concat[ax0, ax1])
                    T_concat[ax0, ax1] = T.if_then_else(T.int64(4) <= ax1, rxplaceholder_1[ax0, ax1 - T.int64(4)], rxplaceholder[ax0, ax1])
    # fmt: on

    mod = LegalizeOps()(Concat)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_concat_input_tuple_var_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Concat:
        @R.function
        def main(t: R.Tuple(R.Tensor(("a", "b0"), "float32"), R.Tensor(("a", "b1"), "float32"), R.Tensor(("a", "b2"), "float32"))) -> R.Tensor(("a", "b0 + b1 + b2"), "float32"):
            a = T.int64()
            b0 = T.int64()
            b1 = T.int64()
            b2 = T.int64()
            gv: R.Tensor((a, b0 + b1 + b2), "float32") = R.concat(t, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(t: R.Tuple(R.Tensor(("a", "b0"), "float32"), R.Tensor(("a", "b1"), "float32"), R.Tensor(("a", "b2"), "float32"))) -> R.Tensor(("a", "b0 + b1 + b2"), "float32"):
            a = T.int64()
            b0 = T.int64()
            b1 = T.int64()
            b2 = T.int64()
            gv: R.Tensor((a, b0), dtype="float32") = t[0]
            gv1: R.Tensor((a, b1), dtype="float32") = t[1]
            gv2: R.Tensor((a, b2), dtype="float32") = t[2]
            gv3 = R.call_tir(Expected.concatenate, (gv, gv1, gv2), R.Tensor((a, ((b0 + b1) + b2)), dtype="float32"))
            return gv3

        @T.prim_func(private=True)
        def concatenate(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_T_concat: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b0 = T.int64()
            b1 = T.int64()
            b2 = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b0], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [a, b1], dtype="float32")
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, [a, b2], dtype="float32")
            T_concat = T.match_buffer(var_T_concat, [a, b0 + b1 + b2], dtype="float32")
            for i0, i1 in T.grid(a, b0 + b1 + b2):
                with T.block("T_concat"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder_2[ax0, ax1 - b0 - b1], rxplaceholder_1[ax0, ax1 - b0], rxplaceholder[ax0, ax1])
                    T.writes(T_concat[ax0, ax1])
                    T_concat[ax0, ax1] = T.if_then_else(T.int64(0) <= ax1 - b0 - b1, rxplaceholder_2[ax0, ax1 - b0 - b1], T.if_then_else(T.int64(0) <= ax1 - b0, rxplaceholder_1[ax0, ax1 - b0], rxplaceholder[ax0, ax1]))
    # fmt: on

    mod = LegalizeOps()(Concat)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_expand_dims():
    # fmt: off
    @tvm.script.ir_module
    class ExpandDims:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32"):
            gv: R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32") = R.expand_dims(x, axis=[-1, 1, -6, 3, 5])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), "float32"):
            gv = R.call_tir(Expected.expand_dims, (x,), R.Tensor((2, 1, 1, 1, 3, 1, 4, 1), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def expand_dims(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), expand_dims: T.Buffer((T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(3), T.int64(1), T.int64(4), T.int64(1)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6, i7 in T.grid(T.int64(2), T.int64(1), T.int64(1), T.int64(1), T.int64(3), T.int64(1), T.int64(4), T.int64(1)):
                with T.block("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1 = T.axis.remap("SSSSSSSS", [i0, i1, i2, i3, i4, i5, i6, i7])
                    T.reads(rxplaceholder[i0_1, i4_1, i6_1])
                    T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1, i6_1, i7_1] = rxplaceholder[i0_1, i4_1, i6_1]
    # fmt: on

    mod = LegalizeOps()(ExpandDims)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_expand_dims_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class ExpandDims:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a", 1, "b", 1, "c", 1), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv: R.Tensor((a, 1, b, 1, c, 1), "float32") = R.expand_dims(x, axis=[1, 3, 5])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a", 1, "b", 1, "c", 1), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv = R.call_tir(Expected.expand_dims, (x,), R.Tensor((a, 1, b, 1, c, 1), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def expand_dims(var_rxplaceholder: T.handle, var_expand_dims: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c], dtype="float32")
            expand_dims = T.match_buffer(var_expand_dims, [a, T.int64(1), b, T.int64(1), c, T.int64(1)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(a, T.int64(1), b, T.int64(1), c, T.int64(1)):
                with T.block("expand_dims"):
                    i0_1, i1_1, i2_1, i3_1, i4_1, i5_1 = T.axis.remap("SSSSSS", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[i0_1, i2_1, i4_1])
                    T.writes(expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1])
                    expand_dims[i0_1, i1_1, i2_1, i3_1, i4_1, i5_1] = rxplaceholder[i0_1, i2_1, i4_1]
    # fmt: on

    mod = LegalizeOps()(ExpandDims)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_flatten():
    # fmt: off
    @tvm.script.ir_module
    class Flatten:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((24,), "float32"):
            gv: R.Tensor((24,), "float32") = R.flatten(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4), "float32")) -> R.Tensor((24,), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), R.Tensor((24,), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(24), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0 in T.serial(T.int64(24)):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(T.int64(24), i0)
                    T.reads(rxplaceholder[ax0 % T.int64(24) // T.int64(12), ax0 % T.int64(12) // T.int64(4), ax0 % T.int64(4)])
                    T.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[ax0 % T.int64(24) // T.int64(12), ax0 % T.int64(12) // T.int64(4), ax0 % T.int64(4)]
    # fmt: on

    mod = LegalizeOps()(Flatten)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_flatten_zero_rank():
    # fmt: off
    @tvm.script.ir_module
    class Flatten:
        @R.function
        def main(x: R.Tensor((), "float32")) -> R.Tensor((1,), "float32"):
            gv: R.Tensor((1,), "float32") = R.flatten(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((), "float32")) -> R.Tensor((1,), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), R.Tensor((1,), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def reshape(rxplaceholder: T.Buffer((), "float32"), T_reshape: T.Buffer(T.int64(1), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0 in T.serial(T.int64(1)):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(T.int64(1), i0)
                    T.reads(rxplaceholder[()])
                    T.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[()]
    # fmt: on

    mod = LegalizeOps()(Flatten)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_flatten_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Flatten:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a * b * c",), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv: R.Tensor((a * b * c,), "float32") = R.flatten(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a * b * c",), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv = R.call_tir(Expected.reshape, (x,), R.Tensor((((a * b) * c),), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def reshape(var_rxplaceholder: T.handle, var_T_reshape: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c], dtype="float32")
            T_reshape = T.match_buffer(var_T_reshape, [a * b * c], dtype="float32")
            for i0 in T.serial(a * b * c):
                with T.block("T_reshape"):
                    ax0 = T.axis.spatial(a * b * c, i0)
                    T.reads(rxplaceholder[ax0 // c // b % a, ax0 // c % b, ax0 % c])
                    T.writes(T_reshape[ax0])
                    T_reshape[ax0] = rxplaceholder[ax0 // c // b % a, ax0 // c % b, ax0 % c]
    # fmt: on

    mod = LegalizeOps()(Flatten)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_permute_dims():
    # fmt: off
    @tvm.script.ir_module
    class PermuteDims:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((2, 4, 3, 1), "float32"):
            gv: R.Tensor((2, 4, 3, 1), "float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((2, 4, 3, 1), "float32"):
            gv = R.call_tir(Expected.transpose, (x,), R.Tensor((2, 4, 3, 1), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def transpose(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"), T_transpose: T.Buffer((T.int64(2), T.int64(4), T.int64(3), T.int64(1)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(4), T.int64(3), T.int64(1)):
                with T.block("T_transpose"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax3, ax0, ax2, ax1])
                    T.writes(T_transpose[ax0, ax1, ax2, ax3])
                    T_transpose[ax0, ax1, ax2, ax3] = rxplaceholder[ax3, ax0, ax2, ax1]
    # fmt: on

    mod = LegalizeOps()(PermuteDims)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_permute_dims_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class PermuteDims:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), "float32")) -> R.Tensor(("b", "d", "c", "a"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            gv: R.Tensor((b, d, c, a), "float32") = R.permute_dims(x, axes=[1, -1, 2, -4])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c", "d"), dtype="float32")) -> R.Tensor(("b", "d", "c", "a"), dtype="float32"):
            b = T.int64()
            d = T.int64()
            c = T.int64()
            a = T.int64()
            gv = R.call_tir(Expected.transpose, (x,), R.Tensor((b, d, c, a), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def transpose(var_rxplaceholder: T.handle, var_T_transpose: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            d = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c, d], dtype="float32")
            T_transpose = T.match_buffer(var_T_transpose, [b, d, c, a], dtype="float32")
            for i0, i1, i2, i3 in T.grid(b, d, c, a):
                with T.block("T_transpose"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax3, ax0, ax2, ax1])
                    T.writes(T_transpose[ax0, ax1, ax2, ax3])
                    T_transpose[ax0, ax1, ax2, ax3] = rxplaceholder[ax3, ax0, ax2, ax1]
    # fmt: on

    mod = LegalizeOps()(PermuteDims)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_reshape():
    # fmt: off
    @tvm.script.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
            gv: R.Tensor((8, 3), "float32") = R.reshape(x, (8, 3))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
            gv = R.call_tir(Expected.reshape, (x,), R.Tensor((8, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def reshape(rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"), T_reshape: T.Buffer((T.int64(8), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(8), T.int64(3)):
                with T.block("T_reshape"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[T.int64(0), (ax0 * T.int64(3) + ax1) % T.int64(24) // T.int64(12), (ax0 * T.int64(3) + ax1) % T.int64(12) // T.int64(4), (ax0 * T.int64(3) + ax1) % T.int64(4)])
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[T.int64(0), (ax0 * T.int64(3) + ax1) % T.int64(24) // T.int64(12), (ax0 * T.int64(3) + ax1) % T.int64(12) // T.int64(4), (ax0 * T.int64(3) + ax1) % T.int64(4)]
    # fmt: on

    mod = LegalizeOps()(Reshape)
    tvm.ir.assert_structural_equal(mod, Expected)

    # fmt: off
    # ShapeExpr might be produced by shape computation
    @tvm.script.ir_module
    class Reshape2:
        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), "float32")) -> R.Tensor((8, 3), "float32"):
            lv: R.Shape((8, 3)) = R.shape((8, 3))
            gv: R.Tensor((8, 3), "float32") = R.reshape(x, lv)
            return gv

    # After lowering, redundant var might be removed by later dead code elimination
    @tvm.script.ir_module
    class Expected2:
        @T.prim_func(private=True)
        def reshape(
            rxplaceholder: T.Buffer((T.int64(1), T.int64(2), T.int64(3), T.int64(4)), "float32"),
            T_reshape: T.Buffer((T.int64(8), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(8), T.int64(3)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(
                        rxplaceholder[
                            T.int64(0),
                            (v_ax0 * T.int64(3) + v_ax1) % T.int64(24) // T.int64(12),
                            (v_ax0 * T.int64(3) + v_ax1) % T.int64(12) // T.int64(4),
                            (v_ax0 * T.int64(3) + v_ax1) % T.int64(4),
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = rxplaceholder[
                        T.int64(0),
                        (v_ax0 * T.int64(3) + v_ax1) % T.int64(24) // T.int64(12),
                        (v_ax0 * T.int64(3) + v_ax1) % T.int64(12) // T.int64(4),
                        (v_ax0 * T.int64(3) + v_ax1) % T.int64(4),
                    ]

        @R.function
        def main(x: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((8, 3), dtype="float32"):
            lv: R.Shape((8, 3)) = R.shape((8, 3))
            gv = R.call_tir(Expected2.reshape, (x,), out_sinfo=R.Tensor((8, 3), dtype="float32"))
            return gv
    # fmt: on

    mod2 = LegalizeOps()(Reshape2)
    tvm.ir.assert_structural_equal(mod2, Expected2)


def test_reshape_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Reshape:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32")) -> R.Tensor(("a // 2", "b * 2"), "float32"):
            a = T.int64()
            b = T.int64()
            gv: R.Tensor((a // 2, b * 2), "float32") = R.reshape(x, (a // 2, b * 2))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32")) -> R.Tensor(("a // 2", "b * 2"), "float32"):
            a = T.int64()
            b = T.int64()
            gv = R.call_tir(Expected.reshape, (x,), R.Tensor(((a // 2), (b * 2)), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def reshape(var_rxplaceholder: T.handle, var_T_reshape: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b], dtype="float32")
            T_reshape = T.match_buffer(var_T_reshape, [a // T.int64(2), b * T.int64(2)], dtype="float32")
            for i0, i1 in T.grid(a // T.int64(2), b * T.int64(2)):
                with T.block("T_reshape"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[(ax0 * b * T.int64(2) + ax1) // b % a, (ax0 * b * T.int64(2) + ax1) % b])
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[(ax0 * b * T.int64(2) + ax1) // b % a, (ax0 * b * T.int64(2) + ax1) % b]
    # fmt: on

    mod = LegalizeOps()(Reshape)
    tvm.ir.assert_structural_equal(mod, Expected)

    # ShapeExpr might be produced by shape computation
    @tvm.script.ir_module
    class Reshape2:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32")) -> R.Tensor(("a // 2", "b * 2"), "float32"):
            a = T.int64()
            b = T.int64()
            lv: R.Shape((a // 2, b * 2)) = R.shape((a // 2, b * 2))
            gv: R.Tensor((a // 2, b * 2), "float32") = R.reshape(x, lv)
            return gv

    # After lowering, redundant var might be removed by later dead code elimination
    @tvm.script.ir_module
    class Expected2:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32")) -> R.Tensor(("a // 2", "b * 2"), "float32"):
            a = T.int64()
            b = T.int64()
            lv: R.Shape((a // 2, b * 2)) = R.shape((a // 2, b * 2))
            gv = R.call_tir(Expected2.reshape, (x,), R.Tensor(((a // 2), (b * 2)), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def reshape(var_rxplaceholder: T.handle, var_T_reshape: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b], dtype="float32")
            T_reshape = T.match_buffer(
                var_T_reshape, [a // T.int64(2), b * T.int64(2)], dtype="float32"
            )
            for i0, i1 in T.grid(a // T.int64(2), b * T.int64(2)):
                with T.block("T_reshape"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(
                        rxplaceholder[
                            (ax0 * b * T.int64(2) + ax1) // b % a,
                            (ax0 * b * T.int64(2) + ax1) % b,
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        (ax0 * b * T.int64(2) + ax1) // b % a, (ax0 * b * T.int64(2) + ax1) % b
                    ]

    mod2 = LegalizeOps()(Reshape2)
    tvm.ir.assert_structural_equal(mod2, Expected2)

    # ShapeExpr might be produced by shape computation
    @I.ir_module
    class Reshape3:
        @R.function
        def main(x: R.Tensor((10, "b"), "float32")) -> R.Tensor((5, "b * 2"), "float32"):
            a = T.int64()
            b = T.int64()
            lv: R.Shape((5, b * 2)) = R.shape((5, b * 2))
            gv: R.Tensor((5, b * 2), "float32") = R.reshape(x, lv)
            return gv

    # After lowering, redundant var might be removed by later dead code elimination
    @I.ir_module
    class Expected3:
        @T.prim_func(private=True)
        def reshape(var_rxplaceholder: T.handle, var_T_reshape: T.handle):
            T.func_attr({"tir.noalias": True})
            b = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(10), b))
            T_reshape = T.match_buffer(var_T_reshape, (T.int64(5), b * T.int64(2)))
            # with T.block("root"):
            for ax0, ax1 in T.grid(T.int64(5), b * T.int64(2)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(
                        rxplaceholder[
                            (v_ax0 * b * T.int64(2) + v_ax1) // b % T.int64(10),
                            (v_ax0 * b * T.int64(2) + v_ax1) % b,
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = rxplaceholder[
                        (v_ax0 * b * T.int64(2) + v_ax1) // b % T.int64(10),
                        (v_ax0 * b * T.int64(2) + v_ax1) % b,
                    ]

        @R.function
        def main(
            x: R.Tensor((10, "b"), dtype="float32")
        ) -> R.Tensor((5, "b * 2"), dtype="float32"):
            b = T.int64()
            lv: R.Shape([5, b * 2]) = R.shape([5, b * 2])
            gv = R.call_tir(
                Expected3.reshape, (x,), out_sinfo=R.Tensor((5, b * 2), dtype="float32")
            )
            return gv

    mod3 = LegalizeOps()(Reshape3)
    tvm.ir.assert_structural_equal(mod3, Expected3)


def test_data_dependent_reshape():
    # fmt: off
    @tvm.script.ir_module
    class DDReshape:
        @R.function
        def main(
            x: R.Tensor([2], dtype="int64"),
            y: R.Tensor([16],dtype='float32'),
        ):
            lv: R.Shape(ndim=2) = R.tensor_to_shape(x)
            gv = R.reshape(y, lv)
            return gv
    # fmt: on

    assert relax.analysis.well_formed(DDReshape)
    mod = relax.transform.DecomposeOpsForInference()(DDReshape)
    out_mod = relax.transform.LegalizeOps()(mod)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(
                x: R.Tensor([2], dtype="int64"),
                y: R.Tensor([16],dtype="float32"),
        ) -> R.Tensor(ndim=2, dtype="float32"):
            M = T.int64()
            N = T.int64()
            gv = R.call_pure_packed("vm.builtin.tensor_to_shape", x, sinfo_args=(R.Shape(ndim=2),))
            _ = R.match_cast(gv, R.Shape([M,N]))
            _ = R.shape([M,N])
            gv_1 = R.call_tir(Expected.reshape, (y,), out_sinfo=R.Tensor([M,N], dtype="float32"))
            return gv_1

        @T.prim_func(private=True)
        def reshape(
            rxplaceholder: T.Buffer(T.int64(16), "float32"),
            var_T_reshape: T.handle,
        ):
            T.func_attr({"tir.noalias": True})
            M = T.int64()
            N = T.int64()
            T_reshape = T.match_buffer(var_T_reshape, [M,N], "float32")
            for i,j in T.grid(M,N):
                with T.block("T_reshape"):
                    vi,vj = T.axis.remap('SS',[i,j])
                    T.reads(rxplaceholder[(vi*N + vj) % 16])
                    T.writes(T_reshape[vi,vj])
                    T_reshape[vi,vj] = rxplaceholder[(vi*N + vj) % 16]

    # fmt: on
    tvm.ir.assert_structural_equal(out_mod, Expected)


def test_split_by_indices():
    # fmt: off
    @tvm.script.ir_module
    class Split:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")]):
            gv: R.Tuple([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")]) = R.split(x, [3, 7], axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")]):
            gv = R.call_tir(Expected.split, (x,), [R.Tensor((2, 3, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 3, 4), "float32")])
            return gv

        @T.prim_func(private=True)
        def split(rxplaceholder: T.Buffer((T.int64(2), T.int64(10), T.int64(4)), "float32"), T_split: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), T_split_1: T.Buffer((T.int64(2), T.int64(4), T.int64(4)), "float32"), T_split_2: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("T_split"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1, ax2])
                    T.writes(T_split[ax0, ax1, ax2])
                    T_split[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(4), T.int64(4)):
                with T.block("T_split_1"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1 + T.int64(3), ax2])
                    T.writes(T_split_1[ax0, ax1, ax2])
                    T_split_1[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(3), ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("T_split_2"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1 + T.int64(7), ax2])
                    T.writes(T_split_2[ax0, ax1, ax2])
                    T_split_2[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(7), ax2]
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_split_by_indices_n_section_indivisible():
    # fmt: off
    @tvm.script.ir_module
    class Split:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 2, 4), "float32")]):
            gv: R.Tuple([R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 4, 4), "float32"), R.Tensor((2, 2, 4), "float32")]) = R.split(x, 3, axis=1)
            return gv
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm.ir.assert_structural_equal(mod, Split)


def test_split_by_indices_n_section_divisible():
    # fmt: off
    @tvm.script.ir_module
    class Split:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")]):
            gv: R.Tuple([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")]) = R.split(x, 2, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 10, 4), "float32")) -> R.Tuple([R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")]):
            gv = R.call_tir(Expected.split, (x,), [R.Tensor((2, 5, 4), "float32"), R.Tensor((2, 5, 4), "float32")])
            return gv

        @T.prim_func(private=True)
        def split(rxplaceholder: T.Buffer((T.int64(2), T.int64(10), T.int64(4)), "float32"), T_split_sections: T.Buffer((T.int64(2), T.int64(5), T.int64(4)), "float32"), T_split_sections_1: T.Buffer((T.int64(2), T.int64(5), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(5), T.int64(4)):
                with T.block("T_split_sections"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1, ax2])
                    T.writes(T_split_sections[ax0, ax1, ax2])
                    T_split_sections[ax0, ax1, ax2] = rxplaceholder[ax0, ax1, ax2]
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(5), T.int64(4)):
                with T.block("T_split_sections_1"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1 + T.int64(5), ax2])
                    T.writes(T_split_sections_1[ax0, ax1, ax2])
                    T_split_sections_1[ax0, ax1, ax2] = rxplaceholder[ax0, ax1 + T.int64(5), ax2]
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_split_by_indices_n_section_divisible_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Split:
        @R.function
        def main(dumb_param: R.Tensor(("n",)), x: R.Tensor(("m", "n * 3"), "float32")) -> R.Tuple([R.Tensor(("m", "n"), "float32"), R.Tensor(("m", "n"), "float32"), R.Tensor(("m", "n"), "float32")]):
            m = T.int64()
            n = T.int64()
            gv: R.Tuple([R.Tensor((m, n), "float32"), R.Tensor((m, n), "float32"), R.Tensor((m, n), "float32")]) = R.split(x, 3, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor(("n",)), x: R.Tensor(("m", "(n * 3)"), "float32")) -> R.Tuple(R.Tensor(("m", "((n * 3) // 3)"), "float32"), R.Tensor(("m", "((((n * 3) // 3) * 2) - ((n * 3) // 3))"), "float32"), R.Tensor(("m", "((n * 3) - (((n * 3) // 3) * 2))"), "float32")):
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.split, (x,), [R.Tensor((m, ((n * 3) // 3)), "float32"), R.Tensor((m, ((((n * 3) // 3) * 2) - ((n * 3) // 3))), "float32"), R.Tensor((m, ((n * 3) - (((n * 3) // 3) * 2))), "float32")], tir_vars=(n,))
            return gv

        @T.prim_func(private=True)
        def split(var_rxplaceholder: T.handle, var_T_split_sections: T.handle, var_T_split_sections_1: T.handle, var_T_split_sections_2: T.handle, n: T.int64):
            T.func_attr({"tir.noalias": True})
            m = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n * T.int64(3)], dtype="float32")
            T_split_sections = T.match_buffer(var_T_split_sections, [m, n * T.int64(3) // T.int64(3)], dtype="float32")
            T_split_sections_1 = T.match_buffer(var_T_split_sections_1, [m, n * T.int64(3) // T.int64(3) * T.int64(2) - n * T.int64(3) // T.int64(3)], dtype="float32")
            T_split_sections_2 = T.match_buffer(var_T_split_sections_2, [m, n * T.int64(3) - n * T.int64(3) // T.int64(3) * T.int64(2)], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_split_sections"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_split_sections[ax0, ax1])
                    T_split_sections[ax0, ax1] = rxplaceholder[ax0, ax1]
            for i0, i1 in T.grid(m, n):
                with T.block("T_split_sections_1"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, n + ax1])
                    T.writes(T_split_sections_1[ax0, ax1])
                    T_split_sections_1[ax0, ax1] = rxplaceholder[ax0, n + ax1]
            for i0, i1 in T.grid(m, n):
                with T.block("T_split_sections_2"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, n * T.int64(2) + ax1])
                    T.writes(T_split_sections_2[ax0, ax1])
                    T_split_sections_2[ax0, ax1] = rxplaceholder[ax0, n * T.int64(2) + ax1]
    # fmt: on

    mod = LegalizeOps()(Split)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_squeeze():
    # fmt: off
    @tvm.script.ir_module
    class Squeeze:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
            gv: R.Tensor((2, 3, 1, 4), "float32") = R.squeeze(x, [1, 4])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
            gv = R.call_tir(Expected.squeeze, (x,), R.Tensor((2, 3, 1, 4), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def squeeze(rxplaceholder: T.Buffer((T.int64(2), T.int64(1), T.int64(3), T.int64(1), T.int64(1), T.int64(4)), "float32"), T_squeeze: T.Buffer((T.int64(2), T.int64(3), T.int64(1), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(1), T.int64(4)):
                with T.block("T_squeeze"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, T.int64(0), ax1, ax2, T.int64(0), ax3])
                    T.writes(T_squeeze[ax0, ax1, ax2, ax3])
                    T_squeeze[ax0, ax1, ax2, ax3] = rxplaceholder[ax0, T.int64(0), ax1, ax2, T.int64(0), ax3]
    # fmt: on

    mod = LegalizeOps()(Squeeze)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_squeeze_no_axis():
    # fmt: off
    @tvm.script.ir_module
    class Squeeze:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) :
            gv: R.Tensor((2, 3, 4), "float32") = R.squeeze(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) :
            gv = R.call_tir(Expected.squeeze, (x,), R.Tensor((2, 3, 4), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def squeeze(rxplaceholder: T.Buffer((T.int64(2), T.int64(1), T.int64(3), T.int64(1), T.int64(1), T.int64(4)), "float32"), T_squeeze: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(4)):
                with T.block("T_squeeze"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, T.int64(0), ax1, T.int64(0), T.int64(0), ax2])
                    T.writes(T_squeeze[ax0, ax1, ax2])
                    T_squeeze[ax0, ax1, ax2] = rxplaceholder[ax0, T.int64(0), ax1, T.int64(0), T.int64(0), ax2]
    # fmt: on

    mod = LegalizeOps()(Squeeze)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_squeeze_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Squeeze:
        @R.function
        def main(x: R.Tensor(("a", 1, "b", 1), "float32")) -> R.Tensor(("a", "b", 1), "float32"):
            a = T.int64()
            b = T.int64()
            gv: R.Tensor((a, b, 1), "float32") = R.squeeze(x, [1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", 1, "b", 1), "float32")) -> R.Tensor(("a", "b", 1), "float32"):
            a = T.int64()
            b = T.int64()
            gv = R.call_tir(Expected.squeeze, (x,), R.Tensor((a, b, 1), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def squeeze(var_rxplaceholder: T.handle, var_T_squeeze: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, T.int64(1), b, T.int64(1)], dtype="float32")
            T_squeeze = T.match_buffer(var_T_squeeze, [a, b, T.int64(1)], dtype="float32")
            for i0, i1, i2 in T.grid(a, b, T.int64(1)):
                with T.block("T_squeeze"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, T.int64(0), ax1, ax2])
                    T.writes(T_squeeze[ax0, ax1, ax2])
                    T_squeeze[ax0, ax1, ax2] = rxplaceholder[ax0, T.int64(0), ax1, ax2]
    # fmt: on

    mod = LegalizeOps()(Squeeze)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_collapse_sum_like():
    # fmt: off
    @tvm.script.ir_module
    class CollapseSumLike:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((1, 3), "float32")) -> R.Tensor((1, 3), "float32"):
            gv: R.Tensor((1, 3), "float32") = R.collapse_sum_like(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((1, 3), "float32")) -> R.Tensor((1, 3), "float32"):
            gv = R.call_tir(Expected.collapse_sum, (x,), R.Tensor((1, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def collapse_sum(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), rxplaceholder_red: T.Buffer((T.int64(1), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2 in T.grid(T.int64(1), T.int64(3), T.int64(2)):
                with T.block("rxplaceholder_red"):
                    ax0, ax1, k0 = T.axis.remap("SSR", [i0, i1, i2])
                    T.reads(rxplaceholder[k0, ax1])
                    T.writes(rxplaceholder_red[ax0, ax1])
                    with T.init():
                        rxplaceholder_red[ax0, ax1] = T.float32(0)
                    rxplaceholder_red[ax0, ax1] = rxplaceholder_red[ax0, ax1] + rxplaceholder[k0, ax1]
    # fmt: on

    mod = LegalizeOps()(CollapseSumLike)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_collapse_sum_to():
    # fmt: off
    @tvm.script.ir_module
    class CollapseSumTo:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")) -> R.Tensor((2, 1), "float32"):
            gv: R.Tensor((2, 1), "float32") = R.collapse_sum_to(x, (2, 1))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 2, 3), dtype="float32")
        ) -> R.Tensor((2, 1), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.collapse_sum, (x,), R.Tensor((2, 1), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def collapse_sum(rxplaceholder: T.Buffer((T.int64(3), T.int64(2), T.int64(3)), "float32"), rxplaceholder_red: T.Buffer((T.int64(2), T.int64(1)), "float32")):
            T.func_attr({"tir.noalias": True})
            for ax0, ax1, k0, k2 in T.grid(T.int64(2), T.int64(1), T.int64(3), T.int64(3)):
                with T.block("rxplaceholder_red"):
                    v_ax0, v_ax1, v_k0, v_k2 = T.axis.remap("SSRR", [ax0, ax1, k0, k2])
                    T.reads(rxplaceholder[v_k0, v_ax0, v_k2])
                    T.writes(rxplaceholder_red[v_ax0, v_ax1])
                    with T.init():
                        rxplaceholder_red[v_ax0, v_ax1] = T.float32(0)
                    rxplaceholder_red[v_ax0, v_ax1] = (rxplaceholder_red[v_ax0, v_ax1] + rxplaceholder[v_k0, v_ax0, v_k2])
    # fmt: on

    mod = LegalizeOps()(CollapseSumTo)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_repeat():
    # fmt: off
    @I.ir_module
    class Repeat:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.repeat(x, 2, 0)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((6, 2, 3), dtype="float32"):
            gv = R.call_tir(Expected.repeat, (x,), out_sinfo=R.Tensor((6, 2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def repeat(rxplaceholder: T.Buffer((T.int64(3), T.int64(2), T.int64(3)), "float32"), T_repeat: T.Buffer((T.int64(6), T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1, ax2 in T.grid(T.int64(6), T.int64(2), T.int64(3)):
                with T.block("T_repeat"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2])
                    T.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                    T_repeat[v_ax0, v_ax1, v_ax2] = rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2]
    # fmt: on

    mod = LegalizeOps()(Repeat)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_repeat_no_axis():
    # fmt: off
    @I.ir_module
    class Repeat:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.repeat(x, 2)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((3, 2, 3), dtype="float32")
        ) -> R.Tensor((36,), dtype="float32"):
            gv = R.call_tir(Expected.repeat, (x,), out_sinfo=R.Tensor((36,), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def repeat(
            rxplaceholder: T.Buffer((T.int64(3), T.int64(2), T.int64(3)), "float32"),
            T_repeat: T.Buffer((T.int64(36),), "float32"),
        ):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            T_reshape = T.alloc_buffer((T.int64(18),))
            for ax0 in range(T.int64(18)):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(T.int64(18), ax0)
                    T.reads(
                        rxplaceholder[
                            v_ax0 % T.int64(18) // T.int64(6),
                            v_ax0 % T.int64(6) // T.int64(3),
                            v_ax0 % T.int64(3),
                        ]
                    )
                    T.writes(T_reshape[v_ax0])
                    T_reshape[v_ax0] = rxplaceholder[
                        v_ax0 % T.int64(18) // T.int64(6),
                        v_ax0 % T.int64(6) // T.int64(3),
                        v_ax0 % T.int64(3),
                    ]
            for ax0 in range(T.int64(36)):
                with T.block("T_repeat"):
                    v_ax0 = T.axis.spatial(T.int64(36), ax0)
                    T.reads(T_reshape[v_ax0 // T.int64(2)])
                    T.writes(T_repeat[v_ax0])
                    T_repeat[v_ax0] = T_reshape[v_ax0 // T.int64(2)]
    # fmt: on

    mod = LegalizeOps()(Repeat)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_repeat_symbolic():
    # fmt: off
    @I.ir_module
    class Repeat:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")):
            gv = R.repeat(x, 2, 0)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def repeat(var_rxplaceholder: T.handle, var_T_repeat: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b, c))
            T_repeat = T.match_buffer(var_T_repeat, (T.int64(2) * a, b, c))
            # with T.block("root"):
            for ax0, ax1, ax2 in T.grid(a * T.int64(2), b, c):
                with T.block("T_repeat"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2])
                    T.writes(T_repeat[v_ax0, v_ax1, v_ax2])
                    T_repeat[v_ax0, v_ax1, v_ax2] = rxplaceholder[v_ax0 // T.int64(2), v_ax1, v_ax2]

        @R.function
        def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor(("2 * a", "b", "c"), dtype="float32"):
            a = T.Var("a", "int64")
            b = T.Var("b", "int64")
            c = T.Var("c", "int64")
            gv = R.call_tir(Expected.repeat, (x,), out_sinfo=R.Tensor((2 * a, b, c), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Repeat)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tile():
    # fmt: off
    @I.ir_module
    class Tile:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.tile(x, (2, 1, 2, 3))
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def tile(rxplaceholder: T.Buffer((T.int64(3), T.int64(2), T.int64(3)), "float32"), T_tile: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(9)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(9)):
                with T.block("T_tile"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax1 % T.int64(3), v_ax2 % T.int64(2), v_ax3 % T.int64(3)])
                    T.writes(T_tile[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_tile[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax1 % T.int64(3), v_ax2 % T.int64(2), v_ax3 % T.int64(3)]

        @R.function
        def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((2, 3, 4, 9), dtype="float32"):
            gv = R.call_tir(Expected.tile, (x,), out_sinfo=R.Tensor((2, 3, 4, 9), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Tile)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tile_symbolic():
    # fmt: off
    @I.ir_module
    class Tile:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")):
            gv = R.tile(x, (2, 1, 2, 3))
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def tile(var_rxplaceholder: T.handle, var_T_tile: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b, c))
            T_tile = T.match_buffer(var_T_tile, (T.int64(2), a, b * T.int64(2), c * T.int64(3)))
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), a, b * T.int64(2), c * T.int64(3)):
                with T.block("T_tile"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax1 % a, v_ax2 % b, v_ax3 % c])
                    T.writes(T_tile[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_tile[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax1 % a, v_ax2 % b, v_ax3 % c]

        @R.function
        def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor((2, "a", "b * 2", "c * 3"), dtype="float32"):
            a = T.Var("a", "int64")
            b = T.Var("b", "int64")
            c = T.Var("c", "int64")
            gv = R.call_tir(Expected.tile, (x,), out_sinfo=R.Tensor((2, a, b * 2, c * 3), dtype="float32"))
            return gv
    # fmt: on
    mod = LegalizeOps()(Tile)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_flip():
    # fmt: off
    @I.ir_module
    class Flip:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            gv = R.flip(x, axis=0)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.flip, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def flip(
            rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            T_reverse_sequence: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_reverse_sequence"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[T.int64(1) - v_ax0, v_ax1])
                    T.writes(T_reverse_sequence[v_ax0, v_ax1])
                    T_reverse_sequence[v_ax0, v_ax1] = rxplaceholder[
                        T.int64(1) - v_ax0, v_ax1
                    ]

    # fmt: on

    mod = LegalizeOps()(Flip)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_flip_symbolic():
    # fmt: off
    @I.ir_module
    class Flip:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32")):
            gv = R.flip(x, axis=1)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(("a", "b"), dtype="float32")
        ) -> R.Tensor(("a", "b"), dtype="float32"):
            a = T.int64()
            b = T.int64()
            cls = Expected
            gv = R.call_tir(cls.flip, (x,), out_sinfo=R.Tensor((a, b), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def flip(var_rxplaceholder: T.handle, var_T_reverse_sequence: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            a, b = T.int64(), T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b))
            T_reverse_sequence = T.match_buffer(var_T_reverse_sequence, (a, b))
            for ax0, ax1 in T.grid(a, b):
                with T.block("T_reverse_sequence"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, b - v_ax1 - T.int64(1)])
                    T.writes(T_reverse_sequence[v_ax0, v_ax1])
                    T_reverse_sequence[v_ax0, v_ax1] = rxplaceholder[
                        v_ax0, b - v_ax1 - T.int64(1)
                    ]

    # fmt: on

    mod = LegalizeOps()(Flip)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_scatter_elements():
    # fmt: off
    @I.ir_module
    class ScatterElements:
        @R.function
        def main(x: R.Tensor((4,4), "float32"), indices: R.Tensor((2,2), "int64"), updates: R.Tensor((2,2), "float32")):
            gv = R.scatter_elements(x, indices, updates, axis=1)
            return gv
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def scatter_elements(
            var_rxplaceholder: T.handle,
            var_rxplaceholder_1: T.handle,
            var_rxplaceholder_2: T.handle,
            out_buf: T.Buffer((T.int64(4), T.int64(4)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            rxplaceholder = T.match_buffer(
                var_rxplaceholder, (T.int64(4), T.int64(4)), offset_factor=1
            )
            rxplaceholder_1 = T.match_buffer(
                var_rxplaceholder_1, (T.int64(2), T.int64(2)), "int64", offset_factor=1
            )
            rxplaceholder_2 = T.match_buffer(
                var_rxplaceholder_2, (T.int64(2), T.int64(2)), offset_factor=1
            )
            with T.block("scatter_elements_generic"):
                for i in T.parallel(T.int64(16)):
                    out_buf[i // T.int64(4), i % T.int64(4)] = rxplaceholder[
                        i // T.int64(4), i % T.int64(4)
                    ]
                for fused in T.parallel(T.int64(2)):
                    for k in range(T.int64(2)):
                        out_buf[
                            (
                                fused * T.int64(4)
                                + (
                                    rxplaceholder_1[
                                        (fused * T.int64(2) + k) // T.int64(2),
                                        (fused * T.int64(2) + k) % T.int64(2),
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * T.int64(2) + k) // T.int64(2),
                                            (fused * T.int64(2) + k) % T.int64(2),
                                        ]
                                        < T.int64(0),
                                    )
                                    * T.int64(4)
                                )
                            )
                            // T.int64(4),
                            (
                                fused * T.int64(4)
                                + (
                                    rxplaceholder_1[
                                        (fused * T.int64(2) + k) // T.int64(2),
                                        (fused * T.int64(2) + k) % T.int64(2),
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * T.int64(2) + k) // T.int64(2),
                                            (fused * T.int64(2) + k) % T.int64(2),
                                        ]
                                        < T.int64(0),
                                    )
                                    * T.int64(4)
                                )
                            )
                            % T.int64(4),
                        ] = rxplaceholder_2[
                            (fused * T.int64(2) + k) // T.int64(2),
                            (fused * T.int64(2) + k) % T.int64(2),
                        ]

        @R.function
        def main(
            x: R.Tensor((4, 4), dtype="float32"),
            indices: R.Tensor((2, 2), dtype="int64"),
            updates: R.Tensor((2, 2), dtype="float32"),
        ) -> R.Tensor((4, 4), dtype="float32"):
            gv = R.call_tir(
                Expected.scatter_elements,
                (x, indices, updates),
                out_sinfo=R.Tensor((4, 4), dtype="float32"),
            )
            return gv

    # fmt: on
    mod = LegalizeOps()(ScatterElements)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_scatter_elements_symbolic():
    # fmt: off
    @I.ir_module
    class ScatterElements:
        @R.function
        def main(x: R.Tensor(("a", "b"), "float32"), indices:R.Tensor(("m", "n"), "int64"), updates:R.Tensor(("m","n"), "float32")):
            gv = R.scatter_elements(x, indices, updates, axis=1)
            return gv
    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def scatter_elements(
            var_rxplaceholder: T.handle,
            var_rxplaceholder_1: T.handle,
            var_rxplaceholder_2: T.handle,
            var_scatter_elements_generic: T.handle,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            a, b = T.int64(), T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b), offset_factor=1)
            m, n = T.int64(), T.int64()
            rxplaceholder_1 = T.match_buffer(
                var_rxplaceholder_1, (m, n), "int64", offset_factor=1
            )
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (m, n), offset_factor=1)
            out_buf = T.match_buffer(var_scatter_elements_generic, (a, b))
            with T.block("scatter_elements_generic"):
                for i in T.parallel(a * b):
                    out_buf[i // b, i % b] = rxplaceholder[i // b, i % b]
                for fused in T.parallel(m):
                    for k in range(n):
                        out_buf[
                            (
                                fused * b
                                + (
                                    rxplaceholder_1[
                                        (fused * n + k) // n, (fused * n + k) % n
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * n + k) // n, (fused * n + k) % n
                                        ]
                                        < T.int64(0),
                                    )
                                    * b
                                )
                            )
                            // b,
                            (
                                fused * b
                                + (
                                    rxplaceholder_1[
                                        (fused * n + k) // n, (fused * n + k) % n
                                    ]
                                    + T.Cast(
                                        "int64",
                                        rxplaceholder_1[
                                            (fused * n + k) // n, (fused * n + k) % n
                                        ]
                                        < T.int64(0),
                                    )
                                    * b
                                )
                            )
                            % b,
                        ] = rxplaceholder_2[(fused * n + k) // n, (fused * n + k) % n]

        @R.function
        def main(
            x: R.Tensor(("a", "b"), dtype="float32"),
            indices: R.Tensor(("m", "n"), dtype="int64"),
            updates: R.Tensor(("m", "n"), dtype="float32"),
        ) -> R.Tensor(("a", "b"), dtype="float32"):
            a = T.int64()
            b = T.int64()
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(
                Expected.scatter_elements,
                (x, indices, updates),
                out_sinfo=R.Tensor((a, b), dtype="float32"),
            )
            return gv
    # fmt: on

    mod = LegalizeOps()(ScatterElements)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layout_transform():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2
    # fmt: off
    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor((10, 21, 30), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value
            )
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def te_layout_transform(A: T.Buffer((T.int64(10), T.int64(21), T.int64(30)), "float32"), te_layout_transform_1: T.Buffer((T.int64(10), T.int64(30), T.int64(7), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1, i2 in T.grid(T.int64(10), T.int64(21), T.int64(30)):
                with T.block("te_layout_transform"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(A[v_i0, v_i1, v_i2])
                    T.writes(te_layout_transform_1[v_i0, v_i2, v_i1 // T.int64(3), v_i1 % T.int64(3)])
                    te_layout_transform_1[v_i0, v_i2, v_i1 // T.int64(3), v_i1 % T.int64(3)] = A[v_i0, v_i1, v_i2]

        @R.function
        def main(x: R.Tensor((10, 21, 30), dtype="float32")) -> R.Tensor((10, 30, 7, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform, (x,), out_sinfo=R.Tensor((10, 30, 7, 3), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layout_transform_with_pad():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2
    # fmt: off
    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor((10, 20, 30), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value
            )
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def te_layout_transform_with_pad(A: T.Buffer((T.int64(10), T.int64(20), T.int64(30)), "float32"), te_layout_transform_with_pad_1: T.Buffer((T.int64(10), T.int64(30), T.int64(7), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for axis0, axis1, axis2, axis3 in T.grid(T.int64(10), T.int64(30), T.int64(7), T.int64(3)):
                with T.block("te_layout_transform_with_pad"):
                    v_axis0, v_axis1, v_axis2, v_axis3 = T.axis.remap("SSSS", [axis0, axis1, axis2, axis3])
                    T.reads(A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])
                    T.writes(te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3])
                    te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3] = T.if_then_else(v_axis2 == T.int64(6) and v_axis3 == T.int64(2), T.float32(2), A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])

        @R.function
        def main(x: R.Tensor((10, 20, 30), dtype="float32")) -> R.Tensor((10, 30, 7, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform_with_pad, (x,), out_sinfo=R.Tensor((10, 30, 7, 3), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layout_transform_symbolic():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2
    # fmt: off
    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value
            )
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def te_layout_transform_with_pad(var_A: T.handle, var_te_layout_transform_with_pad: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            a, b, c = T.int64(), T.int64(), T.int64()
            A = T.match_buffer(var_A, (a, b, c))
            te_layout_transform_with_pad_1 = T.match_buffer(var_te_layout_transform_with_pad, (a, c, (b - b % T.int64(-3)) // T.int64(3), T.int64(3)))
            # with T.block("root"):
            for axis0, axis1, axis2, axis3 in T.grid(a, c, (b - b % T.int64(-3)) // T.int64(3), T.int64(3)):
                with T.block("te_layout_transform_with_pad_with_pad"):
                    v_axis0, v_axis1, v_axis2, v_axis3 = T.axis.remap("SSSS", [axis0, axis1, axis2, axis3])
                    T.reads(A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])
                    T.writes(te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3])
                    te_layout_transform_with_pad_1[v_axis0, v_axis1, v_axis2, v_axis3] = T.if_then_else(b % T.int64(-3) < T.int64(0) and v_axis2 == b // T.int64(3) and b % T.int64(3) <= v_axis3, T.float32(2), A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])

        @R.function
        def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor(("a", "c", "(b - b % -3) // 3", 3), dtype="float32"):
            a = T.int64()
            c = T.int64()
            b = T.int64()
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform_with_pad, (x,), out_sinfo=R.Tensor((a, c, (b - b % -3) // 3, 3), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layout_transform_with_pad_axis_sep():
    transformation = lambda a, b, c: (a, c, b // 3, b % 3)
    pad_value = 2
    axis_separator = [3]
    # fmt: off
    @I.ir_module
    class LayoutTransform:
        @R.function
        def main(x: R.Tensor((10, 20, 30), "float32")):
            gv = R.layout_transform(
                x, index_map=transformation, pad_value=pad_value, axis_separators=axis_separator,
            )
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def te_layout_transform_with_pad_axis_separator(A: T.Buffer((T.int64(10), T.int64(20), T.int64(30)), "float32"), var_te_layout_transform_with_pad_axis_separator: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            te_layout_transform_with_pad_axis_separator_1 = T.match_buffer(var_te_layout_transform_with_pad_axis_separator, (T.int64(10), T.int64(30), T.int64(7), T.int64(3)), axis_separators=[3])
            # with T.block("root"):
            for axis0, axis1, axis2, axis3 in T.grid(T.int64(10), T.int64(30), T.int64(7), T.int64(3)):
                with T.block("te_layout_transform_with_pad_axis_separator"):
                    v_axis0, v_axis1, v_axis2, v_axis3 = T.axis.remap("SSSS", [axis0, axis1, axis2, axis3])
                    T.reads(A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])
                    T.writes(te_layout_transform_with_pad_axis_separator_1[v_axis0, v_axis1, v_axis2, v_axis3])
                    te_layout_transform_with_pad_axis_separator_1[v_axis0, v_axis1, v_axis2, v_axis3] = T.if_then_else(v_axis2 == T.int64(6) and v_axis3 == T.int64(2), T.float32(2), A[v_axis0, v_axis2 * T.int64(3) + v_axis3, v_axis1])

        @R.function
        def main(x: R.Tensor((10, 20, 30), dtype="float32")) -> R.Tensor((10, 30, 7, 3), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.te_layout_transform_with_pad_axis_separator, (x,), out_sinfo=R.Tensor((10, 30, 7, 3), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(LayoutTransform)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_func_struct_info_of_legalized_layout_transform():
    """PrimFunc shape information must be correct

    This is a regression test.  Previously, the legalization of
    `R.layout_transform` produced a PrimFunc with `FuncStructInfo`
    different than its actual signature.  This resulted in errors
    when later passes attempted to infer the StructInfo.
    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"), y: R.Tensor((16,), dtype="float32")
        ) -> R.Tensor((16,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            with R.dataflow():
                lv: R.Tensor((4, 4), dtype="float32") = R.layout_transform(
                    x, index_map=lambda i: (i // 4, i % 4), pad_value=None
                )
                gv: R.Tensor((4, 4), dtype="float32") = lv
                R.output(gv)
            return gv

    After = tvm.ir.transform.Sequential(
        [
            relax.transform.LegalizeOps(),
            relax.transform.ToNonDataflow(),
            relax.transform.RemovePurityChecking(),
            relax.transform.CallTIRRewrite(),
        ]
    )(Before)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((16,), dtype="float32"),
            y: R.Tensor((16,), dtype="float32"),
        ):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            alloc: R.Tensor((4, 4), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([4, 4]), R.dtype("float32"), R.prim_value(0), R.str("global")
            )
            cls.te_layout_transform(x, alloc)
            lv = alloc
            gv = lv
            return gv

        @T.prim_func(private=True)
        def te_layout_transform(
            A: T.Buffer((T.int64(16),), "float32"),
            te_layout_transform: T.Buffer((T.int64(4), T.int64(4)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for i in range(T.int64(16)):
                with T.block("te_layout_transform"):
                    vi = T.axis.spatial(T.int64(16), i)
                    te_layout_transform[vi // T.int64(4), vi % T.int64(4)] = A[vi]

    tvm.ir.assert_structural_equal(Expected, After)


def test_scatter_nd():

    # fmt: off
    @I.ir_module
    class Before:
        @R.function
        def main(
            data: R.Tensor((8,), "float32"),
            indices: R.Tensor((4, 1), "int64"),
            updates: R.Tensor((4,), "float32"),
        ) -> R.Tensor((8,), "float32"):
            gv: R.Tensor((8,), "float32") = R.scatter_nd(data, indices, updates, reduction="update")
            return gv

    After = relax.transform.LegalizeOps()(Before)

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((8,), "float32"),
            indices: R.Tensor((4, 1), "int64"),
            updates: R.Tensor((4,), "float32"),
        ) -> R.Tensor((8,), "float32"):
            gv = R.call_tir(
                Expected.scatter_nd, (data, indices, updates), R.Tensor((8,), dtype="float32")
            )
            return gv

        @T.prim_func(private=True)
        def scatter_nd(var_data: T.handle, var_indices: T.handle, var_updates: T.handle, var_scatter_nd_generic: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            data = T.match_buffer(var_data, (T.int64(8),), offset_factor=1)
            indices = T.match_buffer(var_indices, (T.int64(4), T.int64(1)), "int64")
            updates = T.match_buffer(var_updates, (T.int64(4),), offset_factor=1)
            out_buf = T.match_buffer(var_scatter_nd_generic, (T.int64(8),))
            with T.block("root"):
                T.reads()
                T.writes()
                T_transpose = T.alloc_buffer((T.int64(1), T.int64(4)), "int64")
                for ax0 in range(T.int64(1)):
                    for ax1 in range(T.int64(4)):
                        with T.block("T_transpose"):
                            v_ax0 = T.axis.spatial(T.int64(1), ax0)
                            v_ax1 = T.axis.spatial(T.int64(4), ax1)
                            T.reads(indices[v_ax1, v_ax0])
                            T.writes(T_transpose[v_ax0, v_ax1])
                            T_transpose[v_ax0, v_ax1] = indices[v_ax1, v_ax0]
                with T.block("scatter_nd_generic"):
                    T.reads()
                    T.writes()
                    for i in range(T.int64(8)):
                        out_buf[i] = data[i]
                    for j in range(T.int64(4)):
                        for k in T.parallel(T.int64(1)):
                            out_buf[k + T_transpose[j // T.int64(4), j % T.int64(4)]] = updates[j + k]

    # fmt: on
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
