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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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
        @T.prim_func
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

        @T.prim_func
        def reshape(var_rxplaceholder: T.handle, var_T_reshape: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b], dtype="float32")
            T_reshape = T.match_buffer(var_T_reshape, [a // T.int64(2), b * T.int64(2)], dtype="float32")
            for i0, i1 in T.grid(a // T.int64(2), b * T.int64(2)):
                with T.block("T_reshape"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[(ax0 * (b * T.int64(2)) + ax1) // b % a, (ax0 * (b * T.int64(2)) + ax1) % b])
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[(ax0 * (b * T.int64(2)) + ax1) // b % a, (ax0 * (b * T.int64(2)) + ax1) % b]
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

        @T.prim_func
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
                            (ax0 * (b * T.int64(2)) + ax1) // b % a,
                            (ax0 * (b * T.int64(2)) + ax1) % b,
                        ]
                    )
                    T.writes(T_reshape[ax0, ax1])
                    T_reshape[ax0, ax1] = rxplaceholder[
                        (ax0 * (b * T.int64(2)) + ax1) // b % a, (ax0 * (b * T.int64(2)) + ax1) % b
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
        @T.prim_func
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
                            (v_ax0 * (b * T.int64(2)) + v_ax1) // b % T.int64(10),
                            (v_ax0 * (b * T.int64(2)) + v_ax1) % b,
                        ]
                    )
                    T.writes(T_reshape[v_ax0, v_ax1])
                    T_reshape[v_ax0, v_ax1] = rxplaceholder[
                        (v_ax0 * (b * T.int64(2)) + v_ax1) // b % T.int64(10),
                        (v_ax0 * (b * T.int64(2)) + v_ax1) % b,
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
        def main(x: R.Tensor((3, ), dtype="int64")):
            lv: R.Shape([3,]) = R.tensor_to_shape(x)
            gv = R.reshape(x, lv)
            return gv
    
    assert relax.analysis.well_formed(DDReshape)
    mod = relax.transform.DecomposeCompositeOperator()(DDReshape)
    out_mod = relax.transform.LegalizeOps()(mod)

    @I.ir_module
    class Expected:
        @T.prim_func
        def reshape(
            rxplaceholder: T.Buffer((T.int64(3),), "int64"), var_T_reshape: T.handle
        ):
            T.func_attr({"tir.noalias": True})
            x = T.int64()
            T_reshape = T.match_buffer(var_T_reshape, (x,), "int64")
            # with T.block("root"):
            for ax0 in range(x):
                with T.block("T_reshape"):
                    v_ax0 = T.axis.spatial(x, ax0)
                    T.reads(rxplaceholder[v_ax0 % T.int64(3)])
                    T.writes(T_reshape[v_ax0])
                    T_reshape[v_ax0] = rxplaceholder[v_ax0 % T.int64(3)]

        @R.function
        def main(x: R.Tensor((3,), dtype="int64")) -> R.Tensor((3,), dtype="int64"):
            x_1 = T.int64()
            gv: R.Shape([3]) = R.call_packed(
                "vm.builtin.tensor_to_shape", x, sinfo_args=(R.Shape([3]),)
            )
            y: R.Shape([x_1]) = R.match_cast(gv, R.Shape([x_1]))
            lv: R.Shape([x_1]) = R.shape([x_1])
            gv_1 = R.call_tir(Expected.reshape, (x,), out_sinfo=R.Tensor((x_1,), dtype="int64"))
            return gv_1
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
            gv: R.Tensor((2, 3, 1, 4), "float32") = R.squeeze(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 1, 3, 1, 1, 4), "float32")) -> R.Tensor((2, 3, 1, 4), "float32"):
            gv = R.call_tir(Expected.squeeze, (x,), R.Tensor((2, 3, 4), dtype="float32"))
            return gv

        @T.prim_func
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

        @T.prim_func
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

        @T.prim_func
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


@pytest.mark.skip("TOPI collapse_sum not support symbolic now")
def test_collapse_sum_like_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class CollapseSumLike:
        @R.function
        def main(x: R.Tensor(("a", "b", "a"), "float32"), y: R.Tensor(("b", 1), "float32")) -> R.Tensor(("b", 1), "float32"):
            b = T.int64()
            gv: R.Tensor((b, 1), "float32") = R.collapse_sum_like(x, y)
            return gv

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

        @T.prim_func
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


@pytest.mark.skip("TOPI collapse_sum not support symbolic now")
def test_collapse_sum_to_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class CollapseSumTo:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("b", 1), "float32"):
            b = T.int64()
            gv: R.Tensor((b, 1), "float32") = R.collapse_sum_to(x, (b, 1))
            return gv

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

        @T.prim_func
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

        @T.prim_func
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
        @T.prim_func
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
        @T.prim_func
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
        @T.prim_func
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


def test_cumsum():
    # fmt: off
    @I.ir_module
    class Cumsum:
        @R.function
        def main(x: R.Tensor((3, 2, 3), "float32")):
            gv = R.cumsum(x, axis=1, dtype="int32")
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def cumsum(var_rxplaceholder: T.handle, out_buf: T.Buffer((T.int64(3), T.int64(2), T.int64(3)), "int32")):
            T.func_attr({"tir.noalias": True})
            rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(3), T.int64(2), T.int64(3)), offset_factor=1)
            with T.block("cumsum_generic"):
                T.reads(rxplaceholder[T.int64(0):T.int64(3), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                T.writes(out_buf[T.int64(0):T.int64(3), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                for fused in T.parallel(T.int64(9)):
                    out_buf[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) % T.int64(3)] = T.Cast("int32", rxplaceholder[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3)) % T.int64(3)])
                    for _k in range(T.int64(1)):
                        out_buf[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) % T.int64(3)] = out_buf[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1) - T.int64(1)) * T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1) - T.int64(1)) * T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1) - T.int64(1)) * T.int64(3)) % T.int64(3)] + T.Cast("int32", rxplaceholder[(fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) // T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) // T.int64(3) % T.int64(2), (fused // T.int64(3) * T.int64(2) * T.int64(3) + fused % T.int64(3) + (_k + T.int64(1)) * T.int64(3)) % T.int64(3)])

        @R.function
        def main(x: R.Tensor((3, 2, 3), dtype="float32")) -> R.Tensor((3, 2, 3), dtype="int32"):
            cls = Expected
            gv = R.call_tir(cls.cumsum, (x,), out_sinfo=R.Tensor((3, 2, 3), dtype="int32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Cumsum)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cumsum_symbolic():
    # fmt: off
    @I.ir_module
    class Cumsum:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")):
            gv = R.cumsum(x, axis=1, dtype="int32")
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def cumsum(var_rxplaceholder: T.handle, var_cumsum_generic: T.handle):
            T.func_attr({"tir.noalias": True})
            a, b, c = T.int64(), T.int64(), T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (a, b, c), offset_factor=1)
            out_buf = T.match_buffer(var_cumsum_generic, (a, b, c), "int32")
            with T.block("cumsum_generic"):
                T.reads(rxplaceholder[T.int64(0):a, T.int64(0):b, T.int64(0):c])
                T.writes(out_buf[T.int64(0):a, T.int64(0):b, T.int64(0):c])
                for fused in T.parallel(a * c):
                    out_buf[(fused // c * b * c + fused % c) // c // b, (fused // c * b * c + fused % c) // c % b, (fused // c * b * c + fused % c) % c] = T.Cast("int32", rxplaceholder[(fused // c * b * c + fused % c) // c // b, (fused // c * b * c + fused % c) // c % b, (fused // c * b * c + fused % c) % c])
                    for _k in range(b - T.int64(1)):
                        out_buf[(fused // c * b * c + fused % c + (_k + T.int64(1)) * c) // c // b, (fused // c * b * c + fused % c + (_k + T.int64(1)) * c) // c % b, (fused // c * b * c + fused % c + (_k + T.int64(1)) * c) % c] = out_buf[(fused // c * b * c + fused % c + (_k + T.int64(1) - T.int64(1)) * c) // c // b, (fused // c * b * c + fused % c + (_k + T.int64(1) - T.int64(1)) * c) // c % b, (fused // c * b * c + fused % c + (_k + T.int64(1) - T.int64(1)) * c) % c] + T.Cast("int32", rxplaceholder[(fused // c * b * c + fused % c + (_k + T.int64(1)) * c) // c // b, (fused // c * b * c + fused % c + (_k + T.int64(1)) * c) // c % b, (fused // c * b * c + fused % c + (_k + T.int64(1)) * c) % c])

        @R.function
        def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor(("a", "b", "c"), dtype="int32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            cls = Expected
            gv = R.call_tir(cls.cumsum, (x,), out_sinfo=R.Tensor((a, b, c), dtype="int32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(Cumsum)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
