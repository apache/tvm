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
from tvm.relax.transform import LegalizeOps
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

##################### Neural network #####################


def test_conv1d():
    # fmt: off
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((2, 128, 28), "float32"), w: R.Tensor((64, 16, 3), "float32")) -> R.Tensor((2, 64, 13), "float32"):
            gv: R.Tensor((2, 4, 13), "float32") = R.nn.conv1d(x, w, strides=(2,), padding=(1,), dilation=(2,), groups=8)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 128, 28), dtype="float32"), w: R.Tensor((64, 16, 3), dtype="float32")) -> R.Tensor((2, 64, 13), dtype="float32"):
            gv = R.call_tir(Expected.conv1d, (x, w), out_sinfo=R.Tensor((2, 64, 13), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv1d(A: T.Buffer((T.int64(2), T.int64(128), T.int64(28)), "float32"), B: T.Buffer((T.int64(64), T.int64(16), T.int64(3)), "float32"), group_conv1d_ncw: T.Buffer((T.int64(2), T.int64(64), T.int64(13)), "float32")):
            T.func_attr({"tir.noalias": True})
            pad_temp = T.alloc_buffer((T.int64(2), T.int64(128), T.int64(30)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(128), T.int64(30)):
                with T.block("pad_temp"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(A[v_i0, v_i1, v_i2 - T.int64(1)])
                    T.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(29), A[v_i0, v_i1, v_i2 - T.int64(1)], T.float32(0))
            for nn, ff, yy, rc, ry in T.grid(T.int64(2), T.int64(64), T.int64(13), T.int64(16), T.int64(3)):
                with T.block("group_conv1d_ncw"):
                    v_nn, v_ff, v_yy, v_rc, v_ry = T.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                    T.reads(pad_temp[v_nn, v_ff // T.int64(8) * T.int64(16) + v_rc, v_yy * T.int64(2) + v_ry * T.int64(2)], B[v_ff, v_rc, v_ry])
                    T.writes(group_conv1d_ncw[v_nn, v_ff, v_yy])
                    with T.init():
                        group_conv1d_ncw[v_nn, v_ff, v_yy] = T.float32(0)
                    group_conv1d_ncw[v_nn, v_ff, v_yy] = group_conv1d_ncw[v_nn, v_ff, v_yy] + pad_temp[v_nn, v_ff // T.int64(8) * T.int64(16) + v_rc, v_yy * T.int64(2) + v_ry * T.int64(2)] * B[v_ff, v_rc, v_ry]
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv1d_with_out_dtype():
    # fmt: off
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((2, 3, 28), "float32"), w: R.Tensor((4, 3, 3), "float32")) -> R.Tensor((2, 4, 26), "float16"):
            gv: R.Tensor((2, 4, 26), "float16") = R.nn.conv1d(x, w, out_dtype="float16")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 28), dtype="float32"), w: R.Tensor((4, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 26), dtype="float16"):
            gv = R.call_tir(Expected.conv1d, (x, w), out_sinfo=R.Tensor((2, 4, 26), dtype="float16"))
            return gv

        @T.prim_func(private=True)
        def conv1d(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(28)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(3)), "float32"), conv1d_ncw: T.Buffer((T.int64(2), T.int64(4), T.int64(26)), "float16")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            pad_temp = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(3), T.int64(28)):
                with T.block("pad_temp"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2])
                    T.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = rxplaceholder[v_i0, v_i1, v_i2]
            for nn, ff, yy, rc, ry in T.grid(T.int64(2), T.int64(4), T.int64(26), T.int64(3), T.int64(3)):
                with T.block("conv1d_ncw"):
                    v_nn, v_ff, v_yy, v_rc, v_ry = T.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                    T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry], rxplaceholder_1[v_ff, v_rc, v_ry])
                    T.writes(conv1d_ncw[v_nn, v_ff, v_yy])
                    with T.init():
                        conv1d_ncw[v_nn, v_ff, v_yy] = T.float16(0)
                    conv1d_ncw[v_nn, v_ff, v_yy] = conv1d_ncw[v_nn, v_ff, v_yy] + T.Cast("float16", pad_temp[v_nn, v_rc, v_yy + v_ry]) * T.Cast("float16", rxplaceholder_1[v_ff, v_rc, v_ry])
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv1d_nwc():
    # fmt: off
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor((2, 28, 128), "float32"), w: R.Tensor((64, 128, 3), "float32")) -> R.Tensor((2, 26, 64), "float32"):
            gv: R.Tensor((2, 26, 64), "float32") = R.nn.conv1d(x, w, data_layout="NWC")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 28, 128), dtype="float32"), w: R.Tensor((64, 128, 3), dtype="float32")) -> R.Tensor((2, 26, 64), dtype="float32"):
            gv = R.call_tir(Expected.conv1d, (x, w), out_sinfo=R.Tensor((2, 26, 64), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv1d(rxplaceholder: T.Buffer((T.int64(2), T.int64(28), T.int64(128)), "float32"), rxplaceholder_1: T.Buffer((T.int64(64), T.int64(128), T.int64(3)), "float32"), conv1d_nwc: T.Buffer((T.int64(2), T.int64(26), T.int64(64)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            pad_temp = T.alloc_buffer((T.int64(2), T.int64(28), T.int64(128)))
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(28), T.int64(128)):
                with T.block("pad_temp"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2])
                    T.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = rxplaceholder[v_i0, v_i1, v_i2]
            for nn, yy, ff, ry, rc in T.grid(T.int64(2), T.int64(26), T.int64(64), T.int64(3), T.int64(128)):
                with T.block("conv1d_nwc"):
                    v_nn, v_yy, v_ff, v_ry, v_rc = T.axis.remap("SSSRR", [nn, yy, ff, ry, rc])
                    T.reads(pad_temp[v_nn, v_yy + v_ry, v_rc], rxplaceholder_1[v_ff, v_rc, v_ry])
                    T.writes(conv1d_nwc[v_nn, v_yy, v_ff])
                    with T.init():
                        conv1d_nwc[v_nn, v_yy, v_ff] = T.float32(0)
                    conv1d_nwc[v_nn, v_yy, v_ff] = conv1d_nwc[v_nn, v_yy, v_ff] + pad_temp[v_nn, v_yy + v_ry, v_rc] * rxplaceholder_1[v_ff, v_rc, v_ry]
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv1d_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Conv1d:
        @R.function
        def main(x: R.Tensor(("n", "c", "w"), "float32"), kernel: R.Tensor(("f", "c", "kw"), "float32")) -> R.Tensor(("n", "f", "w - kw + 1"), "float32"):
            n = T.int64()
            w = T.int64()
            f = T.int64()
            kw = T.int64()
            gv: R.Tensor((n, f, w - kw + 1), "float32") = R.nn.conv1d(x, kernel)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("n", "c", "w"), dtype="float32"), kernel: R.Tensor(("f", "c", "kw"), dtype="float32")) -> R.Tensor(("n", "f", "w - kw + 1"), dtype="float32"):
            n = T.int64()
            f = T.int64()
            w = T.int64()
            kw = T.int64()
            c = T.int64()
            gv = R.call_tir(Expected.conv1d, (x, kernel), out_sinfo=R.Tensor((n, f, w + 1 - kw), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv1d(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_conv1d_ncw: T.handle):
            T.func_attr({"tir.noalias": True})
            n, c, w = T.int64(), T.int64(), T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (n, c, w))
            f, kw = T.int64(), T.int64()
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (f, c, kw))
            conv1d_ncw = T.match_buffer(var_conv1d_ncw, (n, f, w + T.int64(1) - kw))
            # with T.block("root"):
            pad_temp = T.alloc_buffer((n, c, w))
            for i0, i1, i2 in T.grid(n, c, w):
                with T.block("pad_temp"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2])
                    T.writes(pad_temp[v_i0, v_i1, v_i2])
                    pad_temp[v_i0, v_i1, v_i2] = rxplaceholder[v_i0, v_i1, v_i2]
            for nn, ff, yy, rc, ry in T.grid(n, f, w + T.int64(1) - kw, c, kw):
                with T.block("conv1d_ncw"):
                    v_nn, v_ff, v_yy, v_rc, v_ry = T.axis.remap("SSSRR", [nn, ff, yy, rc, ry])
                    T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry], rxplaceholder_1[v_ff, v_rc, v_ry])
                    T.writes(conv1d_ncw[v_nn, v_ff, v_yy])
                    with T.init():
                        conv1d_ncw[v_nn, v_ff, v_yy] = T.float32(0)
                    conv1d_ncw[v_nn, v_ff, v_yy] = conv1d_ncw[v_nn, v_ff, v_yy] + pad_temp[v_nn, v_rc, v_yy + v_ry] * rxplaceholder_1[v_ff, v_rc, v_ry]
    # fmt: on

    mod = LegalizeOps()(Conv1d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d():
    # fmt: off
    @tvm.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), "float32"), w: R.Tensor((64, 16, 3, 3), "float32")) -> R.Tensor((2, 64, 13, 13), "float32"):
            gv: R.Tensor((2, 4, 13, 13), "float32") = R.nn.conv2d(x, w, strides=(2, 2), padding=(1, 1), dilation=(2, 2), groups=8)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), "float32"), w: R.Tensor((64, 16, 3, 3), "float32")) -> R.Tensor((2, 64, 13, 13), "float32"):
            gv = R.call_tir(Expected.conv2d, (x, w), R.Tensor((2, 64, 13, 13), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv2d(rxplaceholder: T.Buffer((T.int64(2), T.int64(128), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Buffer((T.int64(64), T.int64(16), T.int64(3), T.int64(3)), "float32"), group_conv2d_nchw: T.Buffer((T.int64(2), T.int64(64), T.int64(13), T.int64(13)), "float32")):
            T.func_attr({"tir.noalias": True})
            pad_temp = T.alloc_buffer([T.int64(2), T.int64(128), T.int64(30), T.int64(30)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(128), T.int64(30), T.int64(30)):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1 - T.int64(1), i3_1 - T.int64(1)])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(T.int64(1) <= i2_1 and i2_1 < T.int64(29) and T.int64(1) <= i3_1 and i3_1 < T.int64(29), rxplaceholder[i0_1, i1_1, i2_1 - T.int64(1), i3_1 - T.int64(1)], T.float32(0), dtype="float32")
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(64), T.int64(13), T.int64(13), T.int64(16), T.int64(3), T.int64(3)):
                with T.block("group_conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    T.reads(pad_temp[nn, ff // T.int64(8) * T.int64(16) + rc, yy * T.int64(2) + ry * T.int64(2), xx * T.int64(2) + rx * T.int64(2)], rxplaceholder_1[ff, rc, ry, rx])
                    T.writes(group_conv2d_nchw[nn, ff, yy, xx])
                    with T.init():
                        group_conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
                    group_conv2d_nchw[nn, ff, yy, xx] = group_conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, ff // T.int64(8) * T.int64(16) + rc, yy * T.int64(2) + ry * T.int64(2), xx * T.int64(2) + rx * T.int64(2)] * rxplaceholder_1[ff, rc, ry, rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d_with_out_dtype():
    # fmt: off
    @tvm.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")) -> R.Tensor((2, 4, 26, 26), "float16"):
            gv: R.Tensor((2, 4, 26, 26), "float16") = R.nn.conv2d(x, w, out_dtype="float16")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")) -> R.Tensor((2, 4, 26, 26), "float16"):
            gv = R.call_tir(Expected.conv2d, (x, w), R.Tensor((2, 4, 26, 26), dtype="float16"))
            return gv

        @T.prim_func(private=True)
        def conv2d(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(3), T.int64(3), T.int64(3)), "float32"), conv2d_nchw: T.Buffer((T.int64(2), T.int64(4), T.int64(26), T.int64(26)), "float16")):
            T.func_attr({"tir.noalias": True})
            pad_temp = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(28), T.int64(28)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(4), T.int64(26), T.int64(26), T.int64(3), T.int64(3), T.int64(3)):
                with T.block("conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    T.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                    T.writes(conv2d_nchw[nn, ff, yy, xx])
                    with T.init():
                        conv2d_nchw[nn, ff, yy, xx] = T.float16(0)
                    conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + T.Cast("float16", pad_temp[nn, rc, yy + ry, xx + rx]) * T.Cast("float16", rxplaceholder_1[ff, rc, ry, rx])
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d_nhwc():
    # fmt: off
    @tvm.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor((2, 28, 28, 128), "float32"), w: R.Tensor((64, 128, 3, 3), "float32")) -> R.Tensor((2, 26, 26, 64), "float32"):
            gv: R.Tensor((2, 26, 26, 64), "float32") = R.nn.conv2d(x, w, data_layout="NHWC")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 28, 28, 128), "float32"), w: R.Tensor((64, 128, 3, 3), "float32")) -> R.Tensor((2, 26, 26, 64), "float32"):
            gv = R.call_tir(Expected.conv2d, (x, w), R.Tensor((2, 26, 26, 64), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv2d(rxplaceholder: T.Buffer((T.int64(2), T.int64(28), T.int64(28), T.int64(128)), "float32"), rxplaceholder_1: T.Buffer((T.int64(64), T.int64(128), T.int64(3), T.int64(3)), "float32"), conv2d_nhwc: T.Buffer((T.int64(2), T.int64(26), T.int64(26), T.int64(64)), "float32")):
            T.func_attr({"tir.noalias": True})
            pad_temp = T.alloc_buffer([T.int64(2), T.int64(28), T.int64(28), T.int64(128)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(28), T.int64(28), T.int64(128)):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(26), T.int64(26), T.int64(64), T.int64(3), T.int64(3), T.int64(128)):
                with T.block("conv2d_nhwc"):
                    nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    T.reads(pad_temp[nn, yy + ry, xx + rx, rc], rxplaceholder_1[ff, rc, ry, rx])
                    T.writes(conv2d_nhwc[nn, yy, xx, ff])
                    with T.init():
                        conv2d_nhwc[nn, yy, xx, ff] = T.float32(0)
                    conv2d_nhwc[nn, yy, xx, ff] = conv2d_nhwc[nn, yy, xx, ff] + pad_temp[nn, yy + ry, xx + rx, rc] * rxplaceholder_1[ff, rc, ry, rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Conv2d:
        @R.function
        def main(x: R.Tensor(("n", "c", "h", "w"), "float32"), kernel: R.Tensor(("f", "c", "kh", "kw"), "float32")) -> R.Tensor(("n", "f", "h - kh + 1", "w - kw + 1"), "float32"):
            n = T.int64()
            h = T.int64()
            w = T.int64()
            f = T.int64()
            kh = T.int64()
            kw = T.int64()
            gv: R.Tensor((n, f, h - kh + 1, w - kw + 1), "float32") = R.nn.conv2d(x, kernel)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("n", "c", "h", "w"), "float32"), kernel: R.Tensor(("f", "c", "kh", "kw"), "float32")) -> R.Tensor(("n", "f", "h - kh + 1", "w - kw + 1"), "float32"):
            n = T.int64()
            f = T.int64()
            h = T.int64()
            kh = T.int64()
            w = T.int64()
            kw = T.int64()
            gv = R.call_tir(Expected.conv2d, (x, kernel), R.Tensor((n, f, h + 1 - kh, w + 1 - kw), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv2d(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_conv2d_nchw: T.handle):
            T.func_attr({"tir.noalias": True})
            c = T.int64()
            f = T.int64()
            h = T.int64()
            kh = T.int64()
            kw = T.int64()
            n = T.int64()
            w = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [n, c, h, w], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [f, c, kh, kw], dtype="float32")
            conv2d_nchw = T.match_buffer(var_conv2d_nchw, [n, f, h + T.int64(1) - kh, w + T.int64(1) - kw], dtype="float32")
            pad_temp = T.alloc_buffer([n, c, h, w], dtype="float32")
            for i0, i1, i2, i3 in T.grid(n, c, h, w):
                with T.block("pad_temp"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, i2_1, i3_1])
                    T.writes(pad_temp[i0_1, i1_1, i2_1, i3_1])
                    pad_temp[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, i1_1, i2_1, i3_1]
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(n, f, h + T.int64(1) - kh, w + T.int64(1) - kw, c, kh, kw):
                with T.block("conv2d_nchw"):
                    nn, ff, yy, xx, rc, ry, rx = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                    T.reads(pad_temp[nn, rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                    T.writes(conv2d_nchw[nn, ff, yy, xx])
                    with T.init():
                        conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
                    conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + pad_temp[nn, rc, yy + ry, xx + rx] * rxplaceholder_1[ff, rc, ry, rx]
    # fmt: on

    mod = LegalizeOps()(Conv2d)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose():
    # fmt: off
    @I.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), "float32"), w: R.Tensor((128, 16, 3, 3), "float32")):
            gv = R.nn.conv2d_transpose(x, w, strides=(2, 3), padding=(1, 1), dilation=(1, 1), output_padding=(1, 2), groups=8)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 128, 28, 28), dtype="float32"), w: R.Tensor((128, 16, 3, 3), dtype="float32")) -> R.Tensor((2, 128, 56, 84), dtype="float32"):
            gv = R.call_tir(Expected.conv2d_transpose, (x, w), out_sinfo=R.Tensor((2, 128, 56, 84), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv2d_transpose(rxplaceholder: T.Buffer((T.int64(2), T.int64(128), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Buffer((T.int64(128), T.int64(16), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(128), T.int64(56), T.int64(84)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            data_dilate = T.alloc_buffer((T.int64(2), T.int64(128), T.int64(55), T.int64(82)))
            data_pad = T.alloc_buffer((T.int64(2), T.int64(128), T.int64(58), T.int64(86)))
            kernel_transform = T.alloc_buffer((T.int64(16), T.int64(128), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(128), T.int64(55), T.int64(82)):
                with T.block("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(3)])
                    T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(2) == T.int64(0) and v_i3 % T.int64(3) == T.int64(0), rxplaceholder[v_i0, v_i1, v_i2 // T.int64(2), v_i3 // T.int64(3)], T.float32(0))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(128), T.int64(58), T.int64(86)):
                with T.block("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
                    T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(56) and T.int64(1) <= v_i3 and v_i3 < T.int64(83), data_dilate[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
            for i, o, h, w in T.grid(T.int64(16), T.int64(128), T.int64(3), T.int64(3)):
                with T.block("kernel_transform"):
                    v_i, v_o, v_h, v_w = T.axis.remap("SSSS", [i, o, h, w])
                    T.reads(rxplaceholder_1[v_o, v_i, T.int64(2) - v_h, T.int64(2) - v_w])
                    T.writes(kernel_transform[v_i, v_o, v_h, v_w])
                    kernel_transform[v_i, v_o, v_h, v_w] = rxplaceholder_1[v_o, v_i, T.int64(2) - v_h, T.int64(2) - v_w]
            for b, c, h, w, dc, dh, dw in T.grid(T.int64(2), T.int64(128), T.int64(56), T.int64(84), T.int64(16), T.int64(3), T.int64(3)):
                with T.block("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = T.axis.remap("SSSSRRR", [b, c, h, w, dc, dh, dw])
                    T.reads(data_pad[v_b, v_c // T.int64(16) * T.int64(16) + v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c % T.int64(16), v_c // T.int64(16) * T.int64(16) + v_dc, v_dh, v_dw])
                    T.writes(compute[v_b, v_c, v_h, v_w])
                    with T.init():
                        compute[v_b, v_c, v_h, v_w] = T.float32(0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_c // T.int64(16) * T.int64(16) + v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c % T.int64(16), v_c // T.int64(16) * T.int64(16) + v_dc, v_dh, v_dw]
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose_with_out_dtype():
    # fmt: off
    @tvm.script.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((3, 4, 3, 3), "float32")):
            gv = R.nn.conv2d_transpose(x, w, out_dtype="float16")
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((3, 4, 3, 3), dtype="float32")) -> R.Tensor((2, 4, 30, 30), dtype="float16"):
            gv = R.call_tir(Expected.conv2d_transpose, (x, w), out_sinfo=R.Tensor((2, 4, 30, 30), dtype="float16"))
            return gv

        @T.prim_func(private=True)
        def conv2d_transpose(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Buffer((T.int64(3), T.int64(4), T.int64(3), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4), T.int64(30), T.int64(30)), "float16")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            data_dilate = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            data_pad = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(32), T.int64(32)))
            kernel_transform = T.alloc_buffer((T.int64(4), T.int64(3), T.int64(3), T.int64(3)))
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2, v_i3])
                    T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = rxplaceholder[v_i0, v_i1, v_i2, v_i3]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(32)):
                with T.block("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2)])
                    T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(2) <= v_i2 and v_i2 < T.int64(30) and T.int64(2) <= v_i3 and v_i3 < T.int64(30), data_dilate[v_i0, v_i1, v_i2 - T.int64(2), v_i3 - T.int64(2)], T.float32(0))
            for o, i, h, w in T.grid(T.int64(4), T.int64(3), T.int64(3), T.int64(3)):
                with T.block("kernel_transform"):
                    v_o, v_i, v_h, v_w = T.axis.remap("SSSS", [o, i, h, w])
                    T.reads(rxplaceholder_1[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w])
                    T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                    kernel_transform[v_o, v_i, v_h, v_w] = rxplaceholder_1[v_i, v_o, T.int64(2) - v_h, T.int64(2) - v_w]
            for b, c, h, w, dc, dh, dw in T.grid(T.int64(2), T.int64(4), T.int64(30), T.int64(30), T.int64(3), T.int64(3), T.int64(3)):
                with T.block("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = T.axis.remap("SSSSRRR", [b, c, h, w, dc, dh, dw])
                    T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                    T.writes(compute[v_b, v_c, v_h, v_w])
                    with T.init():
                        compute[v_b, v_c, v_h, v_w] = T.float16(0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + T.Cast("float16", data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw]) * T.Cast("float16", kernel_transform[v_c, v_dc, v_dh, v_dw])
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_conv2d_transpose_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Conv2dTranspose:
        @R.function
        def main(x: R.Tensor(("n", "c", "h", "w"), "float32"), kernel: R.Tensor(("f", "c", "kh", "kw"), "float32")):
            gv = R.nn.conv2d_transpose(x, kernel, strides=(3, 3))
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("n", "c", "h", "w"), dtype="float32"), kernel: R.Tensor(("f", "c", "kh", "kw"), dtype="float32")) -> R.Tensor(("n", "c", "h * 3 + kh - 3", "w * 3 + kw - 3"), dtype="float32"):
            n = T.int64()
            c = T.int64()
            h = T.int64()
            kh = T.int64()
            w = T.int64()
            kw = T.int64()
            f = T.int64()
            gv = R.call_tir(Expected.conv2d_transpose, (x, kernel), out_sinfo=R.Tensor((n, c, h * 3 + kh - 3, w * 3 + kw - 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def conv2d_transpose(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int64()
            c = T.int64()
            h = T.int64()
            w = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (n, c, h, w))
            f = T.int64()
            kh = T.int64()
            kw = T.int64()
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (f, c, kh, kw))
            compute = T.match_buffer(var_compute, (n, c, h * T.int64(3) + kh - T.int64(3), w * T.int64(3) + kw - T.int64(3)))
            # with T.block("root"):
            data_dilate = T.alloc_buffer((n, c, h * T.int64(3) - T.int64(2), w * T.int64(3) - T.int64(2)))
            data_pad = T.alloc_buffer((n, c, h * T.int64(3) + kh * T.int64(2) - T.int64(4), w * T.int64(3) + kw * T.int64(2) - T.int64(4)))
            kernel_transform = T.alloc_buffer((c, c, kh, kw))
            for i0, i1, i2, i3 in T.grid(n, c, h * T.int64(3) - T.int64(2), w * T.int64(3) - T.int64(2)):
                with T.block("data_dilate"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2 // T.int64(3), v_i3 // T.int64(3)])
                    T.writes(data_dilate[v_i0, v_i1, v_i2, v_i3])
                    data_dilate[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(v_i2 % T.int64(3) == T.int64(0) and v_i3 % T.int64(3) == T.int64(0), rxplaceholder[v_i0, v_i1, v_i2 // T.int64(3), v_i3 // T.int64(3)], T.float32(0))
            for i0, i1, i2, i3 in T.grid(n, c, h * T.int64(3) + kh * T.int64(2) - T.int64(4), w * T.int64(3) + kw * T.int64(2) - T.int64(4)):
                with T.block("data_pad"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(data_dilate[v_i0, v_i1, v_i2 + T.int64(1) - kh, v_i3 + T.int64(1) - kw])
                    T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
                    data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(kh <= v_i2 + T.int64(1) and v_i2 + T.int64(3)< h * T.int64(3) + kh and kw <= v_i3 + T.int64(1) and v_i3 + T.int64(3) < w * T.int64(3) + kw , data_dilate[v_i0, v_i1, v_i2 + T.int64(1) - kh, v_i3 + T.int64(1) - kw], T.float32(0))
            for o, i, h_1, w_1 in T.grid(c, c, kh, kw):
                with T.block("kernel_transform"):
                    v_o, v_i, v_h, v_w = T.axis.remap("SSSS", [o, i, h_1, w_1])
                    T.reads(rxplaceholder_1[v_i, v_o, kh - v_h - T.int64(1), kw - v_w - T.int64(1)])
                    T.writes(kernel_transform[v_o, v_i, v_h, v_w])
                    kernel_transform[v_o, v_i, v_h, v_w] = rxplaceholder_1[v_i, v_o, kh - v_h - T.int64(1), kw - v_w - T.int64(1)]
            for b, c_1, h_1, w_1, dc, dh, dw in T.grid(n, c, h * T.int64(3) + kh - T.int64(3), w * T.int64(3) + kw - T.int64(3), c, kh, kw):
                with T.block("compute"):
                    v_b, v_c, v_h, v_w, v_dc, v_dh, v_dw = T.axis.remap("SSSSRRR", [b, c_1, h_1, w_1, dc, dh, dw])
                    T.reads(data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw], kernel_transform[v_c, v_dc, v_dh, v_dw])
                    T.writes(compute[v_b, v_c, v_h, v_w])
                    with T.init():
                        compute[v_b, v_c, v_h, v_w] = T.float32(0)
                    compute[v_b, v_c, v_h, v_w] = compute[v_b, v_c, v_h, v_w] + data_pad[v_b, v_dc, v_h + v_dh, v_w + v_dw] * kernel_transform[v_c, v_dc, v_dh, v_dw]
    # fmt: on

    mod = LegalizeOps()(Conv2dTranspose)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d():
    # fmt: off
    @tvm.script.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), "float32")) -> R.Tensor((4, 56, 56, 6), "float32"):
            gv: R.Tensor((4, 56, 56, 6), "float32") = R.nn.max_pool2d(x, pool_size=[3, 3], strides=[2, 2], dilation=[1, 1], padding=[1, 1, 1, 1], layout="NHWC")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), "float32")) -> R.Tensor((4, 56, 56, 6), "float32"):
            gv = R.call_tir(Expected.max_pool2d, (x,), R.Tensor((4, 56, 56, 6), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def max_pool2d(rxplaceholder: T.Buffer((T.int64(4), T.int64(112), T.int64(112), T.int64(6)), "float32"), pool_max: T.Buffer((T.int64(4), T.int64(56), T.int64(56), T.int64(6)), "float32")):
            T.func_attr({"tir.noalias": True})
            pad_temp = T.alloc_buffer([T.int64(4), T.int64(114), T.int64(114), T.int64(6)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(114), T.int64(114), T.int64(6)):
                with T.block("pad_temp"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1 - T.int64(1), ax2 - T.int64(1), ax3])
                    T.writes(pad_temp[ax0, ax1, ax2, ax3])
                    pad_temp[ax0, ax1, ax2, ax3] = T.if_then_else(T.int64(1) <= ax1 and ax1 < T.int64(113) and T.int64(1) <= ax2 and ax2 < T.int64(113), rxplaceholder[ax0, ax1 - T.int64(1), ax2 - T.int64(1), ax3], T.float32(-3.4028234663852886e+38), dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(4), T.int64(56), T.int64(56), T.int64(6), T.int64(3), T.int64(3)):
                with T.block("pool_max"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(pad_temp[ax0, ax1 * T.int64(2) + rv0, ax2 * T.int64(2) + rv1, ax3])
                    T.writes(pool_max[ax0, ax1, ax2, ax3])
                    T.block_attr({"schedule_rule":"meta_schedule.pool_max"})
                    with T.init():
                        pool_max[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e+38)
                    pool_max[ax0, ax1, ax2, ax3] = T.max(pool_max[ax0, ax1, ax2, ax3], pad_temp[ax0, ax1 * T.int64(2) + rv0, ax2 * T.int64(2) + rv1, ax3])
    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d_NCHW16c():
    # fmt: off
    @tvm.script.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), "float32")) -> R.Tensor((4, 4, 110, 110, 16), "float32"):
            gv: R.Tensor((4, 4, 110, 110, 16), "float32") = R.nn.max_pool2d(x, pool_size=[3, 3], layout="NCHW16c")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), "float32")) -> R.Tensor((4, 4, 110, 110, 16), "float32"):
            gv = R.call_tir(Expected.max_pool2d, (x,), R.Tensor((4, 4, 110, 110, 16), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def max_pool2d(rxplaceholder: T.Buffer((T.int64(4), T.int64(4), T.int64(112), T.int64(112), T.int64(16)), "float32"), pool_max: T.Buffer((T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16), T.int64(3), T.int64(3)):
                with T.block("pool_max"):
                    ax0, ax1, ax2, ax3, ax4, rv0, rv1 = T.axis.remap("SSSSSRR", [i0, i1, i2, i3, i4, i5, i6])
                    T.reads(rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1, ax4])
                    T.writes(pool_max[ax0, ax1, ax2, ax3, ax4])
                    T.block_attr({"schedule_rule":"meta_schedule.pool_max"})
                    with T.init():
                        pool_max[ax0, ax1, ax2, ax3, ax4] = T.float32(-3.4028234663852886e+38)
                    pool_max[ax0, ax1, ax2, ax3, ax4] = T.max(pool_max[ax0, ax1, ax2, ax3, ax4], rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1, ax4])
    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_max_pool2d_ceil_mode():
    # fmt: off
    @tvm.script.ir_module
    class MaxPool2D:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), "float32")) -> R.Tensor((4, 6, 38, 38), "float32"):
            gv: R.Tensor((4, 6, 38, 38), "float32") = R.nn.max_pool2d(x, pool_size=[3, 3], strides=[3, 3], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=True)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), dtype="float32")) -> R.Tensor((4, 6, 38, 38), dtype="float32"):
            gv = R.call_tir(Expected.max_pool2d, (x,), R.Tensor((4, 6, 38, 38), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def max_pool2d(rxplaceholder: T.Buffer((T.int64(4), T.int64(6), T.int64(112), T.int64(112)), "float32"), pool_max: T.Buffer((T.int64(4), T.int64(6), T.int64(38), T.int64(38)), "float32")):
            T.func_attr({"tir.noalias": True})
            pad_temp = T.alloc_buffer([T.int64(4), T.int64(6), T.int64(116), T.int64(116)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(6), T.int64(116), T.int64(116)):
                with T.block("pad_temp"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2 - T.int64(1), ax3 - T.int64(1)])
                    T.writes(pad_temp[ax0, ax1, ax2, ax3])
                    pad_temp[ax0, ax1, ax2, ax3] = T.if_then_else(T.int64(1) <= ax2 and ax2 < T.int64(113) and T.int64(1) <= ax3 and ax3 < T.int64(113), rxplaceholder[ax0, ax1, ax2 - T.int64(1), ax3 - T.int64(1)], T.float32(-3.4028234663852886e+38), dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(4), T.int64(6), T.int64(38), T.int64(38), T.int64(3), T.int64(3)):
                with T.block("pool_max"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(pad_temp[ax0, ax1, ax2 * T.int64(3) + rv0, ax3 * T.int64(3) + rv1])
                    T.writes(pool_max[ax0, ax1, ax2, ax3])
                    T.block_attr({"schedule_rule":"meta_schedule.pool_max"})
                    with T.init():
                        pool_max[ax0, ax1, ax2, ax3] = T.float32(-3.4028234663852886e+38)
                    pool_max[ax0, ax1, ax2, ax3] = T.max(pool_max[ax0, ax1, ax2, ax3], pad_temp[ax0, ax1, ax2 * T.int64(3) + rv0, ax3 * T.int64(3) + rv1])
    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


@pytest.mark.skip("TOPI pooling casts every shape value to i32.")
def test_max_pool2d_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class MaxPool2D:
        @R.function
        def main(dumb_param: R.Tensor(("kh", "kw")), x: R.Tensor(("n", "c", "h", "w"), "float32")) -> R.Tensor(("n", "c", "h - kh + 1", "w - kw + 1"), "float32"):
            n = T.int64()
            c = T.int64()
            h = T.int64()
            w = T.int64()
            kh = T.int64()
            kw = T.int64()
            gv: R.Tensor((n, c, h - kh + 1, w - kw + 1), "float32") = R.nn.max_pool2d(x, pool_size=[kh, kw])
            return gv

    # fmt: on

    mod = LegalizeOps()(MaxPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d():
    # fmt: off
    @tvm.script.ir_module
    class AvgPool2D:
        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), "float32")) -> R.Tensor((4, 56, 56, 6), "float32"):
            gv: R.Tensor((4, 56, 56, 6), "float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], strides=[2, 2], dilation=[1, 1], padding=[1, 1, 1, 1], layout="NHWC")
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def avg_pool2d(rxplaceholder: T.Buffer((T.int64(4), T.int64(112), T.int64(112), T.int64(6)), "float32"), pool_avg: T.Buffer((T.int64(4), T.int64(56), T.int64(56), T.int64(6)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            pad_temp = T.alloc_buffer((T.int64(4), T.int64(114), T.int64(114), T.int64(6)))
            pool_sum = T.alloc_buffer((T.int64(4), T.int64(56), T.int64(56), T.int64(6)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(114), T.int64(114), T.int64(6)):
                with T.block("pad_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1 - T.int64(1), v_ax2 - T.int64(1), v_ax3])
                    T.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                    pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax1 and v_ax1 < T.int64(113) and T.int64(1) <= v_ax2 and v_ax2 < T.int64(113), rxplaceholder[v_ax0, v_ax1 - T.int64(1), v_ax2 - T.int64(1), v_ax3], T.float32(0))
            for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(4), T.int64(56), T.int64(56), T.int64(6), T.int64(3), T.int64(3)):
                with T.block("pool_sum"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
                    T.reads(pad_temp[v_ax0, v_ax1 * T.int64(2) + v_rv0, v_ax2 * T.int64(2) + v_rv1, v_ax3])
                    T.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    with T.init():
                        pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + pad_temp[v_ax0, v_ax1 * T.int64(2) + v_rv0, v_ax2 * T.int64(2) + v_rv1, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(56), T.int64(56), T.int64(6)):
                with T.block("pool_avg"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_avg"})
                    pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / T.Cast("float32", (T.min(T.int64(1), T.int64(112) - v_ax1 * T.int64(2)) + T.int64(2)) * (T.min(T.int64(1), T.int64(112) - v_ax2 * T.int64(2)) + T.int64(2)))

        @R.function
        def main(x: R.Tensor((4, 112, 112, 6), dtype="float32")) -> R.Tensor((4, 56, 56, 6), dtype="float32"):
            gv = R.call_tir(Expected.avg_pool2d, (x,), out_sinfo=R.Tensor((4, 56, 56, 6), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d_NCHW16c():
    # fmt: off
    @tvm.script.ir_module
    class AvgPool2D:
        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), "float32")) -> R.Tensor((4, 4, 110, 110, 16), "float32"):
            gv: R.Tensor((4, 4, 110, 110, 16), "float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], layout="NCHW16c")
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def avg_pool2d(rxplaceholder: T.Buffer((T.int64(4), T.int64(4), T.int64(112), T.int64(112), T.int64(16)), "float32"), pool_avg: T.Buffer((T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            pool_sum = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)))
            for ax0, ax1, ax2, ax3, ax4, rv0, rv1 in T.grid(T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16), T.int64(3), T.int64(3)):
                with T.block("pool_sum"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv0, v_rv1 = T.axis.remap("SSSSSRR", [ax0, ax1, ax2, ax3, ax4, rv0, rv1])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1, v_ax4])
                    T.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    with T.init():
                        pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.float32(0)
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] + rxplaceholder[v_ax0, v_ax1, v_ax2 + v_rv0, v_ax3 + v_rv1, v_ax4]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(4), T.int64(4), T.int64(110), T.int64(110), T.int64(16)):
                with T.block("pool_avg"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_avg"})
                    pool_avg[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] / T.Cast("float32", (T.min(T.int64(2), T.int64(111) - v_ax2) + T.int64(1)) * (T.min(T.int64(2), T.int64(111) - v_ax3) + T.int64(1)))

        @R.function
        def main(x: R.Tensor((4, 4, 112, 112, 16), dtype="float32")) -> R.Tensor((4, 4, 110, 110, 16), dtype="float32"):
            gv = R.call_tir(Expected.avg_pool2d, (x,), out_sinfo=R.Tensor((4, 4, 110, 110, 16), dtype="float32"))
            return gv
    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_avg_pool2d_ceil_mode():
    # fmt: off
    @tvm.script.ir_module
    class AvgPool2D:
        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), "float32")) -> R.Tensor((4, 6, 38, 38), "float32"):
            gv: R.Tensor((4, 6, 38, 38), "float32") = R.nn.avg_pool2d(x, pool_size=[3, 3], strides=[3, 3], dilation=[1, 1], padding=[1, 1, 1, 1], ceil_mode=True)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def avg_pool2d(rxplaceholder: T.Buffer((T.int64(4), T.int64(6), T.int64(112), T.int64(112)), "float32"), pool_avg: T.Buffer((T.int64(4), T.int64(6), T.int64(38), T.int64(38)), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            pad_temp = T.alloc_buffer((T.int64(4), T.int64(6), T.int64(116), T.int64(116)))
            pool_sum = T.alloc_buffer((T.int64(4), T.int64(6), T.int64(38), T.int64(38)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(6), T.int64(116), T.int64(116)):
                with T.block("pad_temp"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2 - T.int64(1), v_ax3 - T.int64(1)])
                    T.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
                    pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax2 and v_ax2 < T.int64(113) and T.int64(1) <= v_ax3 and v_ax3 < T.int64(113), rxplaceholder[v_ax0, v_ax1, v_ax2 - T.int64(1), v_ax3 - T.int64(1)], T.float32(0))
            for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(4), T.int64(6), T.int64(38), T.int64(38), T.int64(3), T.int64(3)):
                with T.block("pool_sum"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
                    T.reads(pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(3) + v_rv0, v_ax3 * T.int64(3) + v_rv1])
                    T.writes(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    with T.init():
                        pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0)
                    pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] + pad_temp[v_ax0, v_ax1, v_ax2 * T.int64(3) + v_rv0, v_ax3 * T.int64(3) + v_rv1]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(6), T.int64(38), T.int64(38)):
                with T.block("pool_avg"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(pool_sum[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(pool_avg[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.block_attr({"schedule_rule": "meta_schedule.pool_avg"})
                    pool_avg[v_ax0, v_ax1, v_ax2, v_ax3] = pool_sum[v_ax0, v_ax1, v_ax2, v_ax3] / T.Cast("float32", (T.min(T.int64(1), T.int64(112) - v_ax2 * T.int64(3)) + T.int64(2)) * (T.min(T.int64(1), T.int64(112) - v_ax3 * T.int64(3)) + T.int64(2)))

        @R.function
        def main(x: R.Tensor((4, 6, 112, 112), dtype="float32")) -> R.Tensor((4, 6, 38, 38), dtype="float32"):
            gv = R.call_tir(Expected.avg_pool2d, (x,), out_sinfo=R.Tensor((4, 6, 38, 38), dtype="float32"))
            return gv

    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


@pytest.mark.skip("TOPI pooling casts every shape value to i32.")
def test_avg_pool2d_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class AvgPool2D:
        @R.function
        def main(dumb_param: R.Tensor(("kh", "kw")), x: R.Tensor(("n", "c", "h", "w"), "float32")) -> R.Tensor(("n", "c", "h - kh + 1", "w - kw + 1"), "float32"):
            n = T.int64()
            c = T.int64()
            h = T.int64()
            w = T.int64()
            kh = T.int64()
            kw = T.int64()
            gv: R.Tensor((n, c, h - kh + 1, w - kw + 1), "float32") = R.nn.avg_pool2d(x, pool_size=[kh, kw])
            return gv

    # fmt: on

    mod = LegalizeOps()(AvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_adaptive_avg_pool2d():
    # fmt: off
    @tvm.script.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(x: R.Tensor((2, 4, 7, 7, 16), "float32")) -> R.Tensor((2, 4, 1, 1, 16), "float32"):
            gv: R.Tensor((2, 4, 1, 1, 16), "float32") = R.nn.adaptive_avg_pool2d(x, output_size=[1, 1], layout="NCHW16c")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 7, 7, 16), "float32")) -> R.Tensor((2, 4, 1, 1, 16), "float32"):
            gv = R.call_tir(Expected.adaptive_avg_pool2d, (x,), R.Tensor((2, 4, 1, 1, 16), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def adaptive_avg_pool2d(rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(7), T.int64(7), T.int64(16)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16)), "float32")):
            T.func_attr({"tir.noalias": True})
            adaptive_pool_sum = T.alloc_buffer([T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16)], dtype="float32")
            for i0, i1, i2, i3, i4, i5, i6 in T.grid(T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16), T.int64(7), T.int64(7)):
                with T.block("adaptive_pool_sum"):
                    ax0, ax1, ax2, ax3, ax4, rv0, rv1 = T.axis.remap("SSSSSRR", [i0, i1, i2, i3, i4, i5, i6])
                    T.reads(rxplaceholder[ax0, ax1, ax2 * T.int64(7) + rv0, ax3 * T.int64(7) + rv1, ax4])
                    T.writes(adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4])
                    with T.init():
                        adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] = T.float32(0)
                    adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] = adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] + rxplaceholder[ax0, ax1, ax2 * T.int64(7) + rv0, ax3 * T.int64(7) + rv1, ax4]
            for i0, i1, i2, i3, i4 in T.grid(T.int64(2), T.int64(4), T.int64(1), T.int64(1), T.int64(16)):
                with T.block("adaptive_pool_avg"):
                    ax0, ax1, ax2, ax3, ax4 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4])
                    T.writes(adaptive_pool_avg[ax0, ax1, ax2, ax3, ax4])
                    T.block_attr({"schedule_rule":"meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[ax0, ax1, ax2, ax3, ax4] = adaptive_pool_sum[ax0, ax1, ax2, ax3, ax4] * T.float32(0.020408163265306121)
    # fmt: on

    mod = LegalizeOps()(AdaptiveAvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_adaptive_avg_pool2d_without_output_size():
    # fmt: off
    @tvm.script.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(x: R.Tensor((2, 16, 7, 7), "float32")) -> R.Tensor((2, 16, 7, 7), "float32"):
            gv: R.Tensor((2, 16, 7, 7), "float32") = R.nn.adaptive_avg_pool2d(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 16, 7, 7), "float32")) -> R.Tensor((2, 16, 7, 7), "float32"):
            gv = R.call_tir(Expected.adaptive_avg_pool2d, (x,), R.Tensor((2, 16, 7, 7), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def adaptive_avg_pool2d(rxplaceholder: T.Buffer((T.int64(2), T.int64(16), T.int64(7), T.int64(7)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(2), T.int64(16), T.int64(7), T.int64(7)), "float32")):
            T.func_attr({"tir.noalias": True})
            adaptive_pool_sum = T.alloc_buffer([T.int64(2), T.int64(16), T.int64(7), T.int64(7)], dtype="float32")
            for i0, i1, i2, i3, i4, i5 in T.grid(T.int64(2), T.int64(16), T.int64(7), T.int64(7), T.int64(1), T.int64(1)):
                with T.block("adaptive_pool_sum"):
                    ax0, ax1, ax2, ax3, rv0, rv1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, i4, i5])
                    T.reads(rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1])
                    T.writes(adaptive_pool_sum[ax0, ax1, ax2, ax3])
                    with T.init():
                        adaptive_pool_sum[ax0, ax1, ax2, ax3] = T.float32(0)
                    adaptive_pool_sum[ax0, ax1, ax2, ax3] = adaptive_pool_sum[ax0, ax1, ax2, ax3] + rxplaceholder[ax0, ax1, ax2 + rv0, ax3 + rv1]
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(16), T.int64(7), T.int64(7)):
                with T.block("adaptive_pool_avg"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(adaptive_pool_sum[ax0, ax1, ax2, ax3])
                    T.writes(adaptive_pool_avg[ax0, ax1, ax2, ax3])
                    T.block_attr({"schedule_rule":"meta_schedule.adaptive_pool_avg"})
                    adaptive_pool_avg[ax0, ax1, ax2, ax3] = adaptive_pool_sum[ax0, ax1, ax2, ax3]
    # fmt: on

    mod = LegalizeOps()(AdaptiveAvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


@pytest.mark.skip("TOPI pooling casts every shape value to i32.")
def test_adaptive_avg_pool2d_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class AdaptiveAvgPool2D:
        @R.function
        def main(dumb_param: R.Tensor(("oh", "ow")), x: R.Tensor(("n", "c", "h", "w"), "float32")) -> R.Tensor(("n", "c", "oh", "ow"), "float32"):
            n = T.int64()
            c = T.int64()
            oh = T.int64()
            ow = T.int64()
            gv: R.Tensor((n, c, oh, ow), "float32") = R.nn.adaptive_avg_pool2d(x, (oh, ow))
            return gv
    # fmt: on

    mod = LegalizeOps()(AdaptiveAvgPool2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_relu():
    # fmt: off
    @tvm.script.ir_module
    class Relu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.relu(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.relu, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def relu(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.max(rxplaceholder[i0_1, i1_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Relu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_relu_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Relu:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv: R.Tensor((m, n), "float32") = R.nn.relu(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.relu, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def relu(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.max(rxplaceholder[i0_1, i1_1], T.float32(0))
    # fmt: on

    mod = LegalizeOps()(Relu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_leakyrelu():
    # fmt: off
    @tvm.script.ir_module
    class LeakyRelu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.leakyrelu(x, 0.02)
            return gv


    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.leaky_relu, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def leaky_relu(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.Select(T.float32(0) < rxplaceholder[i0_1, i1_1], rxplaceholder[i0_1, i1_1], \
                                                   rxplaceholder[i0_1, i1_1] * T.float32(0.02))
    # fmt: on

    mod = LegalizeOps()(LeakyRelu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_leakyrelu_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class LeakyRelu:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv: R.Tensor((m, n), "float32") = R.nn.leakyrelu(x, 0.03)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.leaky_relu, (x, ), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def leaky_relu(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            compute = T.match_buffer(var_compute, [m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.Select(T.float32(0) < rxplaceholder[i0_1, i1_1], rxplaceholder[i0_1, i1_1], \
                                                    rxplaceholder[i0_1, i1_1] * T.float32(0.03))
    # fmt: on

    mod = LegalizeOps()(LeakyRelu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_gelu():
    # fmt: off
    @tvm.script.ir_module
    class Gelu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.gelu(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.gelu, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def gelu(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            T_multiply_1 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            compute = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_2 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_divide = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_multiply_1[ax0, ax1])
                    T_multiply_1[ax0, ax1] = rxplaceholder[ax0, ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_1[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.erf(T_multiply_1[i0_1, i1_1], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_1"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(compute[ax0, ax1])
                    T.writes(T_multiply_2[ax0, ax1])
                    T_multiply_2[ax0, ax1] = compute[ax0, ax1] * T.float32(0.5)
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_divide"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_2[ax0, ax1])
                    T.writes(T_divide[ax0, ax1])
                    T_divide[ax0, ax1] = T.float32(0.5) + T_multiply_2[ax0, ax1]
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_2"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], T_divide[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * T_divide[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Gelu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_gelu_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Gelu:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv: R.Tensor((m, n), "float32") = R.nn.gelu(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.gelu, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def gelu(var_rxplaceholder: T.handle, var_T_multiply: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            T_multiply = T.match_buffer(var_T_multiply, [m, n], dtype="float32")
            T_multiply_1 = T.alloc_buffer([m, n], dtype="float32")
            compute = T.alloc_buffer([m, n], dtype="float32")
            T_multiply_2 = T.alloc_buffer([m, n], dtype="float32")
            T_add = T.alloc_buffer([m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1])
                    T.writes(T_multiply_1[ax0, ax1])
                    T_multiply_1[ax0, ax1] = rxplaceholder[ax0, ax1] * T.float32(0.70710678118654757)
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_1[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.erf(T_multiply_1[i0_1, i1_1], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("T_multiply_1"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(compute[ax0, ax1])
                    T.writes(T_multiply_2[ax0, ax1])
                    T_multiply_2[ax0, ax1] = compute[ax0, ax1] * T.float32(0.5)
            for i0, i1 in T.grid(m, n):
                with T.block("T_add"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_2[ax0, ax1])
                    T.writes(T_add[ax0, ax1])
                    T_add[ax0, ax1] = T.float32(0.5) + T_multiply_2[ax0, ax1]
            for i0, i1 in T.grid(m, n):
                with T.block("T_multiply_2"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], T_add[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * T_add[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Gelu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_gelu_tanh():
    # fmt: off
    @tvm.script.ir_module
    class GeluTanh:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.gelu_tanh(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            gv = R.call_tir(Expected.gelu_tanh, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def gelu_tanh(A: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            T_multiply_1 = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_multiply_2 = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_multiply_3 = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_multiply_4 = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_add = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_multiply_5 = T.alloc_buffer((T.int64(2), T.int64(3)))
            compute = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_add_1 = T.alloc_buffer((T.int64(2), T.int64(3)))
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = T.float32(0.5) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = T.float32(0.79788456080286541) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_2"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_multiply_3[v_ax0, v_ax1])
                    T_multiply_3[v_ax0, v_ax1] = T.float32(0.044714999999999998) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_3"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_3[v_ax0, v_ax1], A[v_ax0, v_ax1])
                    T.writes(T_multiply_4[v_ax0, v_ax1])
                    T_multiply_4[v_ax0, v_ax1] = T_multiply_3[v_ax0, v_ax1] * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_4[v_ax0, v_ax1])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(1) + T_multiply_4[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_4"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_2[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    T.writes(T_multiply_5[v_ax0, v_ax1])
                    T_multiply_5[v_ax0, v_ax1] = T_multiply_2[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_5[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.tanh(T_multiply_5[v_i0, v_i1])
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(compute[v_ax0, v_ax1])
                    T.writes(T_add_1[v_ax0, v_ax1])
                    T_add_1[v_ax0, v_ax1] = T.float32(1) + compute[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_5"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_1[v_ax0, v_ax1], T_add_1[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = T_multiply_1[v_ax0, v_ax1] * T_add_1[v_ax0, v_ax1]

    mod = LegalizeOps()(GeluTanh)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_gelu_tanh_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class GeluTanh:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv: R.Tensor((m, n), "float32") = R.nn.gelu_tanh(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), dtype="float32")) -> R.Tensor(("m", "n"), dtype="float32"):
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.gelu_tanh, (x,), out_sinfo=R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def gelu_tanh(var_A: T.handle, var_T_multiply: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            m, n = T.int64(), T.int64()
            A = T.match_buffer(var_A, (m, n))
            T_multiply = T.match_buffer(var_T_multiply, (m, n))
            # with T.block("root"):
            T_multiply_1 = T.alloc_buffer((m, n))
            T_multiply_2 = T.alloc_buffer((m, n))
            T_multiply_3 = T.alloc_buffer((m, n))
            T_multiply_4 = T.alloc_buffer((m, n))
            T_add = T.alloc_buffer((m, n))
            T_multiply_5 = T.alloc_buffer((m, n))
            compute = T.alloc_buffer((m, n))
            T_add_1 = T.alloc_buffer((m, n))
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_multiply_1[v_ax0, v_ax1])
                    T_multiply_1[v_ax0, v_ax1] = T.float32(0.5) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_multiply_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_multiply_2[v_ax0, v_ax1])
                    T_multiply_2[v_ax0, v_ax1] = T.float32(0.79788456080286541) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_multiply_2"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(A[v_ax0, v_ax1])
                    T.writes(T_multiply_3[v_ax0, v_ax1])
                    T_multiply_3[v_ax0, v_ax1] = T.float32(0.044714999999999998) * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_multiply_3"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_3[v_ax0, v_ax1], A[v_ax0, v_ax1])
                    T.writes(T_multiply_4[v_ax0, v_ax1])
                    T_multiply_4[v_ax0, v_ax1] = T_multiply_3[v_ax0, v_ax1] * A[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_4[v_ax0, v_ax1])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = T.float32(1) + T_multiply_4[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_multiply_4"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_2[v_ax0, v_ax1], T_add[v_ax0, v_ax1])
                    T.writes(T_multiply_5[v_ax0, v_ax1])
                    T_multiply_5[v_ax0, v_ax1] = T_multiply_2[v_ax0, v_ax1] * T_add[v_ax0, v_ax1]
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(T_multiply_5[v_i0, v_i1])
                    T.writes(compute[v_i0, v_i1])
                    compute[v_i0, v_i1] = T.tanh(T_multiply_5[v_i0, v_i1])
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_add_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(compute[v_ax0, v_ax1])
                    T.writes(T_add_1[v_ax0, v_ax1])
                    T_add_1[v_ax0, v_ax1] = T.float32(1) + compute[v_ax0, v_ax1]
            for ax0, ax1 in T.grid(m, n):
                with T.block("T_multiply_5"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_1[v_ax0, v_ax1], T_add_1[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = T_multiply_1[v_ax0, v_ax1] * T_add_1[v_ax0, v_ax1]


    mod = LegalizeOps()(GeluTanh)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_silu():
    # fmt: off
    @tvm.script.ir_module
    class Silu:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv: R.Tensor((2, 3), "float32") = R.nn.silu(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
            gv = R.call_tir(Expected.silu, (x,), R.Tensor((2, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def silu(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_multiply: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            compute = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1])
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], compute[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * compute[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Silu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_silu_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Silu:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv: R.Tensor((m, n), "float32") = R.nn.silu(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")) -> R.Tensor(("m", "n"), "float32"):
            m = T.int64()
            n = T.int64()
            gv = R.call_tir(Expected.silu, (x,), R.Tensor((m, n), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def silu(var_rxplaceholder: T.handle, var_T_multiply: T.handle):
            T.func_attr({"tir.noalias": True})
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [m, n], dtype="float32")
            T_multiply = T.match_buffer(var_T_multiply, [m, n], dtype="float32")
            compute = T.alloc_buffer([m, n], dtype="float32")
            for i0, i1 in T.grid(m, n):
                with T.block("compute"):
                    i0_1, i1_1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[i0_1, i1_1])
                    T.writes(compute[i0_1, i1_1])
                    compute[i0_1, i1_1] = T.sigmoid(rxplaceholder[i0_1, i1_1])
            for i0, i1 in T.grid(m, n):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], compute[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * compute[ax0, ax1]
    # fmt: on

    mod = LegalizeOps()(Silu)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_softmax():
    # fmt: off
    @tvm.script.ir_module
    class Softmax:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor((2, 3, 16, 32), "float32"):
            gv: R.Tensor((2, 3, 16, 32), "float32") = R.nn.softmax(x, axis=-2)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor((2, 3, 16, 32), "float32"):
            gv = R.call_tir(Expected.softmax, (x,), R.Tensor((2, 3, 16, 32), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def softmax(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"), T_softmax_norm: T.Buffer((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32")):
            T.func_attr({"tir.noalias": True})
            T_softmax_maxelem = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            T_softmax_exp = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(16), T.int64(32)], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with T.block("T_softmax_maxelem"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, k, i2_1])
                    T.writes(T_softmax_maxelem[i0_1, i1_1, i2_1])
                    with T.init():
                        T_softmax_maxelem[i0_1, i1_1, i2_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_1, i1_1, i2_1] = T.max(T_softmax_maxelem[i0_1, i1_1, i2_1], rxplaceholder[i0_1, i1_1, k, i2_1])
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
                with T.block("T_softmax_exp"):
                    i0_2, i1_2, i2_2, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_2, i1_2, i2_2, i3_1], T_softmax_maxelem[i0_2, i1_2, i3_1])
                    T.writes(T_softmax_exp[i0_2, i1_2, i2_2, i3_1])
                    T_softmax_exp[i0_2, i1_2, i2_2, i3_1] = T.exp(rxplaceholder[i0_2, i1_2, i2_2, i3_1] - T_softmax_maxelem[i0_2, i1_2, i3_1], dtype="float32")
            for i0_3, i1_3, i2_3, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with T.block("T_softmax_expsum"):
                    i0_4, i1_4, i2_4, k = T.axis.remap("SSSR", [i0_3, i1_3, i2_3, i3])
                    T.reads(T_softmax_exp[i0_4, i1_4, k, i2_4])
                    T.writes(T_softmax_expsum[i0_4, i1_4, i2_4])
                    with T.init():
                        T_softmax_expsum[i0_4, i1_4, i2_4] = T.float32(0)
                    T_softmax_expsum[i0_4, i1_4, i2_4] = T_softmax_expsum[i0_4, i1_4, i2_4] + T_softmax_exp[i0_4, i1_4, k, i2_4]
            for i0_5, i1_5, i2_5, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
                with T.block("T_softmax_norm"):
                    i0_6, i1_6, i2_6, i3_2 = T.axis.remap("SSSS", [i0_5, i1_5, i2_5, i3])
                    T.reads(T_softmax_exp[i0_6, i1_6, i2_6, i3_2], T_softmax_expsum[i0_6, i1_6, i3_2])
                    T.writes(T_softmax_norm[i0_6, i1_6, i2_6, i3_2])
                    T.block_attr({"axis":2})
                    T_softmax_norm[i0_6, i1_6, i2_6, i3_2] = T_softmax_exp[i0_6, i1_6, i2_6, i3_2] / T_softmax_expsum[i0_6, i1_6, i3_2]
    # fmt: on

    mod = LegalizeOps()(Softmax)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_softmax_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Softmax:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a", "b", "c"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv: R.Tensor((a, b, c), "float32") = R.nn.softmax(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a", "b", "c"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv = R.call_tir(Expected.softmax, (x,), R.Tensor((a, b, c), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def softmax(var_rxplaceholder: T.handle, var_T_softmax_norm: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c], dtype="float32")
            T_softmax_norm = T.match_buffer(var_T_softmax_norm, [a, b, c], dtype="float32")
            T_softmax_maxelem = T.alloc_buffer([a, b], dtype="float32")
            T_softmax_exp = T.alloc_buffer([a, b, c], dtype="float32")
            T_softmax_expsum = T.alloc_buffer([a, b], dtype="float32")
            for i0, i1, i2 in T.grid(a, b, c):
                with T.block("T_softmax_maxelem"):
                    i0_1, i1_1, k = T.axis.remap("SSR", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_1, i1_1, k])
                    T.writes(T_softmax_maxelem[i0_1, i1_1])
                    with T.init():
                        T_softmax_maxelem[i0_1, i1_1] = T.float32(-3.4028234663852886e+38)
                    T_softmax_maxelem[i0_1, i1_1] = T.max(T_softmax_maxelem[i0_1, i1_1], rxplaceholder[i0_1, i1_1, k])
            for i0, i1, i2 in T.grid(a, b, c):
                with T.block("T_softmax_exp"):
                    i0_2, i1_2, i2_1 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[i0_2, i1_2, i2_1], T_softmax_maxelem[i0_2, i1_2])
                    T.writes(T_softmax_exp[i0_2, i1_2, i2_1])
                    T_softmax_exp[i0_2, i1_2, i2_1] = T.exp(rxplaceholder[i0_2, i1_2, i2_1] - T_softmax_maxelem[i0_2, i1_2], dtype="float32")
            for i0_3, i1_3, i2 in T.grid(a, b, c):
                with T.block("T_softmax_expsum"):
                    i0_4, i1_4, k = T.axis.remap("SSR", [i0_3, i1_3, i2])
                    T.reads(T_softmax_exp[i0_4, i1_4, k])
                    T.writes(T_softmax_expsum[i0_4, i1_4])
                    with T.init():
                        T_softmax_expsum[i0_4, i1_4] = T.float32(0)
                    T_softmax_expsum[i0_4, i1_4] = T_softmax_expsum[i0_4, i1_4] + T_softmax_exp[i0_4, i1_4, k]
            for i0_5, i1_5, i2 in T.grid(a, b, c):
                with T.block("T_softmax_norm"):
                    i0_6, i1_6, i2_2 = T.axis.remap("SSS", [i0_5, i1_5, i2])
                    T.reads(T_softmax_exp[i0_6, i1_6, i2_2], T_softmax_expsum[i0_6, i1_6])
                    T.writes(T_softmax_norm[i0_6, i1_6, i2_2])
                    T.block_attr({"axis":2})
                    T_softmax_norm[i0_6, i1_6, i2_2] = T_softmax_exp[i0_6, i1_6, i2_2] / T_softmax_expsum[i0_6, i1_6]
    # fmt: on

    mod = LegalizeOps()(Softmax)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_log_softmax():
    # fmt: off
    @tvm.script.ir_module
    class LogSoftmax:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), "float32")) -> R.Tensor(None, "float32", ndim=4):
            gv: R.Tensor((2, 3, 16, 32), "float32") = R.nn.log_softmax(x, axis=-2)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 16, 32), dtype="float32")) -> R.Tensor((2, 3, 16, 32), dtype="float32"):
            gv = R.call_tir(Expected.log_softmax, (x,), R.Tensor((2, 3, 16, 32), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def log_softmax(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"), compute: T.Buffer((T.int64(2), T.int64(3), T.int64(16), T.int64(32)), "float32"),):
            T.func_attr({"tir.noalias": True})
            T_softmax_maxelem = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            compute_1 = T.alloc_buffer([T.int64(2), T.int64(3), T.int64(32)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with T.block("T_softmax_maxelem"):
                    i0_1, i1_1, i2_1, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, i1_1, k, i2_1])
                    T.writes(T_softmax_maxelem[i0_1, i1_1, i2_1])
                    with T.init():
                        T_softmax_maxelem[i0_1, i1_1, i2_1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[i0_1, i1_1, i2_1] = T.max(T_softmax_maxelem[i0_1, i1_1, i2_1], rxplaceholder[i0_1, i1_1, k, i2_1])
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(32), T.int64(16)):
                with T.block("compute"):
                    i0_2, i1_2, i2_2, k = T.axis.remap("SSSR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_2, i1_2, k, i2_2], T_softmax_maxelem[i0_2, i1_2, i2_2])
                    T.writes(compute_1[i0_2, i1_2, i2_2])
                    with T.init():
                        compute_1[i0_2, i1_2, i2_2] = T.float32(0)
                    compute_1[i0_2, i1_2, i2_2] = compute_1[i0_2, i1_2, i2_2] + T.exp(rxplaceholder[i0_2, i1_2, k, i2_2] - T_softmax_maxelem[i0_2, i1_2, i2_2], dtype="float32")
            for i0_3, i1_3, i2_3, i3 in T.grid(T.int64(2), T.int64(3), T.int64(16), T.int64(32)):
                with T.block("compute_1"):
                    i0_4, i1_4, i2_4, i3_1 = T.axis.remap("SSSS", [i0_3, i1_3, i2_3, i3])
                    T.reads(rxplaceholder[i0_4, i1_4, i2_4, i3_1], T_softmax_maxelem[i0_4, i1_4, i3_1], compute_1[i0_4, i1_4, i3_1])
                    T.writes(compute[i0_4, i1_4, i2_4, i3_1])
                    T.block_attr({"axis": 2})
                    compute[i0_4, i1_4, i2_4, i3_1] = (rxplaceholder[i0_4, i1_4, i2_4, i3_1] - T_softmax_maxelem[i0_4, i1_4, i3_1] - T.log(compute_1[i0_4, i1_4, i3_1], dtype="float32"))
    # fmt: on

    mod = LegalizeOps()(LogSoftmax)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_log_softmax_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class LogSoftmax:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), "float32")) -> R.Tensor(("a", "b", "c"), "float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            gv: R.Tensor((a, b, c), "float32") = R.nn.log_softmax(x)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("a", "b", "c"), dtype="float32")) -> R.Tensor(("a", "b", "c"), dtype="float32"):
            a = T.int64()
            b = T.int64()
            c = T.int64()
            # block 0
            gv = R.call_tir(Expected.log_softmax, (x,), R.Tensor((a, b, c), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def log_softmax(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.func_attr({"tir.noalias": True})
            a = T.int64()
            b = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [a, b, c], dtype="float32")
            compute = T.match_buffer(var_compute, [a, b, c], dtype="float32")
            T_softmax_maxelem = T.alloc_buffer([a, b], dtype="float32")
            compute_1 = T.alloc_buffer([a, b], dtype="float32")
            for i0, i1, k in T.grid(a, b, c):
                with T.block("T_softmax_maxelem"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(rxplaceholder[v_i0, v_i1, v_k])
                    T.writes(T_softmax_maxelem[v_i0, v_i1])
                    with T.init():
                        T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e38)
                    T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], rxplaceholder[v_i0, v_i1, v_k])
            for i0, i1, k in T.grid(a, b, c):
                with T.block("compute"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(rxplaceholder[v_i0, v_i1, v_k], T_softmax_maxelem[v_i0, v_i1])
                    T.writes(compute_1[v_i0, v_i1])
                    with T.init():
                        compute_1[v_i0, v_i1] = T.float32(0)
                    compute_1[v_i0, v_i1] = compute_1[v_i0, v_i1] + T.exp(rxplaceholder[v_i0, v_i1, v_k] - T_softmax_maxelem[v_i0, v_i1], dtype="float32")
            for i0, i1, i2 in T.grid(a, b, c):
                with T.block("compute_1"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1], compute_1[v_i0, v_i1],)
                    T.writes(compute[v_i0, v_i1, v_i2])
                    T.block_attr({"axis": 2})
                    compute[v_i0, v_i1, v_i2] = (rxplaceholder[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1] - T.log(compute_1[v_i0, v_i1], dtype="float32"))
    # fmt: on

    mod = LegalizeOps()(LogSoftmax)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy_with_logits():
    # fmt: off
    @tvm.script.ir_module
    class CrossEntropyWithLogits:
        @R.function
        def main(x: R.Tensor((3,), "float32"), y: R.Tensor((3,), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((), "float32") = R.nn.cross_entropy_with_logits(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((3,), dtype="float32"), y: R.Tensor((3,), dtype="float32")):
            gv = R.call_tir(Expected.cross_entropy_with_logits, (x, y), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def cross_entropy_with_logits(rxplaceholder: T.Buffer(T.int64(3), "float32"), rxplaceholder_1: T.Buffer(T.int64(3), "float32"), T_multiply: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            T_multiply_1 = T.alloc_buffer([T.int64(3)], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")
            for i0 in T.serial(T.int64(3)):
                with T.block("T_multiply"):
                    ax0 = T.axis.spatial(T.int64(3), i0)
                    T.reads(rxplaceholder[ax0], rxplaceholder_1[ax0])
                    T.writes(T_multiply_1[ax0])
                    T_multiply_1[ax0] = rxplaceholder[ax0] * rxplaceholder_1[ax0]
            for i0 in T.serial(T.int64(3)):
                with T.block("T_multiply_red"):
                    k0 = T.axis.reduce(T.int64(3), i0)
                    T.reads(T_multiply_1[k0])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply_1[k0]
            with T.block("T_multiply_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_multiply[()])
                T_multiply[()] = T_multiply_red[()] * T.float32(-1)
    # fmt: on

    mod = LegalizeOps()(CrossEntropyWithLogits)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy_with_logits_batch():
    # fmt: off
    @tvm.script.ir_module
    class CrossEntropyWithLogits:
        @R.function
        def main(x: R.Tensor((2, 3), "float32"), y: R.Tensor((2, 3), "float32")) -> R.Tensor(None, "float32", ndim=2):
            gv: R.Tensor((), "float32") = R.nn.cross_entropy_with_logits(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            gv = R.call_tir(Expected.cross_entropy_with_logits, (x, y), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def cross_entropy_with_logits(rxplaceholder: T.Buffer((T.int64(2), T.int64(3)), "float32"), rxplaceholder_1: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            T_multiply = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")
            T_multiply_1 = T.alloc_buffer([], dtype="float32")
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply"):
                    ax0, ax1 = T.axis.remap("SS", [i0, i1])
                    T.reads(rxplaceholder[ax0, ax1], rxplaceholder_1[ax0, ax1])
                    T.writes(T_multiply[ax0, ax1])
                    T_multiply[ax0, ax1] = rxplaceholder[ax0, ax1] * rxplaceholder_1[ax0, ax1]
            for i0, i1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_multiply_red"):
                    k0, k1 = T.axis.remap("RR", [i0, i1])
                    T.reads(T_multiply[k0, k1])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[k0, k1]
            with T.block("T_multiply_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_multiply_1[()])
                T_multiply_1[()] = T_multiply_red[()] * T.float32(-1)
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_1[()])
                T.writes(T_divide[()])
                T_divide[()] = T_multiply_1[()] * T.float32(0.5)
    # fmt: on

    mod = LegalizeOps()(CrossEntropyWithLogits)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_cross_entropy_with_logits_batch_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class CrossEntropyWithLogits:
        @R.function
        def main(x: R.Tensor(("n", "m"), "float32"), y: R.Tensor(("n", "m"), "float32")) -> R.Tensor(None, "float32", ndim=2):
            n = T.int64()
            m = T.int64()
            gv: R.Tensor((), "float32") = R.nn.cross_entropy_with_logits(x, y)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("n", "m"), dtype="float32"), y: R.Tensor(("n", "m"), dtype="float32")):
            gv = R.call_tir(Expected.cross_entropy_with_logits, (x, y), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def cross_entropy_with_logits(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, T_divide: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            m = T.int64()
            n = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [n, m], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [n, m], dtype="float32")
            T_multiply = T.alloc_buffer([n, m], dtype="float32")
            T_multiply_red = T.alloc_buffer([], dtype="float32")
            T_multiply_1 = T.alloc_buffer([], dtype="float32")
            for ax0, ax1 in T.grid(n, m):
                with T.block("T_multiply"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder[v_ax0, v_ax1], rxplaceholder_1[v_ax0, v_ax1])
                    T.writes(T_multiply[v_ax0, v_ax1])
                    T_multiply[v_ax0, v_ax1] = rxplaceholder[v_ax0, v_ax1] * rxplaceholder_1[v_ax0, v_ax1]
            for k0, k1 in T.grid(n, m):
                with T.block("T_multiply_red"):
                    v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                    T.reads(T_multiply[v_k0, v_k1])
                    T.writes(T_multiply_red[()])
                    with T.init():
                        T_multiply_red[()] = T.float32(0)
                    T_multiply_red[()] = T_multiply_red[()] + T_multiply[v_k0, v_k1]
            with T.block("T_multiply_1"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_red[()])
                T.writes(T_multiply_1[()])
                T_multiply_1[()] = T_multiply_red[()] * T.float32(-1)
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(T_multiply_1[()])
                T.writes(T_divide[()])
                T_divide[()] = T_multiply_1[()] / T.Cast("float32", n)
    # fmt: on

    mod = LegalizeOps()(CrossEntropyWithLogits)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_batch_norm():
    # fmt: off
    @tvm.script.ir_module
    class BatchNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), gamma: R.Tensor((3,), "float32"), beta: R.Tensor((3,), "float32"), moving_mean: R.Tensor((3,), "float32"), moving_var: R.Tensor((3,), "float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), "float32"), R.Tensor((3,), "float32"), R.Tensor((3,), "float32")):
            gv: R.Tuple(R.Tensor((2, 3, 28, 28), "float32"), R.Tensor((3,), "float32"), R.Tensor((3,), "float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def batch_norm(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), rxplaceholder_1: T.Buffer((T.int64(3),), "float32"), rxplaceholder_2: T.Buffer((T.int64(3),), "float32"), rxplaceholder_3: T.Buffer((T.int64(3),), "float32"), rxplaceholder_4: T.Buffer((T.int64(3),), "float32"), T_add: T.Buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)), "float32"), T_add_1: T.Buffer((T.int64(3),), "float32"), T_add_2: T.Buffer((T.int64(3),), "float32")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            rxplaceholder_red = T.alloc_buffer((T.int64(3),))
            T_divide = T.alloc_buffer((T.int64(3),))
            T_reshape = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
            T_subtract = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_subtract_1 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_subtract_2 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_multiply = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_multiply_red = T.alloc_buffer((T.int64(3),))
            T_divide_1 = T.alloc_buffer((T.int64(3),))
            T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
            T_add_3 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
            compute = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
            T_divide_2 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_reshape_2 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
            T_multiply_1 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_reshape_3 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)))
            T_multiply_2 = T.alloc_buffer((T.int64(3),))
            T_multiply_3 = T.alloc_buffer((T.int64(3),))
            T_multiply_4 = T.alloc_buffer((T.int64(3),))
            T_subtract_3 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_subtract_4 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_multiply_5 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(28), T.int64(28)))
            T_multiply_red_1 = T.alloc_buffer((T.int64(3),))
            T_divide_3 = T.alloc_buffer((T.int64(3),))
            T_multiply_6 = T.alloc_buffer((T.int64(3),))
            for ax0, k0, k2, k3 in T.grid(T.int64(3), T.int64(2), T.int64(28), T.int64(28)):
                with T.block("rxplaceholder_red"):
                    v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                    T.reads(rxplaceholder[v_k0, v_ax0, v_k2, v_k3])
                    T.writes(rxplaceholder_red[v_ax0])
                    with T.init():
                        rxplaceholder_red[v_ax0] = T.float32(0)
                    rxplaceholder_red[v_ax0] = rxplaceholder_red[v_ax0] + rxplaceholder[v_k0, v_ax0, v_k2, v_k3]
            for ax0 in range(T.int64(3)):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(rxplaceholder_red[v_ax0])
                    T.writes(T_divide[v_ax0])
                    T_divide[v_ax0] = rxplaceholder_red[v_ax0] * T.float32(0.00063775510204081628)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_divide[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_subtract"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_subtract_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_subtract_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, k0, k2, k3 in T.grid(T.int64(3), T.int64(2), T.int64(28), T.int64(28)):
                with T.block("T_multiply_red"):
                    v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                    T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                    T.writes(T_multiply_red[v_ax0])
                    with T.init():
                        T_multiply_red[v_ax0] = T.float32(0)
                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
            for ax0 in range(T.int64(3)):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = T_multiply_red[v_ax0] * T.float32(0.00063775510204081628)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
            for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_divide_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_1[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_multiply_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1), T.int64(1)):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder_2[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_2[(v_ax1 + v_ax2 + v_ax3) % T.int64(3)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_add_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0 in range(T.int64(3)):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(rxplaceholder_3[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.90000000000000002) * rxplaceholder_3[v_ax0]
            for ax0 in range(T.int64(3)):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(T_divide[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.10000000000000001) * T_divide[v_ax0]
            for ax0 in range(T.int64(3)):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(T_multiply_2[v_ax0], T_multiply_3[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_2[v_ax0] + T_multiply_3[v_ax0]
            for ax0 in range(T.int64(3)):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(rxplaceholder_4[v_ax0])
                    T.writes(T_multiply_4[v_ax0])
                    T_multiply_4[v_ax0] = T.float32(0.90000000000000002) * rxplaceholder_4[v_ax0]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_subtract_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_subtract_4"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(28), T.int64(28)):
                with T.block("T_multiply_5"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_5[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_5[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, k0, k2, k3 in T.grid(T.int64(3), T.int64(2), T.int64(28), T.int64(28)):
                with T.block("T_multiply_red_1"):
                    v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                    T.reads(T_multiply_5[v_k0, v_ax0, v_k2, v_k3])
                    T.writes(T_multiply_red_1[v_ax0])
                    with T.init():
                        T_multiply_red_1[v_ax0] = T.float32(0)
                    T_multiply_red_1[v_ax0] = T_multiply_red_1[v_ax0] + T_multiply_5[v_k0, v_ax0, v_k2, v_k3]
            for ax0 in range(T.int64(3)):
                with T.block("T_divide_3"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(T_multiply_red_1[v_ax0])
                    T.writes(T_divide_3[v_ax0])
                    T_divide_3[v_ax0] = T_multiply_red_1[v_ax0] * T.float32(0.00063775510204081628)
            for ax0 in range(T.int64(3)):
                with T.block("T_multiply_6"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(T_divide_3[v_ax0])
                    T.writes(T_multiply_6[v_ax0])
                    T_multiply_6[v_ax0] = T.float32(0.10000000000000001) * T_divide_3[v_ax0]
            for ax0 in range(T.int64(3)):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads(T_multiply_4[v_ax0], T_multiply_6[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_4[v_ax0] + T_multiply_6[v_ax0]

        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), dtype="float32"), gamma: R.Tensor((3,), dtype="float32"), beta: R.Tensor((3,), dtype="float32"), moving_mean: R.Tensor((3,), dtype="float32"), moving_var: R.Tensor((3,), dtype="float32")) -> R.Tuple(R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")):
            gv = R.call_tir(Expected.batch_norm, (x, gamma, beta, moving_mean, moving_var), out_sinfo=[R.Tensor((2, 3, 28, 28), dtype="float32"), R.Tensor((3,), dtype="float32"), R.Tensor((3,), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(BatchNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_batch_norm_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class BatchNorm:
        @R.function
        def main(x: R.Tensor(("n", "h", "w", "c"), "float32"), gamma: R.Tensor(("c",), "float32"), beta: R.Tensor(("c",), "float32"), moving_mean: R.Tensor(("c",), "float32"), moving_var: R.Tensor(("c",), "float32")) -> R.Tuple(R.Tensor(("n", "h", "w", "c"), "float32"), R.Tensor(("c",), "float32"), R.Tensor(("c",), "float32")):
            n = T.int64()
            h = T.int64()
            w = T.int64()
            c = T.int64()
            gv: R.Tuple(R.Tensor((n, h, w, c), "float32"), R.Tensor((c,), "float32"), R.Tensor((c,), "float32")) = R.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def batch_norm(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_rxplaceholder_3: T.handle, var_rxplaceholder_4: T.handle, var_T_add: T.handle, var_T_add_1: T.handle, var_T_add_2: T.handle):
            T.func_attr({"tir.noalias": True})
            n = T.int64()
            h = T.int64()
            w = T.int64()
            c = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (n, h, w, c))
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (c,))
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (c,))
            rxplaceholder_3 = T.match_buffer(var_rxplaceholder_3, (c,))
            rxplaceholder_4 = T.match_buffer(var_rxplaceholder_4, (c,))
            T_add = T.match_buffer(var_T_add, (n, h, w, c))
            T_add_1 = T.match_buffer(var_T_add_1, (T.max(c, h),))
            T_add_2 = T.match_buffer(var_T_add_2, (T.max(c, h),))
            # with T.block("root"):
            rxplaceholder_red = T.alloc_buffer((h,))
            T_divide = T.alloc_buffer((h,))
            T_reshape = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
            T_subtract = T.alloc_buffer((n, h, w, c))
            T_subtract_1 = T.alloc_buffer((n, h, w, c))
            T_subtract_2 = T.alloc_buffer((n, h, w, c))
            T_multiply = T.alloc_buffer((n, h, w, c))
            T_multiply_red = T.alloc_buffer((h,))
            T_divide_1 = T.alloc_buffer((h,))
            T_reshape_1 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
            T_add_3 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
            compute = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
            T_divide_2 = T.alloc_buffer((n, h, w, c))
            T_reshape_2 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
            T_multiply_1 = T.alloc_buffer((n, h, w, c))
            T_reshape_3 = T.alloc_buffer((T.int64(1), h, T.int64(1), T.int64(1)))
            T_multiply_2 = T.alloc_buffer((c,))
            T_multiply_3 = T.alloc_buffer((h,))
            T_multiply_4 = T.alloc_buffer((c,))
            T_subtract_3 = T.alloc_buffer((n, h, w, c))
            T_subtract_4 = T.alloc_buffer((n, h, w, c))
            T_multiply_5 = T.alloc_buffer((n, h, w, c))
            T_multiply_red_1 = T.alloc_buffer((h,))
            T_divide_3 = T.alloc_buffer((h,))
            T_multiply_6 = T.alloc_buffer((h,))
            for ax0, k0, k2, k3 in T.grid(h, n, w, c):
                with T.block("rxplaceholder_red"):
                    v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                    T.reads(rxplaceholder[v_k0, v_ax0, v_k2, v_k3])
                    T.writes(rxplaceholder_red[v_ax0])
                    with T.init():
                        rxplaceholder_red[v_ax0] = T.float32(0)
                    rxplaceholder_red[v_ax0] = rxplaceholder_red[v_ax0] + rxplaceholder[v_k0, v_ax0, v_k2, v_k3]
            for ax0 in range(h):
                with T.block("T_divide"):
                    v_ax0 = T.axis.spatial(h, ax0)
                    T.reads(rxplaceholder_red[v_ax0])
                    T.writes(T_divide[v_ax0])
                    T_divide[v_ax0] = rxplaceholder_red[v_ax0] / T.Cast("float32", n * w * c)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_divide[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_subtract"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_subtract_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_subtract_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_2[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, k0, k2, k3 in T.grid(h, n, w, c):
                with T.block("T_multiply_red"):
                    v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                    T.reads(T_multiply[v_k0, v_ax0, v_k2, v_k3])
                    T.writes(T_multiply_red[v_ax0])
                    with T.init():
                        T_multiply_red[v_ax0] = T.float32(0)
                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_k0, v_ax0, v_k2, v_k3]
            for ax0 in range(h):
                with T.block("T_divide_1"):
                    v_ax0 = T.axis.spatial(h, ax0)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(T_divide_1[v_ax0])
                    T_divide_1[v_ax0] = T_multiply_red[v_ax0] / T.Cast("float32", n * w * c)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_divide_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % h]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add_3[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(1.0000000000000001e-05)
            for i0, i1, i2, i3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(T_add_3[v_i0, v_i1, v_i2, v_i3])
                    T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                    compute[v_i0, v_i1, v_i2, v_i3] = T.sqrt(T_add_3[v_i0, v_i1, v_i2, v_i3])
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_divide_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_subtract[v_ax0, v_ax1, v_ax2, v_ax3], compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract[v_ax0, v_ax1, v_ax2, v_ax3] / compute[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_1[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_multiply_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] = T_divide_2[v_ax0, v_ax1, v_ax2, v_ax3] * T_reshape_2[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), h, T.int64(1), T.int64(1)):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder_2[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_3[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder_2[(v_ax0 * h + v_ax1 + v_ax2 + v_ax3) % c]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_add_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply_1[v_ax0, v_ax1, v_ax2, v_ax3] + T_reshape_3[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0 in range(c):
                with T.block("T_multiply_2"):
                    v_ax0 = T.axis.spatial(c, ax0)
                    T.reads(rxplaceholder_3[v_ax0])
                    T.writes(T_multiply_2[v_ax0])
                    T_multiply_2[v_ax0] = T.float32(0.90000000000000002) * rxplaceholder_3[v_ax0]
            for ax0 in range(h):
                with T.block("T_multiply_3"):
                    v_ax0 = T.axis.spatial(h, ax0)
                    T.reads(T_divide[v_ax0])
                    T.writes(T_multiply_3[v_ax0])
                    T_multiply_3[v_ax0] = T.float32(0.10000000000000001) * T_divide[v_ax0]
            for ax0 in range(T.max(c, h)):
                with T.block("T_add_2"):
                    v_ax0 = T.axis.spatial(T.max(c, h), ax0)
                    T.reads(T_multiply_2[v_ax0], T_multiply_3[v_ax0])
                    T.writes(T_add_1[v_ax0])
                    T_add_1[v_ax0] = T_multiply_2[v_ax0] + T_multiply_3[v_ax0]
            for ax0 in range(c):
                with T.block("T_multiply_4"):
                    v_ax0 = T.axis.spatial(c, ax0)
                    T.reads(rxplaceholder_4[v_ax0])
                    T.writes(T_multiply_4[v_ax0])
                    T_multiply_4[v_ax0] = T.float32(0.90000000000000002) * rxplaceholder_4[v_ax0]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_subtract_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_subtract_4"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
                    T.writes(T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3] = rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3] - T_reshape[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(n, h, w, c):
                with T.block("T_multiply_5"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3], T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply_5[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply_5[v_ax0, v_ax1, v_ax2, v_ax3] = T_subtract_3[v_ax0, v_ax1, v_ax2, v_ax3] * T_subtract_4[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, k0, k2, k3 in T.grid(h, n, w, c):
                with T.block("T_multiply_red_1"):
                    v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                    T.reads(T_multiply_5[v_k0, v_ax0, v_k2, v_k3])
                    T.writes(T_multiply_red_1[v_ax0])
                    with T.init():
                        T_multiply_red_1[v_ax0] = T.float32(0)
                    T_multiply_red_1[v_ax0] = T_multiply_red_1[v_ax0] + T_multiply_5[v_k0, v_ax0, v_k2, v_k3]
            for ax0 in range(h):
                with T.block("T_divide_3"):
                    v_ax0 = T.axis.spatial(h, ax0)
                    T.reads(T_multiply_red_1[v_ax0])
                    T.writes(T_divide_3[v_ax0])
                    T_divide_3[v_ax0] = T_multiply_red_1[v_ax0] / T.Cast("float32", n * w * c)
            for ax0 in range(h):
                with T.block("T_multiply_6"):
                    v_ax0 = T.axis.spatial(h, ax0)
                    T.reads(T_divide_3[v_ax0])
                    T.writes(T_multiply_6[v_ax0])
                    T_multiply_6[v_ax0] = T.float32(0.10000000000000001) * T_divide_3[v_ax0]
            for ax0 in range(T.max(c, h)):
                with T.block("T_add_3"):
                    v_ax0 = T.axis.spatial(T.max(c, h), ax0)
                    T.reads(T_multiply_4[v_ax0], T_multiply_6[v_ax0])
                    T.writes(T_add_2[v_ax0])
                    T_add_2[v_ax0] = T_multiply_4[v_ax0] + T_multiply_6[v_ax0]

        @R.function
        def main(x: R.Tensor(("n", "h", "w", "c"), dtype="float32"), gamma: R.Tensor(("c",), dtype="float32"), beta: R.Tensor(("c",), dtype="float32"), moving_mean: R.Tensor(("c",), dtype="float32"), moving_var: R.Tensor(("c",), dtype="float32")) -> R.Tuple(R.Tensor(("n", "h", "w", "c"), dtype="float32"), R.Tensor(("T.max(c,h)",), dtype="float32"), R.Tensor(("T.max(c,h)",), dtype="float32")):
            n = T.int64()
            h = T.int64()
            w = T.int64()
            c = T.int64()
            gv = R.call_tir(Expected.batch_norm, (x, gamma, beta, moving_mean, moving_var), out_sinfo=[R.Tensor((n, h, w, c), dtype="float32"), R.Tensor((T.max(c, h),), dtype="float32"), R.Tensor((T.max(c, h),), dtype="float32")])
            return gv
    # fmt: on

    mod = LegalizeOps()(BatchNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layer_norm():
    # fmt: off
    @tvm.script.ir_module
    class LayerNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), gamma: R.Tensor((4, 5), "float32"), beta: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.layer_norm(x, gamma, beta, axes=[-2, -1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), gamma: R.Tensor((4, 5), "float32"), beta: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv = R.call_tir(Expected.layer_norm, (x, gamma, beta), R.Tensor((2, 3, 4, 5), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def layer_norm(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4), T.int64(5)), "float32"), rxplaceholder_2: T.Buffer((T.int64(4), T.int64(5)), "float32"), T_layer_norm: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": True})
            rxplaceholder_red_temp_v0 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            rxplaceholder_red_temp_v1 = T.alloc_buffer([T.int64(2), T.int64(3)], dtype="float32")
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("rxplaceholder_red_temp"):
                    ax0, ax1, k2, k3 = T.axis.remap("SSRR", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, k2, k3])
                    T.writes(rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1])
                    with T.init():
                        rxplaceholder_red_temp_v0[ax0, ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[ax0, ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[ax0, ax1] + rxplaceholder[ax0, ax1, k2, k3]
                    v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[ax0, ax1] + rxplaceholder[ax0, ax1, k2, k3] * rxplaceholder[ax0, ax1, k2, k3]
                    rxplaceholder_red_temp_v0[ax0, ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[ax0, ax1] = v_rxplaceholder_red_temp_v1
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_layer_norm"):
                    ax0, ax1, ax2, ax3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[ax0, ax1, ax2, ax3], rxplaceholder_red_temp_v0[ax0, ax1], rxplaceholder_red_temp_v1[ax0, ax1], rxplaceholder_1[ax2, ax3], rxplaceholder_2[ax2, ax3])
                    T.writes(T_layer_norm[ax0, ax1, ax2, ax3])
                    T_layer_norm[ax0, ax1, ax2, ax3] = (rxplaceholder[ax0, ax1, ax2, ax3] - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.05)) * T.rsqrt(rxplaceholder_red_temp_v1[ax0, ax1] * T.float32(0.05) - rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.05) * (rxplaceholder_red_temp_v0[ax0, ax1] * T.float32(0.05)) + T.float32(1e-05), dtype="float32") * rxplaceholder_1[ax2, ax3] + rxplaceholder_2[ax2, ax3]
    # fmt: on
    mod = LegalizeOps()(LayerNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layer_norm_fp16():
    # fmt: off
    @tvm.script.ir_module
    class LayerNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), gamma: R.Tensor((4, 5), "float16"), beta: R.Tensor((4, 5), "float16")) -> R.Tensor((2, 3, 4, 5), "float16"):
            gv: R.Tensor((2, 3, 4, 5), "float16") = R.nn.layer_norm(x, gamma, beta, axes=[-2, -1])
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def layer_norm(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_T_layer_norm: T.handle):
            T.func_attr({"tir.noalias": True})
            rxplaceholder = T.match_buffer(var_rxplaceholder, (T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(4), T.int64(5)), "float16")
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (T.int64(4), T.int64(5)), "float16")
            T_layer_norm = T.match_buffer(var_T_layer_norm, (T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16")
            with T.block("root"):
                T.reads()
                T.writes()
                rxplaceholder_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(3)))
                rxplaceholder_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(3)))
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for k2 in range(T.int64(4)):
                            for k3 in range(T.int64(5)):
                                with T.block("rxplaceholder_red_temp"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_k2 = T.axis.reduce(T.int64(4), k2)
                                    v_k3 = T.axis.reduce(T.int64(5), k3)
                                    T.reads(rxplaceholder[v_ax0, v_ax1, v_k2, v_k3])
                                    T.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                                    with T.init():
                                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                                    v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T.Cast("float32", rxplaceholder[v_ax0, v_ax1, v_k2, v_k3])
                                    v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T.Cast("float32", rxplaceholder[v_ax0, v_ax1, v_k2, v_k3]) * T.Cast("float32", rxplaceholder[v_ax0, v_ax1, v_k2, v_k3])
                                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
                for ax0 in range(T.int64(2)):
                    for ax1 in range(T.int64(3)):
                        for ax2 in range(T.int64(4)):
                            for ax3 in range(T.int64(5)):
                                with T.block("T_layer_norm"):
                                    v_ax0 = T.axis.spatial(T.int64(2), ax0)
                                    v_ax1 = T.axis.spatial(T.int64(3), ax1)
                                    v_ax2 = T.axis.spatial(T.int64(4), ax2)
                                    v_ax3 = T.axis.spatial(T.int64(5), ax3)
                                    T.reads(rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], rxplaceholder_1[v_ax2, v_ax3], rxplaceholder_2[v_ax2, v_ax3])
                                    T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                                    T_layer_norm[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float16", (T.Cast("float32", rxplaceholder[v_ax0, v_ax1, v_ax2, v_ax3]) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.Cast("float32", T.float16(4) * T.float16(5))) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] / T.Cast("float32", T.float16(4) * T.float16(5)) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.Cast("float32", T.float16(4) * T.float16(5)) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] / T.Cast("float32", T.float16(4) * T.float16(5))) + T.float32(1.0000000000000001e-05))) * rxplaceholder_1[v_ax2, v_ax3] + rxplaceholder_2[v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float16"), gamma: R.Tensor((4, 5), dtype="float16"), beta: R.Tensor((4, 5), dtype="float16")) -> R.Tensor((2, 3, 4, 5), dtype="float16"):
            gv = R.call_tir(Expected.layer_norm, (x, gamma, beta), out_sinfo=R.Tensor((2, 3, 4, 5), dtype="float16"))
            return gv
    # fmt: on
    mod = LegalizeOps()(LayerNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_layer_norm_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class LayerNorm:
        @R.function
        def main(x: R.Tensor(("n", "s", "f"), "float32"), gamma: R.Tensor(("s", "f"), "float32"), beta: R.Tensor(("s", "f"), "float32")) -> R.Tensor(("n", "s", "f"), "float32"):
            n = T.int64()
            s = T.int64()
            f = T.int64()
            gv: R.Tensor((n, s, f), "float32") = R.nn.layer_norm(x, gamma, beta, axes=[1, 2])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("n", "s", "f"), "float32"), gamma: R.Tensor(("s", "f"), "float32"), beta: R.Tensor(("s", "f"), "float32")) -> R.Tensor(("n", "s", "f"), "float32"):
            n = T.int64()
            s = T.int64()
            f = T.int64()
            gv = R.call_tir(Expected.layer_norm, (x, gamma, beta), R.Tensor((n, s, f), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def layer_norm(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_T_layer_norm: T.handle):
            T.func_attr({"tir.noalias": True})
            f = T.int64()
            n = T.int64()
            s = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [n, s, f], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [s, f], dtype="float32")
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, [s, f], dtype="float32")
            T_layer_norm = T.match_buffer(var_T_layer_norm, [n, s, f], dtype="float32")
            rxplaceholder_red_temp_v0 = T.alloc_buffer([n], dtype="float32")
            rxplaceholder_red_temp_v1 = T.alloc_buffer([n], dtype="float32")
            for i0, i1, i2 in T.grid(n, s, f):
                with T.block("rxplaceholder_red_temp"):
                    ax0, k1, k2 = T.axis.remap("SRR", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, k1, k2])
                    T.writes(rxplaceholder_red_temp_v0[ax0], rxplaceholder_red_temp_v1[ax0])
                    with T.init():
                        rxplaceholder_red_temp_v0[ax0] = T.float32(0)
                        rxplaceholder_red_temp_v1[ax0] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[ax0] + rxplaceholder[ax0, k1, k2]
                    v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[ax0] + rxplaceholder[ax0, k1, k2] * rxplaceholder[ax0, k1, k2]
                    rxplaceholder_red_temp_v0[ax0] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[ax0] = v_rxplaceholder_red_temp_v1
            for i0, i1, i2 in T.grid(n, s, f):
                with T.block("T_layer_norm"):
                    ax0, ax1, ax2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(rxplaceholder[ax0, ax1, ax2], rxplaceholder_red_temp_v0[ax0], rxplaceholder_red_temp_v1[ax0], rxplaceholder_1[ax1, ax2], rxplaceholder_2[ax1, ax2])
                    T.writes(T_layer_norm[ax0, ax1, ax2])
                    T_layer_norm[ax0, ax1, ax2] = (rxplaceholder[ax0, ax1, ax2] - rxplaceholder_red_temp_v0[ax0] / (T.Cast("float32", s) * T.Cast("float32", f))) * T.rsqrt(rxplaceholder_red_temp_v1[ax0] / (T.Cast("float32", s) * T.Cast("float32", f)) - rxplaceholder_red_temp_v0[ax0] / (T.Cast("float32", s) * T.Cast("float32", f)) * (rxplaceholder_red_temp_v0[ax0] / (T.Cast("float32", s) * T.Cast("float32", f))) + T.float32(1e-05), dtype="float32") * rxplaceholder_1[ax1, ax2] + rxplaceholder_2[ax1, ax2]
    # fmt: on
    mod = LegalizeOps()(LayerNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_group_norm():
    # fmt: off
    @tvm.script.ir_module
    class GroupNorm:
        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), "float32"), gamma: R.Tensor((4,), "float32"), beta: R.Tensor((4,), "float32")) -> R.Tensor((2, 4, 4, 5), "float32"):
            gv: R.Tensor((2, 4, 4, 5), "float32") = R.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=1, axes=[2, 3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def group_norm(rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Buffer((T.int64(4),), "float32"), rxplaceholder_2: T.Buffer((T.int64(4),), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": True})
            T_reshape_1 = T.alloc_buffer((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)))
            rxplaceholder_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(2)))
            rxplaceholder_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(2)))
            T_reshape_2 = T.alloc_buffer((T.int64(2), T.int64(2)))
            T_reshape_3 = T.alloc_buffer((T.int64(2), T.int64(2)))
            T_group_norm = T.alloc_buffer((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)]
            for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4])
                    T.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                    with T.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    T.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    T.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_group_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - rxplaceholder_red_temp_v0[v_ax0, v_ax1] * T.float32(0.025000000000000001)) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] * T.float32(0.025000000000000001) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] * T.float32(0.025000000000000001) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] * T.float32(0.025000000000000001)) + T.float32(1.0000000000000001e-05)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(4), T.int64(5)):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)]

        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), dtype="float32"), gamma: R.Tensor((4,), dtype="float32"), beta: R.Tensor((4,), dtype="float32")) -> R.Tensor((2, 4, 4, 5), dtype="float32"):
            gv = R.call_tir(Expected.group_norm, (x, gamma, beta), out_sinfo=R.Tensor((2, 4, 4, 5), dtype="float32"))
            return gv
    # fmt: on
    mod = LegalizeOps()(GroupNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_group_norm_fp16():
    # fmt: off
    @tvm.script.ir_module
    class GroupNorm:
        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), "float16"), gamma: R.Tensor((4,), "float16"), beta: R.Tensor((4,), "float16")) -> R.Tensor((2, 4, 4, 5), "float16"):
            gv: R.Tensor((2, 4, 4, 5), "float16") = R.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=1, axes=[2, 3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 4, 4, 5), dtype="float16"), gamma: R.Tensor((4,), dtype="float16"), beta: R.Tensor((4,), dtype="float16")) -> R.Tensor((2, 4, 4, 5), dtype="float16"):
            gv = R.call_tir(Expected.group_norm, (x, gamma, beta), out_sinfo=R.Tensor((2, 4, 4, 5), dtype="float16"))
            return gv

        @T.prim_func(private=True)
        def group_norm(rxplaceholder: T.Buffer((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float16"), rxplaceholder_1: T.Buffer((T.int64(4),), "float16"), rxplaceholder_2: T.Buffer((T.int64(4),), "float16"), T_reshape: T.Buffer((T.int64(2), T.int64(4), T.int64(4), T.int64(5)), "float16")):
            T.func_attr({"tir.noalias": True})
            # with T.block("root"):
            T_reshape_1 = T.alloc_buffer((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)), "float16")
            T_cast = T.alloc_buffer((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)))
            rxplaceholder_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(2)))
            rxplaceholder_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(2)))
            T_reshape_2 = T.alloc_buffer((T.int64(2), T.int64(2)), "float16")
            T_reshape_3 = T.alloc_buffer((T.int64(2), T.int64(2)), "float16")
            T_group_norm = T.alloc_buffer((T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)), "float16")
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = rxplaceholder[((v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) // T.int64(4) + v_ax0) % T.int64(2), (v_ax1 * T.int64(2) + (v_ax4 // T.int64(5) + v_ax3) // T.int64(4) + v_ax2) % T.int64(4), (v_ax4 // T.int64(5) + v_ax3) % T.int64(4), v_ax4 % T.int64(5)]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.Cast("float32", T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
                    T.reads(T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4])
                    T.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                    with T.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_cast[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    T.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = rxplaceholder_1[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(2)):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)])
                    T.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = rxplaceholder_2[(v_ax0 * T.int64(2) + v_ax1) % T.int64(4)]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(2), T.int64(2), T.int64(4), T.int64(5)):
                with T.block("T_group_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.Cast("float16", (T_cast[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - rxplaceholder_red_temp_v0[v_ax0, v_ax1] * T.float32(0.025000000000000001)) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] * T.float32(0.025000000000000001) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] * T.float32(0.025000000000000001) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] * T.float32(0.025000000000000001)) + T.float32(1.0000000000000001e-05))) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(4), T.int64(5)):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) // T.int64(4) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(4) // T.int64(2), ((v_ax3 // T.int64(5) + v_ax2) // T.int64(4) + v_ax1) % T.int64(2), (v_ax3 // T.int64(5) + v_ax2) % T.int64(4), v_ax3 % T.int64(5)]
    # fmt: on

    mod = LegalizeOps()(GroupNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_group_norm_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class GroupNorm:
        @R.function
        def main(s: R.Shape(["c"]), x: R.Tensor(("n", "4 * c", "h", "w"), "float32"), gamma: R.Tensor(("4 * c",), "float32"), beta: R.Tensor(("4 * c",), "float32")) -> R.Tensor(("n", "4 * c", "h", "w"), "float32"):
            n = T.int64()
            c = T.int64()
            h = T.int64()
            w = T.int64()
            gv: R.Tensor((n, 4 * c, h, w), "float32") = R.nn.group_norm(x, gamma, beta, num_groups=4, channel_axis=1, axes=[2, 3])
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def group_norm(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, var_T_reshape: T.handle, c: T.int64):
            T.func_attr({"tir.noalias": True})
            n = T.int64()
            h = T.int64()
            w = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, (n, T.int64(4) * c, h, w))
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, (T.int64(4) * c,))
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, (T.int64(4) * c,))
            T_reshape = T.match_buffer(var_T_reshape, (n, T.int64(4) * c, h, w))
            # with T.block("root"):
            T_reshape_1 = T.alloc_buffer((n, T.int64(4), T.int64(4) * c // T.int64(4), h, w))
            rxplaceholder_red_temp_v0 = T.alloc_buffer((n, T.int64(4)))
            rxplaceholder_red_temp_v1 = T.alloc_buffer((n, T.int64(4)))
            T_reshape_2 = T.alloc_buffer((T.int64(4), T.int64(4) * c // T.int64(4)))
            T_reshape_3 = T.alloc_buffer((T.int64(4), T.int64(4) * c // T.int64(4)))
            T_group_norm = T.alloc_buffer((n, T.int64(4), T.int64(4) * c // T.int64(4), h, w))
            for ax0, ax1, ax2, ax3, ax4 in T.grid(n, T.int64(4), c, h, w):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(rxplaceholder[((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) // w // h // (c * T.int64(4)) % n, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) // w // h % (c * T.int64(4)), ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) // w % h, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) % w])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = rxplaceholder[((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) // w // h // (c * T.int64(4)) % n, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) // w // h % (c * T.int64(4)), ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) // w % h, ((((v_ax0 * T.int64(4) + v_ax1) * c + v_ax2) * h + v_ax3) * w + v_ax4) % w]
            for ax0, ax1, k2, k3, k4 in T.grid(n, T.int64(4), c, h, w):
                with T.block("rxplaceholder_red_temp"):
                    v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4])
                    T.writes(rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1])
                    with T.init():
                        rxplaceholder_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                        rxplaceholder_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
                    v_rxplaceholder_red_temp_v0: T.float32 = rxplaceholder_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    v_rxplaceholder_red_temp_v1: T.float32 = rxplaceholder_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
                    rxplaceholder_red_temp_v0[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v0
                    rxplaceholder_red_temp_v1[v_ax0, v_ax1] = v_rxplaceholder_red_temp_v1
            for ax0, ax1 in T.grid(T.int64(4), c):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder_1[(v_ax0 * c + v_ax1) % (c * T.int64(4))])
                    T.writes(T_reshape_2[v_ax0, v_ax1])
                    T_reshape_2[v_ax0, v_ax1] = rxplaceholder_1[(v_ax0 * c + v_ax1) % (c * T.int64(4))]
            for ax0, ax1 in T.grid(T.int64(4), c):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(rxplaceholder_2[(v_ax0 * c + v_ax1) % (c * T.int64(4))])
                    T.writes(T_reshape_3[v_ax0, v_ax1])
                    T_reshape_3[v_ax0, v_ax1] = rxplaceholder_2[(v_ax0 * c + v_ax1) % (c * T.int64(4))]
            for ax0, ax1, ax2, ax3, ax4 in T.grid(n, T.int64(4), c, h, w):
                with T.block("T_group_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                    T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], rxplaceholder_red_temp_v0[v_ax0, v_ax1], rxplaceholder_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
                    T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h) * T.Cast("float32", w))) * T.rsqrt(rxplaceholder_red_temp_v1[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h) * T.Cast("float32", w)) - rxplaceholder_red_temp_v0[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h) * T.Cast("float32", w)) * (rxplaceholder_red_temp_v0[v_ax0, v_ax1] / (T.Cast("float32", c) * T.Cast("float32", h) * T.Cast("float32", w))) + T.float32(1.0000000000000001e-05)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
            for ax0, ax1, ax2, ax3 in T.grid(n, c * T.int64(4), h, w):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_group_norm[(((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w // h // c // T.int64(4) % n, (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w // h // c % T.int64(4), (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w // h % c, (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w % h, (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) % w])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w // h // c // T.int64(4) % n, (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w // h // c % T.int64(4), (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w // h % c, (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) // w % h, (((v_ax0 * c * T.int64(4) + v_ax1) * h + v_ax2) * w + v_ax3) % w]

        @R.function
        def main(s: R.Shape(["c"]), x: R.Tensor(("n", "4 * c", "h", "w"), dtype="float32"), gamma: R.Tensor(("4 * c",), dtype="float32"), beta: R.Tensor(("4 * c",), dtype="float32")) -> R.Tensor(("n", "4 * c", "h", "w"), dtype="float32"):
            n = T.int64()
            c = T.int64()
            h = T.int64()
            w = T.int64()
            gv = R.call_tir(Expected.group_norm, (x, gamma, beta), out_sinfo=R.Tensor((n, 4 * c, h, w), dtype="float32"), tir_vars=R.shape([c]))
            return gv
    # fmt: on
    mod = LegalizeOps()(GroupNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_rms_norm():
    # fmt: off
    @tvm.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), weight: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.rms_norm(x, weight, axes=[-2, -1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def rms_norm(A: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), B: T.Buffer((T.int64(4), T.int64(5)), "float32"), T_cast: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            T_cast_1 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = T.alloc_buffer((T.int64(2), T.int64(3)))
            rsqrt = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_cast_2 = T.alloc_buffer((T.int64(4), T.int64(5)))
            T_rms_norm = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply_red"):
                    v_ax0, v_ax1, v_k2, v_k3 = T.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    T.reads(T_multiply[v_ax0, v_ax1, v_k2, v_k3])
                    T.writes(T_multiply_red[v_ax0, v_ax1])
                    with T.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("rsqrt"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_red[v_ax0, v_ax1])
                    T.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.050000000000000003) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(5)):
                with T.block("T_cast_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(B[v_ax0, v_ax1])
                    T.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = B[v_ax0, v_ax1]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3], T_cast_2[v_ax2, v_ax3])
                    T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_2[v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_cast_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3] = T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32"), weight: R.Tensor((4, 5), dtype="float32")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), out_sinfo=R.Tensor((2, 3, 4, 5), dtype="float32"))
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_rms_norm_fp16():
    # fmt: off
    @tvm.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float16"), weight: R.Tensor((4, 5), "float16")) -> R.Tensor((2, 3, 4, 5), "float16"):
            gv: R.Tensor((2, 3, 4, 5), "float16") = R.nn.rms_norm(x, weight, axes=[-2, -1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def rms_norm(A: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16"), B: T.Buffer((T.int64(4), T.int64(5)), "float16"), T_cast: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float16")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            T_cast_1 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = T.alloc_buffer((T.int64(2), T.int64(3)))
            rsqrt = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_cast_2 = T.alloc_buffer((T.int64(4), T.int64(5)))
            T_rms_norm = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float32", A[v_ax0, v_ax1, v_ax2, v_ax3])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply_red"):
                    v_ax0, v_ax1, v_k2, v_k3 = T.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    T.reads(T_multiply[v_ax0, v_ax1, v_k2, v_k3])
                    T.writes(T_multiply_red[v_ax0, v_ax1])
                    with T.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("rsqrt"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_red[v_ax0, v_ax1])
                    T.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.050000000000000003) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(5)):
                with T.block("T_cast_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(B[v_ax0, v_ax1])
                    T.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = T.Cast("float32", B[v_ax0, v_ax1])
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3], T_cast_2[v_ax2, v_ax3])
                    T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_2[v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_cast_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3] = T.Cast("float16", T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float16"), weight: R.Tensor((4, 5), dtype="float16")) -> R.Tensor((2, 3, 4, 5), dtype="float16"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), out_sinfo=R.Tensor((2, 3, 4, 5), dtype="float16"))
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_rms_norm_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor(("n", "s", "f"), "float32"), weight: R.Tensor(("s", "f"), "float32")) -> R.Tensor(("n", "s", "f"), "float32"):
            n = T.int64()
            s = T.int64()
            f = T.int64()
            gv: R.Tensor((n, s, f), "float32") = R.nn.rms_norm(x, weight, axes=[1, 2])
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def rms_norm(var_A: T.handle, var_B: T.handle, var_T_cast: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n, s, f = T.int64(), T.int64(), T.int64()
            A = T.match_buffer(var_A, (n, s, f))
            B = T.match_buffer(var_B, (s, f))
            T_cast = T.match_buffer(var_T_cast, (n, s, f))
            # with T.block("root"):
            T_cast_1 = T.alloc_buffer((n, s, f))
            T_multiply = T.alloc_buffer((n, s, f))
            T_multiply_red = T.alloc_buffer((n,))
            rsqrt = T.alloc_buffer((n,))
            T_cast_2 = T.alloc_buffer((s, f))
            T_rms_norm = T.alloc_buffer((n, s, f))
            for ax0, ax1, ax2 in T.grid(n, s, f):
                with T.block("T_cast"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(A[v_ax0, v_ax1, v_ax2])
                    T.writes(T_cast_1[v_ax0, v_ax1, v_ax2])
                    T_cast_1[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2]
            for ax0, ax1, ax2 in T.grid(n, s, f):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_cast_1[v_ax0, v_ax1, v_ax2])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_1[v_ax0, v_ax1, v_ax2]
            for ax0, k1, k2 in T.grid(n, s, f):
                with T.block("T_multiply_red"):
                    v_ax0, v_k1, v_k2 = T.axis.remap("SRR", [ax0, k1, k2])
                    T.reads(T_multiply[v_ax0, v_k1, v_k2])
                    T.writes(T_multiply_red[v_ax0])
                    with T.init():
                        T_multiply_red[v_ax0] = T.float32(0)
                    T_multiply_red[v_ax0] = T_multiply_red[v_ax0] + T_multiply[v_ax0, v_k1, v_k2]
            for ax0 in range(n):
                with T.block("rsqrt"):
                    v_ax0 = T.axis.spatial(n, ax0)
                    T.reads(T_multiply_red[v_ax0])
                    T.writes(rsqrt[v_ax0])
                    rsqrt[v_ax0] = T.rsqrt(T_multiply_red[v_ax0] / (T.Cast("float32", s) * T.Cast("float32", f)) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(s, f):
                with T.block("T_cast_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(B[v_ax0, v_ax1])
                    T.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = B[v_ax0, v_ax1]
            for ax0, ax1, ax2 in T.grid(n, s, f):
                with T.block("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rsqrt[v_ax0], T_cast_1[v_ax0, v_ax1, v_ax2], T_cast_2[v_ax1, v_ax2])
                    T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T_rms_norm[v_ax0, v_ax1, v_ax2] = rsqrt[v_ax0] * T_cast_1[v_ax0, v_ax1, v_ax2] * T_cast_2[v_ax1, v_ax2]
            for ax0, ax1, ax2 in T.grid(n, s, f):
                with T.block("T_cast_2"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2])
                    T_cast[v_ax0, v_ax1, v_ax2] = T_rms_norm[v_ax0, v_ax1, v_ax2]

        @R.function
        def main(x: R.Tensor(("n", "s", "f"), dtype="float32"), weight: R.Tensor(("s", "f"), dtype="float32")) -> R.Tensor(("n", "s", "f"), dtype="float32"):
            n = T.int64()
            s = T.int64()
            f = T.int64()
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), out_sinfo=R.Tensor((n, s, f), dtype="float32"))
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_rms_norm_no_bias():
    # fmt: off
    @tvm.script.ir_module
    class RMSNorm:
        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), "float32"), weight: R.Tensor((4, 5), "float32")) -> R.Tensor((2, 3, 4, 5), "float32"):
            gv: R.Tensor((2, 3, 4, 5), "float32") = R.nn.rms_norm(x, weight, axes=[-2, -1])
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def rms_norm(A: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), B: T.Buffer((T.int64(4), T.int64(5)), "float32"), T_cast: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            T_cast_1 = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            T_multiply_red = T.alloc_buffer((T.int64(2), T.int64(3)))
            rsqrt = T.alloc_buffer((T.int64(2), T.int64(3)))
            T_cast_2 = T.alloc_buffer((T.int64(4), T.int64(5)))
            T_rms_norm = T.alloc_buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_cast"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, k2, k3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_multiply_red"):
                    v_ax0, v_ax1, v_k2, v_k3 = T.axis.remap("SSRR", [ax0, ax1, k2, k3])
                    T.reads(T_multiply[v_ax0, v_ax1, v_k2, v_k3])
                    T.writes(T_multiply_red[v_ax0, v_ax1])
                    with T.init():
                        T_multiply_red[v_ax0, v_ax1] = T.float32(0)
                    T_multiply_red[v_ax0, v_ax1] = T_multiply_red[v_ax0, v_ax1] + T_multiply[v_ax0, v_ax1, v_k2, v_k3]
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("rsqrt"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(T_multiply_red[v_ax0, v_ax1])
                    T.writes(rsqrt[v_ax0, v_ax1])
                    rsqrt[v_ax0, v_ax1] = T.rsqrt(T_multiply_red[v_ax0, v_ax1] * T.float32(0.050000000000000003) + T.float32(1.0000000000000001e-05))
            for ax0, ax1 in T.grid(T.int64(4), T.int64(5)):
                with T.block("T_cast_1"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(B[v_ax0, v_ax1])
                    T.writes(T_cast_2[v_ax0, v_ax1])
                    T_cast_2[v_ax0, v_ax1] = B[v_ax0, v_ax1]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_rms_norm"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(rsqrt[v_ax0, v_ax1], T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3], T_cast_2[v_ax2, v_ax3])
                    T.writes(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3] = rsqrt[v_ax0, v_ax1] * T_cast_1[v_ax0, v_ax1, v_ax2, v_ax3] * T_cast_2[v_ax2, v_ax3]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(3), T.int64(4), T.int64(5)):
                with T.block("T_cast_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_cast[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_cast[v_ax0, v_ax1, v_ax2, v_ax3] = T_rms_norm[v_ax0, v_ax1, v_ax2, v_ax3]

        @R.function
        def main(x: R.Tensor((2, 3, 4, 5), dtype="float32"), weight: R.Tensor((4, 5), dtype="float32")) -> R.Tensor((2, 3, 4, 5), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.rms_norm, (x, weight), out_sinfo=R.Tensor((2, 3, 4, 5), dtype="float32"))
            return gv
    # fmt: on
    mod = LegalizeOps()(RMSNorm)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_attention():
    # fmt: off
    @tvm.script.ir_module
    class Attention:
        @R.function
        def main(q: R.Tensor((4, 16, 32, 8), "float32"), k: R.Tensor((4, 8, 32, 8), "float32"), v: R.Tensor((4, 8, 32, 16), "float32"), bias: R.Tensor((4, 32, 16, 8), "float32")):
            scale = T.FloatImm("float32", 0.1)
            gv: R.Tensor((4, 16, 32, 16), "float32") = R.nn.attention(q, k, v, bias, scale=scale, causal_mask="TopLeft")
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def attention_bias(A: T.Buffer((T.int64(4), T.int64(16), T.int64(32), T.int64(8)), "float32"), B: T.Buffer((T.int64(4), T.int64(8), T.int64(32), T.int64(8)), "float32"), C: T.Buffer((T.int64(4), T.int64(8), T.int64(32), T.int64(16)), "float32"), D: T.Buffer((T.int64(4), T.int64(32), T.int64(16), T.int64(8)), "float32"), T_transpose: T.Buffer((T.int64(4), T.int64(16), T.int64(32), T.int64(16)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            T_transpose_1 = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(16), T.int64(8)))
            T_reshape = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            T_transpose_2 = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(8), T.int64(8)))
            T_reshape_1 = T.alloc_buffer((T.int64(128), T.int64(8), T.int64(8)))
            T_batch_matmul_NT = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            T_multiply = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            T_reshape_2 = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(16), T.int64(8)))
            T_add = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(16), T.int64(8)))
            T_reshape_3 = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            trilu = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            trilu_red = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(1)))
            T_subtract = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            compute = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            trilu_1 = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            trilu_red_1 = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(1)))
            T_divide = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(8)))
            T_transpose_3 = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(8), T.int64(16)))
            T_reshape_4 = T.alloc_buffer((T.int64(128), T.int64(8), T.int64(16)))
            T_batch_matmul_NN = T.alloc_buffer((T.int64(128), T.int64(16), T.int64(16)))
            T_reshape_5 = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(16), T.int64(16)))
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(8)):
                with T.block("T_transpose"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
                    T.writes(T_transpose_1[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("T_reshape"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_transpose_1[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)])
                    T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                    T_reshape[v_ax0, v_ax1, v_ax2] = T_transpose_1[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(8), T.int64(8)):
                with T.block("T_transpose_1"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(B[v_ax0, v_ax2, v_ax1, v_ax3])
                    T.writes(T_transpose_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_2[v_ax0, v_ax1, v_ax2, v_ax3] = B[v_ax0, v_ax2, v_ax1, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(8), T.int64(8)):
                with T.block("T_reshape_1"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_transpose_2[((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(8), v_ax2 % T.int64(8)])
                    T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2])
                    T_reshape_1[v_ax0, v_ax1, v_ax2] = T_transpose_2[((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(8), v_ax2 % T.int64(8)]
            for b, i, j, k in T.grid(T.int64(128), T.int64(16), T.int64(8), T.int64(8)):
                with T.block("T_batch_matmul_NT"):
                    v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k])
                    T.reads(T_reshape[v_b, v_i, v_k], T_reshape_1[v_b, v_j, v_k])
                    T.writes(T_batch_matmul_NT[v_b, v_i, v_j])
                    T.block_attr({"layout_free_placeholders": [T_reshape_1]})
                    with T.init():
                        T_batch_matmul_NT[v_b, v_i, v_j] = T.float32(0)
                    T_batch_matmul_NT[v_b, v_i, v_j] = T_batch_matmul_NT[v_b, v_i, v_j] + T_reshape[v_b, v_i, v_k] * T_reshape_1[v_b, v_j, v_k]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("T_multiply"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_batch_matmul_NT[v_ax0, v_ax1, v_ax2])
                    T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
                    T_multiply[v_ax0, v_ax1, v_ax2] = T_batch_matmul_NT[v_ax0, v_ax1, v_ax2] * T.float32(0.10000000000000001)
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(8)):
                with T.block("T_reshape_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_multiply[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(8) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(8) + v_ax2) % T.int64(16), v_ax3 % T.int64(8)])
                    T.writes(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(8) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(8) + v_ax2) % T.int64(16), v_ax3 % T.int64(8)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(8)):
                with T.block("T_add"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3], D[v_ax0, v_ax1, v_ax2, v_ax3])
                    T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_add[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_2[v_ax0, v_ax1, v_ax2, v_ax3] + D[v_ax0, v_ax1, v_ax2, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("T_reshape_3"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_add[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)])
                    T.writes(T_reshape_3[v_ax0, v_ax1, v_ax2])
                    T_reshape_3[v_ax0, v_ax1, v_ax2] = T_add[((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(8) + v_ax1) // T.int64(16) + v_ax0) % T.int64(32), (v_ax2 // T.int64(8) + v_ax1) % T.int64(16), v_ax2 % T.int64(8)]
            for i0, i1, i2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("trilu"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(T_reshape_3[v_i0, v_i1, v_i2])
                    T.writes(trilu[v_i0, v_i1, v_i2])
                    trilu[v_i0, v_i1, v_i2] = T.Select(v_i2 <= v_i1, T_reshape_3[v_i0, v_i1, v_i2], T.float32(0))
            for ax0, ax1, ax2, k2 in T.grid(T.int64(128), T.int64(16), T.int64(1), T.int64(8)):
                with T.block("trilu_red"):
                    v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                    T.reads(trilu[v_ax0, v_ax1, v_k2])
                    T.writes(trilu_red[v_ax0, v_ax1, v_ax2])
                    with T.init():
                        trilu_red[v_ax0, v_ax1, v_ax2] = T.float32(-3.4028234663852886e+38)
                    trilu_red[v_ax0, v_ax1, v_ax2] = T.max(trilu_red[v_ax0, v_ax1, v_ax2], trilu[v_ax0, v_ax1, v_k2])
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("T_subtract"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(trilu[v_ax0, v_ax1, v_ax2], trilu_red[v_ax0, v_ax1, T.int64(0)])
                    T.writes(T_subtract[v_ax0, v_ax1, v_ax2])
                    T_subtract[v_ax0, v_ax1, v_ax2] = trilu[v_ax0, v_ax1, v_ax2] - trilu_red[v_ax0, v_ax1, T.int64(0)]
            for i0, i1, i2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("compute"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(T_subtract[v_i0, v_i1, v_i2])
                    T.writes(compute[v_i0, v_i1, v_i2])
                    compute[v_i0, v_i1, v_i2] = T.exp(T_subtract[v_i0, v_i1, v_i2])
            for i0, i1, i2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("trilu_1"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(compute[v_i0, v_i1, v_i2])
                    T.writes(trilu_1[v_i0, v_i1, v_i2])
                    trilu_1[v_i0, v_i1, v_i2] = T.Select(v_i2 <= v_i1, compute[v_i0, v_i1, v_i2], T.float32(0))
            for ax0, ax1, ax2, k2 in T.grid(T.int64(128), T.int64(16), T.int64(1), T.int64(8)):
                with T.block("trilu_red_1"):
                    v_ax0, v_ax1, v_ax2, v_k2 = T.axis.remap("SSSR", [ax0, ax1, ax2, k2])
                    T.reads(trilu_1[v_ax0, v_ax1, v_k2])
                    T.writes(trilu_red_1[v_ax0, v_ax1, v_ax2])
                    with T.init():
                        trilu_red_1[v_ax0, v_ax1, v_ax2] = T.float32(0)
                    trilu_red_1[v_ax0, v_ax1, v_ax2] = trilu_red_1[v_ax0, v_ax1, v_ax2] + trilu_1[v_ax0, v_ax1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(16), T.int64(8)):
                with T.block("T_divide"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(trilu_1[v_ax0, v_ax1, v_ax2], trilu_red_1[v_ax0, v_ax1, T.int64(0)])
                    T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                    T_divide[v_ax0, v_ax1, v_ax2] = trilu_1[v_ax0, v_ax1, v_ax2] / trilu_red_1[v_ax0, v_ax1, T.int64(0)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(8), T.int64(16)):
                with T.block("T_transpose_2"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(C[v_ax0, v_ax2, v_ax1, v_ax3])
                    T.writes(T_transpose_3[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose_3[v_ax0, v_ax1, v_ax2, v_ax3] = C[v_ax0, v_ax2, v_ax1, v_ax3]
            for ax0, ax1, ax2 in T.grid(T.int64(128), T.int64(8), T.int64(16)):
                with T.block("T_reshape_4"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(T_transpose_3[((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(16) + v_ax1) % T.int64(8), v_ax2 % T.int64(16)])
                    T.writes(T_reshape_4[v_ax0, v_ax1, v_ax2])
                    T_reshape_4[v_ax0, v_ax1, v_ax2] = T_transpose_3[((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(128) // T.int64(32), ((v_ax2 // T.int64(16) + v_ax1) // T.int64(8) + v_ax0) % T.int64(32), (v_ax2 // T.int64(16) + v_ax1) % T.int64(8), v_ax2 % T.int64(16)]
            for b, i, j, k in T.grid(T.int64(128), T.int64(16), T.int64(16), T.int64(8)):
                with T.block("T_batch_matmul_NN"):
                    v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k])
                    T.reads(T_divide[v_b, v_i, v_k], T_reshape_4[v_b, v_k, v_j])
                    T.writes(T_batch_matmul_NN[v_b, v_i, v_j])
                    T.block_attr({"layout_free_placeholders": [T_reshape_4]})
                    with T.init():
                        T_batch_matmul_NN[v_b, v_i, v_j] = T.float32(0)
                    T_batch_matmul_NN[v_b, v_i, v_j] = T_batch_matmul_NN[v_b, v_i, v_j] + T_divide[v_b, v_i, v_k] * T_reshape_4[v_b, v_k, v_j]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(32), T.int64(16), T.int64(16)):
                with T.block("T_reshape_5"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_batch_matmul_NN[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(16) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(16) + v_ax2) % T.int64(16), v_ax3 % T.int64(16)])
                    T.writes(T_reshape_5[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_reshape_5[v_ax0, v_ax1, v_ax2, v_ax3] = T_batch_matmul_NN[(v_ax0 * T.int64(32) + (v_ax3 // T.int64(16) + v_ax2) // T.int64(16) + v_ax1) % T.int64(128), (v_ax3 // T.int64(16) + v_ax2) % T.int64(16), v_ax3 % T.int64(16)]
            for ax0, ax1, ax2, ax3 in T.grid(T.int64(4), T.int64(16), T.int64(32), T.int64(16)):
                with T.block("T_transpose_3"):
                    v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(T_reshape_5[v_ax0, v_ax2, v_ax1, v_ax3])
                    T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
                    T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = T_reshape_5[v_ax0, v_ax2, v_ax1, v_ax3]

        @R.function
        def main(q: R.Tensor((4, 16, 32, 8), dtype="float32"), k: R.Tensor((4, 8, 32, 8), dtype="float32"), v: R.Tensor((4, 8, 32, 16), dtype="float32"), bias: R.Tensor((4, 32, 16, 8), dtype="float32")) -> R.Tensor((4, 16, 32, 16), dtype="float32"):
            cls = Expected
            gv = R.call_tir(cls.attention_bias, (q, k, v, bias), out_sinfo=R.Tensor((4, 16, 32, 16), dtype="float32"))
            return gv

    # fmt: on
    mod = LegalizeOps()(Attention)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_nll_loss():
    # fmt: off
    @tvm.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64"), weights: R.Tensor((4,), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64"), weights: R.Tensor((4,), dtype="float32"),) -> R.Tensor((), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.nll_loss, (predictions, targets, weights), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def nll_loss(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Buffer((T.int64(2), T.int64(4), T.int64(5)), "int64"), rxplaceholder_2: T.Buffer(T.int64(4), "float32"), T_divide: T.Buffer((), "float32"),):
            # function attr dict
            T.func_attr({"tir.noalias": True})
            # body
            # with T.block("root")
            nll_loss = T.alloc_buffer([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red = T.alloc_buffer([], dtype="float32")
            nll_loss_1 = T.alloc_buffer([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red_1 = T.alloc_buffer([], dtype="float32")
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2], rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]])
                    T.writes(nll_loss[v_ax0, v_ax1, v_ax2])
                    nll_loss[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), (T.float32(0) - rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2]) * rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss_red"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(nll_loss[v_k0, v_k1, v_k2])
                    T.writes(nll_loss_red[()])
                    with T.init():
                        nll_loss_red[()] = T.float32(0)
                    nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss_1"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]])
                    T.writes(nll_loss_1[v_ax0, v_ax1, v_ax2])
                    nll_loss_1[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss_red_1"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(nll_loss_1[v_k0, v_k1, v_k2])
                    T.writes(nll_loss_red_1[()])
                    with T.init():
                        nll_loss_red_1[()] = T.float32(0)
                    nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0, v_k1, v_k2]
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(nll_loss_red[()], nll_loss_red_1[()])
                T.writes(T_divide[()])
                T_divide[()] = nll_loss_red[()] / nll_loss_red_1[()]
    # fmt: on
    mod = LegalizeOps()(NLLLoss)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_nll_no_weight():
    # fmt: off
    @tvm.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor((2, 3, 4, 5), "float32"), targets: R.Tensor((2, 4, 5), "int64")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.nn.nll_loss(predictions, targets, reduction="mean", ignore_index=-1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor((2, 3, 4, 5), dtype="float32"), targets: R.Tensor((2, 4, 5), dtype="int64"),) -> R.Tensor((), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.nll_loss_without_weight, (predictions, targets), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def nll_loss_without_weight(rxplaceholder: T.Buffer((T.int64(2), T.int64(3), T.int64(4), T.int64(5)), "float32"), rxplaceholder_1: T.Buffer((T.int64(2), T.int64(4), T.int64(5)), "int64"), T_divide: T.Buffer((), "float32"),):
            # function attr dict
            T.func_attr({"tir.noalias": True})
            # body
            # with T.block("root")
            T_full = T.alloc_buffer([T.int64(3)], dtype="float32")
            nll_loss = T.alloc_buffer([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red = T.alloc_buffer([], dtype="float32")
            nll_loss_1 = T.alloc_buffer([T.int64(2), T.int64(4), T.int64(5)], dtype="float32")
            nll_loss_red_1 = T.alloc_buffer([], dtype="float32")
            for ax0 in T.serial(T.int64(3)):
                with T.block("T_full"):
                    v_ax0 = T.axis.spatial(T.int64(3), ax0)
                    T.reads()
                    T.writes(T_full[v_ax0])
                    T_full[v_ax0] = T.float32(1)
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2], T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]])
                    T.writes(nll_loss[v_ax0, v_ax1, v_ax2])
                    nll_loss[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), (T.float32(0) - rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2]) * T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss_red"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(nll_loss[v_k0, v_k1, v_k2])
                    T.writes(nll_loss_red[()])
                    with T.init():
                        nll_loss_red[()] = T.float32(0)
                    nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss_1"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]])
                    T.writes(nll_loss_1[v_ax0, v_ax1, v_ax2])
                    nll_loss_1[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), T_full[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0))
            for k0, k1, k2 in T.grid(T.int64(2), T.int64(4), T.int64(5)):
                with T.block("nll_loss_red_1"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(nll_loss_1[v_k0, v_k1, v_k2])
                    T.writes(nll_loss_red_1[()])
                    with T.init():
                        nll_loss_red_1[()] = T.float32(0)
                    nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0, v_k1, v_k2]
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(nll_loss_red[()], nll_loss_red_1[()])
                T.writes(T_divide[()])
                T_divide[()] = nll_loss_red[()] / nll_loss_red_1[()]
    # fmt: on

    mod = LegalizeOps()(NLLLoss)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_nll_no_batch():
    # fmt: off
    @tvm.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor(("C",), "float32"), targets: R.Tensor((), "int64"), weights: R.Tensor(("C",), "float32")) -> R.Tensor((), "float32"):
            gv = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor(("C",), dtype="float32"), targets: R.Tensor((), dtype="int64"), weights: R.Tensor(("C",), dtype="float32")) -> R.Tensor((), dtype="float32"):
            C = T.int64()
            gv = R.call_tir(Expected.nll_loss, (predictions, targets, weights), out_sinfo=R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def nll_loss(var_rxplaceholder: T.handle, rxplaceholder: T.Buffer((), "int64"), var_rxplaceholder_1: T.handle, T_divide: T.Buffer((), "float32")):
            T.func_attr({"tir.noalias": True})
            C = T.int64()
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder, (C,))
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_1, (C,))
            # with T.block("root"):
            nll_loss = T.alloc_buffer(())
            nll_loss_1 = T.alloc_buffer(())
            with T.block("nll_loss"):
                vi = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(rxplaceholder[()], rxplaceholder_1[rxplaceholder[()]], rxplaceholder_2[rxplaceholder[()]])
                T.writes(nll_loss[()])
                nll_loss[()] = T.Select(rxplaceholder[()] != T.int64(1), (T.float32(0) - rxplaceholder_1[rxplaceholder[()]]) * rxplaceholder_2[rxplaceholder[()]], T.float32(0))
            with T.block("nll_loss_1"):
                vi = T.axis.spatial(T.int64(1), T.int64(0))
                T.reads(rxplaceholder[()], rxplaceholder_2[rxplaceholder[()]])
                T.writes(nll_loss_1[()])
                nll_loss_1[()] = T.Select(rxplaceholder[()] != T.int64(1), rxplaceholder_2[rxplaceholder[()]], T.float32(0))
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(nll_loss[()], nll_loss_1[()])
                T.writes(T_divide[()])
                T_divide[()] = nll_loss[()] / nll_loss_1[()]
    # fmt: on

    mod = LegalizeOps()(NLLLoss)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_nll_loss_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class NLLLoss:
        @R.function
        def main(predictions: R.Tensor(("N", "C", "d1", "d2"), "float32"), targets: R.Tensor(("N", "d1", "d2"), "int64"), weights: R.Tensor(("C",), "float32")) -> R.Tensor((), "float32"):
            gv: R.Tensor((), "float32") = R.nn.nll_loss(predictions, targets, weights, reduction="mean", ignore_index=-1)
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(predictions: R.Tensor(("N", "C", "d1", "d2"), dtype="float32"), targets: R.Tensor(("N", "d1", "d2"), dtype="int64"), weights: R.Tensor(("C",), dtype="float32")) -> R.Tensor((), dtype="float32"):
            # block 0
            gv = R.call_tir(Expected.nll_loss, (predictions, targets, weights), R.Tensor((), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def nll_loss(var_rxplaceholder: T.handle, var_rxplaceholder_1: T.handle, var_rxplaceholder_2: T.handle, T_divide: T.Buffer((), "float32"),):
            # function attr dict
            T.func_attr({"tir.noalias": True})
            C = T.int64()
            N = T.int64()
            d1 = T.int64()
            d2 = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [N, C, d1, d2], dtype="float32")
            rxplaceholder_1 = T.match_buffer(var_rxplaceholder_1, [N, d1, d2], dtype="int64")
            rxplaceholder_2 = T.match_buffer(var_rxplaceholder_2, [C], dtype="float32")
            # body
            # with T.block("root")
            nll_loss = T.alloc_buffer([N, d1, d2], dtype="float32")
            nll_loss_red = T.alloc_buffer([], dtype="float32")
            nll_loss_1 = T.alloc_buffer([N, d1, d2], dtype="float32")
            nll_loss_red_1 = T.alloc_buffer([], dtype="float32")
            for ax0, ax1, ax2 in T.grid(N, d1, d2):
                with T.block("nll_loss"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2],rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]],)
                    T.writes(nll_loss[v_ax0, v_ax1, v_ax2])
                    nll_loss[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), (T.float32(0) - rxplaceholder[v_ax0, rxplaceholder_1[v_ax0, v_ax1, v_ax2], v_ax1, v_ax2]) * rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0),)
            for k0, k1, k2 in T.grid(N, d1, d2):
                with T.block("nll_loss_red"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(nll_loss[v_k0, v_k1, v_k2])
                    T.writes(nll_loss_red[()])
                    with T.init():
                        nll_loss_red[()] = T.float32(0)
                    nll_loss_red[()] = nll_loss_red[()] + nll_loss[v_k0, v_k1, v_k2]
            for ax0, ax1, ax2 in T.grid(N, d1, d2):
                with T.block("nll_loss_1"):
                    v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                    T.reads(rxplaceholder_1[v_ax0, v_ax1, v_ax2], rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]],)
                    T.writes(nll_loss_1[v_ax0, v_ax1, v_ax2])
                    nll_loss_1[v_ax0, v_ax1, v_ax2] = T.Select(rxplaceholder_1[v_ax0, v_ax1, v_ax2] != T.int64(-1), rxplaceholder_2[rxplaceholder_1[v_ax0, v_ax1, v_ax2]], T.float32(0),)
            for k0, k1, k2 in T.grid(N, d1, d2):
                with T.block("nll_loss_red_1"):
                    v_k0, v_k1, v_k2 = T.axis.remap("RRR", [k0, k1, k2])
                    T.reads(nll_loss_1[v_k0, v_k1, v_k2])
                    T.writes(nll_loss_red_1[()])
                    with T.init():
                        nll_loss_red_1[()] = T.float32(0)
                    nll_loss_red_1[()] = nll_loss_red_1[()] + nll_loss_1[v_k0, v_k1, v_k2]
            with T.block("T_divide"):
                vi = T.axis.spatial(1, T.int64(0))
                T.reads(nll_loss_red[()], nll_loss_red_1[()])
                T.writes(T_divide[()])
                T_divide[()] = nll_loss_red[()] / nll_loss_red_1[()]
    # fmt: on
    mod = LegalizeOps()(NLLLoss)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_pad():
    @tvm.script.ir_module
    class Pad:
        @R.function
        def main(x: R.Tensor((2, 128, 28), "float32")) -> R.Tensor((2, 130, 30), "float32"):
            gv: R.Tensor((2, 130, 30), "float32") = R.nn.pad(x, (0, 0, 1, 1, 1, 1))
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 128, 28), dtype="float32")
        ) -> R.Tensor((2, 130, 30), dtype="float32"):
            gv = R.call_tir(Expected.pad, (x), out_sinfo=R.Tensor((2, 130, 30), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def pad(
            A: T.Buffer((T.int64(2), T.int64(128), T.int64(28)), "float32"),
            PadInput: T.Buffer((T.int64(2), T.int64(130), T.int64(30)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1, i2 in T.grid(T.int64(2), T.int64(130), T.int64(30)):
                with T.block("PadInput"):
                    v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                    T.reads(A[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1)])
                    T.writes(PadInput[v_i0, v_i1, v_i2])
                    PadInput[v_i0, v_i1, v_i2] = T.if_then_else(
                        T.int64(1) <= v_i1
                        and v_i1 < T.int64(129)
                        and T.int64(1) <= v_i2
                        and v_i2 < T.int64(29),
                        A[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1)],
                        T.float32(0),
                    )

    mod = LegalizeOps()(Pad)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
