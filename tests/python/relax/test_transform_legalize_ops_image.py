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


def test_image_resize2d():
    # fmt: off
    @tvm.script.ir_module
    class Resize2D:
        @R.function
        def main(x: R.Tensor((2, 8, 8, 3), "float32")) -> R.Tensor((2, 16, 16, 3), "float32"):
            gv: R.Tensor((2, 16, 16, 3), "float32") = R.image.resize2d(x, size=(16, 16), layout="NHWC", method="nearest_neighbor", coordinate_transformation_mode="asymmetric")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 8, 8, 3), "float32")) -> R.Tensor((2, 16, 16, 3), "float32"):
            gv = R.call_tir(Expected.resize2d, (x,), R.Tensor((2, 16, 16, 3), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def resize2d(rxplaceholder: T.Buffer((T.int64(2), T.int64(8), T.int64(8), T.int64(3)), "float32"), resize: T.Buffer((T.int64(2), T.int64(16), T.int64(16), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": True})
            for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(16), T.int64(16), T.int64(3)):
                with T.block("resize"):
                    i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                    T.reads(rxplaceholder[i0_1, T.max(T.min(T.Div(i1_1, T.int64(2)), T.int64(7)), T.int64(0)), T.max(T.min(T.Div(i2_1, T.int64(2)), T.int64(7)), T.int64(0)), i3_1])
                    T.writes(resize[i0_1, i1_1, i2_1, i3_1])
                    resize[i0_1, i1_1, i2_1, i3_1] = rxplaceholder[i0_1, T.max(T.min(T.Div(i1_1, T.int64(2)), T.int64(7)), T.int64(0)), T.max(T.min(T.Div(i2_1, T.int64(2)), T.int64(7)), T.int64(0)), i3_1]
    # fmt: on

    mod = LegalizeOps()(Resize2D)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_image_resize2d_symbolic():
    # fmt: off
    @tvm.script.ir_module
    class Resize2D:
        @R.function
        def main(dumb_param: R.Tensor(("oh", "ow")), x: R.Tensor(("n", "c", "h", "w", 16), "float32")) -> R.Tensor(("n", "c", "oh", "ow", 16), "float32"):
            n = T.int64()
            c = T.int64()
            oh = T.int64()
            ow = T.int64()
            gv: R.Tensor((n, c, oh, ow, 16), "float32") = R.image.resize2d(x, size=(oh, ow), layout="NCHW16c", method="nearest_neighbor", coordinate_transformation_mode="asymmetric")
            return gv

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(dumb_param: R.Tensor(("oh", "ow")), x: R.Tensor(("n", "c", "h", "w", 16), "float32")) -> R.Tensor(("n", "c", "oh", "ow", 16), "float32"):
            n = T.int64()
            c = T.int64()
            oh = T.int64()
            ow = T.int64()
            gv = R.call_tir(Expected.resize2d, (x,), R.Tensor((n, c, oh, ow, 16), dtype="float32"))
            return gv

        @T.prim_func(private=True)
        def resize2d(var_rxplaceholder: T.handle, var_resize: T.handle):
            T.func_attr({"tir.noalias": True})
            c = T.int64()
            h = T.int64()
            n = T.int64()
            oh = T.int64()
            ow = T.int64()
            w = T.int64()
            rxplaceholder = T.match_buffer(var_rxplaceholder, [n, c, h, w, T.int64(16)], dtype="float32")
            resize = T.match_buffer(var_resize, [n, c, oh, ow, T.int64(16)], dtype="float32")
            for i0, i1, i2, i3, i4 in T.grid(n, c, oh, ow, T.int64(16)):
                with T.block("resize"):
                    i0_1, i1_1, i2_1, i3_1, i4_1 = T.axis.remap("SSSSS", [i0, i1, i2, i3, i4])
                    T.reads(rxplaceholder[i0_1, i1_1, T.int64(0) : T.max(h, T.int64(1)), T.int64(0) : T.max(w, T.int64(1)), i4_1])
                    T.writes(resize[i0_1, i1_1, i2_1, i3_1, i4_1])
                    resize[i0_1, i1_1, i2_1, i3_1, i4_1] = rxplaceholder[i0_1, i1_1, T.max(T.min(T.Cast("int64", T.round(T.Cast("float32", h) / T.Cast("float32", oh) * T.Cast("float32", i2_1), dtype="float32")), h - T.int64(1)), T.int64(0)), T.max(T.min(T.Cast("int64", T.round(T.Cast("float32", w) / T.Cast("float32", ow) * T.Cast("float32", i3_1), dtype="float32")), w - T.int64(1)), T.int64(0)), i4_1]
    # fmt: on

    mod = LegalizeOps()(Resize2D)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
