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
from tvm.relax.transform import ConvertLayout, Normalize
from tvm.script.parser import ir as I, relax as R, tir as T


def verify(input, expected):
    mod = ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(input)
    mod = Normalize()(mod)
    tvm.ir.assert_structural_equal(mod, expected)


def test_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv)
            return gv

    verify(Input, Expected)


def test_conv2d_onlydim():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor("float32", ndim=4), w: R.Tensor("float32", ndim=4)
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor("float32", ndim=4) = R.nn.conv2d(x, w, out_dtype="float32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)
        ) -> R.Tensor(dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(lv2, axes=[0, 3, 1, 2])
                R.output(gv)
            return gv

    verify(Input, Expected)


def test_conv2d_symbolic():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor("float32", ndim=4), w: R.Tensor("float32", ndim=4)
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                N, C, H, W = T.int64(), T.int64(), T.int64(), T.int64()
                lv0 = R.match_cast(x, R.Tensor((N, C, H, W), "float32"))
                gv: R.Tensor("float32", ndim=4) = R.nn.conv2d(lv0, w, out_dtype="float32")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)
        ) -> R.Tensor(dtype="float32", ndim=4):
            N = T.int64()
            C = T.int64()
            H = T.int64()
            W = T.int64()
            with R.dataflow():
                lv0: R.Tensor((N, C, H, W), dtype="float32") = R.match_cast(
                    x, R.Tensor((N, C, H, W), dtype="float32")
                )
                lv: R.Tensor((N, H, W, C), dtype="float32") = R.permute_dims(lv0, axes=[0, 2, 3, 1])
                lv1: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(lv2, axes=[0, 3, 1, 2])
                R.output(gv)
            return gv

    verify(Input, Expected)


def test_conv2d_matchcast_bias():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor("float32", ndim=4), w: R.Tensor("float32", ndim=4)
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                lv0: R.Tensor("float32", ndim=4) = R.nn.conv2d(x, w, out_dtype="float32")
                N, C, H, W = T.int64(), T.int64(), T.int64(), T.int64()
                lv1 = R.match_cast(lv0, R.Tensor((N, C, H, W), "float32"))
                gv = R.add(lv1, w)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=4), w: R.Tensor(dtype="float32", ndim=4)
        ) -> R.Tensor(dtype="float32", ndim=4):
            N = T.int64()
            H = T.int64()
            W = T.int64()
            C = T.int64()
            with R.dataflow():
                lv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv0: R.Tensor(dtype="float32", ndim=4) = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((N, H, W, C), dtype="float32") = R.match_cast(
                    lv0, R.Tensor((N, H, W, C), dtype="float32")
                )
                lv3: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv4: R.Tensor(dtype="float32", ndim=4) = R.add(lv2, lv3)
                gv: R.Tensor(dtype="float32", ndim=4) = R.permute_dims(lv4, axes=[0, 3, 1, 2])
                R.output(gv)
            return gv

    verify(Input, Expected)


def test_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_relu_conv2d_relu():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 3, 28, 28), "float32") = R.nn.relu(x)
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 3, 28, 28), dtype="float32") = R.nn.relu(x)
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(
                    x0, axes=[0, 2, 3, 1]
                )
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_relu_tanh():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.tanh(gv2)
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.tanh(gv2)
                gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv3)
            return gv3

    verify(Input, Expected)


def test_conv2d_add():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.permute_dims(
                    bias, axes=[0, 2, 3, 1]
                )
                lv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.add(gv, lv2)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv3, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_add_relu_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), "float32"),
            w: R.Tensor((4, 4, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv2)
                gv4: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(gv3, w, out_dtype="float32")
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), dtype="float32"),
            w: R.Tensor((4, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 4), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.permute_dims(
                    bias, axes=[0, 2, 3, 1]
                )
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.add(gv, lv2)
                gv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv2)
                lv3: R.Tensor((4, 3, 3, 4), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv4: R.Tensor((2, 24, 24, 4), dtype="float32") = R.nn.conv2d(
                    gv3,
                    lv3,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv4: R.Tensor((2, 4, 24, 24), dtype="float32") = R.permute_dims(
                    lv4, axes=[0, 3, 1, 2]
                )
                R.output(gv4)
            return gv4

    verify(Input, Expected)


def test_conv2d_fma_relu_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), "float32"),
            w: R.Tensor((4, 4, 3, 3), "float32"),
            scale: R.Tensor((2, 4, 26, 26), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.ewise_fma(gv, scale, bias)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv2)
                gv4: R.Tensor((2, 4, 24, 24), "float32") = R.nn.conv2d(gv3, w, out_dtype="float32")
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), dtype="float32"),
            w: R.Tensor((4, 4, 3, 3), dtype="float32"),
            scale: R.Tensor((2, 4, 26, 26), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 4), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv, axes=[0, 3, 1, 2]
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.ewise_fma(lv2, scale, bias)
                gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv2)
                lv3: R.Tensor((2, 26, 26, 4), dtype="float32") = R.permute_dims(
                    gv3, axes=[0, 2, 3, 1]
                )
                lv4: R.Tensor((4, 3, 3, 4), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv5: R.Tensor((2, 24, 24, 4), dtype="float32") = R.nn.conv2d(
                    lv3,
                    lv4,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv4: R.Tensor((2, 4, 24, 24), dtype="float32") = R.permute_dims(
                    lv5, axes=[0, 3, 1, 2]
                )
                R.output(gv4)
            return gv4

    verify(Input, Expected)


def test_conv2d_sum():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[2, 3])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=2):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 4), dtype="float32") = R.sum(gv, axis=[1, 2], keepdims=False)
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_sum_keepdim():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 1, 1), "float32") = R.sum(gv, axis=[2, 3], keepdims=True)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 1, 4), dtype="float32") = R.sum(gv, axis=[1, 2], keepdims=True)
                gv2: R.Tensor((2, 4, 1, 1), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_sum_negative_dims():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[-2, -1])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 4), dtype="float32") = R.sum(gv, axis=[1, 2])
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_transpose():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv, axes=[3, 2, 1, 0])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((26, 26, 4, 2), dtype="float32") = R.permute_dims(
                    gv, axes=[2, 1, 3, 0]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_expand_dims_scalar():
    @I.ir_module
    class Input:
        @R.function
        def main() -> R.Tensor((1,), dtype="int64"):
            with R.dataflow():
                gv: R.Tensor((1,), dtype="int64") = R.expand_dims(R.const(0, "int64"), axis=[0])
                R.output(gv)
            return gv

    verify(Input, Input)


def test_conv2d_expand_dims():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=6):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=6):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 1, 26, 4), dtype="float32") = R.expand_dims(
                    gv, axis=[-3, 1]
                )
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 1, 5, 3, 2, 4]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_expand_dims_squeeze():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.squeeze(gv2, axis=[1, 3])
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 1, 26, 1, 26, 4), dtype="float32") = R.expand_dims(
                    gv, axis=[-3, 1]
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.squeeze(gv2, axis=[1, 3])
                gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv3)
            return gv3

    verify(Input, Expected)


def test_conv2d_strided_slice():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    gv, begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4], axes=[1, 2, 3]
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 9, 7, 2), dtype="float32") = R.strided_slice(
                    gv, axes=[3, 1, 2], begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4]
                )
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_relu_concat():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 26, 26, 8), dtype="float32") = R.concat((gv, gv2), axis=3)
                gv3: R.Tensor((2, 8, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv3)
            return gv3

    verify(Input, Expected)


def test_conv2d_relu_concat_split():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 26, 26, 8), dtype="float32") = R.concat((gv, gv2), axis=3)
                gv4: R.Tuple(
                    R.Tensor((2, 26, 26, 4), dtype="float32"),
                    R.Tensor((2, 26, 26, 4), dtype="float32"),
                ) = R.split(gv3, indices_or_sections=2, axis=3)
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = gv4[0]
                lv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                lv4: R.Tensor((2, 26, 26, 4), dtype="float32") = gv4[1]
                lv5: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv4, axes=[0, 3, 1, 2]
                )
                gv5 = (lv3, lv5)
                R.output(gv5)
            return gv5

    verify(Input, Expected)


def test_conv2d_maxpool2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0],
                    layout="NCHW",
                    out_layout="NCHW",
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 13, 13, 4), dtype="float32") = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    dilation=[1, 1],
                    padding=[0, 0, 0, 0],
                    ceil_mode=False,
                    layout="NHWC",
                    out_layout="NHWC",
                )
                gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_avgpool2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.adaptive_avg_pool2d(gv, output_size=[13, 13], layout="NCHW")
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 13, 13, 4), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    gv, output_size=[13, 13], layout="NHWC", out_layout="NHWC"
                )
                gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_softmax():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.softmax(gv, axis=1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.softmax(gv, axis=3)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_batchnorm():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            gamma: R.Tensor((4,), dtype="float32"),
            beta: R.Tensor((4,), dtype="float32"),
            moving_mean: R.Tensor((4,), dtype="float32"),
            moving_var: R.Tensor((4,), dtype="float32"),
        ):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = R.nn.batch_norm(gv, gamma, beta, moving_mean, moving_var, axis=1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            gamma: R.Tensor((4,), dtype="float32"),
            beta: R.Tensor((4,), dtype="float32"),
            moving_mean: R.Tensor((4,), dtype="float32"),
            moving_var: R.Tensor((4,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv2: R.Tuple(
                    R.Tensor((2, 26, 26, 4), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = R.nn.batch_norm(
                    gv,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=3,
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = gv2[0]
                lv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                lv4: R.Tensor((4,), dtype="float32") = gv2[1]
                lv5: R.Tensor((4,), dtype="float32") = gv2[2]
                gv3 = (lv3, lv4, lv5)
                R.output(gv3)
            return gv3

    verify(Input, Expected)


def test_conv2d_layernorm():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.layer_norm(
                    gv, gamma, beta, axes=[-2, -1]
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.layer_norm(
                    gv,
                    gamma,
                    beta,
                    axes=[1, 2],
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_resize2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.image.resize2d(gv, (52, 52), layout="NCHW")
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor(None, dtype="float32", ndim=4):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 52, 52, 4), dtype="float32") = R.image.resize2d(
                    gv,
                    (52, 52),
                    roi=[T.float32(0), T.float32(0), T.float32(0), T.float32(0)],
                    layout="NHWC",
                    method="linear",
                    coordinate_transformation_mode="half_pixel",
                    rounding_method="round",
                    cubic_alpha=-0.5,
                    cubic_exclude=0,
                    extrapolation_value=0,
                    out_dtype="void",
                )
                gv2: R.Tensor((2, 4, 52, 52), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_conv2d_unknown_bias_dim():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            w2: R.Tensor(dtype="float32"),
        ) -> R.Tensor(None, "float32"):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = w2 + gv
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            w2: R.Tensor(dtype="float32"),
        ) -> R.Tensor(dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv, axes=[0, 3, 1, 2]
                )
                gv2: R.Tensor(dtype="float32") = R.add(w2, lv2)
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_binary_broadcast():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"),
            w: R.Tensor((4, 3, 3, 3), "float32"),
            bias: R.Tensor((26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv, axes=[0, 3, 1, 2]
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(lv2, bias)
                R.output(gv2)
            return gv2

    verify(Input, Expected)


def test_binary_ewise_scalar():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, R.const(1, "float32"))
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.add(gv, R.const(1, "float32"))
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


if __name__ == "__main__":
    tvm.testing.main()
