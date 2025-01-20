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


def verify(input, expected, extra_ops={}):
    desired_layouts = {"relax.nn.conv2d": ["NHWC", "OHWI"]}
    desired_layouts.update(extra_ops)
    mod = ConvertLayout(desired_layouts)(input)
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
    # Channel not a proper multiple shouldn't alter the mod
    verify(Input, Input, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Input, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


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
                lv3: R.Tensor((N, C, H, W), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                gv: R.Tensor(dtype="float32", ndim=4) = R.add(lv3, w)
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

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), dtype="float32"),
            w: R.Tensor((4, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 1, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 4, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.layout_transform(
                    bias,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.add(gv, lv2)
                gv3: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv2)
                lv3: R.Tensor((1, 4, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                lv4: R.Tensor((2, 1, 24, 24, 4), dtype="float32") = R.nn.conv2d(
                    gv3,
                    lv3,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv4: R.Tensor((2, 4, 24, 24), dtype="float32") = R.layout_transform(
                    lv4,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), dtype="float32"),
            w: R.Tensor((4, 4, 3, 3), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 1, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 4, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.layout_transform(
                    bias,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.add(gv, lv2)
                gv3: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.relu(gv2)
                lv3: R.Tensor((1, 3, 3, 4, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                lv4: R.Tensor((2, 24, 24, 1, 4), dtype="float32") = R.nn.conv2d(
                    gv3,
                    lv3,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv4: R.Tensor((2, 4, 24, 24), dtype="float32") = R.layout_transform(
                    lv4,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv4)
            return gv4

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


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


def test_resize2d_conv2d():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), "float32"), w: R.Tensor((4, 3, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv = R.image.resize2d(x, (52, 52), layout="NCHW")
                gv2: R.Tensor((2, 4, 50, 50), "float32") = R.nn.conv2d(gv, w, out_dtype="float32")
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"), w: R.Tensor((4, 3, 3, 3), dtype="float32")
        ) -> R.Tensor((2, 4, 50, 50), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv: R.Tensor((2, 52, 52, 3), dtype="float32") = R.image.resize2d(
                    lv,
                    R.shape([52, 52]),
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
                lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                lv2: R.Tensor((2, 50, 50, 4), dtype="float32") = R.nn.conv2d(
                    gv,
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
                gv2: R.Tensor((2, 4, 50, 50), dtype="float32") = R.permute_dims(
                    lv2, axes=[0, 3, 1, 2]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected, extra_ops={"relax.image.resize2d": ["NHWC"]})


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


def test_conv2d_NCHW_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(
                    x,
                    w,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv)
            return gv

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv)
            return gv

    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_NHWC_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 28, 28, 16), "float32"), w: R.Tensor((4, 3, 3, 16), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 26, 26, 4), "float32") = R.nn.conv2d(
                    x,
                    w,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_dtype="float32",
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 28, 28, 16), dtype="float32"),
            w: R.Tensor((4, 3, 3, 16), dtype="float32"),
        ) -> R.Tensor((2, 26, 26, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i3 // 4, i1, i2, i3 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i3, i1, i2, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i2, i3, i1 * 4 + i4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv)
            return gv

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 28, 28, 16), dtype="float32"),
            w: R.Tensor((4, 3, 3, 16), dtype="float32"),
        ) -> R.Tensor((2, 26, 26, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1, i2, i3 // 4, i3 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1, i2, i3 * 4 + i4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv)
            return gv

    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})

    @I.ir_module
    class Expected_N2nHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 28, 28, 16), dtype="float32"),
            w: R.Tensor((4, 3, 3, 16), dtype="float32"),
        ) -> R.Tensor((2, 26, 26, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 2, i0 % 2, i1, i2, i3 // 4, i3 % 4),
                        index_dtype="int32",
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 8, 2, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3 // 2, i3 % 2, i0 % 4),
                        index_dtype="int32",
                    ),
                )
                lv2: R.Tensor((1, 2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="N2nHWC4c",
                    kernel_layout="OHWI2i4o",
                    out_layout="N2nHWC4c",
                    out_dtype="float32",
                )
                gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4, i5: (i0 * 2 + i1, i2, i3, i4 * 4 + i5),
                        index_dtype="int32",
                    ),
                )
                R.output(gv)
            return gv

    verify(Input, Expected_N2nHWC4c, {"relax.nn.conv2d": ["N2nHWC4c", "OHWI2i4o"]})


def test_conv2d_symbolic_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor("float32", ndim=4), w: R.Tensor("float32", ndim=4)
        ) -> R.Tensor("float32", ndim=4):
            with R.dataflow():
                N, C, H, W = T.int64(), T.int64(16), T.int64(), T.int64()
                Nw, Cw, Hw, Ww = T.int64(4), T.int64(16), T.int64(), T.int64()
                lv0 = R.match_cast(x, R.Tensor((N, C, H, W), "float32"))
                lv1 = R.match_cast(w, R.Tensor((Nw, Cw, Hw, Ww), "float32"))
                gv: R.Tensor(
                    (N, T.int64(4), H + T.int64(1) - Hw, W + T.int64(1) - Ww), "float32"
                ) = R.nn.conv2d(lv0, lv1, out_dtype="float32")
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
            Hw = T.int64()
            Ww = T.int64()
            with R.dataflow():
                lv0: R.Tensor((N, 16, H, W), dtype="float32") = R.match_cast(
                    x, R.Tensor((N, 16, H, W), dtype="float32")
                )
                lv1: R.Tensor((4, 16, Hw, Ww), dtype="float32") = R.match_cast(
                    w, R.Tensor((4, 16, Hw, Ww), dtype="float32")
                )
                lv: R.Tensor((N, 4, H, W, 4), dtype="float32") = R.layout_transform(
                    lv0,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1_1: R.Tensor((1, 16, Hw, Ww, 4), dtype="float32") = R.layout_transform(
                    lv1,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv2: R.Tensor((N, 1, H + 1 - Hw, W + 1 - Ww, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1_1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv: R.Tensor((N, 4, H + 1 - Hw, W + 1 - Ww), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv)
            return gv

    verify(Input, Expected, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_matchcast_bias_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor("float32", ndim=4),
            w: R.Tensor("float32", ndim=4),
            bias: R.Tensor("float32", ndim=4),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                N, C, H, W = T.int64(), T.int64(16), T.int64(), T.int64()
                Nw, Cw, Hw, Ww = T.int64(4), T.int64(16), T.int64(), T.int64()
                lv0 = R.match_cast(x, R.Tensor((N, C, H, W), "float32"))
                lv1 = R.match_cast(w, R.Tensor((Nw, Cw, Hw, Ww), "float32"))
                lv2: R.Tensor("float32", ndim=4) = R.nn.conv2d(lv0, lv1, out_dtype="float32")
                Nb, Cb, Hb, Wb = T.int64(), T.int64(), T.int64(), T.int64()
                lv_bias = R.match_cast(bias, R.Tensor((Nb, Cb, Hb, Wb), "float32"))
                gv = R.add(lv2, lv_bias)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=4),
            w: R.Tensor(dtype="float32", ndim=4),
            bias: R.Tensor(dtype="float32", ndim=4),
        ) -> R.Tensor(dtype="float32", ndim=4):
            N, C, H, W = T.int64(), T.int64(16), T.int64(), T.int64()
            Nw, Cw, Hw, Ww = T.int64(4), T.int64(16), T.int64(), T.int64()
            Nb, Cb, Hb, Wb = T.int64(), T.int64(), T.int64(), T.int64()
            with R.dataflow():
                lv0: R.Tensor((N, 16, H, W), dtype="float32") = R.match_cast(
                    x, R.Tensor((N, 16, H, W), dtype="float32")
                )
                lv1: R.Tensor((4, 16, Hw, Ww), dtype="float32") = R.match_cast(
                    w, R.Tensor((4, 16, Hw, Ww), dtype="float32")
                )
                lv: R.Tensor((N, H, W, 4, 4), dtype="float32") = R.layout_transform(
                    lv0,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1_1: R.Tensor((1, Hw, Ww, 16, 4), dtype="float32") = R.layout_transform(
                    lv1,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv2: R.Tensor((N, H + 1 - Hw, W + 1 - Ww, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1_1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv_bias: R.Tensor((Nb, Cb, Hb, Wb), dtype="float32") = R.match_cast(
                    bias, R.Tensor((Nb, Cb, Hb, Wb), dtype="float32")
                )
                lv2_1: R.Tensor(
                    (Nb, Hb, Wb, (Cb - Cb % -4) // 4, 4), dtype="float32"
                ) = R.layout_transform(
                    lv_bias,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv3: R.Tensor(dtype="float32", ndim=5) = R.add(lv2, lv2_1)
                gv: R.Tensor(dtype="float32", ndim=4) = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=4),
            w: R.Tensor(dtype="float32", ndim=4),
            bias: R.Tensor(dtype="float32", ndim=4),
        ) -> R.Tensor(dtype="float32", ndim=4):
            N, C, H, W = T.int64(), T.int64(16), T.int64(), T.int64()
            Nw, Cw, Hw, Ww = T.int64(4), T.int64(16), T.int64(), T.int64()
            Nb, Cb, Hb, Wb = T.int64(), T.int64(), T.int64(), T.int64()
            with R.dataflow():
                lv0: R.Tensor((N, 16, H, W), dtype="float32") = R.match_cast(
                    x, R.Tensor((N, 16, H, W), dtype="float32")
                )
                lv1: R.Tensor((4, 16, Hw, Ww), dtype="float32") = R.match_cast(
                    w, R.Tensor((4, 16, Hw, Ww), dtype="float32")
                )
                lv: R.Tensor((N, 4, H, W, 4), dtype="float32") = R.layout_transform(
                    lv0,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1_1: R.Tensor((1, 16, Hw, Ww, 4), dtype="float32") = R.layout_transform(
                    lv1,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                lv2: R.Tensor((N, 1, H + 1 - Hw, W + 1 - Ww, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1_1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv_bias: R.Tensor((Nb, Cb, Hb, Wb), dtype="float32") = R.match_cast(
                    bias, R.Tensor((Nb, Cb, Hb, Wb), dtype="float32")
                )
                lv2_1: R.Tensor(
                    (Nb, (Cb - Cb % -4) // 4, Hb, Wb, 4), dtype="float32"
                ) = R.layout_transform(
                    lv_bias,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv3: R.Tensor(dtype="float32", ndim=5) = R.add(lv2, lv2_1)
                gv: R.Tensor(dtype="float32", ndim=4) = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv)
            return gv

    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})
    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_layout_incompatible_fallback():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor("float32", ndim=4),
            w: R.Tensor("float32", ndim=4),
            bias: R.Tensor("float32", ndim=4),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                N, C, H, W = T.int64(), T.int64(15), T.int64(), T.int64()
                Nw, Cw, Hw, Ww = T.int64(4), T.int64(15), T.int64(), T.int64()
                lv0 = R.match_cast(x, R.Tensor((N, C, H, W), "float32"))
                lv1 = R.match_cast(w, R.Tensor((Nw, Cw, Hw, Ww), "float32"))
                lv2: R.Tensor("float32", ndim=4) = R.nn.conv2d(lv0, lv1, out_dtype="float32")
                Nb, Cb, Hb, Wb = T.int64(), T.int64(), T.int64(), T.int64()
                lv_bias = R.match_cast(bias, R.Tensor((Nb, Cb, Hb, Wb), "float32"))
                gv = R.add(lv2, lv_bias)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(dtype="float32", ndim=4),
            w: R.Tensor(dtype="float32", ndim=4),
            bias: R.Tensor(dtype="float32", ndim=4),
        ) -> R.Tensor(dtype="float32", ndim=4):
            N, C, H, W = T.int64(), T.int64(15), T.int64(), T.int64()
            Nw, Cw, Hw, Ww = T.int64(4), T.int64(15), T.int64(), T.int64()
            Nb, Cb, Hb, Wb = T.int64(), T.int64(), T.int64(), T.int64()
            with R.dataflow():
                lv0: R.Tensor((N, 15, H, W), dtype="float32") = R.match_cast(
                    x, R.Tensor((N, 15, H, W), dtype="float32")
                )
                lv1: R.Tensor((4, 15, Hw, Ww), dtype="float32") = R.match_cast(
                    w, R.Tensor((4, 15, Hw, Ww), dtype="float32")
                )
                lv2: R.Tensor((N, 4, H + 1 - Hw, W + 1 - Ww), dtype="float32") = R.nn.conv2d(
                    lv0,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="float32",
                )
                lv_bias: R.Tensor((Nb, Cb, Hb, Wb), dtype="float32") = R.match_cast(
                    bias, R.Tensor((Nb, Cb, Hb, Wb), dtype="float32")
                )
                gv: R.Tensor(dtype="float32", ndim=4) = R.add(lv2, lv_bias)
                R.output(gv)
            return gv

    verify(Input, Expected, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})
    verify(Input, Expected, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_relu_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_relu_conv2d_relu_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                x0: R.Tensor((2, 16, 28, 28), "float32") = R.nn.relu(x)
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x0, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                x0: R.Tensor((2, 16, 28, 28), dtype="float32") = R.nn.relu(x)
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x0,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                x0: R.Tensor((2, 16, 28, 28), dtype="float32") = R.nn.relu(x)
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x0,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.relu(gv)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_relu_tanh_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 26, 26), "float32") = R.tanh(gv2)
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.tanh(gv2)
                gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.tanh(gv2)
                gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv3)
            return gv3

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_add_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((2, 4, 26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.layout_transform(
                    bias,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv3: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.add(gv, lv2)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.layout_transform(
                    bias,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv3: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.add(gv, lv2)
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_fma_relu_conv2d_sub_indexed():
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
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 4, 28, 28), dtype="float32"),
            w: R.Tensor((4, 4, 3, 3), dtype="float32"),
            scale: R.Tensor((2, 4, 26, 26), dtype="float32"),
            bias: R.Tensor((2, 4, 26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 24, 24), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 1, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 4, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.ewise_fma(lv2, scale, bias)
                gv3: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.relu(gv2)
                lv3: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.layout_transform(
                    gv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv4: R.Tensor((1, 4, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                lv5: R.Tensor((2, 1, 24, 24, 4), dtype="float32") = R.nn.conv2d(
                    lv3,
                    lv4,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv4: R.Tensor((2, 4, 24, 24), dtype="float32") = R.layout_transform(
                    lv5,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv4)
            return gv4

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_sum_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[2, 3])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 4), dtype="float32") = R.sum(gv, axis=[2, 3], keepdims=False)
                gv2: R.Tensor((2, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2: (i0, i1 * 4 + i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 4), dtype="float32") = R.sum(gv, axis=[1, 2], keepdims=False)
                gv2: R.Tensor((2, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2: (i0, i1 * 4 + i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_sum_keepdims_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 1, 1), "float32") = R.sum(gv, axis=[2, 3], keepdims=True)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 1, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 1, 1, 4), dtype="float32") = R.sum(
                    gv, axis=[2, 3], keepdims=True
                )
                gv2: R.Tensor((2, 4, 1, 1), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 1, 1), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 1, 1, 4), dtype="float32") = R.sum(
                    gv, axis=[1, 2], keepdims=True
                )
                gv2: R.Tensor((2, 4, 1, 1), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_sum_reduce_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 26), "float32") = R.sum(gv, axis=[1, 2])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26), dtype="float32") = R.sum(gv, axis=[1, 2, 4], keepdims=False)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26), dtype="float32") = R.sum(gv, axis=[1, 3, 4], keepdims=False)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW2n4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4, 28, 28, 2, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 2, i1 // 4, i2, i3, i0 % 2, i1 % 4),
                        index_dtype="int32",
                    ),
                )
                lv1: R.Tensor((1, 8, 3, 3, 2, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1 // 2, i2, i3, i1 % 2, i0 % 4),
                        index_dtype="int32",
                    ),
                )
                gv: R.Tensor((1, 1, 26, 26, 2, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW2n4c",
                    kernel_layout="OIHW2i4o",
                    out_layout="NCHW2n4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 26, 2), dtype="float32") = R.sum(
                    gv, axis=[1, 2, 5], keepdims=False
                )
                gv2: R.Tensor((2, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2: (i0 * 2 + i2, i1), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})
    verify(Input, Expected_NCHW2n4c, {"relax.nn.conv2d": ["NCHW2n4c", "OIHW2i4o"]})


def test_conv2d_sum_negative_dims_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4), "float32") = R.sum(gv, axis=[-2, -1])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 4), dtype="float32") = R.sum(gv, axis=[2, 3], keepdims=False)
                gv2: R.Tensor((2, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2: (i0, i1 * 4 + i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 4), dtype="float32") = R.sum(gv, axis=[1, 2], keepdims=False)
                gv2: R.Tensor((2, 4), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2: (i0, i1 * 4 + i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_transpose_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((26, 26, 4, 2), "float32") = R.permute_dims(gv, axes=[3, 2, 1, 0])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((26, 26, 4, 2), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((26, 26, 4, 2), dtype="float32") = R.permute_dims(
                    lv2, axes=[3, 2, 1, 0]
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((26, 26, 4, 2), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((26, 26, 4, 2), dtype="float32") = R.permute_dims(
                    lv2, axes=[3, 2, 1, 0]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_expand_dims_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=6):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), "float32") = R.expand_dims(gv, axis=(-3, 1))
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 1, 4, 1, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), dtype="float32") = R.expand_dims(
                    lv2, axis=[-3, 1]
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 1, 4, 1, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 1, 4, 1, 26, 26), dtype="float32") = R.expand_dims(
                    lv2, axis=[-3, 1]
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_squeeze_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((1, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=3):
            with R.dataflow():
                gv: R.Tensor((1, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((4, 26, 26), "float32") = R.squeeze(gv, axis=[0])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((1, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((1, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((4, 26, 26), dtype="float32") = R.squeeze(lv2, axis=[0])
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((1, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((1, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                gv: R.Tensor((1, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((1, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                    pad_value=None,
                    axis_separators=[],
                    input_axis_separators=[],
                )
                gv2: R.Tensor((4, 26, 26), dtype="float32") = R.squeeze(lv2, axis=[0])
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_strided_slice_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    gv, begin=[0, 0, 0], end=[4, 26, 26], strides=[2, 3, 4], axes=[1, 2, 3]
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 2, 9, 7), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    lv2,
                    (R.prim_value(1), R.prim_value(2), R.prim_value(3)),
                    (R.prim_value(0), R.prim_value(0), R.prim_value(0)),
                    (R.prim_value(4), R.prim_value(26), R.prim_value(26)),
                    (R.prim_value(2), R.prim_value(3), R.prim_value(4)),
                    assume_inbound=False,
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 2, 9, 7), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 2, 9, 7), dtype="float32") = R.strided_slice(
                    lv2,
                    (R.prim_value(1), R.prim_value(2), R.prim_value(3)),
                    (R.prim_value(0), R.prim_value(0), R.prim_value(0)),
                    (R.prim_value(4), R.prim_value(26), R.prim_value(26)),
                    (R.prim_value(2), R.prim_value(3), R.prim_value(4)),
                    assume_inbound=False,
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_relu_concat_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 8, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 2, 26, 26, 4), dtype="float32") = R.concat((gv, gv2), axis=1)
                gv3: R.Tensor((2, 8, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 8, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 26, 26, 2, 4), dtype="float32") = R.concat((gv, gv2), axis=3)
                gv3: R.Tensor((2, 8, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv3)
            return gv3

    @I.ir_module
    class Expected_N4cHWC:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 8, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 % 4, i2, i3, i1 // 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 4, 3, 3, 16), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i0 % 4, i2, i3, i1), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 4, 26, 26, 1), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="N4cHWC",
                    kernel_layout="O4oHWI",
                    out_layout="N4cHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 4, 26, 26, 1), dtype="float32") = R.nn.relu(gv)
                lv2: R.Tensor((2, 4, 26, 26, 2), dtype="float32") = R.concat((gv, gv2), axis=4)
                gv3: R.Tensor((2, 8, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i4 * 4 + i1, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv3)
            return gv3

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})
    # Concat axis after sub index
    verify(Input, Expected_N4cHWC, {"relax.nn.conv2d": ["N4cHWC", "O4oHWI"]})


def test_conv2d_relu_concat_split_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=2, axis=1)
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 2, 26, 26, 4), dtype="float32") = R.concat((gv, gv2), axis=1)
                lv2: R.Tuple(
                    R.Tensor((2, 1, 26, 26, 4), dtype="float32"),
                    R.Tensor((2, 1, 26, 26, 4), dtype="float32"),
                ) = R.split(gv3, indices_or_sections=2, axis=1)
                lv3: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = lv2[0]
                lv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                lv5: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = lv2[1]
                lv6: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv5,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv4: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                ) = (lv4, lv6)
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 26, 26, 2, 4), dtype="float32") = R.concat((gv, gv2), axis=3)
                lv2: R.Tuple(
                    R.Tensor((2, 26, 26, 1, 4), dtype="float32"),
                    R.Tensor((2, 26, 26, 1, 4), dtype="float32"),
                ) = R.split(gv3, indices_or_sections=2, axis=3)
                lv3: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = lv2[0]
                lv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                lv5: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = lv2[1]
                lv6: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv5,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                gv4: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                ) = (lv4, lv6)
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected_N4cHWC:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4, 26, 26), dtype="float32"), R.Tensor((2, 4, 26, 26), dtype="float32")
        ):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 % 4, i2, i3, i1 // 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 4, 3, 3, 16), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i0 % 4, i2, i3, i1), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 4, 26, 26, 1), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="N4cHWC",
                    kernel_layout="O4oHWI",
                    out_layout="N4cHWC",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 4, 26, 26, 1), dtype="float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 4, 26, 26, 2), dtype="float32") = R.concat((gv, gv2), axis=4)
                lv2: R.Tuple(
                    R.Tensor((2, 4, 26, 26, 1), dtype="float32"),
                    R.Tensor((2, 4, 26, 26, 1), dtype="float32"),
                ) = R.split(gv3, indices_or_sections=2, axis=4)
                lv3: R.Tensor((2, 4, 26, 26, 1), dtype="float32") = lv2[0]
                lv4: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i4 * 4 + i1, i2, i3), index_dtype="int32"
                    ),
                )
                lv5: R.Tensor((2, 4, 26, 26, 1), dtype="float32") = lv2[1]
                lv6: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv5,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i4 * 4 + i1, i2, i3), index_dtype="int32"
                    ),
                )
                gv4: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                ) = (lv4, lv6)
                R.output(gv4)
            return gv4

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})
    verify(Input, Expected_N4cHWC, {"relax.nn.conv2d": ["N4cHWC", "O4oHWI"]})


def test_conv2d_relu_concat_split_sub_indexed_div_exception():
    @I.ir_module
    class Input:
        @R.function
        def main(x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 8, 26, 26), "float32") = R.concat((gv, gv2), axis=1)
                gv4 = R.split(gv3, indices_or_sections=4, axis=1)
                R.output(gv4)
            return gv4

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 2, 26, 26), dtype="float32"),
            R.Tensor((2, 2, 26, 26), dtype="float32"),
            R.Tensor((2, 2, 26, 26), dtype="float32"),
            R.Tensor((2, 2, 26, 26), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                gv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.relu(gv)
                gv3: R.Tensor((2, 2, 26, 26, 4), dtype="float32") = R.concat((gv, gv2), axis=1)
                lv2: R.Tensor((2, 8, 26, 26), dtype="float32") = R.layout_transform(
                    gv3,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv4: R.Tuple(
                    R.Tensor((2, 2, 26, 26), dtype="float32"),
                    R.Tensor((2, 2, 26, 26), dtype="float32"),
                    R.Tensor((2, 2, 26, 26), dtype="float32"),
                    R.Tensor((2, 2, 26, 26), dtype="float32"),
                ) = R.split(lv2, indices_or_sections=4, axis=1)
                R.output(gv4)
            return gv4

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_maxpool2d_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
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
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 13, 13), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 13, 13, 4), dtype="float32") = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    layout="NCHW4c",
                    out_layout="NCHW4c",
                )
                gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 13, 13), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 13, 13, 1, 4), dtype="float32") = R.nn.max_pool2d(
                    gv,
                    pool_size=[2, 2],
                    strides=[2, 2],
                    padding=[0, 0, 0, 0],
                    layout="NHWC4c",
                    out_layout="NHWC4c",
                )
                gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_avgpool2d_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.adaptive_avg_pool2d(gv, output_size=[13, 13], layout="NCHW")
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 13, 13), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 13, 13, 4), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    gv, output_size=[13, 13], layout="NCHW4c", out_layout="NCHW4c"
                )
                gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NHWC4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 13, 13), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 28, 28, 4, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i2, i3, i1 // 4, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 3, 3, 16, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i2, i3, i1, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 26, 26, 1, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NHWC4c",
                    kernel_layout="OHWI4o",
                    out_layout="NHWC4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 13, 13, 1, 4), dtype="float32") = R.nn.adaptive_avg_pool2d(
                    gv, output_size=[13, 13], layout="NHWC4c", out_layout="NHWC4c"
                )
                gv2: R.Tensor((2, 4, 13, 13), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i3 * 4 + i4, i1, i2), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})
    verify(Input, Expected_NHWC4c, {"relax.nn.conv2d": ["NHWC4c", "OHWI4o"]})


def test_conv2d_softmax_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.nn.softmax(gv, axis=1)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.nn.softmax(lv2, axis=1)
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_batchnorm_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
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
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
            gamma: R.Tensor((4,), dtype="float32"),
            beta: R.Tensor((4,), dtype="float32"),
            moving_mean: R.Tensor((4,), dtype="float32"),
            moving_var: R.Tensor((4,), dtype="float32"),
        ) -> R.Tuple(
            R.Tensor((2, 4, 26, 26), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
            R.Tensor((4,), dtype="float32"),
        ):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tuple(
                    R.Tensor((2, 4, 26, 26), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                    R.Tensor((4,), dtype="float32"),
                ) = R.nn.batch_norm(
                    lv2,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    axis=1,
                    epsilon=1.0000000000000001e-05,
                    center=True,
                    scale=True,
                    momentum=0.10000000000000001,
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_layernorm_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
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
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
            gamma: R.Tensor((26, 26), dtype="float32"),
            beta: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.layer_norm(
                    gv,
                    gamma,
                    beta,
                    axes=[2, 3],
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_resize2d_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = R.image.resize2d(gv, (52, 52), layout="NCHW")
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 52, 52), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 4, 52, 52), dtype="float32") = R.image.resize2d(
                    lv2,
                    R.shape([52, 52]),
                    layout="NCHW",
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_conv2d_unknown_bias_dim_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            w2: R.Tensor(dtype="float32"),
        ) -> R.Tensor(None, "float32"):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2 = w2 + gv
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
            w2: R.Tensor(dtype="float32"),
        ) -> R.Tensor(dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor(dtype="float32") = R.add(w2, lv2)
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_binary_broadcast_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"),
            w: R.Tensor((4, 16, 3, 3), "float32"),
            bias: R.Tensor((26, 26), "float32"),
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, bias)
                R.output(gv2)
            return gv2

    @tvm.script.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    gv,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(lv2, bias)
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


def test_binary_ewise_scalar_sub_indexed():
    @I.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), "float32"), w: R.Tensor((4, 16, 3, 3), "float32")
        ) -> R.Tensor(None, "float32", ndim=4):
            with R.dataflow():
                gv: R.Tensor((2, 4, 26, 26), "float32") = R.nn.conv2d(x, w, out_dtype="float32")
                gv2: R.Tensor((2, 4, 26, 26), "float32") = R.add(gv, R.const(1, "float32"))
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected_NCHW4c:
        @R.function
        def main(
            x: R.Tensor((2, 16, 28, 28), dtype="float32"),
            w: R.Tensor((4, 16, 3, 3), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4, 28, 28, 4), dtype="float32") = R.layout_transform(
                    x,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0, i1 // 4, i2, i3, i1 % 4), index_dtype="int32"
                    ),
                )
                lv1: R.Tensor((1, 16, 3, 3, 4), dtype="float32") = R.layout_transform(
                    w,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3: (i0 // 4, i1, i2, i3, i0 % 4), index_dtype="int32"
                    ),
                )
                gv: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    lv,
                    lv1,
                    data_layout="NCHW4c",
                    kernel_layout="OIHW4o",
                    out_layout="NCHW4c",
                    out_dtype="float32",
                )
                lv2: R.Tensor((2, 1, 26, 26, 4), dtype="float32") = R.add(
                    gv, R.const(1.0, "float32")
                )
                gv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.layout_transform(
                    lv2,
                    index_map=T.index_map(
                        lambda i0, i1, i2, i3, i4: (i0, i1 * 4 + i4, i2, i3), index_dtype="int32"
                    ),
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected_NCHW4c, {"relax.nn.conv2d": ["NCHW4c", "OIHW4o"]})


if __name__ == "__main__":
    tvm.testing.main()
