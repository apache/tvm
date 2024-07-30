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
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Conv2dReLUx2
        with R.dataflow():
            lv: R.Tensor(
                (1, 64, 56, 56), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu(data, weight1)
            gv: R.Tensor(
                (1, 64, 54, 54), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu1(lv, weight2)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_nn_conv2d_relax_nn_relu(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1,
                weight11,
                padding=[1, 1, 1, 1],
            )
            gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_nn_conv2d_relax_nn_relu1(
        conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv1,
                weight21,
                padding=[0, 0, 0, 0],
            )
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv2)
            R.output(gv2)
        return gv2


@tvm.script.ir_module
class Conv2dReLUx2_merged:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Conv2dReLUx2_merged
        with R.dataflow():
            gv: R.Tensor(
                (1, 64, 54, 54), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu_relax_nn_conv2d_relax_nn_relu1_dnnl(
                data, weight1, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu_relax_nn_conv2d_relax_nn_relu1_dnnl(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Codegen": "dnnl"})

        @R.function
        def lv(
            data11: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight111: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "dnnl.conv2d_relu"})
            with R.dataflow():
                lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    data11,
                    weight111,
                    padding=[1, 1, 1, 1],
                )
                gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
                R.output(gv1)
            return gv1

        lv2: R.Tensor((1, 64, 56, 56), dtype="float32") = lv(data1, weight11)

        @R.function
        def lv11(
            conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight211: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            R.func_attr({"Composite": "dnnl.conv2d_relu"})
            with R.dataflow():
                lv21: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                    conv1,
                    weight211,
                    padding=[0, 0, 0, 0],
                )
                gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv21)
                R.output(gv2)
            return gv2

        gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = lv11(lv2, weight21)
        return gv3


@tvm.script.ir_module
class Diamond:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Diamond
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_conv2d(
                data, weight
            )
            lv3: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_relu(lv2)
            lv4: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_gelu(lv2)
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_add(lv3, lv4)
            R.output(gv2)
        return gv2

    @R.function(private=True)
    def fused_relax_nn_gelu(
        lv: R.Tensor((1, 64, 54, 54), dtype="float32")
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.gelu"})
        with R.dataflow():
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_nn_relu(
        lv1: R.Tensor((1, 64, 54, 54), dtype="float32")
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
        with R.dataflow():
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_add(
        lv5: R.Tensor((1, 64, 54, 54), dtype="float32"),
        gelu1: R.Tensor((1, 64, 54, 54), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
        with R.dataflow():
            gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(lv5, gelu1)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.conv2d"})
        with R.dataflow():
            gv4: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                data1,
                weight1,
                padding=[0, 0, 0, 0],
            )
            R.output(gv4)
        return gv4


@tvm.script.ir_module
class Diamond_merged:
    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu_relax_nn_gelu_relax_add_compiler_A(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        # function attr dict
        R.func_attr({"Codegen": "compiler_A"})

        # block 0
        @R.function
        def lv(
            data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.conv2d"})
            # block 0
            with R.dataflow():
                gv4: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                    data1,
                    weight1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NCHW",
                    kernel_layout="OIHW",
                    out_layout="NCHW",
                    out_dtype="",
                )
                R.output(gv4)
            return gv4

        lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = lv(data, weight)

        @R.function
        def lv1(
            lv11: R.Tensor((1, 64, 54, 54), dtype="float32")
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.relu"})
            # block 0
            with R.dataflow():
                gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv11)
                R.output(gv1)
            return gv1

        lv3: R.Tensor((1, 64, 54, 54), dtype="float32") = lv1(lv2)

        @R.function
        def lv21(
            lv4: R.Tensor((1, 64, 54, 54), dtype="float32")
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.gelu"})
            # block 0
            with R.dataflow():
                gv: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv4)
                R.output(gv)
            return gv

        lv41: R.Tensor((1, 64, 54, 54), dtype="float32") = lv21(lv2)

        @R.function
        def lv31(
            lv5: R.Tensor((1, 64, 54, 54), dtype="float32"),
            gelu1: R.Tensor((1, 64, 54, 54), dtype="float32"),
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.add"})
            # block 0
            with R.dataflow():
                gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(lv5, gelu1)
                R.output(gv3)
            return gv3

        gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = lv31(lv3, lv41)
        return gv2

    @R.function
    def main(
        data2: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Diamond_merged
        with R.dataflow():
            gv5: R.Tensor(
                (1, 64, 54, 54), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu_relax_nn_gelu_relax_add_compiler_A(
                data2, weight2
            )
            R.output(gv5)
        return gv5


@tvm.script.ir_module
class Diamond_cyclic_dep:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Diamond_cyclic_dep
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_conv2d(
                data, weight
            )
            lv3: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_relu(lv2)
            lv4: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_gelu(lv2)
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_add(lv3, lv4)
            R.output(gv2)
        return gv2

    @R.function(private=True)
    def fused_relax_nn_gelu(
        lv: R.Tensor((1, 64, 54, 54), dtype="float32")
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_B.gelu"})
        with R.dataflow():
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_nn_relu(
        lv1: R.Tensor((1, 64, 54, 54), dtype="float32")
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
        with R.dataflow():
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_add(
        lv5: R.Tensor((1, 64, 54, 54), dtype="float32"),
        gelu1: R.Tensor((1, 64, 54, 54), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
        with R.dataflow():
            gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(lv5, gelu1)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.conv2d"})
        with R.dataflow():
            gv4: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                data1,
                weight1,
                padding=[0, 0, 0, 0],
            )
            R.output(gv4)
        return gv4


@tvm.script.ir_module
class Diamond_cyclic_dep_merged:
    @R.function
    def main(
        data2: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Diamond_cyclic_dep_merged
        with R.dataflow():
            lv4: R.Tuple(
                R.Tensor((1, 64, 54, 54), dtype="float32"),
                R.Tensor((1, 64, 54, 54), dtype="float32"),
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu_compiler_A(data2, weight2)
            lv12: R.Tensor((1, 64, 54, 54), dtype="float32") = lv4[0]
            lv22: R.Tensor((1, 64, 54, 54), dtype="float32") = lv4[1]
            lv31: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_gelu1_compiler_B(
                lv12
            )
            gv5: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_add1_compiler_A(
                lv22, lv31
            )
            R.output(gv5)
        return gv5

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu_compiler_A(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((1, 64, 54, 54), dtype="float32"), R.Tensor((1, 64, 54, 54), dtype="float32")
    ):
        R.func_attr({"Codegen": "compiler_A"})

        @R.function
        def lv(
            data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.conv2d"})
            with R.dataflow():
                gv4: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                    data1,
                    weight1,
                    padding=[0, 0, 0, 0],
                )
                R.output(gv4)
            return gv4

        gv: R.Tensor((1, 64, 54, 54), dtype="float32") = lv(data, weight)

        @R.function
        def lv1(
            lv11: R.Tensor((1, 64, 54, 54), dtype="float32")
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.relu"})
            with R.dataflow():
                gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv11)
                R.output(gv1)
            return gv1

        gv11: R.Tensor((1, 64, 54, 54), dtype="float32") = lv1(gv)

        return (gv, gv11)

    @R.function
    def fused_relax_nn_gelu1_compiler_B(
        lv2: R.Tensor((1, 64, 54, 54), dtype="float32")
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Codegen": "compiler_B"})

        @R.function
        def lv21(
            lv3: R.Tensor((1, 64, 54, 54), dtype="float32")
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            R.func_attr({"Composite": "compiler_B.gelu"})
            with R.dataflow():
                gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv3)
                R.output(gv2)
            return gv2

        gv3: R.Tensor((1, 64, 54, 54), dtype="float32") = lv21(lv2)
        return gv3

    @R.function
    def fused_relax_add1_compiler_A(
        lv32: R.Tensor((1, 64, 54, 54), dtype="float32"),
        lv41: R.Tensor((1, 64, 54, 54), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Codegen": "compiler_A"})

        @R.function
        def lv33(
            lv5: R.Tensor((1, 64, 54, 54), dtype="float32"),
            gelu1: R.Tensor((1, 64, 54, 54), dtype="float32"),
        ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.add"})
            with R.dataflow():
                gv31: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(lv5, gelu1)
                R.output(gv31)
            return gv31

        gv6: R.Tensor((1, 64, 54, 54), dtype="float32") = lv33(lv32, lv41)
        return gv6


@tvm.script.ir_module
class MultipleProducers:
    @R.function
    def main(
        x1: R.Tensor((10,), dtype="float32"), x2: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        cls = MultipleProducers
        with R.dataflow():
            lv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(x1)
            lv2: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(x2)
            lv3: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(lv1)
            lv4: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(lv2)
            gv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv3, lv4)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_nn_relu(
        x11: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
        with R.dataflow():
            gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(x11)
            R.output(gv2)
        return gv2

    @R.function(private=True)
    def fused_relax_nn_gelu(
        x21: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.gelu"})
        with R.dataflow():
            gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x21)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_add(
        lv: R.Tensor((10,), dtype="float32"), gelu1: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
        with R.dataflow():
            gv: R.Tensor((10,), dtype="float32") = R.add(lv, gelu1)
            R.output(gv)
        return gv


@tvm.script.ir_module
class MultipleProducers_merged:
    @R.function
    def fused_relax_nn_relu_relax_nn_gelu_relax_nn_relu_relax_nn_gelu_relax_add_compiler_A(
        x1: R.Tensor((10,), dtype="float32"), x2: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        # function attr dict
        R.func_attr({"Codegen": "compiler_A"})

        # block 0
        @R.function
        def lv(x11: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.relu"})
            # block 0
            with R.dataflow():
                gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(x11)
                R.output(gv2)
            return gv2

        lv1: R.Tensor((10,), dtype="float32") = lv(x1)

        @R.function
        def lv11(x21: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.gelu"})
            # block 0
            with R.dataflow():
                gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x21)
                R.output(gv3)
            return gv3

        lv2: R.Tensor((10,), dtype="float32") = lv11(x2)
        lv3: R.Tensor((10,), dtype="float32") = lv(lv1)
        lv4: R.Tensor((10,), dtype="float32") = lv11(lv2)

        @R.function
        def lv21(
            lv5: R.Tensor((10,), dtype="float32"), gelu1: R.Tensor((10,), dtype="float32")
        ) -> R.Tensor((10,), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.add"})
            # block 0
            with R.dataflow():
                gv: R.Tensor((10,), dtype="float32") = R.add(lv5, gelu1)
                R.output(gv)
            return gv

        gv1: R.Tensor((10,), dtype="float32") = lv21(lv3, lv4)
        return gv1

    @R.function
    def main(
        x12: R.Tensor((10,), dtype="float32"), x22: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        cls = MultipleProducers_merged
        with R.dataflow():
            gv4: R.Tensor(
                (10,), dtype="float32"
            ) = cls.fused_relax_nn_relu_relax_nn_gelu_relax_nn_relu_relax_nn_gelu_relax_add_compiler_A(
                x12, x22
            )
            R.output(gv4)
        return gv4


@tvm.script.ir_module
class MultipleProducersCyclic:
    @R.function
    def main(x1: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        cls = MultipleProducersCyclic
        with R.dataflow():
            lv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(x1)
            lv2: R.Tensor((10,), dtype="float32") = R.nn.relu(lv1)
            lv3: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(lv2)
            gv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv1, lv3)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_nn_relu(
        x11: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
        with R.dataflow():
            gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(x11)
            R.output(gv2)
        return gv2

    @R.function(private=True)
    def fused_relax_nn_gelu(
        x21: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.gelu"})
        with R.dataflow():
            gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x21)
            R.output(gv3)
        return gv3

    @R.function(private=True)
    def fused_relax_add(
        lv: R.Tensor((10,), dtype="float32"), gelu1: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
        with R.dataflow():
            gv: R.Tensor((10,), dtype="float32") = R.add(lv, gelu1)
            R.output(gv)
        return gv


@tvm.script.ir_module
class MultipleProducersCyclic_merged:
    @R.function
    def main(x1: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
        cls = MultipleProducersCyclic_merged
        with R.dataflow():
            lv: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu1_compiler_A(x1)
            lv2: R.Tensor((10,), dtype="float32") = R.nn.relu(lv)
            gv: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu_relax_add_compiler_A(
                lv2, lv
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_relu1_compiler_A(
        x11: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        # function attr dict
        R.func_attr({"Codegen": "compiler_A"})

        # block 0
        @R.function
        def lv1(x111: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.relu"})
            # block 0
            with R.dataflow():
                gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(x111)
                R.output(gv2)
            return gv2

        gv1: R.Tensor((10,), dtype="float32") = lv1(x11)
        return gv1

    @R.function
    def fused_relax_nn_gelu_relax_add_compiler_A(
        lv21: R.Tensor((10,), dtype="float32"), lv11: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        # function attr dict
        R.func_attr({"Codegen": "compiler_A"})
        # block 0

        @R.function
        def lv12(x21: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.gelu"})
            # block 0
            with R.dataflow():
                gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x21)
                R.output(gv3)
            return gv3

        lv3: R.Tensor((10,), dtype="float32") = lv12(lv21)

        @R.function
        def lv22(
            lv4: R.Tensor((10,), dtype="float32"), gelu1: R.Tensor((10,), dtype="float32")
        ) -> R.Tensor((10,), dtype="float32"):
            # function attr dict
            R.func_attr({"Composite": "compiler_A.add"})
            # block 0
            with R.dataflow():
                gv4: R.Tensor((10,), dtype="float32") = R.add(lv4, gelu1)
                R.output(gv4)
            return gv4

        gv5: R.Tensor((10,), dtype="float32") = lv22(lv11, lv3)

        return gv5


@tvm.script.ir_module
class MergeCompilerRegionsExample:
    @R.function
    def main(
        x1: R.Tensor((10,), dtype="float32"),
        x2: R.Tensor((10,), dtype="float32"),
        x3: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((10,), dtype="float32"):
        cls = MergeCompilerRegionsExample
        with R.dataflow():
            lv: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(x1, x2)
            lv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(x3)
            lv11: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv, lv1)
            lv12: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu(lv11)
            lv2: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(lv11)
            lv21: R.Tensor((10,), dtype="float32") = cls.fused_relax_add(lv12, lv2)
            gv1: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_relu(lv21)
            R.output(gv1)
        return gv1

    @R.function(private=True)
    def fused_relax_nn_relu(
        add2: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.relu"})
        with R.dataflow():
            gv: R.Tensor((10,), dtype="float32") = R.nn.relu(add2)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_relax_add(
        x11: R.Tensor((10,), dtype="float32"), x21: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_A.add"})
        with R.dataflow():
            gv2: R.Tensor((10,), dtype="float32") = R.add(x11, x21)
            R.output(gv2)
        return gv2

    @R.function(private=True)
    def fused_relax_nn_gelu(
        x31: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "compiler_B.gelu"})
        with R.dataflow():
            gv3: R.Tensor((10,), dtype="float32") = R.nn.gelu(x31)
            R.output(gv3)
        return gv3


@tvm.script.ir_module
class MergeCompilerRegionsExampleRef:
    @R.function
    def fused_relax_add_relax_add_relax_nn_relu_compiler_A(
        x1: R.Tensor((10,), dtype="float32"),
        x2: R.Tensor((10,), dtype="float32"),
        lv: R.Tensor((10,), dtype="float32"),
    ) -> R.Tuple(R.Tensor((10,), dtype="float32"), R.Tensor((10,), dtype="float32")):
        R.func_attr({"Codegen": "compiler_A"})

        @R.function
        def lv1(
            x11: R.Tensor((10,), dtype="float32"), x21: R.Tensor((10,), dtype="float32")
        ) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.add"})
            with R.dataflow():
                gv: R.Tensor((10,), dtype="float32") = R.add(x11, x21)
                R.output(gv)
            return gv

        lv2: R.Tensor((10,), dtype="float32") = lv1(x1, x2)
        gv1: R.Tensor((10,), dtype="float32") = lv1(lv2, lv)

        @R.function
        def lv11(add2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.relu"})
            with R.dataflow():
                gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(add2)
                R.output(gv2)
            return gv2

        gv11: R.Tensor((10,), dtype="float32") = lv11(gv1)
        return (gv1, gv11)

    @R.function
    def fused_relax_add_relax_nn_relu_compiler_A(
        lv12: R.Tensor((10,), dtype="float32"), lv3: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Codegen": "compiler_A"})

        @R.function
        def lv21(
            x11: R.Tensor((10,), dtype="float32"), x21: R.Tensor((10,), dtype="float32")
        ) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.add"})
            with R.dataflow():
                gv: R.Tensor((10,), dtype="float32") = R.add(x11, x21)
                R.output(gv)
            return gv

        lv22: R.Tensor((10,), dtype="float32") = lv21(lv12, lv3)

        @R.function
        def lv31(add2: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"Composite": "compiler_A.relu"})
            with R.dataflow():
                gv2: R.Tensor((10,), dtype="float32") = R.nn.relu(add2)
                R.output(gv2)
            return gv2

        gv3: R.Tensor((10,), dtype="float32") = lv31(lv22)

        return gv3

    @R.function
    def fused_relax_nn_gelu1_compiler_B(
        x3: R.Tensor((10,), dtype="float32")
    ) -> R.Tensor((10,), dtype="float32"):
        R.func_attr({"Codegen": "compiler_B"})

        @R.function
        def lv4(x31: R.Tensor((10,), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"Composite": "compiler_B.gelu"})
            with R.dataflow():
                gv4: R.Tensor((10,), dtype="float32") = R.nn.gelu(x31)
                R.output(gv4)
            return gv4

        gv5: R.Tensor((10,), dtype="float32") = lv4(x3)
        return gv5

    @R.function
    def main(
        x12: R.Tensor((10,), dtype="float32"),
        x22: R.Tensor((10,), dtype="float32"),
        x32: R.Tensor((10,), dtype="float32"),
    ) -> R.Tensor((10,), dtype="float32"):
        cls = MergeCompilerRegionsExampleRef
        with R.dataflow():
            lv5: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu1_compiler_B(x32)
            lv13: R.Tuple(
                R.Tensor((10,), dtype="float32"), R.Tensor((10,), dtype="float32")
            ) = cls.fused_relax_add_relax_add_relax_nn_relu_compiler_A(x12, x22, lv5)
            lv23: R.Tensor((10,), dtype="float32") = lv13[0]
            lv32: R.Tensor((10,), dtype="float32") = lv13[1]
            lv41: R.Tensor((10,), dtype="float32") = cls.fused_relax_nn_gelu1_compiler_B(lv23)
            gv6: R.Tensor((10,), dtype="float32") = cls.fused_relax_add_relax_nn_relu_compiler_A(
                lv41, lv32
            )
            R.output(gv6)
        return gv6


@tvm.script.ir_module
class ModuleWithNonComposite:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        cls = ModuleWithNonComposite
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(data, weight)
            conv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
            R.output(conv)
        return conv

    @R.function(private=True)
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Composite": "tensorrt.conv2d", "Primitive": 1})
        with R.dataflow():
            gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1,
                weight1,
                padding=[1, 1, 1, 1],
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ModuleWithNonComposite_ref:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        cls = ModuleWithNonComposite_ref
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d1_tensorrt(
                data, weight
            )
            conv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
            R.output(conv)
        return conv

    @R.function
    def fused_relax_nn_conv2d1_tensorrt(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Codegen": "tensorrt"})

        @R.function
        def lv1(
            data2: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "tensorrt.conv2d"})
            with R.dataflow():
                gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    data2,
                    weight2,
                    padding=[1, 1, 1, 1],
                )
                R.output(gv)
            return gv

        gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = lv1(data1, weight1)
        return gv1


def check(mod, expected):
    partitioned = relax.transform.MergeCompositeFunctions()(mod)
    tvm.ir.assert_structural_equal(partitioned, expected)


def test_conv2d_relu_x2():
    check(Conv2dReLUx2, Conv2dReLUx2_merged)


def test_diamond_cyclic_dep():
    """
    O = Offloaded to A
    X = Offloaded to B

       O         O
      / \\      /               \\
     O   X --> O    +       +    X
     \\ /             \\ /
       O                O

    We cannot merge all 'O' since it would create a cyclic dependency between the group of `X`.
    """
    check(Diamond_cyclic_dep, Diamond_cyclic_dep_merged)


def test_diamond():
    """
    O = Offloaded to A

       O         O
      / \\      / \\
     O   O --> O   O
     \\ /      \\ /
       O         O

    """
    check(Diamond, Diamond_merged)


def test_merge_producers():
    """
    Test merging multiple producer groups into a single representative group.
     O   O
     |   |
     O   O
     \\ /
       O
    """
    check(MultipleProducers, MultipleProducers_merged)


def test_merge_producers_cyclic_dep():
    """
    Test when multiple producer groups being blocked to merge due to circular dependency
    in the result.
       O
       |\\
       | X
       | |
       | O
       |/
       O
    """
    check(MultipleProducersCyclic, MultipleProducersCyclic_merged)


def test_merge_compiler_regions_example():
    """
    A tricky example from https://discuss.tvm.apache.org/t/relay-improved-graph-partitioning-algorithm/5830
    See also the corresponding test case for Relay MergeCompilerRegions in relay/test_pass_merge_compiler_regions.py.
    """
    check(
        MergeCompilerRegionsExample,
        MergeCompilerRegionsExampleRef,
    )


def test_mixed_non_composite():
    check(ModuleWithNonComposite, ModuleWithNonComposite_ref)


def test_reshape():
    # Verify that the non-CallNode input (shape in reshape) can be handled properly.
    @I.ir_module
    class Module:
        @R.function(private=True)
        def fused_relax_matmul(
            lv: R.Tensor((1, 784), dtype="float32"), lv1: R.Tensor((784, 512), dtype="float32")
        ) -> R.Tensor((1, 512), dtype="float32"):
            R.func_attr({"Composite": "tensorrt.matmul", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((1, 512), dtype="float32") = R.matmul(lv, lv1, out_dtype="float32")
                R.output(gv)
            return gv

        @R.function(private=True)
        def fused_relax_reshape(
            inp_0: R.Tensor((1, 1, 28, 28), dtype="float32"), param_0: R.Shape([1, 784])
        ) -> R.Tensor((1, 784), dtype="float32"):
            R.func_attr({"Composite": "tensorrt.reshape", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((1, 784), dtype="float32") = R.reshape(inp_0, param_0)
                R.output(gv)
            return gv

        @R.function
        def main(
            inp_0: R.Tensor((1, 1, 28, 28), dtype="float32"),
            linear_relu_stack_0_weight: R.Tensor((512, 784), dtype="float32"),
        ) -> R.Tensor((1, 512), dtype="float32"):
            cls = Module
            with R.dataflow():
                lv: R.Tensor((1, 784), dtype="float32") = cls.fused_relax_reshape(
                    inp_0, R.shape([1, 784])
                )
                lv1: R.Tensor((784, 512), dtype="float32") = R.permute_dims(
                    linear_relu_stack_0_weight, axes=None
                )
                lv_1: R.Tensor((1, 512), dtype="float32") = cls.fused_relax_matmul(lv, lv1)
                gv: R.Tensor((1, 512), dtype="float32") = lv_1
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def fused_relax_reshape_relax_matmul_tensorrt(
            inp_0: R.Tensor((1, 1, 28, 28), dtype="float32"),
            lv1: R.Tensor((784, 512), dtype="float32"),
        ) -> R.Tensor((1, 512), dtype="float32"):
            R.func_attr({"Codegen": "tensorrt"})
            # from tvm.script import relax as R

            @R.function
            def lv_1(
                inp_0_1: R.Tensor((1, 1, 28, 28), dtype="float32"), param_0_1: R.Shape([1, 784])
            ) -> R.Tensor((1, 784), dtype="float32"):
                R.func_attr({"Composite": "tensorrt.reshape"})
                with R.dataflow():
                    gv: R.Tensor((1, 784), dtype="float32") = R.reshape(inp_0_1, param_0_1)
                    R.output(gv)
                return gv

            lv_1: R.Tensor((1, 784), dtype="float32") = lv_1(inp_0, R.shape([1, 784]))

            @R.function
            def lv1_1_1(
                lv_2: R.Tensor((1, 784), dtype="float32"),
                lv1_2: R.Tensor((784, 512), dtype="float32"),
            ) -> R.Tensor((1, 512), dtype="float32"):
                R.func_attr({"Composite": "tensorrt.matmul"})
                with R.dataflow():
                    gv: R.Tensor((1, 512), dtype="float32") = R.matmul(
                        lv_2, lv1_2, out_dtype="float32"
                    )
                    R.output(gv)
                return gv

            lv_2: R.Tensor((1, 512), dtype="float32") = lv1_1_1(lv_1, lv1)
            gv: R.Tensor((1, 512), dtype="float32") = lv_2
            return gv

        @R.function
        def main(
            inp_0: R.Tensor((1, 1, 28, 28), dtype="float32"),
            linear_relu_stack_0_weight: R.Tensor((512, 784), dtype="float32"),
        ) -> R.Tensor((1, 512), dtype="float32"):
            cls = Expected
            with R.dataflow():
                lv1: R.Tensor((784, 512), dtype="float32") = R.permute_dims(
                    linear_relu_stack_0_weight, axes=None
                )
                gv: R.Tensor(
                    (1, 512), dtype="float32"
                ) = cls.fused_relax_reshape_relax_matmul_tensorrt(inp_0, lv1)
                R.output(gv)
            return gv

    check(Module, Expected)


def test_handle_existence_of_call_tir():
    """MergeCompositeFunctions should accept R.call_tir as input

    No merging is required in this case, since the two composite
    functions have `R.call_tir` between them.  This is a regression
    test, as previously the `Tuple` used to express of `R.call_tir`
    caused a segfault.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(A: R.Tensor([10], dtype="float32")) -> R.Tensor([10], dtype="float32"):
            cls = Before
            with R.dataflow():
                B = cls.fused_relax_nn_relu(A)
                C = R.call_tir(cls.relu, (B,), out_sinfo=R.Tensor([10], dtype="float32"))
                D = cls.fused_relax_nn_gelu(C)
                R.output(D)
            return D

        @R.function(private=True)
        def fused_relax_nn_relu(
            Input: R.Tensor([10], dtype="float32")
        ) -> R.Tensor([10], dtype="float32"):
            R.func_attr({"Composite": "compiler_A.relu", "Primitive": 1})
            with R.dataflow():
                Output = R.nn.relu(Input)
                R.output(Output)
            return Output

        @T.prim_func(private=True)
        def relu(
            Input: T.Buffer(T.int64(10), "float32"),
            Output: T.Buffer(T.int64(10), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for i in range(T.int64(10)):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    Output[vi] = T.max(Input[vi], T.float32(0))

        @R.function(private=True)
        def fused_relax_nn_gelu(
            Input: R.Tensor([10], dtype="float32")
        ) -> R.Tensor([10], dtype="float32"):
            R.func_attr({"Composite": "compiler_A.gelu", "Primitive": 1})
            with R.dataflow():
                Output = R.nn.gelu(Input)
                R.output(Output)
            return Output

    @I.ir_module
    class Expected:
        @R.function
        def main(A: R.Tensor([10], dtype="float32")) -> R.Tensor([10], dtype="float32"):
            cls = Expected
            with R.dataflow():
                B = cls.fused_relax_nn_relu1_compiler_A(A)
                C = R.call_tir(cls.relu, (B,), out_sinfo=R.Tensor([10], dtype="float32"))
                D = cls.fused_relax_nn_gelu1_compiler_A(C)
                R.output(D)
            return D

        @R.function
        def fused_relax_nn_relu1_compiler_A(
            Input: R.Tensor([10], dtype="float32")
        ) -> R.Tensor([10], dtype="float32"):
            R.func_attr({"Codegen": "compiler_A"})

            @R.function
            def composite_lambda(
                Input: R.Tensor([10], dtype="float32")
            ) -> R.Tensor([10], dtype="float32"):
                R.func_attr({"Composite": "compiler_A.relu"})
                with R.dataflow():
                    Output = R.nn.relu(Input)
                    R.output(Output)
                return Output

            Output = composite_lambda(Input)
            return Output

        @T.prim_func(private=True)
        def relu(
            Input: T.Buffer(T.int64(10), "float32"),
            Output: T.Buffer(T.int64(10), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for i in range(T.int64(10)):
                with T.block("compute"):
                    vi = T.axis.remap("S", [i])
                    Output[vi] = T.max(Input[vi], T.float32(0))

        @R.function
        def fused_relax_nn_gelu1_compiler_A(
            Input: R.Tensor([10], dtype="float32")
        ) -> R.Tensor([10], dtype="float32"):
            R.func_attr({"Codegen": "compiler_A"})

            @R.function
            def composite_lambda(
                Input: R.Tensor([10], dtype="float32")
            ) -> R.Tensor([10], dtype="float32"):
                R.func_attr({"Composite": "compiler_A.gelu"})
                with R.dataflow():
                    Output = R.nn.gelu(Input)
                    R.output(Output)
                return Output

            Output = composite_lambda(Input)
            return Output

    After = relax.transform.MergeCompositeFunctions()(Before)
    tvm.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    pytest.main([__file__])
