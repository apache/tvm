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
import numpy as np
import pytest

import tvm
from tvm import relax
from tvm.relax.dpl.pattern import is_op, make_fused_bias_activation_pattern, wildcard
from tvm.relax.transform import PatternCheckContext
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.script.ir_module
class Conv2dReLU:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.relu(R.nn.conv2d(data, weight1, padding=(1, 1)))
            R.output(conv1)

        return conv1


@tvm.script.ir_module
class Conv2dReLU_composite_annotated:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        cls = Conv2dReLU_composite_annotated
        with R.dataflow():
            gv: R.Tensor(
                (1, 64, 56, 56), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu_dnnl(data, weight1)
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu_dnnl(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr(
            {"Codegen": "dnnl", "global_symbol": "fused_relax_nn_conv2d_relax_nn_relu_dnnl"}
        )

        @R.function
        def gv1(
            data2: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight12: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
            with R.dataflow():
                lv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    data2,
                    weight12,
                    padding=[1, 1, 1, 1],
                )
                gv2: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
                R.output(gv2)
            return gv2

        gv11: R.Tensor((1, 64, 56, 56), dtype="float32") = gv1(data1, weight11)
        return gv11


@tvm.script.ir_module
class Conv2dReLUx2:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.relu(R.nn.conv2d(data, weight1, padding=(1, 1)))
            conv2 = R.nn.relu(R.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2)

        return conv2


@tvm.script.ir_module
class Conv2dReLUx2Partitioned:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Conv2dReLUx2Partitioned
        with R.dataflow():
            lv: R.Tensor(
                (1, 64, 56, 56), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu(data, weight1)
            gv: R.Tensor(
                (1, 64, 54, 54), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu1(lv, weight2)
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1, weight11, padding=[1, 1, 1, 1]
            )
            gv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu1(
        conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv1, weight21, padding=[0, 0, 0, 0]
            )
            gv2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv2)
            R.output(gv2)
        return gv2


@tvm.script.ir_module
class Conv2dReLUx2Partitioned_only_conv2d:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Conv2dReLUx2Partitioned_only_conv2d
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(
                data, weight1
            )
            conv1: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(lv)
            lv1: R.Tensor((1, 64, 54, 54), dtype="float32") = cls.fused_relax_nn_conv2d1(
                conv1, weight2
            )
            conv2d: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(conv2d)
        return conv2d

    @R.function
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d"})
        with R.dataflow():
            gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1, weight11, padding=[1, 1, 1, 1]
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d1(
        conv11: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d"})
        with R.dataflow():
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv11, weight21, padding=[0, 0, 0, 0]
            )
            R.output(gv1)
        return gv1


@tvm.script.ir_module
class Conv2dConv2dReLU:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight1: R.Tensor((64, 64, 3, 3), "float32"),
        weight2: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.conv2d(data, weight1, padding=(1, 1))
            conv2d = R.nn.relu(R.nn.conv2d(conv1, weight2, padding=(0, 0)))
            R.output(conv2d)

        return conv2d


@tvm.script.ir_module
class Conv2dConv2dReLUPartitioned:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        weight2: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        cls = Conv2dConv2dReLUPartitioned
        with R.dataflow():
            lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(
                data, weight1
            )
            gv: R.Tensor(
                (1, 64, 54, 54), dtype="float32"
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu(lv, weight2)
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu(
        conv1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight21: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            lv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(
                conv1, weight21, padding=[0, 0, 0, 0]
            )
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(lv1)
            R.output(gv1)
        return gv1

    @R.function
    def fused_relax_nn_conv2d(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight11: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d"})
        with R.dataflow():
            gv2: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                data1, weight11, padding=[1, 1, 1, 1]
            )
            R.output(gv2)
        return gv2


@tvm.script.ir_module
class BranchTupleOutput:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.conv2d(data, weight)
            relu1 = R.nn.relu(conv1)
            gelu1 = R.nn.gelu(relu1)
            gelu2 = R.nn.gelu(conv1)
            out = relax.op.add(gelu1, gelu2)
            R.output(out)

        return out


@tvm.script.ir_module
class BranchTupleOutputPartitioned:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 64, 54, 54), dtype="float32"):
        with R.dataflow():
            cls = BranchTupleOutputPartitioned
            lv: R.Tuple(
                R.Tensor((1, 64, 54, 54), dtype="float32"),
                R.Tensor((1, 64, 54, 54), dtype="float32"),
            ) = cls.fused_relax_nn_conv2d_relax_nn_relu(data, weight)
            lv1: R.Tensor((1, 64, 54, 54), dtype="float32") = lv[1]  # conv1
            lv2: R.Tensor((1, 64, 54, 54), dtype="float32") = lv[0]  # relu(conv1)
            gelu1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv2)
            gelu2: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.gelu(lv1)
            out: R.Tensor((1, 64, 54, 54), dtype="float32") = R.add(gelu1, gelu2)
            R.output(out)
        return out

    @R.function
    def fused_relax_nn_conv2d_relax_nn_relu(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((1, 64, 54, 54), dtype="float32"), R.Tensor((1, 64, 54, 54), dtype="float32")
    ):
        R.func_attr({"Primitive": 1, "Composite": "dnnl.conv2d_relu"})
        with R.dataflow():
            gv: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.conv2d(data1, weight1)
            gv1: R.Tensor((1, 64, 54, 54), dtype="float32") = R.nn.relu(gv)
            R.output(gv, gv1)
        return (gv1, gv)


@tvm.script.ir_module
class Branch:
    @R.function
    def main(
        data: R.Tensor((1, 64, 56, 56), "float32"),
        weight: R.Tensor((64, 64, 3, 3), "float32"),
    ):
        with R.dataflow():
            conv1 = R.nn.conv2d(data, weight)
            relu1 = R.nn.relu(conv1)
            gelu1 = R.nn.gelu(conv1)

            out = relax.op.add(relu1, gelu1)
            R.output(out)

        return out


@tvm.script.ir_module
class Conv2dx2:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), "float16"),
        weight1: R.Tensor((16, 3, 3, 16), "float16"),
        weight2: R.Tensor((16, 3, 3, 16), "float16"),
    ):
        with R.dataflow():
            conv1 = relax.op.nn.conv2d(
                data, weight1, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
            )
            conv2 = relax.op.nn.conv2d(
                conv1, weight2, padding=(1, 1), data_layout="NHWC", kernel_layout="OHWI"
            )
            R.output(conv2)

        return conv2


@tvm.script.ir_module
class Conv2dx2_partitioned:
    @R.function
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        cls = Conv2dx2_partitioned
        with R.dataflow():
            lv: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_cutlass(
                data, weight1
            )
            gv: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_cutlass(
                lv, weight2
            )
            R.output(gv)
        return gv

    @R.function
    def fused_relax_nn_conv2d_cutlass(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        R.func_attr({"Codegen": "cutlass", "global_symbol": "fused_relax_nn_conv2d_cutlass"})

        @R.function
        def gv(
            data_1: R.Tensor((16, 32, 32, 16), dtype="float16"),
            weight1_1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
            R.func_attr({"Composite": "cutlass.conv2d", "Primitive": 1})
            with R.dataflow():
                gv_1: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(
                    data_1,
                    weight1_1,
                    padding=[1, 1, 1, 1],
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                )
                R.output(gv_1)
            return gv_1

        gv1: R.Tensor((16, 32, 32, 16), dtype="float16") = gv(data, weight1)
        return gv1


conv2d_pat = make_fused_bias_activation_pattern("relax.nn.conv2d", activation=None)
conv2d_relu_pat = make_fused_bias_activation_pattern("relax.nn.conv2d", activation="relax.nn.relu")


def check(mod, patterns, expected, bind_constants=True, annotate_codegen=False):
    partitioned = relax.transform.FuseOpsByPattern(patterns, bind_constants, annotate_codegen)(mod)
    tvm.ir.assert_structural_equal(partitioned, expected)


def test_partition_conv2d_relu():
    check(Conv2dReLUx2, [("dnnl.conv2d_relu", conv2d_relu_pat)], Conv2dReLUx2Partitioned)


def test_partition_multiple_patterns():
    check(
        Conv2dConv2dReLU,
        [("dnnl.conv2d_relu", conv2d_relu_pat), ("dnnl.conv2d", conv2d_pat)],
        Conv2dConv2dReLUPartitioned,
    )


def test_partition_order():
    check(
        Conv2dReLUx2,
        [("dnnl.conv2d", conv2d_pat), ("dnnl.conv2d_relu", conv2d_relu_pat)],
        Conv2dReLUx2Partitioned_only_conv2d,
    )


def test_branch_tuple_output():
    check(BranchTupleOutput, [("dnnl.conv2d_relu", conv2d_relu_pat)], BranchTupleOutputPartitioned)


def test_cyclic_dependency():
    conv_pat = make_fused_bias_activation_pattern("relax.nn.conv2d")
    relu_pat = is_op("relax.nn.relu")(conv_pat)
    add_pat = is_op("relax.add")(relu_pat, wildcard())

    with pytest.raises(tvm.error.TVMError) as err:
        relax.transform.FuseOpsByPattern(
            [("compiler_A.conv2d_relu_add", add_pat)], bind_constants=True
        )(Branch)

    assert "A cyclic dependency detected" in str(err.value)


def test_bind_params():
    weight_np = np.random.randn(64, 64, 3, 3).astype("float32")
    mod = tvm.transform.Sequential(
        [
            relax.transform.BindParams("main", {"weight1": weight_np}),
            relax.transform.FuseOpsByPattern(
                [("dnnl.conv2d_relu", conv2d_relu_pat)], bind_constants=True
            ),
        ]
    )(Conv2dReLU)

    assert "fused_relax_nn_conv2d_relax_nn_relu" in [var.name_hint for var in mod.functions.keys()]

    for gvar, f in mod.functions.items():
        if gvar.name_hint == "fused_relax_nn_conv2d_relax_nn_relu":
            conv2d = f.body.blocks[0].bindings[0].value
            assert isinstance(conv2d.args[1], relax.Constant)


def test_annotate_codegen():
    check(
        Conv2dReLU,
        [("dnnl.conv2d_relu", conv2d_relu_pat)],
        Conv2dReLU_composite_annotated,
        annotate_codegen=True,
    )


def test_multiple_calls_same_extern():
    pat = make_fused_bias_activation_pattern("relax.nn.conv2d", with_bias=False, activation=None)
    check(Conv2dx2, [("cutlass.conv2d", pat)], Conv2dx2_partitioned, annotate_codegen=True)


def test_ignore_call_tir():
    @I.ir_module
    class Conv2dReLUCallTIR:
        @T.prim_func
        def relu(
            data: T.Buffer((64, 64, 56, 56), "float32"), out: T.Buffer((64, 64, 56, 56), "float32")
        ):
            for ax0, ax1, ax2, ax3 in T.grid(64, 64, 56, 56):
                with T.block("root"):
                    i, j, k, l = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    out[i, j, k, l] = T.max(data[i, j, k, l], 0.0)

        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), "float32"),
            weight1: R.Tensor((64, 64, 3, 3), "float32"),
        ):
            with R.dataflow():
                conv1 = R.nn.conv2d(data, weight1, padding=(1, 1))
                relu1 = R.call_tir(
                    Conv2dReLUCallTIR.relu, (conv1,), R.Tensor((64, 64, 56, 56), "float32")
                )
                R.output(relu1)

            return relu1

    @I.ir_module
    class Conv2dReLUCallTIR_partitioned:
        @T.prim_func
        def relu(
            data: T.Buffer((64, 64, 56, 56), "float32"), out: T.Buffer((64, 64, 56, 56), "float32")
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(64, 64, 56, 56):
                with T.block("root"):
                    i, j, k, l = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(data[i, j, k, l])
                    T.writes(out[i, j, k, l])
                    out[i, j, k, l] = T.max(data[i, j, k, l], T.float32(0))

        @R.function
        def fused_relax_nn_conv2d(
            data: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "cutlass.conv2d", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    data,
                    weight1,
                    padding=(1, 1),
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((64, 64, 56, 56), dtype="float32"):
            cls = Conv2dReLUCallTIR_partitioned
            with R.dataflow():
                lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(
                    data, weight1
                )
                relu1 = R.call_tir(
                    cls.relu, (lv,), out_sinfo=R.Tensor((64, 64, 56, 56), dtype="float32")
                )
                R.output(relu1)
            return relu1

    pat = make_fused_bias_activation_pattern("relax.nn.conv2d", with_bias=False, activation=None)
    check(Conv2dReLUCallTIR, [("cutlass.conv2d", pat)], Conv2dReLUCallTIR_partitioned)


def test_unused():
    @I.ir_module
    class Conv2dReLU:
        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), "float32"),
            weight1: R.Tensor((64, 64, 3, 3), "float32"),
        ):
            with R.dataflow():
                conv1 = R.nn.conv2d(data, weight1, padding=(1, 1))
                relu = R.nn.relu(data)
                R.output(conv1)

            return conv1

    @I.ir_module
    class Conv2dReLU_partitioned:
        @R.function
        def fused_relax_nn_conv2d(
            data: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "cutlass.conv2d", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.conv2d(
                    data, weight1, padding=(1, 1)
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            cls = Conv2dReLU_partitioned
            with R.dataflow():
                gv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(
                    data, weight1
                )
                relu: R.Tensor((1, 64, 56, 56), dtype="float32") = R.nn.relu(data)
                R.output(gv)
            return gv

    pat = make_fused_bias_activation_pattern("relax.nn.conv2d", with_bias=False, activation=None)
    check(Conv2dReLU, [("cutlass.conv2d", pat)], Conv2dReLU_partitioned)


def test_check_pattern():
    lhs = wildcard()
    rhs = wildcard()
    out = is_op("relax.nn.conv2d")(lhs, rhs)
    annotation_patterns = {"root": out, "lhs": lhs, "rhs": rhs}

    def pred(context: PatternCheckContext):
        lhs = context.annotated_expr["lhs"]
        rhs = context.annotated_expr["rhs"]
        expr = context.annotated_expr["root"]
        assert isinstance(lhs, relax.expr.Var) and lhs.name_hint == "data"
        assert isinstance(rhs, relax.expr.Var) and rhs.name_hint == "weight1"
        assert isinstance(expr, relax.expr.Call) and expr.op.name == "relax.nn.conv2d"
        return False

    check(
        Conv2dReLU, [("cutlass.conv2d", out, annotation_patterns, pred)], Conv2dReLU
    )  # expect no partitioning


def test_bind_constants():
    weight = np.random.randn(64, 64, 3, 3).astype("float32")

    @I.ir_module
    class Conv2dWithConstantWeight:
        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), "float32"),
            weight1: R.Tensor((64, 64, 3, 3), "float32"),
        ):
            with R.dataflow():
                conv1 = R.nn.conv2d(data, R.const(weight), padding=(1, 1))
                R.output(conv1)
            return conv1

    @I.ir_module
    class Conv2dWithConstantWeight_partitioned:
        @R.function
        def fused_relax_nn_conv2d(
            data: R.Tensor((1, 64, 56, 56), dtype="float32"),
            param_0: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "cutlass.conv2d", "Primitive": 1})
            with R.dataflow():
                gv = R.nn.conv2d(data, param_0, padding=(1, 1))
                R.output(gv)
            return gv

        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            cls = Conv2dWithConstantWeight_partitioned
            with R.dataflow():
                gv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(
                    data, R.const(weight)
                )
                R.output(gv)
            return gv

    pat = make_fused_bias_activation_pattern("relax.nn.conv2d", with_bias=False, activation=None)
    check(
        Conv2dWithConstantWeight,
        [("cutlass.conv2d", pat)],
        Conv2dWithConstantWeight_partitioned,
        bind_constants=False,
    )


if __name__ == "__main__":
    pytest.main([__file__])
