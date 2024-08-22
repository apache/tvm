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
from tvm.relax.dpl.pattern import (
    is_op,
    is_tuple_get_item,
    make_fused_bias_activation_pattern,
    wildcard,
)
from tvm.relax.transform import PatternCheckContext
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.backend.contrib.cublas import partition_for_cublas
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
            {
                "Codegen": "dnnl",
                "global_symbol": "fused_relax_nn_conv2d_relax_nn_relu_dnnl",
            }
        )

        @R.function
        def gv1(
            data2: R.Tensor((1, 64, 56, 56), dtype="float32"),
            weight12: R.Tensor((64, 64, 3, 3), dtype="float32"),
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            R.func_attr({"Composite": "dnnl.conv2d_relu"})
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

    @R.function(private=True)
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

    @R.function(private=True)
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

    @R.function(private=True)
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

    @R.function(private=True)
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

    @R.function(private=True)
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

    @R.function(private=True)
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

    @R.function(private=True)
    def fused_relax_nn_conv2d_relax_nn_relu(
        data1: R.Tensor((1, 64, 56, 56), dtype="float32"),
        weight1: R.Tensor((64, 64, 3, 3), dtype="float32"),
    ) -> R.Tuple(
        R.Tensor((1, 64, 54, 54), dtype="float32"),
        R.Tensor((1, 64, 54, 54), dtype="float32"),
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
    def main2(
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
            conv3 = Conv2dx2.main2(data, weight1, weight2)
            result = conv2 + conv3
            R.output(result)

        return result


@tvm.script.ir_module
class Conv2dx2_partitioned:
    @R.function
    def fused_relax_nn_conv2d_cutlass(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        R.func_attr({"Codegen": "cutlass", "global_symbol": "fused_relax_nn_conv2d_cutlass"})

        @R.function
        def gv_1(
            data_1: R.Tensor((16, 32, 32, 16), dtype="float16"),
            weight1_1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
            R.func_attr({"Composite": "cutlass.conv2d"})
            with R.dataflow():
                gv_2: R.Tensor((16, 32, 32, 16), dtype="float16") = R.nn.conv2d(
                    data_1,
                    weight1_1,
                    strides=[1, 1],
                    padding=[1, 1, 1, 1],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="void",
                )
                R.output(gv_2)
            return gv_2

        gv1: R.Tensor((16, 32, 32, 16), dtype="float16") = gv_1(data, weight1)
        return gv1

    @R.function
    def main2(
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
    def main(
        data: R.Tensor((16, 32, 32, 16), dtype="float16"),
        weight1: R.Tensor((16, 3, 3, 16), dtype="float16"),
        weight2: R.Tensor((16, 3, 3, 16), dtype="float16"),
    ) -> R.Tensor((16, 32, 32, 16), dtype="float16"):
        cls = Conv2dx2_partitioned
        with R.dataflow():
            lv1: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_cutlass(
                data, weight1
            )
            lv2: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.fused_relax_nn_conv2d_cutlass(
                lv1, weight2
            )
            conv3: R.Tensor((16, 32, 32, 16), dtype="float16") = cls.main2(data, weight1, weight2)
            result: R.Tensor((16, 32, 32, 16), dtype="float16") = R.add(lv2, conv3)
            R.output(result)
        return result


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
    check(
        BranchTupleOutput,
        [("dnnl.conv2d_relu", conv2d_relu_pat)],
        BranchTupleOutputPartitioned,
    )


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


@pytest.mark.parametrize("annotate_codegen", [True, False])
def test_no_op_if_no_patterns_match(annotate_codegen):
    """If no matches occur, FuseOpsByPattern is a no-op"""
    check(
        Conv2dReLU,
        [],
        Conv2dReLU,
        annotate_codegen=annotate_codegen,
    )


@pytest.mark.parametrize("annotate_codegen", [True, False])
def test_unmatched_calls_may_include_lambda_functions(annotate_codegen):
    """If no matches occur, FuseOpsByPattern is a no-op

    This is a regression test.  Previous implementations of
    CompositeFunctionAnnotator assumed that all lambda functions
    resulted from FuseOps, and would contain the `kComposite`
    attribute.
    """

    @tvm.script.ir_module
    class Module:
        @R.function
        def main(
            data: R.Tensor((1, 64, 56, 56), "float32"),
            weight1: R.Tensor((64, 64, 3, 3), "float32"),
        ):
            with R.dataflow():
                conv1 = R.nn.relu(R.nn.conv2d(data, weight1, padding=(1, 1)))
                R.output(conv1)

            return conv1

        @R.function
        def unrelated_function(A: R.Tensor([16, 16], dtype="float16")):
            @R.function
            def inner_func(B: R.Tensor([16, 16], dtype="float16")):
                with R.dataflow():
                    C = R.multiply(B, R.const(2, "float16"))
                    R.output(C)
                return C

            D = inner_func(A)
            return D

    check(
        Module,
        [],
        Module,
        annotate_codegen=annotate_codegen,
    )


def test_compare_with_merge_composite_path():
    x = relax.Var("x", relax.TensorStructInfo([10, 10], "float32"))
    y = relax.Var("y", relax.TensorStructInfo([10, 10], "float32"))
    bb = relax.BlockBuilder()
    with bb.function("main", [x, y]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.multiply(x, y))
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    mod = bb.get()
    mod = relax.transform.CanonicalizeBindings()(mod)

    # Currently, we have two paths for BYOC.
    # Path1. [FuseOpsByPattern(patterns, annotate_codegen=True), RunCodegen()]
    # Path2. [FuseOpsByPattern(patterns, annotate_codegen=False), MergeCompositeFunctions(), RunCodegen()]
    # For consistency, both paths should have same interface with RunCodegen().
    # As each path has different naming convention due to the difference in the algorithm,
    # we compare with expected form of each path rather than directly applying structural equality check between two paths.
    patterns = [("cutlass.multiply", is_op("relax.multiply")(wildcard(), wildcard()))]
    mod1 = relax.transform.FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=True)(
        mod
    )
    assert tvm.relax.analysis.well_formed(mod1)

    @I.ir_module
    class Expected1:
        @R.function
        def fused_relax_multiply_cutlass(
            x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Codegen": "cutlass"})
            # from tvm.script import relax as R

            @R.function
            def gv(
                x_1: R.Tensor((10, 10), dtype="float32"),
                y_1: R.Tensor((10, 10), dtype="float32"),
            ) -> R.Tensor((10, 10), dtype="float32"):
                R.func_attr({"Composite": "cutlass.multiply"})
                with R.dataflow():
                    gv_1: R.Tensor((10, 10), dtype="float32") = R.multiply(x_1, y_1)
                    R.output(gv_1)
                return gv_1

            gv1: R.Tensor((10, 10), dtype="float32") = gv(x, y)
            return gv1

        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            cls = Expected1
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_multiply_cutlass(x, y)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(mod1, Expected1)

    mod2 = relax.transform.FuseOpsByPattern(patterns, bind_constants=True, annotate_codegen=False)(
        mod
    )
    mod2 = relax.transform.MergeCompositeFunctions()(mod2)
    assert tvm.relax.analysis.well_formed(mod2)

    @I.ir_module
    class Expected2:
        @R.function
        def fused_relax_multiply1_cutlass(
            x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Codegen": "cutlass"})
            # from tvm.script import relax as R

            @R.function
            def gv(
                x_1: R.Tensor((10, 10), dtype="float32"),
                y_1: R.Tensor((10, 10), dtype="float32"),
            ) -> R.Tensor((10, 10), dtype="float32"):
                R.func_attr({"Composite": "cutlass.multiply"})
                with R.dataflow():
                    gv_1: R.Tensor((10, 10), dtype="float32") = R.multiply(x_1, y_1)
                    R.output(gv_1)

                return gv_1

            gv_1: R.Tensor((10, 10), dtype="float32") = gv(x, y)
            return gv_1

        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32"), y: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            cls = Expected2
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_multiply1_cutlass(x, y)
                R.output(gv)
            return gv

    tvm.ir.assert_structural_equal(mod2, Expected2)


def test_multiple_entries_multiple_calls_same_extern():
    pat = make_fused_bias_activation_pattern("relax.nn.conv2d", with_bias=False, activation=None)
    check(Conv2dx2, [("cutlass.conv2d", pat)], Conv2dx2_partitioned, annotate_codegen=True)


def test_ignore_call_tir():
    @I.ir_module
    class Conv2dReLUCallTIR:
        @T.prim_func
        def relu(
            data: T.Buffer((1, 64, 56, 56), "float32"),
            out: T.Buffer((1, 64, 56, 56), "float32"),
        ):
            for ax0, ax1, ax2, ax3 in T.grid(1, 64, 56, 56):
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
                    Conv2dReLUCallTIR.relu,
                    (conv1,),
                    R.Tensor((1, 64, 56, 56), "float32"),
                )
                R.output(relu1)

            return relu1

    @I.ir_module
    class Conv2dReLUCallTIR_partitioned:
        @T.prim_func
        def relu(
            data: T.Buffer((1, 64, 56, 56), "float32"),
            out: T.Buffer((1, 64, 56, 56), "float32"),
        ):
            # with T.block("root"):
            for ax0, ax1, ax2, ax3 in T.grid(1, 64, 56, 56):
                with T.block("root"):
                    i, j, k, l = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                    T.reads(data[i, j, k, l])
                    T.writes(out[i, j, k, l])
                    out[i, j, k, l] = T.max(data[i, j, k, l], T.float32(0))

        @R.function(private=True)
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
        ) -> R.Tensor((1, 64, 56, 56), dtype="float32"):
            cls = Conv2dReLUCallTIR_partitioned
            with R.dataflow():
                lv: R.Tensor((1, 64, 56, 56), dtype="float32") = cls.fused_relax_nn_conv2d(
                    data, weight1
                )
                relu1 = R.call_tir(
                    cls.relu,
                    (lv,),
                    out_sinfo=R.Tensor((1, 64, 56, 56), dtype="float32"),
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
        @R.function(private=True)
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
        @R.function(private=True)
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


def test_split():
    @R.function
    def func(inp: R.Tensor((16, 32), "float32")):
        R.func_attr({"global_symbol": "main"})
        with R.dataflow():
            tup = R.split(inp, [16], axis=1)
            out = R.add(tup[0], tup[1])
            R.output(out)
        return out

    @tvm.script.ir_module
    class Expected1:
        @R.function(private=True)
        def fused_relax_split(
            inp: R.Tensor((16, 32), dtype="float32")
        ) -> R.Tuple(R.Tensor((16, 16), dtype="float32"), R.Tensor((16, 16), dtype="float32")):
            R.func_attr({"Composite": "x.split", "Primitive": 1})
            with R.dataflow():
                gv: R.Tuple(
                    R.Tensor((16, 16), dtype="float32"),
                    R.Tensor((16, 16), dtype="float32"),
                ) = R.split(inp, indices_or_sections=[16], axis=1)
                R.output(gv)
            return gv

        @R.function
        def main(inp: R.Tensor((16, 32), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
            cls = Expected1
            with R.dataflow():
                lv: R.Tuple(
                    R.Tensor((16, 16), dtype="float32"),
                    R.Tensor((16, 16), dtype="float32"),
                ) = cls.fused_relax_split(inp)
                lv1: R.Tensor((16, 16), dtype="float32") = lv[0]
                lv2: R.Tensor((16, 16), dtype="float32") = lv[1]
                out: R.Tensor((16, 16), dtype="float32") = R.add(lv1, lv2)
                R.output(out)
            return out

    @I.ir_module
    class Expected2:
        @R.function(private=True)
        def fused_relax_split_relax_add(
            inp: R.Tensor((16, 32), dtype="float32")
        ) -> R.Tensor((16, 16), dtype="float32"):
            R.func_attr({"Composite": "x.split", "Primitive": 1})
            with R.dataflow():
                tup: R.Tuple(
                    R.Tensor((16, 16), dtype="float32"),
                    R.Tensor((16, 16), dtype="float32"),
                ) = R.split(inp, indices_or_sections=[16], axis=1)
                lv1: R.Tensor((16, 16), dtype="float32") = tup[0]
                lv2: R.Tensor((16, 16), dtype="float32") = tup[1]
                gv: R.Tensor((16, 16), dtype="float32") = R.add(lv1, lv2)
                R.output(gv)
            return gv

        @R.function
        def main(inp: R.Tensor((16, 32), dtype="float32")) -> R.Tensor((16, 16), dtype="float32"):
            cls = Expected2
            with R.dataflow():
                gv: R.Tensor((16, 16), dtype="float32") = cls.fused_relax_split_relax_add(inp)
                R.output(gv)
            return gv

    mod = tvm.IRModule({"main": func})

    split = is_op("relax.split")(wildcard())
    it1 = is_tuple_get_item(split, 0)
    it2 = is_tuple_get_item(split, 1)
    add = is_op("relax.add")(it1, it2)

    check(mod, [("x.split", split)], Expected1)
    check(mod, [("x.split", add)], Expected2)


def test_clip():
    @R.function
    def func1(x: R.Tensor((10, 10), "float32")):
        R.func_attr({"global_symbol": "main"})
        with R.dataflow():
            gv = R.clip(x, 0, 4)
            R.output(gv)
        return gv

    @I.ir_module
    class Expected1:
        @R.function(private=True)
        def fused_relax_clip(
            x: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Composite": "x.clip", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = R.clip(
                    x, R.prim_value(0), R.prim_value(4)
                )
                R.output(gv)
            return gv

        @R.function
        def main(x: R.Tensor((10, 10), dtype="float32")) -> R.Tensor((10, 10), dtype="float32"):
            cls = Expected1
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_clip(x)
                R.output(gv)
            return gv

    mod1 = tvm.IRModule({"main": func1})
    pat_clip = is_op("relax.clip")(wildcard(), wildcard(), wildcard())

    check(mod1, [("x.clip", pat_clip)], Expected1)

    @R.function
    def func2(x: R.Tensor((10, 10), "float32")):
        R.func_attr({"global_symbol": "main"})
        with R.dataflow():
            gv0 = R.clip(x, 0, 4)
            gv1 = R.clip(x, 1, 3)
            R.output(gv0, gv1)
        return gv0, gv1

    @I.ir_module
    class Expected2:
        @R.function(private=True)
        def fused_relax_clip(
            x: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Composite": "x.clip", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = R.clip(
                    x, R.prim_value(0), R.prim_value(4)
                )
                R.output(gv)
            return gv

        @R.function(private=True)
        def fused_relax_clip1(
            x: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tensor((10, 10), dtype="float32"):
            R.func_attr({"Composite": "x.clip", "Primitive": 1})
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = R.clip(
                    x, R.prim_value(1), R.prim_value(3)
                )
                R.output(gv)
            return gv

        @R.function
        def main(
            x: R.Tensor((10, 10), dtype="float32")
        ) -> R.Tuple(R.Tensor((10, 10), dtype="float32"), R.Tensor((10, 10), dtype="float32")):
            cls = Expected2
            with R.dataflow():
                gv: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_clip(x)
                gv1: R.Tensor((10, 10), dtype="float32") = cls.fused_relax_clip1(x)
                R.output(gv, gv1)
            return (gv, gv1)

    mod = tvm.IRModule({"main": func2})
    check(mod, [("x.clip", pat_clip)], Expected2)


def test_matmul_add3():
    @I.ir_module
    class Module:
        @R.function
        def main(
            x: R.Tensor((32, 8), dtype="float16"),
            y: R.Tensor((8, 8), dtype="float16"),
            x2: R.Tensor((32, 8), dtype="float16"),
            y2: R.Tensor((8, 8), dtype="float16"),
            bias: R.Tensor((8,), dtype="float16"),
            residual: R.Tensor((32, 8), dtype="float16"),
        ) -> R.Tensor((32, 8), dtype="float16"):
            with R.dataflow():
                lv_: R.Tensor((32, 8), dtype="float16") = R.matmul(x2, y2, out_dtype="float16")
                lv: R.Tensor((32, 8), dtype="float16") = R.matmul(x, y, out_dtype="float16")
                lv1: R.Tensor((32, 8), dtype="float16") = R.add(lv, bias)
                lv2: R.Tensor((32, 8), dtype="float16") = R.add(lv1, lv_)
                out: R.Tensor((32, 8), dtype="float16") = R.add(lv2, residual)
                R.output(out)
            return out

    mod = partition_for_cutlass(Module)
    func_names = [name.name_hint for (name, _) in mod.functions.items()]
    assert "fused_relax_matmul_relax_add_relax_add_cutlass" in func_names


def test_intermediate_var_to_var_binding():
    """test the intermediate binding y1 will break the fusion"""

    @I.ir_module
    class Module:
        @R.function
        def main(
            x: R.Tensor((1, 16), dtype="float16"), w: R.Tensor((16, 16), dtype="float16")
        ) -> R.Tensor((1, 16), dtype="float16"):
            with R.dataflow():
                w1: R.Tensor((16, 16), dtype="float16") = R.permute_dims(w, axes=None)
                y: R.Tensor((1, 16), dtype="float16") = R.matmul(x, w1)
                y1: R.Tensor((1, 16), dtype="float16") = y
                out: R.Tensor((1, 16), dtype="float16") = R.add(x, y1)
                R.output(out)
            return out

    mod = partition_for_cublas(Module)
    func_names = [name.name_hint for (name, _) in mod.functions.items()]
    assert "fused_relax_permute_dims_relax_matmul_cublas" in func_names  # add is not fused


def test_multple_runs():
    check(
        Conv2dReLU_composite_annotated,
        [("dnnl.conv2d_relu", conv2d_relu_pat)],
        Conv2dReLU_composite_annotated,
        annotate_codegen=True,
    )


@pytest.mark.skip_well_formed_check_before_transform
def test_error_on_repeated_variable_definitions():
    """Raise error for SSA violations

    Internally, `FuseOpsByPattern` makes a mapping from relax
    variables to the fused group containing that variable.  If the
    input module violates SSA, this map may be ill-formed.

    While not strictly necessary for FuseOps to handle ill-formed
    inputs, checking it at this level provides better error handling
    than propagating it to downstream passes.
    """
    mod = Conv2dReLU.clone()
    mod["copy"] = mod["main"].with_attr("global_symbol", "copy")

    patterns = [("dnnl.conv2d_relu", conv2d_relu_pat)]

    with pytest.raises(ValueError):
        relax.transform.FuseOpsByPattern(patterns)(mod)


def test_matmul_symbolic_var():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1024], "float16"),
            w1: R.Tensor([1024, 1024], "float16"),
            w2: R.Tensor([1024, "M"], "float16"),
        ):
            with R.dataflow():
                matmul1 = R.matmul(x, w1)
                matmul2 = R.matmul(x, w2)
                out = (matmul1, matmul2)
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor(["batch_size", 1024], "float16"),
            w1: R.Tensor([1024, 1024], "float16"),
            w2: R.Tensor([1024, "M"], "float16"),
        ) -> R.Tuple(
            R.Tensor(["batch_size", 1024], "float16"),
            R.Tensor(["batch_size", "M"], "float16"),
        ):
            cls = Expected
            with R.dataflow():
                matmul1 = cls.fused_relax_matmul_cublas(x, w1)
                matmul2 = cls.fused_relax_matmul1_cublas(x, w2)
                out = (matmul1, matmul2)
                R.output(out)
            return out

        @R.function
        def fused_relax_matmul_cublas(
            x: R.Tensor(["batch_size", 1024], "float16"),
            w1: R.Tensor([1024, 1024], "float16"),
        ) -> R.Tensor(["batch_size", 1024], "float16"):
            batch_size = T.int64()
            R.func_attr({"Codegen": "cublas"})

            @R.function
            def inner_func(
                x: R.Tensor([batch_size, 1024], "float16"),
                w1: R.Tensor([1024, 1024], "float16"),
            ) -> R.Tensor([batch_size, 1024], "float16"):
                R.func_attr({"Composite": "cublas.matmul"})
                with R.dataflow():
                    out = R.matmul(x, w1)
                    R.output(out)
                return out

            out = inner_func(x, w1)
            return out

        @R.function
        def fused_relax_matmul1_cublas(
            x: R.Tensor(["batch_size", 1024], "float16"),
            w2: R.Tensor([1024, "M"], "float16"),
        ) -> R.Tensor(["batch_size", "M"], "float16"):
            batch_size = T.int64()
            M = T.int64()
            R.func_attr({"Codegen": "cublas"})

            @R.function
            def inner_func(
                x: R.Tensor([batch_size, 1024], "float16"),
                w2: R.Tensor((1024, M), "float16"),
            ) -> R.Tensor([batch_size, M], "float16"):
                R.func_attr({"Composite": "cublas.matmul"})
                with R.dataflow():
                    out = R.matmul(x, w2)
                    R.output(out)
                return out

            out = inner_func(x, w2)
            return out

    patterns = relax.backend.pattern_registry.get_patterns_with_prefix("cublas.matmul")
    After = relax.transform.FuseOpsByPattern(patterns, bind_constants=False, annotate_codegen=True)(
        Before
    )
    tvm.ir.assert_structural_equal(Expected, After)


def test_match_maximal_subgraph():
    @R.function
    def func(
        x: R.Tensor((32, 8), dtype="int32"),
        y: R.Tensor((8, 8), dtype="int32"),
        bias: R.Tensor((8,), dtype="int32"),
    ) -> R.Tensor((32, 8), dtype="int32"):
        R.func_attr({"global_symbol": "main"})
        with R.dataflow():
            lv0 = R.matmul(x, y, out_dtype="int32")
            lv1 = R.add(lv0, bias)
            lv2 = R.clip(lv1, -128, 127)
            R.output(lv2)
        return lv2

    mod = tvm.IRModule({"main": func})

    matmul = is_op("relax.matmul")(wildcard(), wildcard())
    matmul_add = is_op("relax.add")(matmul, wildcard())
    pattern = matmul_add | is_op("relax.clip")(matmul_add, wildcard(), wildcard())

    partitioned = relax.transform.FuseOpsByPattern([("orclip", pattern)])(mod)
    func_names = [name.name_hint for (name, _) in partitioned.functions.items()]
    assert "fused_relax_matmul_relax_add_relax_clip" in func_names


def test_dataflow_inside_branch():
    """Fusion may apply within internal dataflow

    While relax::DataflowBlock instances may not contain flow control
    or impure functions, they may be contained within flow control
    structures.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor([1024, 1024], "float16"),
            w: R.Tensor([1024, 1024], "float16"),
            transpose_weights: R.Prim("bool"),
        ):
            if transpose_weights:
                with R.dataflow():
                    w_t = R.permute_dims(w)
                    out = R.matmul(x, w_t)
                    R.output(out)
            else:
                with R.dataflow():
                    out = R.matmul(x, w)
                    R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor([1024, 1024], "float16"),
            w: R.Tensor([1024, 1024], "float16"),
            transpose_weights: R.Prim("bool"),
        ):
            cls = Expected
            if transpose_weights:
                with R.dataflow():
                    out_then = cls.fused_relax_permute_dims_relax_matmul_cublas(w, x)
                    R.output(out_then)
                out = out_then
            else:
                with R.dataflow():
                    out_else = cls.fused_relax_matmul_cublas(x, w)
                    R.output(out_else)
                out = out_else
            return out

        @R.function
        def fused_relax_permute_dims_relax_matmul_cublas(
            w: R.Tensor((1024, 1024), dtype="float16"),
            x: R.Tensor((1024, 1024), dtype="float16"),
        ) -> R.Tensor((1024, 1024), dtype="float16"):
            R.func_attr({"Codegen": "cublas"})

            @R.function
            def local_func(
                w_1: R.Tensor((1024, 1024), dtype="float16"),
                x_1: R.Tensor((1024, 1024), dtype="float16"),
            ) -> R.Tensor((1024, 1024), dtype="float16"):
                R.func_attr({"Composite": "cublas.matmul_transposed"})
                with R.dataflow():
                    w_t = R.permute_dims(w_1)
                    out = R.matmul(x_1, w_t)
                    R.output(out)
                return out

            output = local_func(w, x)
            return output

        @R.function
        def fused_relax_matmul_cublas(
            x: R.Tensor((1024, 1024), dtype="float16"),
            w: R.Tensor((1024, 1024), dtype="float16"),
        ) -> R.Tensor((1024, 1024), dtype="float16"):
            R.func_attr({"Codegen": "cublas"})

            @R.function
            def local_func(
                x_1: R.Tensor((1024, 1024), dtype="float16"),
                w_1: R.Tensor((1024, 1024), dtype="float16"),
            ) -> R.Tensor((1024, 1024), dtype="float16"):
                R.func_attr({"Composite": "cublas.matmul"})
                with R.dataflow():
                    out = R.matmul(x_1, w_1)
                    R.output(out)
                return out

            output = local_func(x, w)
            return output

    patterns = relax.backend.pattern_registry.get_patterns_with_prefix("cublas.matmul")
    After = relax.transform.FuseOpsByPattern(
        patterns,
        bind_constants=False,
        annotate_codegen=True,
    )(Before)
    tvm.ir.assert_structural_equal(Expected, After)


if __name__ == "__main__":
    pytest.main([__file__])
