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
# pylint: disable=invalid-name, unused-argument

import pytest

pytest.importorskip("ethosu.vela")

import math

import numpy as np
import tensorflow as tf
import tflite.Model

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu import legalize, preprocess
from tvm.relay import dataflow_pattern
from tvm.relay.op.contrib import ethosu
from tvm.relay.backend.contrib.ethosu import util, codegen
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.frontend.tflite import get_pad_value
from tvm.relay.expr_functor import ExprVisitor

from . import infra


def partition_ethosu_by_table(mod, pattern_table):
    """In case only the legalization part is supported for an operator, we don't
    want to add the operator's pattern to the pattern table so that the compiler
    wouldn't attempt to offload an operator without full stack support."""
    mod = relay.transform.InferType()(mod)
    mod = mod = codegen.replicate_pads(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.MergeComposite(pattern_table)(mod)
    mod = relay.transform.AnnotateTarget("ethos-u")(mod)
    mod = relay.transform.MergeCompilerRegions()(mod)
    mod = relay.transform.InferType()(mod)
    mod = relay.transform.PartitionGraph()(mod)
    mod = relay.transform.InferType()(mod)
    mod = preprocess.preprocess_ext_io()(mod)
    return mod


def relu_n1_to_1(x):
    """
    The specific pattern will be replaced into RELU_N1_TO_1 by tflite.
    """
    return tf.math.maximum(-1.0, tf.math.minimum(x, 1.0))


def test_split_indices_legalize():
    def create_graph(axis):
        x = relay.var("x", shape=(1, 50, 50, 3))
        x_relu = relay.nn.relu(x)
        split_output = relay.split(x_relu, [5, 20, 45], axis).tuple_value
        return relay.Function([x], split_output)

    def expected_mod_axis1():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethos_u_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 5, 50, 3), float32],\
                                                               Tensor[(1, 15, 50, 3), float32],\
                                                               Tensor[(1, 25, 50, 3), float32],\
                                                               Tensor[(1, 5, 50, 3), float32]) {
          %0 = nn.relu(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 5, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 5, 50, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 5, 0, 0], end=[1, 20, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 15, 50, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 20, 0, 0], end=[1, 45, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 25, 50, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 45, 0, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 5, 50, 3), float32] */;
          (%1, %2, %3, %4)
        }
        """
        return tvm.relay.fromtext(expected_ir_string)

    def expected_mod_axis2():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethos_u_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 50, 5, 3), float32],\
                                                               Tensor[(1, 50, 15, 3), float32],\
                                                               Tensor[(1, 50, 25, 3), float32],\
                                                               Tensor[(1, 50, 5, 3), float32]) {
          %0 = nn.relu(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 50, 5, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 5, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 0, 5, 0], end=[1, 50, 20, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 15, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 0, 20, 0], end=[1, 50, 45, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 25, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 0, 45, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 5, 3), float32] */;
          (%1, %2, %3, %4)
        }
        """
        return tvm.relay.fromtext(expected_ir_string)

    rewrite_split = [legalize.PartitionedSplitRewriter(), legalize.SplitRewriter()]

    mod_axis1 = tvm.IRModule()
    func = create_graph(1)
    for r in rewrite_split:
        func = dataflow_pattern.rewrite(r, func)
    mod_axis1["tvmgen_default_ethos_u_main_0"] = func
    expected_axis1 = expected_mod_axis1()
    tvm.ir.assert_structural_equal(mod_axis1, expected_axis1)

    mod_axis2 = tvm.IRModule()
    func = create_graph(2)
    for r in rewrite_split:
        func = dataflow_pattern.rewrite(r, func)
    mod_axis2["tvmgen_default_ethos_u_main_0"] = func
    expected_axis2 = expected_mod_axis2()
    tvm.ir.assert_structural_equal(mod_axis2, expected_axis2)


def test_split_sections_legalize():
    def create_graph(axis, sections):
        x = relay.var("x", shape=(1, 50, 50, 3))
        x_abs = relay.abs(x)
        split_output = relay.split(x_abs, sections, axis).tuple_value
        outputs = list()
        for section_idx in range(sections):
            split_single_out = relay.TupleGetItem(split_output, section_idx)
            tanh = relay.tanh(split_single_out)
            outputs.append(tanh)
        tuple_out = relay.Tuple(outputs)
        return relay.Function([x], tuple_out)

    def expected_mod_axis1():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethos_u_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32],\
                                                               Tensor[(1, 10, 50, 3), float32]) {
          %0 = abs(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 10, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 10, 0, 0], end=[1, 20, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 20, 0, 0], end=[1, 30, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 30, 0, 0], end=[1, 40, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %5 = strided_slice(%0, begin=[0, 40, 0, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %6 = (%1, %2, %3, %4, %5);
          %7 = %6.0;
          %8 = tanh(%7) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %9 = %6.1;
          %10 = tanh(%9) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %11 = %6.2;
          %12 = tanh(%11) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %13 = %6.3;
          %14 = tanh(%13) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          %15 = %6.4;
          %16 = tanh(%15) /* ty=Tensor[(1, 10, 50, 3), float32] */;
          (%8, %10, %12, %14, %16)
        }
        """
        return tvm.relay.fromtext(expected_ir_string)

    def expected_mod_axis2():
        expected_ir_string = """
        #[version = "0.0.5"]
        def @tvmgen_default_ethos_u_main_0(%x: Tensor[(1, 50, 50, 3), float32]) -> (Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32],\
                                                               Tensor[(1, 50, 10, 3), float32]) {
          %0 = abs(%x) /* ty=Tensor[(1, 50, 50, 3), float32] */;
          %1 = strided_slice(%0, begin=[0, 0, 0, 0], end=[1, 50, 10, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %2 = strided_slice(%0, begin=[0, 0, 10, 0], end=[1, 50, 20, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %3 = strided_slice(%0, begin=[0, 0, 20, 0], end=[1, 50, 30, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %4 = strided_slice(%0, begin=[0, 0, 30, 0], end=[1, 50, 40, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %5 = strided_slice(%0, begin=[0, 0, 40, 0], end=[1, 50, 50, 3], strides=[1], axes=None)\
           /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %6 = (%1, %2, %3, %4, %5);
          %7 = %6.0;
          %8 = tanh(%7) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %9 = %6.1;
          %10 = tanh(%9) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %11 = %6.2;
          %12 = tanh(%11) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %13 = %6.3;
          %14 = tanh(%13) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          %15 = %6.4;
          %16 = tanh(%15) /* ty=Tensor[(1, 50, 10, 3), float32] */;
          (%8, %10, %12, %14, %16)
        }
        """
        return tvm.relay.fromtext(expected_ir_string)

    rewrite_split = [legalize.PartitionedSplitRewriter(), legalize.SplitRewriter()]

    mod_axis1 = tvm.IRModule()
    func = create_graph(1, 5)
    for r in rewrite_split:
        func = dataflow_pattern.rewrite(r, func)
    mod_axis1["tvmgen_default_ethos_u_main_0"] = func
    expected_axis1 = expected_mod_axis1()
    tvm.ir.assert_structural_equal(mod_axis1, expected_axis1)

    mod_axis2 = tvm.IRModule()
    func = create_graph(2, 5)
    for r in rewrite_split:
        func = dataflow_pattern.rewrite(r, func)
    mod_axis2["tvmgen_default_ethos_u_main_0"] = func
    expected_axis2 = expected_mod_axis2()
    tvm.ir.assert_structural_equal(mod_axis2, expected_axis2)


INVERSE_LAYOUT_TRANSFORM_OHWI_MAP = {
    "HWIO": [1, 2, 3, 0],
    "HWOI": [1, 2, 0, 3],
    "OWHI": [0, 1, 2, 3],
}


@pytest.mark.parametrize("ifm_shape", [(1, 299, 299, 3), (1, 55, 55, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 2), (1, 3)])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 1)), ((3, 2), (1, 1))])
@pytest.mark.parametrize("activation", [None, "RELU"])
def test_tflite_conv2d_legalize(ifm_shape, kernel_shape, padding, strides, dilation, activation):
    dtype = "int8"

    def create_tflite_graph_single():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, input_shape):
                op = tf.nn.conv2d(
                    input_shape,
                    filters=tf.constant(
                        np.random.uniform(size=(kernel_shape[0], kernel_shape[1], 3, 3)),
                        dtype=tf.float32,
                    ),
                    strides=strides,
                    padding=padding,
                    data_format="NHWC",
                    dilations=dilation,
                )
                if activation:
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        ofm_channels = op.attrs.ofm_channels

        # check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ofm_channels

        # check OFM
        ofm = op.checked_type
        expected_ofm_shape = infra.compute_ofm_shape(
            ifm_shape, padding, kernel_shape, strides, dilation
        )
        assert list(ofm.shape) == list(expected_ofm_shape)
        assert str(ofm.dtype) == dtype
        assert ofm.shape[3] == ofm_channels

        # check weights
        weights_ohwi = op.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert weights_ohwi.shape[0] == ofm_channels
        assert weights_ohwi.shape[1] == kernel_shape[0]
        assert weights_ohwi.shape[2] == kernel_shape[1]
        assert weights_ohwi.shape[3] == 3

        # Check that scale_bias matches weight tensor
        assert list(op.args[2].checked_type.shape)[0] == ofm_channels

        expected_padding = infra.compute_padding_shape(
            ifm_shape,
            expected_ofm_shape,
            padding,
            (kernel_shape[0], kernel_shape[1]),
            strides,
            dilation,
        )
        assert list(op.attrs.padding) == list(expected_padding)
        assert list(op.attrs.strides) == list(strides)
        assert list(op.attrs.dilation) == list(dilation)
        if activation == "RELU":
            assert str(op.attrs.activation) == "CLIP"

    conv2d_pattern_table = [
        (
            ethosu.QnnConv2DParams.composite_name,
            ethosu.qnn_conv2d_pattern(),
            lambda pat: ethosu.QnnConv2DParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph_single()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, conv_params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], conv_params)
    mod = partition_ethosu_by_table(mod, conv2d_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.Conv2DRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    verify(mod["tvmgen_default_ethos_u_main_0"])


def test_tflite_conv2d_with_separate_padding_legalize():
    dtype = "int8"
    ifm_shape = (1, 55, 34, 3)
    kernel_shape = (3, 2)
    strides = (1, 1)
    dilation = (2, 1)
    padding = (0, 0, 1, 1)

    def create_tflite_graph_single():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                tf_strides = [1, strides[0], strides[1], 1]
                op = tf.pad(
                    x,
                    [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]],
                    "CONSTANT",
                )
                weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 3]
                weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                return tf.nn.conv2d(
                    op,
                    weight,
                    strides=tf_strides,
                    padding="VALID",
                    dilations=dilation,
                )

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        ofm_channels = op.attrs.ofm_channels

        # check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ofm_channels

        # check OFM
        ofm = op.checked_type
        expected_ofm_shape = infra.compute_ofm_shape(
            ifm_shape, padding, kernel_shape, strides, dilation
        )
        assert list(ofm.shape) == list(expected_ofm_shape)
        assert str(ofm.dtype) == dtype
        assert ofm.shape[3] == ofm_channels

        # check weights
        weights_ohwi = op.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert weights_ohwi.shape[0] == ofm_channels
        assert weights_ohwi.shape[1] == kernel_shape[0]
        assert weights_ohwi.shape[2] == kernel_shape[1]
        assert weights_ohwi.shape[3] == 3

        # Check that scale_bias matches weight tensor
        assert list(op.args[2].checked_type.shape)[0] == ofm_channels

        assert list(op.attrs.padding) == list(padding)
        assert list(op.attrs.strides) == list(strides)
        assert list(op.attrs.dilation) == list(dilation)

    conv2d_pattern_table = [
        (
            ethosu.QnnConv2DParams.composite_name,
            ethosu.qnn_conv2d_pattern(),
            lambda pat: ethosu.QnnConv2DParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph_single()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, conv_params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], conv_params)
    mod = partition_ethosu_by_table(mod, conv2d_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.Conv2DRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    verify(mod["tvmgen_default_ethos_u_main_0"])


def test_tflite_conv2d_with_separate_channel_padding_legalize():
    dtype = "int8"
    ifm_shape = (1, 55, 34, 3)
    kernel_shape = (3, 2)
    strides = (1, 1)
    dilation = (2, 1)
    padding_ch = (1, 1)

    class ArePadOnGraph(ExprVisitor):
        """
        Visits the Graph recursively and checks if it contains 'nn.pad' op
        """

        def __init__(self):
            ExprVisitor.__init__(self)
            self.on_graph = False

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                if str(call.op.name) == "nn.pad":
                    self.on_graph = True

            return super().visit_call(call)

        def are_pad_on_graph(self, subgraph) -> bool:
            """
            This function recursively visits the graph and checks if 'nn.pad' op is on graph
            """
            self.visit(subgraph)
            return self.on_graph

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                tf_strides = [1, strides[0], strides[1], 1]
                op = tf.pad(
                    x,
                    [[0, 0], [0, 0], [0, 0], [padding_ch[0], padding_ch[1]]],
                    "CONSTANT",
                )
                # HWIO
                weight_shape = [
                    kernel_shape[0],
                    kernel_shape[1],
                    ifm_shape[3] + padding_ch[0] + padding_ch[1],
                    3,
                ]
                weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                return tf.nn.conv2d(
                    op,
                    weight,
                    strides=tf_strides,
                    padding="VALID",
                    dilations=dilation,
                )

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):

        assert ArePadOnGraph().are_pad_on_graph(ext_func.body) == True

    conv2d_pattern_table = [
        (
            ethosu.ChannelPadParams.composite_name,
            ethosu.pad_pattern(),
            lambda pat: ethosu.ChannelPadParams(pat).is_valid(),
        ),
        (
            ethosu.QnnConv2DParams.composite_name,
            ethosu.qnn_conv2d_pattern(),
            lambda pat: ethosu.QnnConv2DParams(pat).is_valid(),
        ),
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, conv_params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], conv_params)
    mod = partition_ethosu_by_table(mod, conv2d_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.Conv2DRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 299, 299, 3), (1, 123, 17, 7)])
@pytest.mark.parametrize("kernel_shape", [(7, 3), (22, 5)])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 1)), ((3, 2), (1, 1))])
@pytest.mark.parametrize("activation", ["RELU", None])
def test_tflite_depthwise_conv_2d_legalize(
    ifm_shape, kernel_shape, padding, strides, dilation, activation
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def depthwise_conv2d(self, x):
                weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 1]
                weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                # The input strides to the TensorFlow API needs to be of shape 1x4
                tf_strides = [1, strides[0], strides[1], 1]
                op = tf.nn.depthwise_conv2d(
                    x, weight, strides=tf_strides, padding=padding, dilations=dilation
                )
                if activation:
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.depthwise_conv2d.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        ofm_channels = op.attrs.ofm_channels

        # check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ofm_channels

        # check OFM
        ofm = op.checked_type
        expected_ofm_shape = infra.compute_ofm_shape(
            ifm_shape, padding, kernel_shape, strides, dilation
        )
        assert list(ofm.shape) == list(expected_ofm_shape)
        assert str(ofm.dtype) == dtype
        assert ofm.shape[3] == ofm_channels

        # check weights
        weights_ohwi = op.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert weights_ohwi.shape[0] == ofm_channels
        assert weights_ohwi.shape[1] == kernel_shape[0]
        assert weights_ohwi.shape[2] == kernel_shape[1]
        assert weights_ohwi.shape[3] == 1  # only depth multiplier 1 is supported

        # Check that scale_bias matches weight tensor
        assert list(op.args[2].checked_type.shape)[0] == ofm_channels

        expected_padding = infra.compute_padding_shape(
            ifm_shape, expected_ofm_shape, padding, kernel_shape, strides, dilation
        )
        assert list(op.attrs.padding) == list(expected_padding)
        assert op.attrs.ofm_channels == ofm_channels
        assert list(op.attrs.strides) == list(strides)
        assert list(op.attrs.dilation) == list(dilation)
        if activation == "RELU":
            assert str(op.attrs.activation) == "CLIP"

    depthwise_pattern_table = [
        (
            ethosu.QnnDepthwiseConv2DParams.composite_name,
            ethosu.qnn_depthwise_conv2d_pattern(),
            lambda pat: ethosu.QnnDepthwiseConv2DParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, depthwise_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.DepthwiseConv2DRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


def test_tflite_depthwise_conv2d_with_separate_padding_legalize():
    dtype = "int8"
    ifm_shape = (1, 23, 32, 7)
    kernel_shape = (1, 2)
    strides = (3, 2)
    dilation = (1, 1)
    padding = (0, 0, 1, 1)

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                tf_strides = [1, strides[0], strides[1], 1]
                op = tf.pad(
                    x,
                    [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]],
                    "CONSTANT",
                )
                weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 1]
                weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                return tf.nn.depthwise_conv2d(
                    op,
                    weight,
                    strides=tf_strides,
                    padding="VALID",
                    dilations=dilation,
                )

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        ofm_channels = op.attrs.ofm_channels

        # check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ofm_channels

        # check OFM
        ofm = op.checked_type
        expected_ofm_shape = infra.compute_ofm_shape(
            ifm_shape, padding, kernel_shape, strides, dilation
        )
        assert list(ofm.shape) == list(expected_ofm_shape)
        assert str(ofm.dtype) == dtype
        assert ofm.shape[3] == ofm_channels

        # check weights
        weights_ohwi = op.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert weights_ohwi.shape[0] == ofm_channels
        assert weights_ohwi.shape[1] == kernel_shape[0]
        assert weights_ohwi.shape[2] == kernel_shape[1]
        assert weights_ohwi.shape[3] == 1  # only depth multiplier 1 is supported

        # Check that scale_bias matches weight tensor
        assert list(op.args[2].checked_type.shape)[0] == ofm_channels

        assert list(op.attrs.padding) == list(padding)
        assert op.attrs.ofm_channels == ofm_channels
        assert list(op.attrs.strides) == list(strides)
        assert list(op.attrs.dilation) == list(dilation)

    depthwise_pattern_table = [
        (
            ethosu.QnnDepthwiseConv2DParams.composite_name,
            ethosu.qnn_depthwise_conv2d_pattern(),
            lambda pat: ethosu.QnnDepthwiseConv2DParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, depthwise_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.DepthwiseConv2DRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 55, 55, 3), (1, 23, 32, 7)])
@pytest.mark.parametrize("padding", [(0, 1, 0, 0), (1, 1, 1, 1), (1, 1, 5, 5)])
@pytest.mark.parametrize("const_value", [0, 5, 125, -5])
def test_tflite_separate_padding_legalize(ifm_shape, padding, const_value):
    dtype = "int8"
    kernel_shape = (1, 1)
    strides = (1, 1)
    dilation = (1, 1)

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                return tf.pad(
                    x,
                    [[0, 0], [padding[0], padding[2]], [padding[1], padding[3]], [0, 0]],
                    "CONSTANT",
                    const_value,
                )

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        ofm_channels = op.attrs.ofm_channels

        # check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ofm_channels

        # check OFM
        ofm = op.checked_type
        expected_ofm_shape = infra.compute_ofm_shape(
            ifm_shape, padding, kernel_shape, strides, dilation
        )
        assert list(ofm.shape) == list(expected_ofm_shape)
        assert str(ofm.dtype) == dtype
        assert ofm.shape[3] == ofm_channels

        # check weights
        weights_ohwi = op.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert weights_ohwi.shape[0] == ofm_channels
        assert weights_ohwi.shape[1] == kernel_shape[0]
        assert weights_ohwi.shape[2] == kernel_shape[1]
        assert weights_ohwi.shape[3] == 1  # only depth multiplier 1 is supported

        # Check that scale_bias matches weight tensor
        assert list(op.args[2].checked_type.shape)[0] == ofm_channels

        assert list(op.attrs.padding) == list(padding)
        assert op.attrs.ofm_channels == ofm_channels
        assert list(op.attrs.strides) == list(strides)
        assert list(op.attrs.dilation) == list(dilation)

    pad_pattern_table = [
        (
            ethosu.PadParams.composite_name,
            ethosu.pad_pattern(),
            lambda pat: ethosu.PadParams(pat).is_valid(),
        ),
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, pad_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.PadRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 55, 55, 3), (1, 23, 32, 7)])
@pytest.mark.parametrize("channel_padding", [(0, 1), (1, 1), (5, 2)])
@pytest.mark.parametrize("const_value", [0, 5, 125, -5])
def test_tflite_separate_channel_padding_legalize(ifm_shape, channel_padding, const_value):
    dtype = "int8"
    padding = (0, 0, 0, 0)

    class AreConcatenateOnGraph(ExprVisitor):
        """
        Visits the Graph recursively and checks if it contains 'concatenate' op
        """

        def __init__(self):
            ExprVisitor.__init__(self)
            self.on_graph = False

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                if str(call.op.name) == "concatenate":
                    self.on_graph = True

            return super().visit_call(call)

        def are_concatenate_on_graph(self, subgraph) -> bool:
            """
            This function recursively visits the graph and checks if 'concatenate' op is on graph
            """
            self.visit(subgraph)
            return self.on_graph

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                return tf.pad(
                    x,
                    [
                        [0, 0],
                        [padding[0], padding[2]],
                        [padding[1], padding[3]],
                        [channel_padding[0], channel_padding[1]],
                    ],
                    "CONSTANT",
                    const_value,
                )

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func, channel_padding):

        op = ext_func.body

        pad_before = 0
        pad_after = 0
        if channel_padding[0] == 0 and channel_padding[1] > 0:
            pad_after = ext_func.body.args[0][1].args[0].checked_type.shape[3]
            ifm = ext_func.body.args[0][0].args[0].checked_type
        if channel_padding[0] > 0 and channel_padding[1] == 0:
            pad_before = ext_func.body.args[0][0].args[0].checked_type.shape[3]
            ifm = ext_func.body.args[0][1].args[0].checked_type
        if channel_padding[0] > 0 and channel_padding[1] > 0:
            pad_before = ext_func.body.args[0][0].args[0].checked_type.shape[3]
            ifm = ext_func.body.args[0][1].args[0].checked_type
            pad_after = ext_func.body.args[0][2].args[0].checked_type.shape[3]

        # check IFM
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ifm_shape[3]

        # check OFM
        ofm = op.checked_type
        expected_ofm_shape = list(ifm_shape)
        expected_ofm_shape[3] = channel_padding[0] + ifm_shape[3] + channel_padding[1]
        assert list(ofm.shape) == expected_ofm_shape
        assert str(ofm.dtype) == dtype

        # check padding
        assert [pad_before, pad_after] == list(channel_padding)

        # check if relay contains 'concatenate' op
        assert AreConcatenateOnGraph().are_concatenate_on_graph(ext_func.body) == True

    pad_pattern_table = [
        (
            ethosu.ChannelPadParams.composite_name,
            ethosu.pad_pattern(),
            lambda pat: ethosu.ChannelPadParams(pat).is_valid(),
        ),
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, pad_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ChannelPadRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"], channel_padding)


@pytest.mark.parametrize("pooling_type", ["MAX", "AVG"])
@pytest.mark.parametrize("ifm_shape", [[1, 3, 4, 3], [1, 4, 5, 2]])
@pytest.mark.parametrize(
    "pool_shape, strides, activation_function, padding",
    [([1, 2], [1, 2], "NONE", "SAME"), ([2, 3], [2, 3], "RELU", "VALID")],
)
def test_tflite_pool2d_legalize(
    ifm_shape, pooling_type, strides, pool_shape, activation_function, padding
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                if pooling_type == "MAX":
                    op = tf.nn.max_pool(x, pool_shape, strides, padding)
                elif pooling_type == "AVG":
                    op = tf.nn.avg_pool(x, pool_shape, strides, padding)
                if activation_function == "RELU":
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        ofm_shape = infra.compute_ofm_shape(ifm_shape, padding, pool_shape, strides)
        op = ext_func.body
        assert list(op.args[0].checked_type.shape) == ifm_shape
        assert op.args[0].checked_type.dtype == dtype
        assert list(op.checked_type.shape) == ofm_shape
        assert op.checked_type.dtype == dtype
        assert op.attrs.pooling_type == pooling_type
        assert list(op.attrs.strides) == strides
        assert list(op.attrs.padding) == infra.compute_padding_shape(
            ifm_shape, ofm_shape, padding, pool_shape, strides
        )
        assert list(op.attrs.pool_shape) == pool_shape
        assert op.attrs.ofm_channels == ifm_shape[3]
        if activation_function == "RELU":
            assert str(op.attrs.activation) == "CLIP"

    if pooling_type == "MAX":
        rewriter = legalize.MaxPoolingRewriter()
        pattern_table = [
            (
                ethosu.MaxPool2DParams.composite_name,
                ethosu.qnn_maxpool2d_pattern(),
                lambda pat: ethosu.MaxPool2DParams(pat).is_valid(),
            ),
        ]
    elif pooling_type == "AVG":
        rewriter = legalize.AvgPoolingRewriter()
        pattern_table = [
            (
                ethosu.AvgPool2DParams.composite_name,
                ethosu.qnn_avgpool2d_pattern(),
                lambda pat: ethosu.AvgPool2DParams(pat).is_valid(),
            ),
        ]
    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"x": ifm_shape},
        dtype_dict={"x": dtype},
    )
    mod = partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("pooling_type", ["MAX", "AVG"])
@pytest.mark.parametrize(
    "ifm_shape, pool_shape, strides, activation_function, padding",
    [
        ([1, 4, 4, 3], [4, 4], [4, 4], "NONE", "SAME"),
        ([1, 4, 4, 3], [4, 4], [4, 4], "RELU", "VALID"),
        ([1, 25, 5, 64], [25, 5], [25, 5], "NONE", "VALID"),
        ([1, 25, 5, 64], [25, 5], [25, 5], "RELU", "SAME"),
    ],
)
def test_tflite_pool2d_same_ifm_and_kernel_shape_legalize(
    pooling_type, ifm_shape, pool_shape, strides, activation_function, padding
):
    dtype = "int8"
    strides_legalized = [1, 1]

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                if pooling_type == "MAX":
                    op = tf.nn.max_pool(x, pool_shape, strides, padding)
                elif pooling_type == "AVG":
                    op = tf.nn.avg_pool(x, pool_shape, strides, padding)
                if activation_function == "RELU":
                    op = tf.nn.relu(op)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def expected_mod():

        expected_ir_string = ""

        if activation_function == "NONE" and pooling_type == "AVG":
            expected_ir_string = f"""
            #[version = "0.0.5"]
            def @main(%x: Tensor[{str(tuple(ifm_shape))}, {dtype}], output_tensor_names=\
                ["Identity"]) -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), {dtype}] {{
                @tvmgen_default_ethos_u_main_0(%x)
            }}

            def @tvmgen_default_ethos_u_main_0(%y: Tensor[{str(tuple(ifm_shape))}, {dtype}], \
                Compiler="ethos-u", Primitive=1, Inline=1, \
                    global_symbol="tvmgen_default_ethos_u_main_0") -> Tensor[(1, 1, 1, \
                        {str(ifm_shape[3])}), {dtype}] {{
                %2 = fn (%z: Tensor[{str(tuple(ifm_shape))}, {dtype}], \
                    PartitionedFromPattern="cast_nn.avg_pool2d_cast_", \
                        Composite="ethos-u.avgpool2d") -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), \
                            {dtype}] {{
                    %0 = cast(%z, dtype="int32") ;
                    %1 = nn.avg_pool2d(%0, pool_size={str(pool_shape)}, strides={str(strides)}, \
                        padding=[0, 0, 0, 0], layout="NHWC") ;
                    cast(%1, dtype="{dtype}")
                }} ;
                %2(%y)
            }}
            """

        if activation_function == "RELU" and pooling_type == "AVG":
            expected_ir_string = f"""
            #[version = "0.0.5"]
            def @main(%x: Tensor[{str(tuple(ifm_shape))}, {dtype}], output_tensor_names=\
                ["Identity"]) -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), {dtype}] {{
                @tvmgen_default_ethos_u_main_0(%x)
            }}

            def @tvmgen_default_ethos_u_main_0(%y: Tensor[{str(tuple(ifm_shape))}, {dtype}], \
                Compiler="ethos-u", Primitive=1, Inline=1, \
                    global_symbol="tvmgen_default_ethos_u_main_0") -> Tensor[(1, 1, 1, \
                        {str(ifm_shape[3])}), {dtype}] {{
                %3 = fn (%z: Tensor[{str(tuple(ifm_shape))}, {dtype}], \
                    PartitionedFromPattern="cast_nn.avg_pool2d_cast_clip_", \
                        Composite="ethos-u.avgpool2d") -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), \
                            {dtype}] {{
                    %0 = cast(%z, dtype="int32") ;
                    %1 = nn.avg_pool2d(%0, pool_size={str(pool_shape)}, strides={str(strides)}, \
                        padding=[0, 0, 0, 0], layout="NHWC") ;
                    %2 = cast(%1, dtype="{dtype}") ;
                    clip(%2, a_min=-128f, a_max=127f)
                }} ;
                %3(%y)
            }}
            """

        if activation_function == "NONE" and pooling_type == "MAX":
            expected_ir_string = f"""
            #[version = "0.0.5"]
            def @main(%x: Tensor[{str(tuple(ifm_shape))}, {dtype}], output_tensor_names=\
                ["Identity"]) -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), {dtype}] {{
                @tvmgen_default_ethos_u_main_0(%x)
            }}

            def @tvmgen_default_ethos_u_main_0(%y: Tensor[{str(tuple(ifm_shape))}, {dtype}], \
                Compiler="ethos-u", Primitive=1, Inline=1, \
                    global_symbol="tvmgen_default_ethos_u_main_0") -> Tensor[(1, 1, 1, \
                        {str(ifm_shape[3])}), {dtype}] {{
                %0 = fn (%z: Tensor[{str(tuple(ifm_shape))}, {dtype}], \
                    PartitionedFromPattern="nn.max_pool2d_", \
                        Composite="ethos-u.maxpool2d") -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), \
                            {dtype}] {{
                    nn.max_pool2d(%z, pool_size={str(pool_shape)}, strides={str(strides)}, \
                        padding=[0, 0, 0, 0], layout="NHWC")
                }} ;
                %0(%y)
            }}
            """

        if activation_function == "RELU" and pooling_type == "MAX":
            expected_ir_string = f"""
            #[version = "0.0.5"]
            def @main(%x: Tensor[{str(tuple(ifm_shape))}, {dtype}] , output_tensor_names=\
                ["Identity"]) -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), {dtype}] {{
                @tvmgen_default_ethos_u_main_0(%x)
            }}

            def @tvmgen_default_ethos_u_main_0(%y: Tensor[{str(tuple(ifm_shape))}, {dtype}] , \
                Compiler="ethos-u", Primitive=1, Inline=1, \
                    global_symbol="tvmgen_default_ethos_u_main_0") -> Tensor[(1, 1, 1, \
                        {str(ifm_shape[3])}), {dtype}] {{
                %1 = fn (%z: Tensor[{str(tuple(ifm_shape))}, {dtype}] , \
                    PartitionedFromPattern="nn.max_pool2d_clip_", \
                        Composite="ethos-u.maxpool2d") -> Tensor[(1, 1, 1, {str(ifm_shape[3])}), \
                            {dtype}] {{
                    %0 = nn.max_pool2d(%z, pool_size={str(pool_shape)}, strides={str(strides)}, \
                        padding=[0, 0, 0, 0], layout="NHWC");
                    clip(%0, a_min=-128f, a_max=127f)
                }};
                %1(%y)
            }}
            """

        return tvm.relay.fromtext(expected_ir_string)

    def verify(ext_func):
        ofm_shape = infra.compute_ofm_shape(ifm_shape, padding, pool_shape, strides)
        op = ext_func.body
        assert list(op.args[0].checked_type.shape) == ifm_shape
        assert op.args[0].checked_type.dtype == dtype
        assert list(op.checked_type.shape) == ofm_shape
        assert op.checked_type.dtype == dtype
        assert op.attrs.pooling_type == pooling_type
        assert list(op.attrs.strides) == strides_legalized
        assert list(op.attrs.padding) == infra.compute_padding_shape(
            ifm_shape, ofm_shape, padding, pool_shape, strides
        )
        assert list(op.attrs.padding) == infra.compute_padding_shape(
            ifm_shape, ofm_shape, padding, pool_shape, strides_legalized
        )
        assert list(op.attrs.pool_shape) == pool_shape
        assert op.attrs.ofm_channels == ifm_shape[3]
        if activation_function == "RELU":
            assert str(op.attrs.activation) == "CLIP"

    if pooling_type == "MAX":
        rewriter = legalize.MaxPoolingRewriter()
        pattern_table = [
            (
                ethosu.MaxPool2DParams.composite_name,
                ethosu.qnn_maxpool2d_pattern(),
                lambda pat: ethosu.MaxPool2DParams(pat).is_valid(),
            ),
        ]

    if pooling_type == "AVG":
        rewriter = legalize.AvgPoolingRewriter()
        pattern_table = [
            (
                ethosu.AvgPool2DParams.composite_name,
                ethosu.qnn_avgpool2d_pattern(),
                lambda pat: ethosu.AvgPool2DParams(pat).is_valid(),
            ),
        ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"x": ifm_shape},
        dtype_dict={"x": dtype},
    )
    mod = partition_ethosu_by_table(mod, pattern_table)

    expected = expected_mod()
    tvm.ir.assert_structural_equal(mod, expected)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("operator_type", ["ADD", "SUB", "MUL", "MIN", "MAX"])
@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, reversed_operands",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], False),
        ([1, 2, 3, 4], [1, 1, 3, 1], False),
        ([1, 1, 3, 1], [1, 2, 3, 4], True),
        ([1, 4, 4], [4, 1], False),
        ([4], [4], False),
        ([4], [1, 2, 3, 4], True),
        ([1, 4, 4], [4, 1], False),
    ],
)
@pytest.mark.parametrize("activation_function", [None, tf.nn.relu])
def test_tflite_binary_elemwise_legalize(
    operator_type,
    ifm_shape,
    ifm2_shape,
    reversed_operands,
    activation_function,
):
    np.random.seed(0)
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x, y):
                if operator_type == "ADD":
                    op = tf.math.add(x, y)
                elif operator_type == "SUB":
                    op = tf.math.subtract(x, y)
                elif operator_type == "MUL":
                    op = tf.math.multiply(x, y)
                elif operator_type == "MIN":
                    op = tf.math.minimum(x, y)
                elif operator_type == "MAX":
                    op = tf.math.maximum(x, y)
                if activation_function:
                    op = activation_function(op)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32), tf.TensorSpec(ifm2_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                data2 = np.random.rand(*tuple(ifm2_shape)) * 2
                yield [data.astype(np.float32), data2.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        out_shape = ifm2_shape if reversed_operands else ifm_shape
        shapes = [ifm_shape, ifm2_shape]
        ifm_index, ifm2_index = (1, 0) if reversed_operands else (0, 1)
        op = ext_func.body

        has_reshaped_output = False
        has_separate_requantize = False
        shapes_padded = [[1] * (4 - len(s)) + s for s in shapes]
        out_padded = [1] * (4 - len(out_shape)) + out_shape
        if op.op.name == "contrib.ethosu.identity":
            op = op.args[0]
            has_separate_requantize = True
        if op.op.name == "reshape":
            has_reshaped_output = True
            op = op.args[0]

        assert list(op.args[0].checked_type.shape) == shapes_padded[ifm_index]
        assert list(op.args[1].checked_type.shape) == shapes_padded[ifm2_index]
        assert op.args[0].checked_type.dtype == dtype
        assert list(op.checked_type.shape) == out_padded
        assert op.checked_type.dtype == dtype
        assert op.attrs.operator_type == operator_type
        assert op.attrs.reversed_operands == reversed_operands
        if activation_function != None:
            assert str(op.attrs.activation) == "CLIP"

            if operator_type in ["MIN", "MAX"]:
                if has_separate_requantize:
                    # In case when requantize cannot be fused with MIN/MAX + CLIP due to hardware constraints
                    # there should be default quantization values since requantize is separate operation.
                    assert float(op.attrs.ifm_scale) == 1.0
                    assert int(op.attrs.ifm_zero_point) == 0
                    assert float(op.attrs.ifm2_scale) == 1.0
                    assert int(op.attrs.ifm2_zero_point) == 0
                    assert float(op.attrs.ofm_scale) == 1.0
                    assert int(op.attrs.ofm_zero_point) == 0
                else:
                    # MIN and MAX with an activation must have a requantize operation
                    # baked into the output. To check the extra requantize node was
                    # picked up by the pattern, we can make sure the quantization
                    # information is not default.
                    assert float(op.attrs.ifm_scale) != 1.0
                    assert int(op.attrs.ifm_zero_point) != 0
                    assert float(op.attrs.ifm2_scale) != 1.0
                    assert int(op.attrs.ifm2_zero_point) != 0
                    assert float(op.attrs.ofm_scale) != 1.0
                    assert int(op.attrs.ofm_zero_point) != 0

        if has_reshaped_output:
            assert list(ext_func.body.checked_type.shape) == out_shape

    if operator_type == "ADD":
        rewriter = legalize.AddRewriter()
        pattern_table = [
            (
                ethosu.AddParams.composite_name,
                ethosu.qnn_add_pattern(),
                lambda pat: ethosu.AddParams(pat).is_valid(),
            ),
        ]
    elif operator_type == "SUB":
        rewriter = legalize.SubRewriter()
        pattern_table = [
            (
                ethosu.SubParams.composite_name,
                ethosu.qnn_subtract_pattern(),
                lambda pat: ethosu.SubParams(pat).is_valid(),
            ),
        ]
    elif operator_type == "MUL":
        rewriter = legalize.MulRewriter()
        pattern_table = [
            (
                ethosu.MulParams.composite_name,
                ethosu.qnn_mul_pattern(),
                lambda pat: ethosu.MulParams(pat).is_valid(),
            ),
        ]
    elif operator_type == "MIN":
        rewriter = [legalize.MinRewriter(), legalize.RequantizeRewriter()]
        pattern_table = [
            (
                ethosu.MinParams.composite_name,
                ethosu.minimum_clip_requantize_pattern(),
                lambda pat: ethosu.MinParams(pat).is_valid(),
            ),
            (
                ethosu.MinParams.composite_name,
                ethosu.minimum_pattern(),
                lambda pat: ethosu.MinParams(pat).is_valid(),
            ),
            (
                ethosu.RequantizeParams.composite_name,
                ethosu.requantize_pattern(),
                lambda pat: ethosu.RequantizeParams(pat).is_valid(),
            ),
        ]
    elif operator_type == "MAX":
        rewriter = [legalize.MaxRewriter(), legalize.RequantizeRewriter()]
        pattern_table = [
            (
                ethosu.MaxParams.composite_name,
                ethosu.maximum_clip_requantize_pattern(),
                lambda pat: ethosu.MaxParams(pat).is_valid(),
            ),
            (
                ethosu.MaxParams.composite_name,
                ethosu.maximum_pattern(),
                lambda pat: ethosu.MaxParams(pat).is_valid(),
            ),
            (
                ethosu.RequantizeParams.composite_name,
                ethosu.requantize_pattern(),
                lambda pat: ethosu.RequantizeParams(pat).is_valid(),
            ),
        ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"x": ifm_shape, "y": ifm2_shape},
        dtype_dict={"x": dtype, "y": dtype},
    )
    mod = partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


# This test is for checking the case when requantize cannot be fused with MIN/MAX + CLIP due to hardware constraints.
def test_tflite_max_relu_n1_to_1_legalize():
    ifm_shape = [1, 4, 8, 16]
    test_tflite_binary_elemwise_legalize("MAX", ifm_shape, ifm_shape, False, relu_n1_to_1)


def test_binary_add_from_constant_scalar():
    dtype = "uint8"
    ifm_shape = (1, 4, 4, 8)

    def create_graph():
        inp = relay.var("input", shape=ifm_shape, dtype=dtype)
        scalar = relay.const(np.ones((1, 1, 1, 1), dtype=dtype), dtype=dtype)
        add = relay.qnn.op.add(
            inp,
            scalar,
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
            relay.const(1.0, dtype="float32"),
            relay.const(0, dtype="int32"),
        )
        func = relay.Function(relay.analysis.free_vars(add), add)
        return tvm.IRModule.from_expr(func)

    def verify(ext_func):
        op = ext_func.body
        assert list(op.args[0].checked_type.shape) == [1, 4, 4, 8]
        assert list(op.args[1].checked_type.shape) == [1, 1, 1, 1]
        assert op.args[0].checked_type.dtype == "uint8"
        assert list(op.checked_type.shape) == [1, 4, 4, 8]
        assert op.checked_type.dtype == "uint8"
        assert op.attrs.operator_type == "ADD"

    rewriter = legalize.AddRewriter()
    pattern_table = [
        (
            ethosu.AddParams.composite_name,
            ethosu.qnn_add_pattern(),
            lambda pat: ethosu.AddParams(pat).is_valid(),
        ),
    ]

    mod = create_graph()
    mod = partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape, ifm2_shape, reversed_operands",
    [
        ([1, 2, 3, 4], [1, 2, 3, 4], False),
        ([1, 2, 3, 4], [1, 1, 3, 1], False),
        ([1, 1, 3, 1], [1, 2, 3, 4], True),
    ],
)
def test_ethosu_left_shift_binary_elemwise_legalize(ifm_shape, ifm2_shape, reversed_operands):
    dtype = "int32"
    operator_type = "SHL"

    def create_graph():
        input1 = relay.var("x1", shape=ifm_shape, dtype=dtype)
        input2 = relay.var("x2", shape=ifm2_shape, dtype=dtype)
        c1 = relay.left_shift(input1, input2)
        f = relay.Function([input1, input2], c1)
        mod = tvm.IRModule()
        mod["main"] = f
        return mod

    def verify(ext_func):
        out_shape = ifm2_shape if reversed_operands else ifm_shape
        shapes = [ifm_shape, ifm2_shape]
        ifm_index, ifm2_index = (1, 0) if reversed_operands else (0, 1)
        op = ext_func.body
        assert list(op.args[0].checked_type.shape) == shapes[ifm_index]
        assert list(op.args[1].checked_type.shape) == shapes[ifm2_index]
        assert op.args[0].checked_type.dtype == dtype
        assert list(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == dtype
        assert op.attrs.operator_type == operator_type
        assert op.attrs.reversed_operands == reversed_operands
        assert str(op.attrs.activation) == "NONE"

    rewriter = legalize.ShlRewriter()
    pattern_table = [
        (
            ethosu.ShlParams.composite_name,
            ethosu.shl_pattern(),
            lambda pat: ethosu.ShlParams(pat).is_valid(),
        ),
    ]

    mod = create_graph()
    mod = partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape, new_shape",
    [
        ((1, 4, 1, 2), (4, 2)),
        ((1, 5, 1, 20), (100,)),
        ((12, 20), (1, 6, 4, 10)),
        ((30,), (10, 1, 3)),
    ],
)
def test_relay_reshape_legalize(ifm_shape, new_shape):

    ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
    reshape = relay.op.reshape(ifm, new_shape)
    func = relay.Function([ifm], reshape)
    mod = tvm.IRModule()
    mod["main"] = func
    mod = relay.transform.InferType()(mod)

    reshape_pattern_table = [
        (
            ethosu.ReshapeParams.composite_name,
            ethosu.reshape_pattern(),
            lambda pat: ethosu.ReshapeParams(pat).is_valid(),
        ),
    ]

    mod = partition_ethosu_by_table(mod, reshape_pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ReshapeRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.NoOpRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    ext_func = mod["tvmgen_default_ethos_u_main_0"]

    identity = ext_func.body
    assert identity.op.name == "contrib.ethosu.identity"

    # check that the reshape is still there
    reshape = identity.args[0]
    assert reshape.op.name == "reshape"

    # check that identity's output shape matches reshape's output shape
    assert tuple(identity.checked_type.shape) == new_shape


@pytest.mark.parametrize(
    "ifm_shape, begin, size",
    [
        ([1, 10, 50, 4], [0, 5, 11, 2], [1, 5, 11, 1]),
        ([15, 17, 3], [3, 0, 1], [8, 17, 2]),
        ([7, 6043], [0, 704], [1, 2860]),
        ([5000], [123], [2151]),
    ],
)
def test_tflite_slice(ifm_shape, begin, size):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def slice_func(self, x):
                return tf.slice(x, begin, size)

        model = Model()

        # Save the model
        concrete_func = model.slice_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        identity = ext_func.body
        assert identity.op.name == "contrib.ethosu.identity"

        # check that the strided_slice is still there
        strided_slice = identity.args[0]
        assert strided_slice.op.name == "strided_slice"

        # check that identity's output shape matches strided slice's output shape
        assert list(identity.checked_type.shape) == size

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    strided_slice_pattern_table = [
        (
            ethosu.StridedSliceParams.composite_name,
            ethosu.strided_slice_pattern(),
            lambda pat: ethosu.StridedSliceParams(pat).is_valid(),
        ),
    ]
    mod = partition_ethosu_by_table(mod, strided_slice_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.StridedSliceRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.NoOpRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape, begin, end",
    [([1, 1, 5, 8], [0, 0, 0, 0], [1, 1, 2, 3]), ([1, 3, 3], [0, 1, 2], [1, 2, 3])],
)
def test_tflite_strided_slice(ifm_shape, begin, end):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def strided_slice_func(self, x):
                return tf.strided_slice(x, begin, end)

        model = Model()

        # Save the model
        concrete_func = model.strided_slice_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        identity = ext_func.body
        assert identity.op.name == "contrib.ethosu.identity"

        # check that the strided_slice is still there
        strided_slice = identity.args[0]
        assert strided_slice.op.name == "strided_slice"

        # check that identity's output shape matches strided slice's output shape
        size = list(np.array(end) - np.array(begin))
        assert list(identity.checked_type.shape) == size

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    strided_slice_pattern_table = [
        (
            ethosu.StridedSliceParams.composite_name,
            ethosu.strided_slice_pattern(),
            lambda pat: ethosu.StridedSliceParams(pat).is_valid(),
        ),
    ]
    mod = partition_ethosu_by_table(mod, strided_slice_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.StridedSliceRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.NoOpRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("operator_type", ["ABS"])
@pytest.mark.parametrize(
    "ifm_shape",
    [[1, 2, 3, 4], [1, 7, 3], [8, 3, 1], [11, 22], [300]],
)
def test_tflite_unary_elemwise_legalize(
    operator_type,
    ifm_shape,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def abs_func(self, x):
                if operator_type == "ABS":
                    op = tf.math.abs(x)
                return op

        model = Model()

        # Save the model
        concrete_func = model.abs_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        out_shape = ifm_shape
        func_body = ext_func.body

        # If we legalized the unary elementwise op into 4D
        if func_body.op.name == "reshape":
            reshape = func_body
            unary = func_body.args[0]
            reshape2 = unary.args[0]

            # Check the input to the reshape
            reshape2_in_shape = [i for i in reshape2.args[0].checked_type.shape]
            assert reshape2_in_shape == ifm_shape

            # Check that the unary elementwise operator is 4D after reshape
            assert len(unary.checked_type.shape) == 4
            assert unary.args[0].checked_type.dtype == dtype

            # Check that the output of the graph has the same shape as input
            reshape_out_shape = [i for i in reshape.checked_type.shape]
            assert reshape_out_shape == ifm_shape
            assert unary.attrs.operator_type == operator_type

        else:
            unary = func_body

            # Check the IFM
            assert list(unary.args[0].checked_type.shape) == ifm_shape
            assert unary.args[0].checked_type.dtype == dtype

            # Check the OFM
            assert list(unary.checked_type.shape) == out_shape
            assert unary.checked_type.dtype == dtype

            # operator type check
            assert unary.attrs.operator_type == operator_type

    if operator_type == "ABS":
        rewriter = legalize.AbsRewriter()
        pattern_table = [
            (
                ethosu.AbsParams.composite_name,
                ethosu.abs_pattern(),
                lambda pat: ethosu.AbsParams(pat).is_valid(),
            ),
        ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


def test_tflite_tanh_legalize():
    dtype = "int8"
    ifm_shape = (1, 241, 132, 7)

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tanh_func(self, x):
                op = tf.math.tanh(x)
                return op

        model = Model()
        concrete_func = model.tanh_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod = ethosu.partition_for_ethosu(mod, params)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.TanhRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    func_body = mod["tvmgen_default_ethos_u_main_0"].body
    assert func_body.op.name == "contrib.ethosu.identity"
    assert func_body.attrs.activation == "TANH"
    assert tuple(func_body.args[0].checked_type.shape) == (ifm_shape)
    assert tuple(func_body.args[1].checked_type.shape) == (256,)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "ifm_shape, axis, keep_dims, use_same_quantization",
    [
        # mean to average pool
        [(1, 8, 16, 16), (1,), True, True],
        [(1, 8, 16, 16), (2,), False, True],
        [(1, 8, 16, 16), (1, 2), False, True],
        [(3, 3, 4), (0,), True, True],
        [(3, 3, 4), (1,), False, True],
        [(8, 5), (0,), False, True],
        [(8, 5), (1,), True, True],
        # mean to depthwise
        [(1, 8, 16, 16), (1,), True, False],
        [(1, 8, 16, 16), (2,), True, False],
        [(1, 8, 16, 16), (1, 2), False, False],
        [(8, 4), (0,), False, False],
        [(1, 65, 2, 1), (1, 2), True, False],  # special case when h > 64
    ],
)
def test_mean(ifm_shape, axis, keep_dims, use_same_quantization, dtype):
    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                op = tf.math.reduce_mean(x, axis=axis, keepdims=keep_dims)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod

    def create_relay_graph_with_same_quantization():
        ifm = relay.var("input", shape=ifm_shape, dtype=dtype)
        cast = relay.cast(ifm, dtype="int32")
        mean = relay.mean(cast, axis=axis, keepdims=keep_dims)
        requantize = relay.qnn.op.requantize(
            mean,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
            out_dtype=dtype,
        )

        func = relay.Function(relay.analysis.free_vars(requantize), requantize)
        mod = tvm.IRModule.from_expr(func)
        return mod

    def verify(ext_func):
        out_var = ext_func.body

        next_op = out_var
        pooling_op = None
        depthwise_op = None
        if (
            isinstance(next_op, relay.expr.Call)
            and isinstance(next_op.op, tvm.ir.op.Op)
            and next_op.op.name == "reshape"
        ):
            next_op = next_op.args[0]
        if util.is_named_ethosu_op(next_op, "pooling"):
            pooling_op = next_op
            next_op = next_op.args[0]
        if util.is_named_ethosu_op(next_op, "depthwise_conv2d"):
            depthwise_op = next_op
            next_op = next_op.args[0]
        while (
            isinstance(next_op, relay.expr.Call)
            and isinstance(next_op.op, tvm.ir.op.Op)
            and next_op.op.name == "reshape"
        ):
            next_op = next_op.args[0]
        in_var = next_op

        def calculate_expected_output_shape():
            for i in range(len(ifm_shape)):
                if i in axis:
                    if keep_dims:
                        yield 1
                else:
                    yield ifm_shape[i]

        out_shape = tuple(calculate_expected_output_shape())

        # check IFM
        assert tuple(in_var.checked_type.shape) == ifm_shape

        if use_same_quantization:
            assert in_var.checked_type.dtype == dtype
        else:
            # in_var's dtype is equal to int8 due to TFLite's requantize
            assert in_var.checked_type.dtype == "int8"

        # check OFM
        assert tuple(out_var.checked_type.shape) == out_shape
        if use_same_quantization:
            assert out_var.checked_type.dtype == dtype
        else:
            # out_var's dtype is equal to int8 due to TFLite's requantize
            assert out_var.checked_type.dtype == "int8"

        # check expected legalization case
        if pooling_op:
            attrs = pooling_op.attrs
            assert (
                attrs.ifm_scale == attrs.ofm_scale and attrs.ifm_zero_point == attrs.ofm_zero_point
            )
        else:
            assert depthwise_op
            attrs = depthwise_op.attrs
            assert (
                attrs.ifm_scale != attrs.ofm_scale or attrs.ifm_zero_point != attrs.ofm_zero_point
            )

    rewriter = legalize.MeanRewriter()
    pattern_table = [
        (
            ethosu.MeanParams.composite_name,
            ethosu.mean_pattern(),
            lambda pat: ethosu.MeanParams(pat).is_valid(),
        ),
    ]

    mod = (
        create_relay_graph_with_same_quantization()
        if use_same_quantization
        else create_tflite_graph()
    )
    mod = partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape, axis, keepdims, relu",
    [
        [(1, 4, 2, 8), 3, False, False],
        [(1, 4, 4, 1), 3, False, True],
        [(3, 5, 7), 2, False, True],
        [(1, 4, 2, 8), 3, True, False],
        [(3, 5, 7), 2, True, False],
    ],
)
def test_ethosu_sum(ifm_shape, axis, keepdims, relu):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                op = tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)
                return tf.nn.relu(op) if relu else op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod

    def verify(ext_func):
        out_var = ext_func.body

        binary_elementwise_op = None
        pooling_op = None
        next_op = out_var
        if (
            isinstance(next_op, relay.expr.Call)
            and isinstance(next_op.op, tvm.ir.op.Op)
            and next_op.op.name == "reshape"
        ):
            next_op = next_op.args[0]
        binary_elementwise_op = next_op
        pooling_op = binary_elementwise_op.args[0]
        next_op = pooling_op.args[0]
        if (
            isinstance(next_op, relay.expr.Call)
            and isinstance(next_op.op, tvm.ir.op.Op)
            and next_op.op.name == "reshape"
        ):
            next_op = next_op.args[0]
        in_var = next_op

        def calculate_expected_output_shape():
            for i in range(len(ifm_shape)):
                if i != axis:
                    yield ifm_shape[i]
                elif keepdims:
                    yield 1

        out_shape = tuple(calculate_expected_output_shape())

        # check IFM
        assert tuple(in_var.checked_type.shape) == ifm_shape
        assert in_var.checked_type.dtype == dtype

        # check OFM
        assert tuple(out_var.checked_type.shape) == out_shape
        assert out_var.checked_type.dtype == dtype

        # check expected legalization case
        assert pooling_op
        attrs = pooling_op.attrs
        assert attrs.pooling_type == "SUM"
        if relu:
            assert attrs.activation == "CLIP"

        assert binary_elementwise_op
        attrs = binary_elementwise_op.attrs
        assert attrs.operator_type == "MUL"
        assert attrs.ifm_channels == attrs.ifm2_channels == 1
        assert attrs.ofm_dtype == "int8"

    rewriter = legalize.SumRewriter()
    pattern_table = [
        (
            ethosu.SumParams.composite_name,
            ethosu.sum_pattern(),
            lambda pat: ethosu.SumParams(pat).is_valid(),
        ),
    ]

    mod = create_tflite_graph()
    mod = partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "shapes, axis",
    [
        ([(2, 3), (4, 3)], 0),
        ([(10, 2, 1), (10, 14, 1)], 1),
        ([(10,), (13,), (14,)], 0),
        ([(1, 5, 2, 1), (1, 5, 7, 1), (1, 5, 3, 1)], 2),
    ],
)
def test_tflite_concat_legalize(shapes, axis):
    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, shapes, axis):
                op = tf.concat(shapes, axis)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            [tf.TensorSpec(shape, tf.float32) for shape in shapes], axis
        )

        def representative_dataset():
            for _ in range(100):
                datas = [np.random.rand(*shape) for shape in shapes]
                yield [data.astype(np.float32) for data in datas]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        new_concat_axis = np.sum(shape[axis] for shape in shapes)
        out_shape = list(shapes[0])
        out_shape[axis] = new_concat_axis

        op = ext_func.body
        for i, _ in enumerate(shapes):
            assert list(op.args[0][i].checked_type.shape) == list(shapes[i])

        assert list(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == "int8"

    concat_pattern_table = [
        (
            ethosu.ConcatParams.composite_name,
            ethosu.concat_pattern(),
            lambda pat: ethosu.ConcatParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={("ifm" + str(i)): shape for i, shape in enumerate(shapes)},
        dtype_dict={("ifm" + str(i)): "int8" for i, _ in enumerate(shapes)},
    )
    mod = partition_ethosu_by_table(relay_module, concat_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ConcatRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.NoOpRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]
    verify(mod["tvmgen_default_ethos_u_main_0"])


def test_tflite_sigmoid_legalize():
    dtype = "int8"
    ifm_shape = (1, 237, 91, 7)

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def sigmoid_func(self, x):
                op = tf.math.sigmoid(x)
                return op

        model = Model()
        concrete_func = model.sigmoid_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_output_type = tf.int8
        converter.inference_input_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod = ethosu.partition_for_ethosu(mod, params)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.SigmoidRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    func_body = mod["tvmgen_default_ethos_u_main_0"].body
    assert func_body.op.name == "contrib.ethosu.identity"
    assert func_body.attrs.activation == "SIGMOID"
    assert tuple(func_body.args[0].checked_type.shape) == (ifm_shape)
    assert tuple(func_body.args[1].checked_type.shape) == (256,)


@pytest.mark.parametrize(
    "ifm_shape, num_or_size_splits, axis",
    [
        ((1, 4, 6, 8), 3, 2),
        ((4, 6, 8), 2, 0),
        ((5, 15), 3, 1),
        ((3, 7), 1, 1),
        ((100,), 25, 0),
    ],
)
def test_tflite_split_legalize(ifm_shape, num_or_size_splits, axis):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x, num_or_size_splits, axis):
                op = tf.split(x, num_or_size_splits, axis=axis)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32), num_or_size_splits, axis
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        # dig out the split
        single_output_split = num_or_size_splits == 1
        split = (
            ext_func.body.tuple_value
            if single_output_split
            else ext_func.body.args[0][0].args[0].tuple_value
        )
        assert split.op.name == "split"

        # Split is specified by number of equal chunks
        assert split.attrs.indices_or_sections == num_or_size_splits

        assert split.attrs.axis == axis

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = ethosu.partition_for_ethosu(mod)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.PartitionedSplitRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape, num_or_size_splits, axis",
    [
        ((1, 4, 6, 8), (1, 3, 4), 3),
        ((10, 18, 4), (1, 4, 3, 2), 0),
        ((22, 7), (4, -1), 1),
        ((25,), (25,), 0),
    ],
)
def test_tflite_split_v_legalize(ifm_shape, num_or_size_splits, axis):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x, num_or_size_splits, axis):
                # TF split gets converted into TFLite's split_v
                op = tf.split(x, num_or_size_splits, axis=axis)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32), num_or_size_splits, axis
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        # dig out the split
        single_output_split = len(num_or_size_splits) == 1
        split = (
            ext_func.body.tuple_value
            if single_output_split
            else ext_func.body.args[0][0].args[0].tuple_value
        )
        assert split.op.name == "split"

        # Split is specified by the size of sections, so converting num_or_size_splits
        # into the indices where the tensor is split at since this is how split is represented
        # in Relay
        split_sections = [] if single_output_split else [num_or_size_splits[0]]
        for split_size in num_or_size_splits[1:-1]:
            sec = split_sections[-1] + split_size
            split_sections.append(sec)
        assert list(split.attrs.indices_or_sections) == split_sections

        assert split.attrs.axis == axis

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = ethosu.partition_for_ethosu(mod)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.PartitionedSplitRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape,ifm_scale,ifm_zp,ofm_scale,ofm_zp",
    [[(1, 8, 8, 3), 1.0, 0, 1.0, 0], [(1, 20, 30, 3), 1.345, 34, 0.32, -23]],
)
def test_ethosu_requantize(ifm_shape, ifm_scale, ifm_zp, ofm_scale, ofm_zp):
    dtype = "int8"

    def create_model():
        ifm = relay.var("ifm", shape=ifm_shape, dtype="int8")
        requantize = relay.qnn.op.requantize(
            ifm,
            relay.const(ifm_scale, dtype="float32"),
            relay.const(ifm_zp, dtype="int32"),
            relay.const(ofm_scale, dtype="float32"),
            relay.const(ofm_zp, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], requantize))

    def verify(ext_func):
        op = ext_func.body

        # Check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype

        # Check OFM
        ofm = op.checked_type
        assert list(ofm.shape) == list(ifm_shape)
        assert str(ofm.dtype) == dtype

        # Check quantization params
        assert math.isclose(op.attrs.ifm_scale, ifm_scale, abs_tol=1e-7)
        assert op.attrs.ifm_zero_point == ifm_zp
        assert math.isclose(op.attrs.ofm_scale, ofm_scale, abs_tol=1e-7)
        assert op.attrs.ofm_zero_point == ofm_zp

    rewriter = legalize.RequantizeRewriter()
    pattern_table = [
        (
            ethosu.RequantizeParams.composite_name,
            ethosu.requantize_pattern(),
            lambda pat: ethosu.RequantizeParams(pat).is_valid(),
        ),
    ]

    mod = create_model()
    mod = partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


def test_multiple_requantize_offload():
    """
    Testing requantize offload in the case one requantize operation is part of
    an existing pattern (in this case Mean: cast->mean->requantize) and the
    other is a stand-alone requantize.
    """

    def create_model():
        ifm = relay.var("input", shape=(1, 3, 3, 4), dtype="int8")
        cast = relay.cast(ifm, dtype="int32")
        mean = relay.mean(cast, axis=1, keepdims=True)
        requantize = relay.qnn.op.requantize(
            mean,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
        )
        requantize = relay.qnn.op.requantize(
            requantize,
            input_scale=relay.const(1.0, dtype="float32"),
            input_zero_point=relay.const(0, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(0, dtype="int32"),
        )
        return tvm.IRModule.from_expr(relay.Function([ifm], requantize))

    def verify(ext_func):
        # If mean operation and separate requantize were offloaded correctly,
        # there should only be a pooling operation followed by an identity
        # operation leagalized.
        op = ext_func.body
        assert op.op.name == "contrib.ethosu.identity"
        op = op.args[0]
        assert ext_func.body.args[0].op.name == "contrib.ethosu.pooling"
        op = op.args[0]
        assert isinstance(op, relay.Var)

    mod = create_model()
    mod = ethosu.partition_for_ethosu(mod)
    mod = legalize.LegalizeEthosU()(mod)
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape,axis", [((2,), 0), ((1, 3, 3), 2)])
def test_tflite_expand_dims(ifm_shape, axis):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                return tf.expand_dims(x, axis=axis)

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32)
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        expected_shape = list(ifm_shape)
        expected_shape.insert(axis, 1)

        # Check IFM
        assert list(op.args[0].checked_type.shape) == list(ifm_shape)
        assert op.args[0].checked_type.dtype == dtype

        # Check OFM
        assert list(op.checked_type.shape) == expected_shape
        assert op.checked_type.dtype == dtype

        # Check op
        assert op.op.name == "reshape"

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = ethosu.partition_for_ethosu(mod)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ExpandDimsRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ReshapeRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape,axis", [((1, 1, 2, 1), 0), ((1, 3, 3, 1), 3), ((1, 1, 2, 1), None)]
)
def test_tflite_squeeze(ifm_shape, axis):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                return tf.squeeze(x, axis=axis)

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32)
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        op = ext_func.body
        expected_shape = list(ifm_shape)
        if isinstance(axis, int):
            expected_shape = ifm_shape[:axis] + ifm_shape[axis + 1 :]
        else:
            expected_shape = list(filter(lambda a: a != 1, expected_shape))

        # Check IFM
        assert list(op.args[0].checked_type.shape) == list(ifm_shape)
        assert op.args[0].checked_type.dtype == dtype

        # Check OFM
        assert list(op.checked_type.shape) == list(expected_shape)
        assert op.checked_type.dtype == dtype

        # Check op
        assert op.op.name == "reshape"

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = ethosu.partition_for_ethosu(mod)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.SqueezeRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.ReshapeRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape,size,half_pixel",
    [
        [(1, 2, 2, 1), (4, 4), False],
        [(1, 2, 2, 1), (4, 4), True],
        [(1, 4, 7, 3), (8, 14), False],
        [(1, 3, 5, 3), (3, 5), False],
        [(1, 6, 6, 96), (12, 12), False],
        [(1, 6, 6, 96), (12, 12), True],
    ],
)
def test_tflite_resize2d_nearest_neighbor(ifm_shape, size, half_pixel):
    align_corners = False
    dtype = "int8"

    def create_tflite_graph():
        @tf.function
        def resize_model(x):
            return tf.compat.v1.image.resize_nearest_neighbor(
                x,
                size,
                align_corners=align_corners,
                half_pixel_centers=half_pixel,
            )

        concrete_func = resize_model.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod

    def verify(ext_func):
        op = ext_func.body
        in_var = op.args[0]

        # check IFM
        assert tuple(in_var.checked_type.shape) == ifm_shape
        assert in_var.checked_type.dtype == dtype

        # check OFM
        attrs = dict(op.attrs)
        out_shape = (ifm_shape[0], size[0], size[1], ifm_shape[3])
        assert tuple(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == dtype

        # Check Op attributes
        if size[0] == ifm_shape[1] and size[1] == ifm_shape[2]:
            assert op.op.name == "contrib.ethosu.identity"
        else:
            assert attrs["pooling_type"] == "AVG"
            assert attrs["upscale"] == "NEAREST"

    rewriter = legalize.Resize2dRewriter()
    pattern_table = [
        (
            ethosu.Resize2dParams.composite_name,
            ethosu.resize2d_pattern(),
            lambda pat: ethosu.Resize2dParams(pat).is_valid(),
        ),
    ]

    mod = create_tflite_graph()
    mod = partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape,size,align_corners",
    [
        [(1, 2, 2, 1), (4, 4), False],
        [(1, 4, 7, 3), (8, 14), False],
        [(1, 2, 2, 1), (3, 3), True],
        [(1, 4, 7, 3), (7, 13), True],
        [(1, 3, 5, 3), (3, 5), False],
    ],
)
def test_tflite_resize2d_bilinear(ifm_shape, size, align_corners):
    dtype = "int8"

    def create_tflite_graph():
        @tf.function
        def resize_model(x):
            return tf.compat.v1.image.resize_bilinear(
                x, size, align_corners=align_corners, half_pixel_centers=False
            )

        concrete_func = resize_model.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model, 0)

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod

    def verify(ext_func):
        op = ext_func.body
        in_var = op.args[0]

        # check IFM
        assert tuple(in_var.checked_type.shape) == ifm_shape
        assert in_var.checked_type.dtype == dtype

        # check OFM
        attrs = dict(op.attrs)
        out_shape = (ifm_shape[0], size[0], size[1], ifm_shape[3])
        assert tuple(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == dtype

        # Check Op attributes
        if size[0] == ifm_shape[1] and size[1] == ifm_shape[2]:
            assert op.op.name == "contrib.ethosu.identity"
        else:
            assert attrs["pooling_type"] == "AVG"
            assert attrs["upscale"] == "NEAREST"

            # Check padding
            if align_corners:
                assert list(attrs["padding"]) == [0, 0, 0, 0]
            else:
                assert list(attrs["padding"]) == [0, 0, 1, 1]

    rewriter = legalize.Resize2dRewriter()
    pattern_table = [
        (
            ethosu.Resize2dParams.composite_name,
            ethosu.resize2d_pattern(),
            lambda pat: ethosu.Resize2dParams(pat).is_valid(),
        ),
    ]

    mod = create_tflite_graph()
    mod = partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape,ofm_shape,kernel_shape,padding",
    [
        [(1, 2, 2, 1), (1, 4, 4, 1), (3, 3), "SAME"],
        [(1, 2, 2, 1), (1, 9, 9, 1), (7, 7), "VALID"],
        [(1, 2, 4, 3), (1, 4, 8, 3), (3, 3), "SAME"],
        [(1, 10, 5, 3), (1, 21, 13, 3), (3, 5), "VALID"],
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_tflite_transpose_convolution(ifm_shape, ofm_shape, kernel_shape, padding, has_bias):
    dtype = "int8"
    dilations = (1, 1)
    strides = (2, 2)

    def create_tflite_graph():
        @tf.function
        def conv2d_transpose(x):
            bias_shape = ofm_shape[3]
            bias = tf.constant(np.random.uniform(size=bias_shape), dtype=tf.float32)
            weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], ofm_shape[3]]
            weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
            tf_strides = [1, strides[0], strides[1], 1]
            op = tf.nn.conv2d_transpose(
                x,
                weight,
                output_shape=ofm_shape,
                strides=tf_strides,
                padding=padding,
                dilations=dilations,
            )
            if has_bias:
                op = tf.nn.bias_add(op, bias)
            return op

        concrete_func = conv2d_transpose.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model, 0)

        mod, params = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod, params

    def verify(ext_func):
        strided_slice = ext_func.body
        conv = strided_slice.args[0]
        ofm_channels = conv.attrs.ofm_channels

        # Check IFM
        ifm = conv.args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype
        assert ifm.shape[3] == ofm_channels

        # Check OFM
        ofm = strided_slice.checked_type
        assert list(ofm.shape) == list(ofm_shape)
        assert str(ofm.dtype) == dtype
        assert ofm.shape[3] == ofm_channels

        # Check weights
        weights_ohwi = conv.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert list(weights_ohwi.shape) == [
            ofm_channels,
            kernel_shape[0],
            kernel_shape[1],
            ifm_shape[3],
        ]

        # Check that scale_bias matches weight tensor
        assert list(conv.args[2].checked_type.shape)[0] == ofm_channels

        # Calculate expected padding for conv2d op
        if padding == "VALID":
            expected_padding = [0, 0, 0, 0]
        elif padding == "SAME":
            pad_top, pad_bottom = get_pad_value(ofm_shape[1], kernel_shape[0], strides[0])
            pad_left, pad_right = get_pad_value(ofm_shape[2], kernel_shape[1], strides[1])
            expected_padding = [pad_top, pad_left, pad_bottom, pad_right]
        pad_top = kernel_shape[0] - 1 - expected_padding[0]
        pad_left = kernel_shape[1] - 1 - expected_padding[1]
        pad_bottom = kernel_shape[0] - 1 - expected_padding[2]
        pad_right = kernel_shape[1] - 1 - expected_padding[3]
        if strides == [2, 2]:
            pad_bottom -= 1
            pad_right -= 1
        expected_padding = [pad_top, pad_left, pad_bottom, pad_right]
        assert list(conv.attrs.padding) == list(expected_padding)

        assert list(conv.attrs.strides) == [1, 1]

    rewriter = legalize.Conv2DTransposeRewriter()
    pattern_table = [
        (
            ethosu.QnnConv2DTransposeParams.composite_name,
            ethosu.qnn_conv2d_transpose_pattern(),
            lambda pat: ethosu.QnnConv2DTransposeParams(pat).is_valid(),
        ),
    ]

    mod, params = create_tflite_graph()
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shapes,axis",
    [
        ([(1, 2, 2), (1, 2, 2), (1, 2, 2)], 2),
        ([(5, 4), (5, 4)], 1),
        ([(1,), (1,)], 0),
        ([(3, 1), (3, 1), (3, 1), (3, 1)], 0),
    ],
)
def test_tflite_pack(ifm_shapes, axis):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, inputs, axis):
                return tf.stack(inputs, axis=axis)

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            [tf.TensorSpec(shape, tf.float32) for shape in ifm_shapes], axis
        )

        def representative_dataset():
            for _ in range(100):
                datas = [np.random.rand(*shape) for shape in ifm_shapes]
                yield [data.astype(np.float32) for data in datas]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        new_pack_axis = len(ifm_shapes)
        ifm_shape = list(ifm_shapes[0])
        op = ext_func.body

        after_reshape = ifm_shape[:axis] + [1] + ifm_shape[axis:]
        out_shape = ifm_shape[:axis] + [new_pack_axis] + ifm_shape[axis:]

        assert op.op.name == "concatenate"

        # Check shapes after expand_dims (legalized as reshape)
        for i in range(len(ifm_shapes)):
            assert list(op.args[0][i].checked_type.shape) == after_reshape
            assert op.args[0][i].checked_type.dtype == dtype

        # Check output
        assert list(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == dtype

    pack_pattern_table = [
        (
            ethosu.ConcatParams.composite_name,
            ethosu.concat_pattern(),
            lambda pat: ethosu.ConcatParams(pat).is_valid(),
        ),
        (
            ethosu.ExpandDimsParams.composite_name,
            ethosu.expand_dims_pattern(),
            lambda pat: ethosu.ExpandDimsParams(pat).is_valid(),
        ),
        (
            ethosu.ReshapeParams.composite_name,
            ethosu.reshape_pattern(),
            lambda pat: ethosu.ReshapeParams(pat).is_valid(),
        ),
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    relay_module, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={("ifm" + str(i)): shape for i, shape in enumerate(ifm_shapes)},
        dtype_dict={("ifm" + str(i)): dtype for i, _ in enumerate(ifm_shapes)},
    )
    mod = partition_ethosu_by_table(relay_module, pack_pattern_table)

    seq = [
        legalize.ConcatRewriter(),
        legalize.ExpandDimsRewriter(),
        legalize.ReshapeRewriter(),
        legalize.NoOpRewriter(),
    ]
    for legalizer in seq:
        mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
            legalizer, mod["tvmgen_default_ethos_u_main_0"]
        )
    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize(
    "ifm_shape,axis",
    [[(1, 2, 3, 4), 1], [(2, 3), 1], [(5, 6, 7), 2]],
)
def test_tflite_unpack(ifm_shape, axis):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x, axis):
                return tf.unstack(x, axis=axis)

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32), axis
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        outputs = ext_func.body.args[0].fields
        shape = list(ifm_shape)
        unpacked_shape = shape[:axis] + shape[axis + 1 :]
        split_shape = shape[:axis] + [1] + shape[axis + 1 :]

        assert len(outputs) == shape[axis]

        for i, output in enumerate(outputs):
            expr = output.args[0].args[0]
            expr = expr.tuple_value[expr.index]
            expr = expr.args[0]

            # Checking expected unpacked output shape.
            # Squeeze is legalized to a reshape.
            assert expr.op.name == "reshape"
            assert list(expr.checked_type.shape) == unpacked_shape
            assert output.checked_type.dtype == dtype

            expr = expr.args[0]
            expr = expr.tuple_value[expr.index]
            expr = expr.args[0]

            # Check input is split correctly
            assert list(expr.args[0].checked_type.shape) == shape
            assert list(expr.checked_type.shape) == split_shape
            assert expr.checked_type.dtype == dtype

            # Check split attrs
            begin_shape = [0] * len(ifm_shape)
            begin_shape[axis] = i
            assert list(expr.attrs.begin) == begin_shape
            end_shape = shape[:axis] + [i + 1] + shape[axis + 1 :]
            assert list(expr.attrs.end) == end_shape
            assert list(expr.attrs.strides) == [1]

    pack_pattern_table = [
        (
            ethosu.SplitParams.composite_name,
            ethosu.split_pattern(),
            lambda pat: ethosu.SplitParams(pat).is_valid(),
        ),
        (
            ethosu.SqueezeParams.composite_name,
            ethosu.squeeze_pattern(),
            lambda pat: ethosu.SqueezeParams(pat).is_valid(),
        ),
        (
            ethosu.ReshapeParams.composite_name,
            ethosu.reshape_pattern(),
            lambda pat: ethosu.ReshapeParams(pat).is_valid(),
        ),
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = partition_ethosu_by_table(mod, pack_pattern_table)

    seq = [
        legalize.PartitionedSplitRewriter(),
        legalize.SplitRewriter(),
        legalize.SqueezeRewriter(),
        legalize.ReshapeRewriter(),
        legalize.NoOpRewriter(),
    ]
    for legalizer in seq:

        mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
            legalizer, mod["tvmgen_default_ethos_u_main_0"]
        )
    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 15, 15, 3), (1, 8, 9, 1)])
@pytest.mark.parametrize("alpha", [0.2, 0.634])
def test_tflite_leaky_relu(ifm_shape, alpha):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def leaky_relu_func(self, x):
                return tf.nn.leaky_relu(x, alpha=alpha)

        model = Model()
        concrete_func = model.leaky_relu_func.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32),
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    def verify(ext_func):
        func_body = ext_func.body
        assert func_body.op.name == "contrib.ethosu.identity"
        assert func_body.attrs.activation == "LUT"
        assert tuple(func_body.args[0].checked_type.shape) == (ifm_shape)
        assert tuple(func_body.args[1].checked_type.shape) == (256,)

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod = ethosu.partition_for_ethosu(mod)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.LeakyReLURewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod["tvmgen_default_ethos_u_main_0"] = relay.transform.InferType()(mod)[
        "tvmgen_default_ethos_u_main_0"
    ]
    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 14), (1, 151)])
@pytest.mark.parametrize("ofm_channels", [32, 64])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("activation_function", ["RELU", "NONE"])
def test_tflite_fully_connected(
    ifm_shape,
    ofm_channels,
    use_bias,
    activation_function,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def fully_connected(self, x):
                bias_shape = ofm_channels
                bias = tf.constant(np.random.uniform(size=bias_shape), dtype=tf.float32)
                w = tf.constant(
                    np.random.uniform(size=[ifm_shape[1], ofm_channels]),
                    dtype=tf.float32,
                )
                x = tf.matmul(x, w)
                if use_bias:
                    x = tf.nn.bias_add(x, bias)
                if activation_function:
                    x = tf.nn.relu(x)
                return x

        model = Model()
        concrete_func = model.fully_connected.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        op = ext_func.body.args[0]
        ofm_channels = op.attrs.ofm_channels

        # check IFM
        ifm = op.args[0].checked_type
        assert list(ifm.shape) == [1, 1] + list(ifm_shape)
        assert str(ifm.dtype) == dtype

        # check OFM
        ofm = op.checked_type
        assert list(ofm.shape) == [1, 1, 1, ofm_channels]
        assert str(ofm.dtype) == dtype

        # check weights
        weights_ohwi = op.args[1].data.asnumpy()
        assert str(weights_ohwi.dtype) == dtype
        assert list(weights_ohwi.shape) == [ofm_channels, 1, 1, ifm_shape[1]]

        # Check that scale_bias matches weight tensor
        assert list(op.args[2].checked_type.shape)[0] == ofm_channels

        assert list(op.attrs.padding) == [0, 0, 0, 0]
        assert list(op.attrs.strides) == [1, 1]
        assert list(op.attrs.dilation) == [1, 1]
        if activation_function == "RELU":
            assert str(op.attrs.activation) == "CLIP"

    fc_pattern_table = [
        (
            ethosu.FullyConnectedParams.composite_name,
            ethosu.qnn_fc_pattern(),
            lambda pat: ethosu.FullyConnectedParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, fc_params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], fc_params)
    mod = partition_ethosu_by_table(mod, fc_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.FullyConnectedRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 5, 5, 3), (1, 12, 9, 1)])
def test_tflite_hard_swish(ifm_shape):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                op = tf.keras.layers.Lambda(
                    lambda x: x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
                )(x)
                return op

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, tf.float32)
        )

        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        return tflite_model

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod = ethosu.partition_for_ethosu(mod, params)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.HardSwishRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    func_body = mod["tvmgen_default_ethos_u_main_0"].body
    assert func_body.op.name == "contrib.ethosu.identity"
    assert func_body.attrs.activation == "LUT"
    assert tuple(func_body.args[0].checked_type.shape) == (ifm_shape)
    assert tuple(func_body.args[1].checked_type.shape) == (256,)


def test_tflite_softmax():
    np.random.seed(0)
    dtype = "int8"
    ifm_shape = (1, 12)

    def create_tflite_graph():
        @tf.function
        def softmax(x):
            return tf.nn.softmax(x)

        concrete_func = softmax.get_concrete_function(tf.TensorSpec(ifm_shape, dtype=tf.float32))
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.uniform(low=-1, high=2, size=tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        out_op = ext_func.body
        ops = []
        # List of expected operations, their type and activation parameters if it exists
        expected_ops_params = [
            ("reshape", None, [None, None, None, None, None, None]),
            ("reshape", None, [None, None, None, None, None, None]),
            ("contrib.ethosu.pooling", "MAX", [0.011756093241274357, -43, None, None, 0.0, -43]),
            (
                "contrib.ethosu.binary_elementwise",
                "SUB",
                [0.011756093241274357, -43, 0.0, -43, 1.0, 127],
            ),
            ("contrib.ethosu.binary_elementwise", "SHR", [1.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.pooling", "SUM", [0.0, 0, None, None, 0.0, -43]),
            ("contrib.ethosu.unary_elementwise", "CLZ", [0.0, 0, None, None, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "SUB", [0.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "SHL", [0.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "SUB", [0.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "SHL", [0.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "ADD", [0.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "ADD", [2.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "SUB", [2.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [2.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "ADD", [1.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "SUB", [2.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [2.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "ADD", [1.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "SUB", [2.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [2.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "ADD", [1.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 0.0, 0, 1.0, 0]),
            ("contrib.ethosu.binary_elementwise", "MUL", [1.0, 0, 1.0, 0, 2.0, 0]),
            ("contrib.ethosu.binary_elementwise", "SUB", [0.0, 0, 0.0, 0, 0.0, -43]),
            ("contrib.ethosu.binary_elementwise", "SHR", [2.0, 0, 0.0, 0, 0.00390625, -128]),
            ("reshape", None, [None, None, None, None, None, None]),
        ]

        def get_attr_value(op, attr_name):
            if hasattr(op.attrs, attr_name):
                return op.attrs[attr_name]
            else:
                return None

        def get_op_type(op):
            if hasattr(op.attrs, "pooling_type"):
                return op.attrs.pooling_type
            elif hasattr(op.attrs, "operator_type"):
                return op.attrs.operator_type
            return None

        def get_activation_params(op):
            activation_params = []
            activation_params.append(get_attr_value(op, "ifm_scale"))
            activation_params.append(get_attr_value(op, "ifm_zero_point"))
            activation_params.append(get_attr_value(op, "ifm2_scale"))
            activation_params.append(get_attr_value(op, "ifm2_zero_point"))
            activation_params.append(get_attr_value(op, "ofm_scale"))
            activation_params.append(get_attr_value(op, "ofm_zero_point"))
            return activation_params

        def _visit(stmt):
            if isinstance(stmt, relay.expr.Call):
                ops.append(stmt)

        relay.analysis.post_order_visit(out_op, _visit)

        # check IFM
        ifm = ops[0].args[0].checked_type
        assert list(ifm.shape) == list(ifm_shape)
        assert str(ifm.dtype) == dtype

        # check OFM
        ofm = out_op.checked_type
        assert list(ofm.shape) == list(ifm_shape)
        assert ofm.dtype == dtype

        # check operations
        for op, expected_op_params in zip(ops, expected_ops_params):
            activation_params = get_activation_params(op)
            expected_op_name, expected_op_type, expected_activation_params = expected_op_params
            assert op.op.name == expected_op_name
            assert expected_op_type == get_op_type(op)
            for activation_param, expected_activation_param in zip(
                activation_params, expected_activation_params
            ):
                if isinstance(activation_param, float):
                    assert math.isclose(expected_activation_param, activation_param, abs_tol=1e-7)
                else:
                    assert expected_activation_param == activation_param

    softmax_pattern_table = [
        (
            ethosu.SoftMaxParams.composite_name,
            ethosu.softmax_pattern(),
            lambda pat: ethosu.SoftMaxParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, softmax_pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.SoftmaxRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    mod = relay.transform.InferType()(mod)

    verify(mod["tvmgen_default_ethos_u_main_0"])


@pytest.mark.parametrize("ifm_shape", [(1, 55, 55, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 3)])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (1, 1))])
@pytest.mark.parametrize("op_padding", ["SAME", "VALID"])
@pytest.mark.parametrize("sep_padding", [(0, 0, 1, 1), (7, 5, 4, 5)])
@pytest.mark.parametrize(
    "op_pairs", [("conv2d", "conv2d"), ("depthwise", "depthwise"), ("conv2d", "depthwise")]
)
def test_tflite_shared_pad_legalize(
    ifm_shape,
    kernel_shape,
    strides,
    dilation,
    op_padding,
    sep_padding,
    op_pairs,
):
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def tf_function(self, x):
                def make_depthwise_or_conv2d(pair_idx):
                    if op_pairs[pair_idx] == "depthwise":
                        weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 1]
                        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                        return tf.nn.depthwise_conv2d(
                            x, weight, strides=tf_strides, padding=op_padding, dilations=dilation
                        )
                    weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], 3]
                    weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                    return tf.nn.conv2d(
                        x,
                        weight,
                        strides=tf_strides,
                        padding=op_padding,
                        dilations=dilation,
                    )

                x = tf.pad(
                    x,
                    [
                        [0, 0],
                        [sep_padding[0], sep_padding[2]],
                        [sep_padding[1], sep_padding[3]],
                        [0, 0],
                    ],
                    "CONSTANT",
                )

                # The input strides to the TensorFlow API needs to be of shape 1x4
                tf_strides = [1, strides[0], strides[1], 1]

                x1 = make_depthwise_or_conv2d(0)
                x2 = make_depthwise_or_conv2d(1)

                x3 = tf.math.add(x1, x2)
                return x3

        model = Model()
        concrete_func = model.tf_function.get_concrete_function(
            tf.TensorSpec(ifm_shape, dtype=tf.float32)
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                data = np.random.rand(*tuple(ifm_shape))
                yield [data.astype(np.float32)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    conv2d_pattern_table = [
        (
            ethosu.QnnConv2DParams.composite_name,
            ethosu.qnn_conv2d_pattern(),
            lambda pat: ethosu.QnnConv2DParams(pat).is_valid(),
        ),
        (
            ethosu.QnnDepthwiseConv2DParams.composite_name,
            ethosu.qnn_depthwise_conv2d_pattern(),
            lambda pat: ethosu.QnnDepthwiseConv2DParams(pat).is_valid(),
        ),
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, conv2d_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        [legalize.Conv2DRewriter(), legalize.DepthwiseConv2DRewriter()],
        mod["tvmgen_default_ethos_u_main_0"],
    )
    mod["tvmgen_default_ethos_u_main_1"] = dataflow_pattern.rewrite(
        [legalize.Conv2DRewriter(), legalize.DepthwiseConv2DRewriter()],
        mod["tvmgen_default_ethos_u_main_1"],
    )

    if op_pairs[0] == "depthwise":
        assert (
            mod["tvmgen_default_ethos_u_main_0"].body.op.name == "contrib.ethosu.depthwise_conv2d"
        )
    else:
        assert mod["tvmgen_default_ethos_u_main_0"].body.op.name == "contrib.ethosu.conv2d"

    if op_pairs[1] == "depthwise":
        assert (
            mod["tvmgen_default_ethos_u_main_1"].body.op.name == "contrib.ethosu.depthwise_conv2d"
        )
    else:
        assert mod["tvmgen_default_ethos_u_main_1"].body.op.name == "contrib.ethosu.conv2d"


def test_tflite_matmul():
    ifm_shape = [1, 4]
    ifm2_shape = [2, 4]
    ifm_shapes = [ifm_shape, ifm2_shape]
    ofm_shape = [ifm_shape[0], ifm2_shape[0]]
    dtype = "int8"

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def matmul(self, x, y):
                res = tf.matmul(x, y, transpose_b=True)
                return res

        model = Model()
        concrete_func = model.matmul.get_concrete_function(
            *[tf.TensorSpec(shape, tf.float32) for shape in ifm_shapes]
        )
        # Convert the model
        def representative_dataset():
            for _ in range(100):
                datas = [np.random.rand(*shape) for shape in ifm_shapes]
                yield [data.astype(np.float32) for data in datas]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()
        return tflite_model

    def verify(ext_func):
        ofm = ext_func.body
        ops = []

        def _visit(stmt):
            if isinstance(stmt, relay.expr.Call):
                ops.append(stmt)

        relay.analysis.post_order_visit(ofm, _visit)
        ofm_checked_type = ofm.checked_type
        ofm_channels = ofm_shape[-1]

        # check IFM
        ifm = ops[1].checked_type
        assert list(ifm.shape) == ifm_shape
        assert str(ifm.dtype) == dtype

        # check IFM2
        ifm2 = ops[3].checked_type
        assert list(ifm2.shape) == ifm2_shape
        assert str(ifm2.dtype) == dtype

        # check split
        split = ops[4]
        split_checked_types = list(split.checked_type.fields)
        assert split.op.name == "split"
        assert split.attrs.axis == 0
        assert int(split.attrs.indices_or_sections) == ofm_channels
        for split_checked_type in split_checked_types:
            assert list(split_checked_type.shape) == ifm_shape
            assert str(split_checked_type.dtype) == dtype

        # check MUL
        mul_ops = [ops[6], ops[10]]
        for mul_op in mul_ops:
            assert mul_op.op.name == "contrib.ethosu.binary_elementwise"
            assert mul_op.attrs.operator_type == "MUL"
            assert mul_op.attrs.ofm_dtype == "int32"

        # check reduce sum
        reduce_sum_ops = [ops[7], ops[11]]
        for reduce_sum_op in reduce_sum_ops:
            assert reduce_sum_op.op.name == "contrib.ethosu.pooling"
            assert reduce_sum_op.attrs.pooling_type == "SUM"
            assert list(reduce_sum_op.checked_type.shape) == [1, 1, 1, 1]

        # check concatenation
        concatenation = ofm.args[0]
        concatenation_shape = concatenation.checked_type.shape
        assert concatenation.op.name == "concatenate"
        assert list(concatenation_shape) == [1, 1, 1, ofm_channels]

        # check OFM
        assert ofm.op.name == "reshape"
        assert list(ofm_checked_type.shape) == ofm_shape
        assert str(ofm_checked_type.dtype) == dtype

    matmul_pattern_table = [
        (
            ethosu.MatMulParams.composite_name,
            ethosu.matmul_pattern(),
            lambda pat: ethosu.MatMulParams(pat).is_valid(),
        )
    ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={("ifm" + str(i)): shape for i, shape in enumerate(ifm_shapes)},
        dtype_dict={("ifm" + str(i)): dtype for i, _ in enumerate(ifm_shapes)},
    )

    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = partition_ethosu_by_table(mod, matmul_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.MatMulRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )

    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    tvm.testing.main()
