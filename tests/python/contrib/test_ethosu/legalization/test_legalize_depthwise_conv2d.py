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

import numpy as np
import tensorflow as tf
import tflite.Model

from tvm.relay.backend.contrib.ethosu import legalize
from tvm import relay
from tvm.relay import dataflow_pattern
from tvm.relay.op.contrib import ethosu
from tvm.relay.build_module import bind_params_by_name
from tests.python.contrib.test_ethosu import infra
from tests.python.contrib.test_ethosu.legalization import legalize_infra


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
    mod = legalize_infra.partition_ethosu_by_table(mod, depthwise_pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        legalize.DepthwiseConv2DRewriter(), mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
