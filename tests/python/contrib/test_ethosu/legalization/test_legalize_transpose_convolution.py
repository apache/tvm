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
from tests.python.contrib.test_ethosu.legalization import legalize_infra
from tvm.relay.frontend.tflite import get_pad_value


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
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
