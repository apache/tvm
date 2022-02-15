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
from tests.python.contrib.test_ethosu import infra
from tests.python.contrib.test_ethosu.legalization import legalize_infra


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
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
