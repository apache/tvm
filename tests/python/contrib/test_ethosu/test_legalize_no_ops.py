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

from tvm import relay
from tvm.relay.backend.contrib.ethosu import legalize
from tvm.relay.op.contrib import ethosu
from tvm.relay.build_module import bind_params_by_name

# There's a bug in TFLite converter which doesn't allow us to create single operator
# reshape and strided_slice graphs, so in order to have some testing coverage for these
# operators starting from TFLite, we test them alongside other operators
def test_tflite_reshape_and_strided_slice():
    dtype = "int8"
    ifm_shape = [1, 8, 3, 6]

    def create_tflite_graph():
        class Model(tf.Module):
            @tf.function
            def model_func(self, x):
                weight_shape = [3, 3, 6, 1]  # HWO1
                weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
                op = tf.nn.depthwise_conv2d(x, weight, strides=[1, 1, 1, 1], padding="SAME")
                op = tf.nn.relu(op)
                op = tf.reshape(op, [1, 8, 6, 3])
                op = tf.nn.pool(op, [2, 2], "MAX")
                op = tf.strided_slice(op, [0, 2, 3, 1], [1, 6, 5, 2])
                return op

        model = Model()
        concrete_func = model.model_func.get_concrete_function(
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

    def verify(func):
        # This TFLite graph gets lowered into
        # deptwhise_conv2d -> clip -> reshape -> max_pool -> strided_slice -> reshape
        # which gets legalized into ethosu_depthwise_conv2d -> reshape -> ehtosu_identity
        # -> ethosu_pooling -> strided_slice -> identity -> reshape -> identity

        identity3 = func.body
        reshape2 = identity3.args[0]
        identity2 = reshape2.args[0]
        strided_slice = identity2.args[0]
        max_pool = strided_slice.args[0]
        identity1 = max_pool.args[0]
        reshape1 = identity1.args[0]
        depthwise_conv2d = reshape1.args[0]

        assert identity3.op.name == "contrib.ethosu.identity"
        assert reshape2.op.name == "reshape"
        assert identity2.op.name == "contrib.ethosu.identity"
        assert strided_slice.op.name == "strided_slice"
        assert max_pool.op.name == "contrib.ethosu.pooling"
        assert identity1.op.name == "contrib.ethosu.identity"
        assert reshape1.op.name == "reshape"
        assert depthwise_conv2d.op.name == "contrib.ethosu.depthwise_conv2d"

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"input": ifm_shape},
        dtype_dict={"input": dtype},
    )
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = ethosu.partition_for_ethosu(mod)
    mod = legalize.LegalizeEthosU()(mod)

    verify(mod["tvmgen_default_ethos_u_main_0"])
