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


if __name__ == "__main__":
    pytest.main([__file__])
