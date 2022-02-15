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
from tests.python.contrib.test_ethosu.legalization import legalize_infra


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
    mod = legalize_infra.partition_ethosu_by_table(relay_module, concat_pattern_table)

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


if __name__ == "__main__":
    pytest.main([__file__])
