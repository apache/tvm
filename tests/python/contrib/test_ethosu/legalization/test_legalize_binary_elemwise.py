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
@pytest.mark.parametrize("activation_function", ["NONE", "RELU"])
def test_tflite_binary_elemwise_legalize(
    operator_type,
    ifm_shape,
    ifm2_shape,
    reversed_operands,
    activation_function,
):
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
                if activation_function == "RELU":
                    op = tf.nn.relu(op)
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
        shapes_padded = [[1] * (4 - len(s)) + s for s in shapes]
        out_padded = [1] * (4 - len(out_shape)) + out_shape
        if op.op.name != "contrib.ethosu.binary_elementwise":
            has_reshaped_output = True
            op = op.args[0]

        assert list(op.args[0].checked_type.shape) == shapes_padded[ifm_index]
        assert list(op.args[1].checked_type.shape) == shapes_padded[ifm2_index]
        assert op.args[0].checked_type.dtype == dtype
        assert list(op.checked_type.shape) == out_padded
        assert op.checked_type.dtype == dtype
        assert op.attrs.operator_type == operator_type
        assert op.attrs.reversed_operands == reversed_operands
        if activation_function == "RELU":
            assert str(op.attrs.activation) == "CLIP"

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
        rewriter = legalize.MinRewriter()
        pattern_table = [
            (
                ethosu.MinParams.composite_name,
                ethosu.minimum_pattern(),
                lambda pat: ethosu.MinParams(pat).is_valid(),
            ),
        ]
    elif operator_type == "MAX":
        rewriter = legalize.MaxRewriter()
        pattern_table = [
            (
                ethosu.MaxParams.composite_name,
                ethosu.maximum_pattern(),
                lambda pat: ethosu.MaxParams(pat).is_valid(),
            ),
        ]

    tflite_graph = create_tflite_graph()
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, _ = relay.frontend.from_tflite(
        tflite_model,
        shape_dict={"x": ifm_shape, "y": ifm2_shape},
        dtype_dict={"x": dtype, "y": dtype},
    )
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)

    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
