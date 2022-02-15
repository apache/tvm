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
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
