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
    mod = legalize_infra.partition_ethosu_by_table(mod, pack_pattern_table)

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


if __name__ == "__main__":
    pytest.main([__file__])
