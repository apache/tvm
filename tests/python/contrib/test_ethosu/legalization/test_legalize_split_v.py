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


if __name__ == "__main__":
    pytest.main([__file__])
