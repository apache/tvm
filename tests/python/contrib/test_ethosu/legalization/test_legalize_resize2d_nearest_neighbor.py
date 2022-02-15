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
    "ifm_shape,size",
    [
        [(1, 2, 2, 1), (4, 4)],
        [(1, 4, 7, 3), (8, 14)],
        [(1, 3, 5, 3), (3, 5)],
    ],
)
def test_tflite_resize2d_nearest_neighbor(ifm_shape, size):
    align_corners = False
    dtype = "int8"

    def create_tflite_graph():
        @tf.function
        def resize_model(x):
            return tf.compat.v1.image.resize_nearest_neighbor(
                x, size, align_corners=align_corners, half_pixel_centers=False
            )

        concrete_func = resize_model.get_concrete_function(
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

        mod, _ = relay.frontend.from_tflite(
            tflite_model,
            shape_dict={"input": ifm_shape},
            dtype_dict={"input": dtype},
        )
        return mod

    def verify(ext_func):
        op = ext_func.body
        in_var = op.args[0]

        # check IFM
        assert tuple(in_var.checked_type.shape) == ifm_shape
        assert in_var.checked_type.dtype == dtype

        # check OFM
        attrs = dict(op.attrs)
        out_shape = (ifm_shape[0], size[0], size[1], ifm_shape[3])
        assert tuple(op.checked_type.shape) == out_shape
        assert op.checked_type.dtype == dtype

        # Check Op attributes
        if size[0] == ifm_shape[1] and size[1] == ifm_shape[2]:
            assert op.op.name == "contrib.ethosu.identity"
        else:
            assert attrs["pooling_type"] == "AVG"
            assert attrs["upscale"] == "NEAREST"

    rewriter = legalize.Resize2dRewriter()
    pattern_table = [
        (
            ethosu.Resize2dParams.composite_name,
            ethosu.resize2d_pattern(),
            lambda pat: ethosu.Resize2dParams(pat).is_valid(),
        ),
    ]

    mod = create_tflite_graph()
    mod = legalize_infra.partition_ethosu_by_table(mod, pattern_table)
    mod["tvmgen_default_ethos_u_main_0"] = dataflow_pattern.rewrite(
        rewriter, mod["tvmgen_default_ethos_u_main_0"]
    )
    verify(mod["tvmgen_default_ethos_u_main_0"])


if __name__ == "__main__":
    pytest.main([__file__])
