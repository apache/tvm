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
from tests.python.contrib.test_ethosu.end_to_end import comparison_infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32", "ethos-u65-256"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize(
    "ifm_shape,ofm_shape,kernel_shape,padding",
    [
        [(1, 2, 2, 1), (1, 4, 4, 1), (3, 3), "SAME"],
        [(1, 2, 2, 1), (1, 9, 9, 1), (7, 7), "VALID"],
        [(1, 2, 4, 3), (1, 4, 8, 3), (5, 3), "SAME"],
        [(1, 10, 5, 3), (1, 21, 13, 3), (3, 5), "VALID"],
    ],
)
@pytest.mark.parametrize("has_bias", [False, True])
def test_tflite_transpose_convolution(
    accel_type, ifm_shape, ofm_shape, kernel_shape, padding, has_bias
):
    dilations = (1, 1)
    strides = (2, 2)

    @tf.function
    def conv2d_transpose(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], ifm_shape[3], ofm_shape[3]]
        weight = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        bias_shape = ofm_shape[3]
        bias = tf.constant(np.random.uniform(size=bias_shape), dtype=tf.float32)
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

    comparison_infra._compare_tvm_with_tflite(conv2d_transpose, [ifm_shape], accel_type=accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
