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


@pytest.mark.parametrize("ifm_shape", [(1, 299, 299, 3), (1, 55, 55, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 2), (1, 3)])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 1)), ((3, 2), (1, 1))])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("activation", ["NONE", "RELU"])
def test_ethosu_conv2d_single(
    ifm_shape,
    kernel_shape,
    strides,
    dilation,
    padding,
    accel_type,
    activation,
):
    @tf.function
    def conv2d_single(x):
        # Use tf.nn API to create the model
        tf_strides = [1, strides[0], strides[1], 1]
        op = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[kernel_shape[0], kernel_shape[1], 3, 3]),
                dtype=tf.float32,
            ),
            strides=tf_strides,
            padding=padding,
            dilations=dilation,
        )
        if activation:
            op = tf.nn.relu(op)
        return op

    comparison_infra._compare_tvm_with_tflite(conv2d_single, [ifm_shape], accel_type)


@pytest.mark.parametrize("ifm_shape", [(1, 214, 227, 3), (1, 27, 42, 3)])
@pytest.mark.parametrize("kernel_shape", [(3, 2), (1, 3)])
@pytest.mark.parametrize("strides, dilation", [((1, 1), (2, 1)), ((3, 2), (1, 1))])
@pytest.mark.parametrize("padding", ["SAME", "VALID"])
@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
@pytest.mark.parametrize("activation", ["NONE", "RELU"])
def test_ethosu_conv2d_double(
    ifm_shape,
    kernel_shape,
    strides,
    dilation,
    padding,
    accel_type,
    activation,
):
    @tf.function
    def conv2d_double(x):
        # Use tf.nn API to create the model with two convolutions
        op = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[kernel_shape[0], kernel_shape[1], 3, 3]),
                dtype=tf.float32,
            ),
            strides=strides,
            padding=padding,
            data_format="NHWC",
            dilations=dilation,
        )
        # Second convolution
        op2 = tf.nn.conv2d(
            op,
            filters=tf.constant(
                np.random.uniform(size=(kernel_shape[0], kernel_shape[1], 3, 3)),
                dtype=tf.float32,
            ),
            strides=strides,
            padding=padding,
            data_format="NHWC",
            dilations=dilation,
        )
        if activation:
            op2 = tf.nn.relu(op2)
        return op2

    comparison_infra._compare_tvm_with_tflite(conv2d_double, [ifm_shape], accel_type)


if __name__ == "__main__":
    pytest.main([__file__])
