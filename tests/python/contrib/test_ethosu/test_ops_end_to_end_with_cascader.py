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

from . import infra


ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32"]


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_ethosu_cascade_conv2d(accel_type):
    np.random.seed(0)
    ifm_shape = (1, 5, 39, 3)

    @tf.function
    def tf_graph(x):
        ofm_channels = 5
        conv2d = tf.nn.conv2d(
            x,
            filters=tf.constant(
                np.random.uniform(size=[3, 2, ifm_shape[3], ofm_channels]),  # HWIO
                dtype=tf.float32,
            ),
            strides=(1, 1),
            padding="VALID",
            dilations=(2, 1),
        )
        conv2d = tf.nn.conv2d(
            conv2d,
            filters=tf.constant(
                np.random.uniform(size=(1, 1, ofm_channels, 3)),  # HWIO
                dtype=tf.float32,
            ),
            strides=(3, 2),
            padding="SAME",
            dilations=(1, 1),
        )

        return conv2d

    infra.compare_tvm_with_tflite(tf_graph, [ifm_shape], accel_type, enable_cascader=True)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_ethosu_cascade_pooling_depthwise2d(accel_type):
    np.random.seed(0)
    ifm_shape = (1, 11, 11, 3)

    @tf.function
    def tf_graph(x):
        # Use tf.nn API to create the model with two convolutions
        max_pool = tf.nn.max_pool(x, (3, 3), (1, 2), "SAME")
        depthwise2d = tf.nn.depthwise_conv2d(
            max_pool,
            tf.constant(np.random.uniform(size=(3, 2, ifm_shape[3], 1)), dtype=tf.float32),  # HWC1
            strides=(1, 2, 2, 1),
            padding="VALID",
            dilations=(1, 1),
        )
        relu = tf.nn.relu(depthwise2d)
        avg_pool = tf.nn.avg_pool(relu, (1, 1), (2, 2), "VALID")

        return avg_pool

    infra.compare_tvm_with_tflite(tf_graph, [ifm_shape], accel_type, enable_cascader=True)


@pytest.mark.parametrize("accel_type", ACCEL_TYPES)
def test_ethosu_cascade_pooling_depthwise2d_conv2d(accel_type):
    np.random.seed(0)
    ifm_shape = (1, 33, 42, 3)

    @tf.function
    def tf_graph(x):
        # Use tf.nn API to create the model with two convolutions
        depthwise2d = tf.nn.depthwise_conv2d(
            x,
            tf.constant(np.random.uniform(size=(4, 4, ifm_shape[3], 1)), dtype=tf.float32),  # HWC1
            strides=(1, 2, 1, 1),
            padding="VALID",
            dilations=(1, 1),
        )
        ofm_channels = 5
        conv2d = tf.nn.conv2d(
            depthwise2d,
            filters=tf.constant(
                np.random.uniform(size=[3, 2, ifm_shape[3], ofm_channels]),  # HWIO
                dtype=tf.float32,
            ),
            strides=(2, 2),
            padding="SAME",
            dilations=(2, 1),
        )
        relu = tf.nn.relu(conv2d)
        max_pool = tf.nn.max_pool(relu, (3, 3), (1, 2), "SAME")
        depthwise2d = tf.nn.depthwise_conv2d(
            max_pool,
            tf.constant(np.random.uniform(size=(4, 4, ofm_channels, 1)), dtype=tf.float32),  # HWC1
            strides=(1, 1, 1, 1),
            padding="SAME",
            dilations=(3, 2),
        )
        avg_pool = tf.nn.avg_pool(depthwise2d, (2, 3), (1, 1), "VALID")

        return avg_pool

    infra.compare_tvm_with_tflite(tf_graph, [ifm_shape], accel_type, enable_cascader=True)


# TODO(ekalda): Currently cascader fails whenever there is an identity op in the graph, so add
# more tests with other operators once that is fixed.
