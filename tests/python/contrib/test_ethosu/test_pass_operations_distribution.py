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
import pytest

pytest.importorskip("ethosu.vela")

import numpy as np

import tvm
from tvm import relay
from tests.python.contrib.test_ethosu.infra import get_tflite_graph
from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.relay.analysis.operations_distribution import analyze_operations_distribution
from tvm.relay.transform.suffixes import tag_suffixes


def test_operations_distribution_ethos():

    tflite = pytest.importorskip("tflite")
    tensorflow = pytest.importorskip("tensorflow")

    import tensorflow as tf

    inp = (224, 224, 9)
    input_shape = (1, *inp)
    kernel_shape = (3, 3)
    padding = (1, 1, 1, 1)
    padding_out = (1, 33, 33, 1)

    @tf.function
    def simple_net(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], input_shape[3], 3]
        weights = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weight_shape[2] = 3
        weights1 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weights2 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        op = tf.nn.conv2d(
            x,
            filters=weights,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op1 = tf.nn.conv2d(
            op,
            filters=weights1,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op2 = tf.nn.conv2d(
            op,
            filters=weights2,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op = tf.concat([op1, op2], 1)
        op = tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[1]], [padding_out[2], padding[3]], [0, 0]],
            "CONSTANT",
        )
        return op

    _, tflite_graph = get_tflite_graph(simple_net, [input_shape])

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_graph, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(tflite_model)

    mod = tag_suffixes(mod)
    mod = partition_for_ethosu(mod, params)
    operations_distribution = analyze_operations_distribution(mod)

    expected = {
        "Identity": ["generic", "generic"],
        "concat": ["ethos-u", "ethos-u.concat"],
        "Conv2D_11": ["ethos-u", "ethos-u.qnn_conv2d"],
        "Conv2D1": ["ethos-u", "ethos-u.qnn_conv2d"],
        "Conv2D_221": ["ethos-u", "ethos-u.qnn_conv2d"],
    }

    assert operations_distribution == expected


def test_operations_distribution_generic():

    tflite = pytest.importorskip("tflite")
    tensorflow = pytest.importorskip("tensorflow")

    import tensorflow as tf

    inp = (224, 224, 9)
    input_shape = (1, *inp)
    kernel_shape = (3, 3)
    padding = (1, 1, 1, 1)
    padding_out = (1, 33, 33, 1)
    dilations_out = 32

    @tf.function
    def simple_net(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], input_shape[3], 3]
        weights = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        op = tf.nn.conv2d(
            x,
            filters=weights,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=dilations_out,
        )
        op = tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[2]], [padding[1], padding[3]], [0, 0]],
            "CONSTANT",
        )
        op = tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[2]], [padding[1], padding[3]], [0, 0]],
            "CONSTANT",
        )
        return tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[2]], [padding[1], padding[3]], [0, 0]],
            "CONSTANT",
        )

    _, tflite_graph = get_tflite_graph(simple_net, [input_shape])

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_graph, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)

    mod, params = relay.frontend.from_tflite(tflite_model)

    mod = tag_suffixes(mod)
    mod = partition_for_ethosu(mod, params)
    operations_distribution = analyze_operations_distribution(mod)

    expected = {
        "Identity": ["generic", "generic"],
        "Pad_1": ["generic", "generic"],
        "Pad": ["generic", "generic"],
        "Conv2D2": ["generic", "generic"],
    }

    assert operations_distribution == expected


if __name__ == "__main__":
    tvm.testing.main()
