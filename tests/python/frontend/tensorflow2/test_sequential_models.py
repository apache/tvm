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
# pylint: disable=import-self, invalid-name, unused-argument, too-many-lines, len-as-condition, broad-except
# pylint: disable=import-outside-toplevel, redefined-builtin
"""TF2 to relay converter test: testing models built with tf.keras.Sequential()"""

import tempfile
import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from common import compare_tf_tvm
from common import run_tf_code


def run_sequential_model(model_fn, input_shape):
    def get_input(shape):
        _input = np.random.uniform(0, 1, shape).astype(dtype="float32")
        return _input

    def save_and_reload(_model):
        with tempfile.TemporaryDirectory() as model_path:
            tf.saved_model.save(_model, model_path)
            loaded = tf.saved_model.load(model_path)
            func = loaded.signatures["serving_default"]
            frozen_func = convert_variables_to_constants_v2(func)
        return frozen_func

    def model_graph(model, input_shape):
        _input = get_input(input_shape)
        f = save_and_reload(model(input_shape))
        _output = run_tf_code(f, _input)
        gdef = f.graph.as_graph_def(add_shapes=True)
        return gdef, _input, _output

    compare_tf_tvm(*model_graph(model_fn, input_shape), runtime="vm")


def test_dense_model():
    def dense_model(input_shape, num_units=128):
        return tf.keras.Sequential(
            [tf.keras.layers.Flatten(input_shape=input_shape[1:]), tf.keras.layers.Dense(num_units)]
        )

    run_sequential_model(dense_model, input_shape=(1, 28, 28))


def test_mnist_model():
    def mnist_model(input_shape):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=input_shape[1:]),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

    run_sequential_model(mnist_model, input_shape=(1, 28, 28))


def test_conv2d_model():
    def conv2d_model(input_shape, kernel=(3, 3), filters=16):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape[1:], batch_size=1),
                tf.keras.layers.Conv2D(filters, kernel),
            ]
        )
        return model

    run_sequential_model(conv2d_model, input_shape=(1, 32, 32, 3))


def test_maxpool_model():
    def maxpool_model(input_shape, pool_size=(2, 2)):
        model = tf.keras.Sequential(
            [tf.keras.layers.MaxPool2D(pool_size=pool_size, input_shape=input_shape[1:])]
        )
        return model

    run_sequential_model(maxpool_model, input_shape=(1, 32, 32, 3))


def test_maxpool_batchnorm_model():
    def maxpool_batchnorm_model(input_shape, pool_size=(2, 2)):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.MaxPool2D(pool_size=pool_size, input_shape=input_shape[1:]),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        return model

    run_sequential_model(maxpool_batchnorm_model, input_shape=(1, 32, 32, 3))


def test_tensorlist_stack_model():
    def tensorlist_stack_model(input_shape):
        class TensorArrayStackLayer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()

            def call(self, inputs):
                inputs = tf.squeeze(inputs)
                outputs = tf.TensorArray(
                    tf.float32,
                    size=inputs.shape[0],
                    infer_shape=False,
                    element_shape=inputs.shape[1:],
                )
                outputs = outputs.unstack(inputs)

                return outputs.stack()

        input_shape = (3, 32)
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=input_shape, batch_size=1), TensorArrayStackLayer()]
        )
        return model

    run_sequential_model(tensorlist_stack_model, input_shape=(3, 32))


def test_tensorlist_read_model():
    def tensorlist_read_model(input_shape):
        class TensorArrayReadLayer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()

            def call(self, inputs):
                inputs = tf.squeeze(inputs)
                outputs = tf.TensorArray(
                    tf.float32,
                    size=inputs.shape[0],
                    infer_shape=False,
                    element_shape=inputs.shape[1:],
                )
                for i in range(inputs.shape[0]):
                    outputs = outputs.write(i, inputs[i, :])

                return outputs.read(0)

        input_shape = (3, 32)
        model = tf.keras.Sequential(
            [tf.keras.layers.Input(shape=input_shape, batch_size=1), TensorArrayReadLayer()]
        )
        return model

    run_sequential_model(tensorlist_read_model, input_shape=(3, 32))


if __name__ == "__main__":
    tvm.testing.main()
