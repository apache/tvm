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
"""OpenCL ML network tests."""

import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from test_clml.infrastructure import build_and_run, Device
import pytest


def _build_and_run_network(mod, params, inputs, data, device, atol, rtol, tvm_log=""):
    """Helper function to build and run a network."""

    outputs = []
    for clml in [True, False]:
        outputs.append(
            build_and_run(mod, data, 1, params, device, enable_clml=clml, tune_log=tvm_log)[0][0]
        )
    return outputs


def _get_keras_model(keras_model, inputs_dict, data):
    """Convert Keras graph to relay."""
    inputs = {}
    for name, (shape, _) in inputs_dict.items():
        inputs[keras_model.input_names[0]] = shape

    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    def get_bottom_top_model(model, layer_name):
        layer = model.get_layer(layer_name)
        bottom_input = model.layers[0].input
        bottom_output = layer.output
        bottom_model = Model(bottom_input, bottom_output)
        return bottom_model

    keras_model = get_bottom_top_model(keras_model, "predictions")
    ref_output = keras_model.predict(data["input_1"].transpose(0, 2, 3, 1))

    mod, params = relay.frontend.from_keras(keras_model, inputs, layout="NCHW")
    return mod, params, ref_output


@pytest.mark.parametrize("dtype", ["float16"])
@tvm.testing.requires_openclml
def test_mobilenet(device, dtype):
    def get_model():
        from tensorflow.keras.applications import MobileNet
        import tensorflow as tf

        tf.keras.backend.clear_session()

        mobilenet = MobileNet(
            include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
        )
        inputs = {mobilenet.input_names[0]: ((1, 3, 224, 224), "float32")}

        data = {}
        np.random.seed(0)

        for name, (shape, dtype) in inputs.items():
            if dtype == "uint8":
                low, high = 0, 1
            else:
                low, high = -1, 1
            data[name] = np.random.uniform(low, high, shape).astype(dtype)

        mod, params, ref_outputs = _get_keras_model(mobilenet, inputs, data)
        return mod, params, inputs, data, ref_outputs

    mod, params, inputs, input_data, ref_outputs = get_model()
    outputs = _build_and_run_network(
        mod, params, inputs, input_data, device=device, atol=1e-5, rtol=1e-5
    )

    # test
    print("OpenCL:", outputs[0].asnumpy().shape)
    print("CLML:", outputs[1].asnumpy().shape)

    opencl_sort = np.argsort(outputs[1].asnumpy()).flatten()
    clml_sort = np.argsort(outputs[0].asnumpy()).flatten()

    tvm.testing.assert_allclose(opencl_sort[:10], clml_sort[:10], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float16"])
@tvm.testing.requires_openclml
def test_inception_v3(device, dtype):
    def get_model():
        from tensorflow.keras.applications import InceptionV3
        import tensorflow as tf

        tf.keras.backend.clear_session()

        inceptionV3 = InceptionV3(
            include_top=True, weights=None, input_shape=(299, 299, 3), classes=1000
        )
        inputs = {inceptionV3.input_names[0]: ((1, 3, 299, 299), "float16")}

        data = {}
        np.random.seed(0)
        for name, (shape, dtype) in inputs.items():
            if dtype == "uint8":
                low, high = 0, 1
            else:
                low, high = -2, 1
            data[name] = np.random.uniform(low, high, shape).astype(dtype)

        mod, params, ref_outputs = _get_keras_model(inceptionV3, inputs, data)
        return mod, params, inputs, data, ref_outputs

    mod, params, inputs, input_data, ref_outputs = get_model()
    outputs = _build_and_run_network(
        mod, params, inputs, input_data, device=device, atol=1e-5, rtol=1e-5
    )

    opencl_sort = np.argsort(outputs[1].asnumpy()).flatten()
    clml_sort = np.argsort(outputs[0].asnumpy()).flatten()

    tvm.testing.assert_allclose(opencl_sort[:5], clml_sort[:5], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", ["float16"])
@tvm.testing.requires_openclml
def test_resnet50v2(device, dtype):
    def get_model():
        from tensorflow.keras.applications import ResNet50V2
        import tensorflow as tf

        tf.keras.backend.clear_session()

        model = ResNet50V2(include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000)
        inputs_dict = {model.input_names[0]: ((1, 3, 224, 224), "float32")}

        data = {}
        np.random.seed(0)

        for name, (shape, dtype) in inputs_dict.items():
            if dtype == "uint8":
                low, high = 0, 1
            else:
                low, high = -1, 1
            data[name] = np.random.uniform(low, high, shape).astype(dtype)

        """Convert Keras graph to relay."""
        inputs = {}
        for name, (shape, _) in inputs_dict.items():
            inputs[model.input_names[0]] = shape

        ref_outputs = model.predict(data["input_1"].transpose(0, 2, 3, 1))

        mod, params = relay.frontend.from_keras(model, inputs, layout="NCHW")

        return mod, params, inputs, data, ref_outputs

    mod, params, inputs, input_data, ref_outputs = get_model()
    outputs = _build_and_run_network(
        mod, params, inputs, input_data, device=device, atol=1e-5, rtol=1e-5
    )

    # test
    print("OpenCL:", outputs[0].asnumpy().shape)
    print("CLML:", outputs[1].asnumpy().shape)

    opencl_sort = np.argsort(outputs[1].asnumpy()).flatten()
    clml_sort = np.argsort(outputs[0].asnumpy()).flatten()

    tvm.testing.assert_allclose(opencl_sort[:10], clml_sort[:10], rtol=1e-5, atol=1e-5)
