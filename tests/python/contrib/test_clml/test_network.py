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

import numpy as np
import pytest
from tvm import testing
from tvm import relay

import tvm
from test_clml.infrastructure import skip_runtime_test, build_and_run
from test_clml.infrastructure import Device


def _build_and_run_network(mod, params, inputs, data, device, atol, rtol):
    """Helper function to build and run a network."""

    outputs = []
    for clml in [True, False]:
        outputs.append(
            build_and_run(
                mod,
                data,
                1,
                params,
                device,
                enable_clml=clml,
            )[0]
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
        bottom_output = bottom_input
        for layer in model.layers:
            bottom_output = layer(bottom_output)
            if layer.name == layer_name:
                break
        bottom_model = Model(bottom_input, bottom_output)
        return bottom_model

    keras_model = get_bottom_top_model(keras_model, "predictions")
    ref_output = keras_model.predict(data["input_1"].transpose(0, 2, 3, 1))

    mod, params = relay.frontend.from_keras(keras_model, inputs, layout="NCHW")
    return mod, params, ref_output


def test_mobilenet():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()
    dtype = "float16"

    def get_model():
        from tensorflow.keras.applications import MobileNet

        mobilenet = MobileNet(
            include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
        )
        mobilenet.load_weights("mobilenet_1_0_224_tf.h5")
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
    print("OpenCL:", outputs[0][0].asnumpy().shape)
    print("CLML:", outputs[1][0].asnumpy().shape)

    opencl_sort = np.argsort(outputs[1][0].asnumpy()).flatten()
    clml_sort = np.argsort(outputs[0][0].asnumpy()).flatten()

    tvm.testing.assert_allclose(opencl_sort[:10], clml_sort[:10], rtol=1e-5, atol=1e-5)


"""
    tvm.testing.assert_allclose(
         ref_outputs, outputs[1][0].asnumpy(), rtol=1e-5, atol=1e-5)
    print("OpenCL to Keras looks good")
    tvm.testing.assert_allclose(
         outputs[0][0].asnumpy(), outputs[1][0].asnumpy(), rtol=1e-5, atol=1e-5)
    print("OpenCL to CLML looks good")
    exit(0)

    tvm.testing.assert_allclose(
         ref_outputs.transpose(0, 3, 1, 2), outputs[1][0].asnumpy(), rtol=1e-5, atol=1e-5)
    print("OpenCL to Keras looks good")
    tvm.testing.assert_allclose(
         outputs[0][0].asnumpy(), outputs[1][0].asnumpy(), rtol=1e-5, atol=1e-5)
    print("OpenCL to CLML looks good")
"""


if __name__ == "__main__":
    test_mobilenet()
