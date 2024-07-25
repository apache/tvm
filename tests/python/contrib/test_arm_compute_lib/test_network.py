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
"""Arm Compute Library network tests."""

from packaging.version import parse

import numpy as np
import pytest
from tvm import relay

from test_arm_compute_lib.infrastructure import Device, skip_runtime_test, build_and_run, verify


def _build_and_run_network(mod, params, inputs, device, tvm_ops, acl_partitions, atol, rtol):
    """Helper function to build and run a network."""
    data = {}
    np.random.seed(0)

    for name, (shape, dtype) in inputs.items():
        if dtype == "uint8":
            low, high = 0, 255
        else:
            low, high = -127, 128
        data[name] = np.random.uniform(low, high, shape).astype(dtype)

    outputs = []
    for acl in [False, True]:
        outputs.append(
            build_and_run(
                mod,
                data,
                1,
                params,
                device,
                enable_acl=acl,
                tvm_ops=tvm_ops,
                acl_partitions=acl_partitions,
            )[0]
        )
    verify(outputs, atol=atol, rtol=rtol, verify_saturation=False)


def _get_tflite_model(tflite_model_path, inputs_dict):
    """Convert TFlite graph to relay."""
    try:
        import tflite.Model
    except ImportError:
        pytest.skip("Missing Tflite support")

    with open(tflite_model_path, "rb") as f:
        tflite_model_buffer = f.read()

    try:
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buffer, 0)
    except AttributeError:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buffer, 0)
    shape_dict = {}
    dtype_dict = {}
    for input in inputs_dict:
        input_shape, input_dtype = inputs_dict[input]
        shape_dict[input] = input_shape
        dtype_dict[input] = input_dtype

    return relay.frontend.from_tflite(tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict)


def _get_keras_model(keras_model, inputs_dict):
    """Convert Keras graph to relay."""
    inputs = {}
    for name, (shape, _) in inputs_dict.items():
        inputs[keras_model.input_names[0]] = shape
    return relay.frontend.from_keras(keras_model, inputs, layout="NHWC")


def test_vgg16():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()

    def get_model():
        try:
            from keras.applications import VGG16
        except ImportError:
            pytest.skip("Missing Keras Package")

        vgg16 = VGG16(include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000)
        inputs = {vgg16.input_names[0]: ((1, 224, 224, 3), "float32")}
        mod, params = _get_keras_model(vgg16, inputs)
        return mod, params, inputs

    _build_and_run_network(
        *get_model(),
        device=device,
        tvm_ops=4,
        acl_partitions=21,
        atol=0.002,
        rtol=0.01,
    )


def test_mobilenet():
    keras = pytest.importorskip("keras")
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    device = Device()

    def get_model():
        try:
            from keras.applications import MobileNet
        except ImportError:
            pytest.skip("Missing keras module")

        mobilenet = MobileNet(
            include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000
        )
        inputs = {mobilenet.input_names[0]: ((1, 224, 224, 3), "float32")}
        mod, params = _get_keras_model(mobilenet, inputs)
        return mod, params, inputs

    if parse(keras.__version__) < parse("2.9"):
        # This can be removed after we migrate to TF/Keras >= 2.9
        expected_tvm_ops = 56
        expected_acl_partitions = 31
    else:
        # In Keras >= 2.7, one reshape operator was removed
        # from the MobileNet model, so it impacted this test
        # which now needs to be reduce in by 1
        # The change in Keras is `b6abfaed1326e3c`
        expected_tvm_ops = 55
        expected_acl_partitions = 30

    _build_and_run_network(
        *get_model(),
        device=device,
        tvm_ops=expected_tvm_ops,
        acl_partitions=expected_acl_partitions,
        atol=0.002,
        rtol=0.01,
    )


def test_quantized_mobilenet():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    try:
        import tvm.relay.testing.tf as tf_testing
    except ImportError:
        pytest.skip("Missing Tflite support")

    device = Device()

    def get_model():
        model_path = tf_testing.get_workload_official(
            "https://storage.googleapis.com/download.tensorflow.org/"
            "models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
            "mobilenet_v1_1.0_224_quant.tflite",
        )
        inputs = {"input": ((1, 224, 224, 3), "uint8")}
        mod, params = _get_tflite_model(model_path, inputs_dict=inputs)
        return mod, params, inputs

    _build_and_run_network(
        *get_model(),
        device=device,
        tvm_ops=3,
        acl_partitions=30,
        atol=10,
        rtol=0,
    )


def test_squeezenet():
    Device.load("test_config.json")

    if skip_runtime_test():
        return

    try:
        import tvm.relay.testing.tf as tf_testing
    except ImportError:
        pytest.skip("Missing TF Support")

    device = Device()

    def get_model():
        model_path = tf_testing.get_workload_official(
            "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz",
            "squeezenet.tflite",
        )
        inputs = {"Placeholder": ((1, 224, 224, 3), "float32")}
        mod, params = _get_tflite_model(model_path, inputs_dict=inputs)
        return mod, params, inputs

    _build_and_run_network(
        *get_model(),
        device=device,
        tvm_ops=9,
        acl_partitions=31,
        atol=8,
        rtol=0,
    )


if __name__ == "__main__":
    test_vgg16()
    test_mobilenet()
    test_quantized_mobilenet()
    test_squeezenet()
