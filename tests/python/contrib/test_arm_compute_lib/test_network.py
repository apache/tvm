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

import numpy as np

from tvm import relay

from .infrastructure import skip_runtime_test, build_and_run, verify
from .infrastructure import Device


def _build_and_run_keras_network(mod, params, inputs, device, tvm_ops, acl_partitions):
    """Helper function to build and run a network from the Keras frontend."""
    data = {}
    np.random.seed(0)
    for name, shape in inputs.items():
        data[name] = np.random.uniform(-128, 127, shape).astype("float32")

    outputs = []
    for acl in [False, True]:
        outputs.append(build_and_run(mod, data, 1, params,
                                     device, enable_acl=acl,
                                     tvm_ops=tvm_ops,
                                     acl_partitions=acl_partitions)[0])
    verify(outputs, atol=0.002, rtol=0.01)


def test_vgg16():
    if skip_runtime_test():
        return

    device = Device()

    def get_model():
        from keras.applications import VGG16
        vgg16 = VGG16(include_top=True, weights='imagenet',
                      input_shape=(224, 224, 3), classes=1000)
        inputs = {vgg16.input_names[0]: (1, 224, 224, 3)}
        mod, params = relay.frontend.from_keras(vgg16, inputs, layout="NHWC")
        return mod, params, inputs

    _build_and_run_keras_network(*get_model(), device=device,
                                 tvm_ops=10, acl_partitions=18)


def test_mobilenet():
    if skip_runtime_test():
        return

    device = Device()

    def get_model():
        from keras.applications import MobileNet
        mobilenet = MobileNet(include_top=True, weights='imagenet',
                              input_shape=(224, 224, 3), classes=1000)
        inputs = {mobilenet.input_names[0]: (1, 224, 224, 3)}
        mod, params = relay.frontend.from_keras(mobilenet, inputs, layout="NHWC")
        return mod, params, inputs

    _build_and_run_keras_network(*get_model(), device=device,
                                 tvm_ops=74, acl_partitions=17)


if __name__ == "__main__":
    test_vgg16()
    test_mobilenet()
