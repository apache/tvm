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
from tests.python.relay.aot.aot_test_utils import (
    generate_ref_data,
    get_relay_module_and_inputs_from_tflite_file,
)
import numpy as np

import tvm
import tvm.micro as micro
from tvm import relay
from tvm.relay.backend.contrib.ethosu import util
from tvm.relay.op.contrib.ethosu import partition_for_ethosu

import tvm.relay.testing.tf as tf_testing

from . import infra

MOBILENET_V1_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
    "mobilenet_v1_1.0_224_quant.tflite",
)
MOBILENET_V2_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz",
    "mobilenet_v2_1.0_224_quant.tflite",
)


@pytest.mark.parametrize(
    "accel_type, enable_usmp, model_url",
    [
        ("ethos-u55-256", True, MOBILENET_V1_URL),
        ("ethos-u55-256", False, MOBILENET_V1_URL),
        ("ethos-u65-256", True, MOBILENET_V1_URL),
        ("ethos-u65-256", False, MOBILENET_V1_URL),
        ("ethos-u55-128", True, MOBILENET_V1_URL),
        ("ethos-u55-64", True, MOBILENET_V1_URL),
        ("ethos-u55-32", True, MOBILENET_V1_URL),
        ("ethos-u55-256", False, MOBILENET_V2_URL),
    ],
)
def test_forward_mobilenet(accel_type, enable_usmp, model_url):
    """Test the Mobilenet V1 TF Lite model."""
    np.random.seed(23)
    tflite_model_file = tf_testing.get_workload_official(*model_url)
    relay_mod, input_data, params = get_relay_module_and_inputs_from_tflite_file(tflite_model_file)

    output_data = generate_ref_data(relay_mod, input_data)

    mod = partition_for_ethosu(relay_mod, params)
    compiled_models = infra.build_source(
        mod, input_data, output_data, accel_type, output_tolerance=10, enable_usmp=enable_usmp
    )
    infra.verify_source(compiled_models, accel_type)
