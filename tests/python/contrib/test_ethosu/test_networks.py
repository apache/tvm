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
    convert_to_relay,
    generate_ref_data,
)
import numpy as np

import tvm
import tvm.micro as micro
from tvm import relay
from tvm.relay.backend.contrib import ethosu
from tvm.relay.backend.contrib.ethosu import util
import tvm.relay.testing.tf as tf_testing

from . import infra

ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32"]


def test_forward_mobilenet_v1(accel_type="ethos-u55-256"):
    """Test the Mobilenet V1 TF Lite model."""
    np.random.seed(23)
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/"
        "models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        "mobilenet_v1_1.0_224_quant.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    input_tensor = "input"
    input_dtype = "uint8"
    input_shape = (1, 224, 224, 3)
    in_min, in_max = util.get_range_for_dtype_str(input_dtype)
    input_data = np.random.randint(in_min, high=in_max, size=input_shape, dtype=input_dtype)

    relay_mod, params = convert_to_relay(tflite_model_buf, input_data, "input")
    input_data = {input_tensor: input_data}
    output_data = generate_ref_data(relay_mod, input_data)

    mod = ethosu.partition_for_ethosu(relay_mod, params)
    compiled_models = infra.build_source(mod, input_data, output_data, accel_type)
    infra.verify_source(compiled_models, accel_type)


if __name__ == "__main__":
    test_forward_mobilenet_v1()
