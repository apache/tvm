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
# pylint: disable=invalid-name, unused-argument, wrong-import-position
import pytest

pytest.importorskip("ethosu.vela")

import numpy as np

from tvm.relay.op.contrib.ethosu import partition_for_ethosu
from tvm.micro import model_library_format as mlf

from tests.python.relay.aot.aot_test_utils import convert_to_relay

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
    "accel_type, model_url, workspace_size",
    [
        ("ethos-u65-256", MOBILENET_V1_URL, 1423344),
        ("ethos-u65-256", MOBILENET_V2_URL, 2185584),
        ("ethos-u55-256", MOBILENET_V1_URL, 1423344),
        ("ethos-u55-256", MOBILENET_V2_URL, 2185584),
        ("ethos-u55-128", MOBILENET_V2_URL, 2185584),
        ("ethos-u55-64", MOBILENET_V2_URL, 2185584),
        ("ethos-u55-32", MOBILENET_V2_URL, 2185584),
    ],
)
def test_networks_without_usmp(accel_type, model_url, workspace_size):
    np.random.seed(23)
    tflite_model_buf = infra.get_tflite_model(model_url)
    input_data, output_data = infra.generate_ref_data_tflite(tflite_model_buf)
    mod, params = convert_to_relay(tflite_model_buf)
    mod = partition_for_ethosu(mod, params)
    compiled_models = infra.build_source(
        mod, input_data, output_data, accel_type, enable_usmp=False
    )
    mlf_memory_map = mlf._build_function_memory_map(
        compiled_models[0].executor_factory.function_metadata
    )
    assert mlf_memory_map["main"][0]["workspace_size_bytes"] == workspace_size
    infra.verify_source(compiled_models, accel_type, enable_usmp=False)


@pytest.mark.parametrize(
    "accel_type, model_url, workspace_size",
    [
        ("ethos-u65-256", MOBILENET_V1_URL, 1205872),
        ("ethos-u55-256", MOBILENET_V2_URL, 1507152),
    ],
)
def test_networks_with_usmp(accel_type, model_url, workspace_size):
    np.random.seed(23)
    tflite_model_buf = infra.get_tflite_model(model_url)
    input_data, output_data = infra.generate_ref_data_tflite(tflite_model_buf)
    mod, params = convert_to_relay(tflite_model_buf)
    mod = partition_for_ethosu(mod, params)
    compiled_models = infra.build_source(mod, input_data, output_data, accel_type, enable_usmp=True)
    allocated_pool_info = list(
        dict(compiled_models[0].executor_factory.executor_codegen_metadata.pool_inputs).values()
    )[0]
    assert allocated_pool_info.allocated_size == workspace_size
    infra.verify_source(compiled_models, accel_type, enable_usmp=True)


if __name__ == "__main__":
    pytest.main([__file__])
