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

"""CMSIS-NN: testing with networks"""

import pytest
import numpy as np

import tvm.testing
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib import cmsisnn
from tvm.testing.aot import AOTTestModel, get_dtype_range, compile_and_run, generate_ref_data
from tvm.micro.testing.aot_test_utils import (
    AOT_CORSTONE300_RUNNER,
    AOT_USMP_CORSTONE300_RUNNER,
)
from .utils import skip_if_no_reference_system

# pylint: disable=import-outside-toplevel
def _convert_to_relay(
    tflite_model_buf,
    input_data,
    input_node,
):
    """Converts TFLite model to Relay module and params"""

    def convert_to_list(x):
        if not isinstance(x, list):
            x = [x]
        return x

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError:
        raise ImportError("The tflite package must be installed")

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, name in enumerate(input_node):
        shape_dict[name] = input_data[i].shape
        dtype_dict[name] = input_data[i].dtype.name

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
    )

    return mod, params


@skip_if_no_reference_system
@tvm.testing.requires_package("tflite")
@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("test_runner", [AOT_CORSTONE300_RUNNER, AOT_USMP_CORSTONE300_RUNNER])
def test_cnn_small(test_runner):
    """Download a small network and tests TVM via CMSIS-NN output against TFLite output"""
    # download the model
    base_url = (
        "https://github.com/ARM-software/ML-zoo/raw/"
        "48a22ee22325d15d2371a6df24eb7d67e21dcc97"
        "/models/keyword_spotting/cnn_small/tflite_int8"
    )
    file_to_download = "cnn_s_quantized.tflite"
    file_saved = "cnn_s_quantized_15Dec2021.tflite"
    model_file = download_testdata("{}/{}".format(base_url, file_to_download), file_saved)

    with open(model_file, "rb") as f:
        tflite_model_buf = f.read()

    input_shape = (1, 490)
    dtype = "int8"
    in_min, in_max = get_dtype_range(dtype)
    rng = np.random.default_rng(12345)
    input_data = rng.integers(in_min, high=in_max, size=input_shape, dtype=dtype)

    orig_mod, params = _convert_to_relay(tflite_model_buf, input_data, "input")
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate CMSIS-NN output against CPU output
    interface_api = "c"
    use_unpacked_api = True
    inputs = {"input": input_data}
    params = {}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(
            module=cmsisnn_mod,
            inputs=inputs,
            outputs=output_list,
            params=params,
            output_tolerance=1,
        ),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


@tvm.testing.requires_package("tflite")
def test_keyword_scramble():
    """Download keyword_scrambled and test for Relay conversion.
    In future, this test can be extended for CMSIS-NN"""
    # download the model
    base_url = (
        "https://github.com/tensorflow/tflite-micro/raw/"
        "de8f61a074460e1fa5227d875c95aa303be01240/"
        "tensorflow/lite/micro/models"
    )
    file_to_download = "keyword_scrambled.tflite"
    file_saved = "keyword_scrambled.tflite"
    model_file = download_testdata("{}/{}".format(base_url, file_to_download), file_saved)

    with open(model_file, "rb") as f:
        tflite_model_buf = f.read()

    input_shape = (1, 96)
    dtype = "int8"
    in_min, in_max = get_dtype_range(dtype)
    rng = np.random.default_rng(12345)
    input_data = rng.integers(in_min, high=in_max, size=input_shape, dtype=dtype)

    with pytest.raises(tvm.error.OpNotImplemented):
        _, _ = _convert_to_relay(tflite_model_buf, input_data, "input")


if __name__ == "__main__":
    tvm.testing.main()
