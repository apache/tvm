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

import sys

import pytest
import numpy as np

import tvm.testing
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib import cmsisnn

from utils import skip_if_no_reference_system, get_range_for_dtype_str
from tests.python.relay.aot.aot_test_utils import (
    AOTTestModel,
    AOT_CORSTONE300_RUNNER,
    generate_ref_data,
    compile_and_run,
)


def convert_to_relay(
    tflite_model_buf,
    input_data,
    input_node,
):
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
    for i, e in enumerate(input_node):
        shape_dict[e] = input_data[i].shape
        dtype_dict[e] = input_data[i].dtype.name

    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict
    )

    return mod, params


@skip_if_no_reference_system
@tvm.testing.requires_package("tflite")
@tvm.testing.requires_cmsisnn
def test_cnn_small():
    # download the model
    base_url = "https://github.com/ARM-software/ML-zoo/raw/master/models/keyword_spotting/cnn_small/tflite_int8"
    file_to_download = "cnn_s_quantized.tflite"
    model_file = download_testdata("{}/{}".format(base_url, file_to_download), file_to_download)

    with open(model_file, "rb") as f:
        tflite_model_buf = f.read()

    input_shape = (1, 490)
    in_min, in_max = get_range_for_dtype_str("int8")
    input_data = np.random.randint(in_min, high=in_max, size=input_shape).astype(np.float32)

    orig_mod, params = convert_to_relay(tflite_model_buf, input_data, "input")
    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(orig_mod, params)

    # validate CMSIS-NN output against CPU output
    interface_api = "c"
    use_unpacked_api = True
    test_runner = AOT_CORSTONE300_RUNNER
    inputs = {"input": input_data}
    params = {}
    output_list = generate_ref_data(orig_mod["main"], inputs, params)
    compile_and_run(
        AOTTestModel(module=cmsisnn_mod, inputs=inputs, outputs=output_list, params=params),
        test_runner,
        interface_api,
        use_unpacked_api,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
