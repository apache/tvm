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
import pytest
import numpy as np

from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCResult
from tvm.driver.tvmc.result_utils import get_top_results
from tvm.runtime.module import BenchmarkResult


def test_generate_tensor_data_zeros():
    expected_shape = (2, 3)
    expected_dtype = "uint8"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "zeros")

    assert sut.shape == (2, 3)


def test_generate_tensor_data_ones():
    expected_shape = (224, 224)
    expected_dtype = "uint8"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "ones")

    assert sut.shape == (224, 224)


def test_generate_tensor_data_random():
    expected_shape = (2, 3)
    expected_dtype = "uint8"
    sut = tvmc.runner.generate_tensor_data(expected_shape, expected_dtype, "random")

    assert sut.shape == (2, 3)


def test_generate_tensor_data__type_unknown():
    with pytest.raises(tvmc.TVMCException) as e:
        tvmc.runner.generate_tensor_data((2, 3), "float32", "whatever")


def test_format_times__contains_header():
    fake_result = TVMCResult(outputs=None, times=BenchmarkResult([0.6, 1.2, 0.12, 0.42]))
    sut = fake_result.format_times()
    assert "std (ms)" in sut


def test_get_top_results_keep_results():
    fake_outputs = {"output_0": np.array([[1, 2, 3, 4], [5, 6, 7, 8]])}
    fake_result = TVMCResult(outputs=fake_outputs, times=None)
    number_of_results_wanted = 3
    sut = get_top_results(fake_result, number_of_results_wanted)

    expected_number_of_lines = 2
    assert len(sut) == expected_number_of_lines

    expected_number_of_results_per_line = 3
    assert len(sut[0]) == expected_number_of_results_per_line
    assert len(sut[1]) == expected_number_of_results_per_line


@tvm.testing.slow
def test_run_tflite_module__with_profile__valid_input(
    tflite_mobilenet_v1_1_quant, tflite_compile_model, imagenet_cat
):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    inputs = np.load(imagenet_cat)

    tflite_compiled_model = tflite_compile_model(tflite_mobilenet_v1_1_quant)
    result = tvmc.run(
        tflite_compiled_model,
        inputs=inputs,
        hostname=None,
        device="cpu",
        profile=True,
    )

    # collect the top 5 results
    top_5_results = get_top_results(result, 5)
    top_5_ids = top_5_results[0]

    # IDs were collected from this reference:
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/
    # java/demo/app/src/main/assets/labels_mobilenet_quant_v1_224.txt
    tiger_cat_mobilenet_id = 283

    assert (
        tiger_cat_mobilenet_id in top_5_ids
    ), "tiger cat is expected in the top-5 for mobilenet v1"
    assert type(result.outputs) is dict
    assert type(result.times) is BenchmarkResult
    assert "output_0" in result.outputs.keys()
