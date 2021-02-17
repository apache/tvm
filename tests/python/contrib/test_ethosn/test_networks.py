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

"""Ethos-N integration end-to-end network tests"""

import pytest

pytest.importorskip("tflite")
pytest.importorskip("tensorflow")

from tvm import relay
from tvm.relay.op.contrib.ethosn import ethosn_available
from tvm.contrib import download
import tvm.relay.testing.tf as tf_testing
import tflite.Model
from . import infrastructure as tei


def _get_tflite_model(tflite_model_path, inputs_dict, dtype):
    with open(tflite_model_path, "rb") as f:
        tflite_model_buffer = f.read()

    try:
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buffer, 0)
    except AttributeError:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buffer, 0)
    shape_dict = {}
    dtype_dict = {}
    for input in inputs_dict:
        input_shape = inputs_dict[input]
        shape_dict[input] = input_shape
        dtype_dict[input] = dtype

    return relay.frontend.from_tflite(
        tflite_model,
        shape_dict=shape_dict,
        dtype_dict=dtype_dict,
    )


def _test_image_network(
    model_url,
    model_sub_path,
    input_dict,
    compile_hash,
    output_count,
    host_ops=0,
    npu_partitions=1,
    run=False,
):
    """Test an image network.

    Parameters
    ----------
    model_url : str
        The URL to the model.
    model_sub_path : str
        The name of the model file.
    input_dict : dict
        The input dict.
    compile_hash : str, set
        The compile hash(es) to check the compilation output against.
    output_count : int
        The expected number of outputs.
    host_ops : int
        The expected number of host operators.
    npu_partitions : int
        The expected number of Ethos-N partitions.
    run : bool
        Whether or not to try running the network. If hardware isn't
        available, the run will still take place but with a mocked
        inference function, so the results will be incorrect. This is
        therefore just to test the runtime flow is working rather than
        to check the correctness/accuracy.

    """
    if not ethosn_available():
        return

    def get_model():
        if model_url[-3:] in ("tgz", "zip"):
            model_path = tf_testing.get_workload_official(
                model_url,
                model_sub_path,
            )
        else:
            model_path = download.download_testdata(
                model_url,
                model_sub_path,
            )
        return _get_tflite_model(model_path, input_dict, "uint8")

    inputs = {}
    for input_name in input_dict:
        input_shape = input_dict[input_name]
        inputs[input_name] = tei.get_real_image(input_shape[1], input_shape[2])

    mod, params = get_model()
    m = tei.build(mod, params, npu=True, expected_host_ops=host_ops, npu_partitions=npu_partitions)
    tei.assert_lib_hash(m.get_lib(), compile_hash)
    if run:
        tei.run(m, inputs, output_count, npu=True)


def test_mobilenet_v1():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"81637c89339201a07dc96e3b5dbf836a"}
    if tei.get_ethosn_api_version() == 2008:
        _compile_hash = {"47e216d8ab2bf491708ccf5620bc0d02"}
        if tei.get_ethosn_variant() == 3:
            _compile_hash = {"2436f523e263f66a063cef902f2f43d7"}
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        model_sub_path="mobilenet_v1_1.0_224_quant.tflite",
        input_dict={"input": (1, 224, 224, 3)},
        compile_hash=_compile_hash,
        output_count=1,
        host_ops=3,
        npu_partitions=1,
        run=True,
    )


def test_inception_v3():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"de0e175af610ebd45ccb03d170dc9664"}
    if tei.get_ethosn_api_version() == 2008:
        _compile_hash = {"8c9d75659cd7bc9ff6dd6d490d28f9b2"}
        if tei.get_ethosn_variant() == 3:
            _compile_hash = {"cdd4d7f6453d722ea73224ff9d6a115a"}
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/tflite_11_05_08/inception_v3_quant.tgz",
        model_sub_path="inception_v3_quant.tflite",
        input_dict={"input": (1, 299, 299, 3)},
        compile_hash=_compile_hash,
        output_count=1,
        host_ops=0,
        npu_partitions=1,
    )


def test_inception_v4():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    if not tei.get_ethosn_variant() == 0:
        pytest.skip("Ethos-N78 20.08 does not support inception_v4 in the default configuration.")
    _compile_hash = {"06bf6cb56344f3904bcb108e54edfe87"}
    if tei.get_ethosn_api_version() == 2008:
        _compile_hash = {"798292bfa596ca7c32086396b494b46c"}
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/inception_v4_299_quant_20181026.tgz",
        model_sub_path="inception_v4_299_quant.tflite",
        input_dict={"input": (1, 299, 299, 3)},
        compile_hash=_compile_hash,
        output_count=1,
        host_ops=3,
        npu_partitions=1,
    )


def test_ssd_mobilenet_v1():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"29aec6b184b09454b4323271aadf89b1", "6211d96103880b016baa85e638abddef"}
    if tei.get_ethosn_api_version() == 2008:
        _compile_hash = {"5999f26e140dee0d7866491997ef78c5", "24e3a690a7e95780052792d5626c85be"}
        if tei.get_ethosn_variant() == 3:
            _compile_hash = {"da871b3f03a93df69d704ed44584d6cd", "9f52411d301f3cba3f6e4c0f1c558e87"}
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        model_sub_path="detect.tflite",
        input_dict={"normalized_input_image_tensor": (1, 300, 300, 3)},
        compile_hash=_compile_hash,
        output_count=4,
        host_ops=28,
        npu_partitions=2,
    )
