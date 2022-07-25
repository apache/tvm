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

"""Arm(R) Ethos(TM)-N integration end-to-end network tests"""

import pytest

pytest.importorskip("tflite")
pytest.importorskip("tensorflow")

from tvm import relay
from tvm.testing import requires_ethosn
from tvm.contrib import download
from tvm.testing import requires_ethosn

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


@requires_ethosn
def test_mobilenet_v1():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"393a19dfb980345cdd3bbeddbc36424d"}
    if tei.get_ethosn_api_version() == 2111:
        _compile_hash = {"5d1c6a6bd4df8963866cc90405bf92dd"}
    if tei.get_ethosn_api_version() == 2102:
        _compile_hash = {"46ccafc840633633aca441645e41b444"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"e4ed29dceb1187505948ab17fc3cc6d6"}
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


@requires_ethosn
def test_resnet_50_int8():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    if tei.get_ethosn_api_version() > 2011:
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"c0a01c547ed1b2e3308094508fa1bfea", "434f0c65c41e24d5482142c88b3438fe"}
            _test_image_network(
                model_url="https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/"
                "models/Quantized/resnet_50_quantized.tflite",
                model_sub_path="resnet_50_quantized.tflite",
                input_dict={"input": (1, 224, 224, 3)},
                compile_hash=_compile_hash,
                output_count=1,
                host_ops=11,
                npu_partitions=2,
            )


@requires_ethosn
def test_inception_v3():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"2c7ff5487e1a21e62b3b42eec624fed4"}
    if tei.get_ethosn_api_version() == 2111:
        _compile_hash = {"e6abe33a7bc4a4170da53eefa6577bba"}
    if tei.get_ethosn_api_version() == 2102:
        _compile_hash = {"43dc2097127eb224c0191b1a15f8acca"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"7db23387bdc5af6eaa1ae3f7d456caf0"}
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


@requires_ethosn
def test_inception_v4():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"4245dbd02e1432dc261a67fc8e632a00"}
    if tei.get_ethosn_api_version() == 2111:
        _compile_hash = {"42e43c323ed8202f7b720ba9029bbcb7"}
    if tei.get_ethosn_api_version() == 2102:
        _compile_hash = {"fab6c2297502f95d33079c6ce1a737f9"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"8da68849b75613ac3dffd3fff2dd87da"}
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


@requires_ethosn
def test_ssd_mobilenet_v1():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"5ee8ed6af9a7f31fc14957b51a8e7423", "e6a91ccc47ba4c6b4614fcd676bd726f"}
    if tei.get_ethosn_api_version() == 2111:
        # TODO(Leo-arm): review split operator
        _compile_hash = {"a37f900601b9493bd142e8aed16205e5", "afb68ca8f452d1f4a674b457b5e30f59"}
    if tei.get_ethosn_api_version() == 2102:
        _compile_hash = {"7795b6c67178da9d1f9b98063bad75b1", "10826406ae724e52f360a06c35ced09d"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"928dc6ae5ce49a4ad63ca87f7575970f", "b092f9820f7e9341fc53daa781b98772"}
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
