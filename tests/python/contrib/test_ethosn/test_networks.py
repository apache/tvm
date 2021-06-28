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
    _compile_hash = {"5d3cee6ecc488c40ecf533c5cbacc534"}
    if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
        _compile_hash = {"896c28b4f06341ea638ead3a593e1aed"}
    if tei.get_ethosn_api_version() == 2011:
        _compile_hash = {"9298b6c51e2a82f70e91dd11dd6af412"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"407eb47346c8afea2d15e8f0d1c079f2"}
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
    _compile_hash = {"1bc66e83c3de5a9773a719b179c65b1a"}
    if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
        _compile_hash = {"551cde850c6ef960d19be4f317fb8e68"}
    if tei.get_ethosn_api_version() == 2011:
        _compile_hash = {"d44eece5027ff56e5e7fcf014367378d"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"1ba555b4bc60c428018a0f2de9d90532"}
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
    _compile_hash = {"578b8ee279911b49912a77a64f5ff620"}
    if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
        _compile_hash = {"30f078bd42757e8686eafa1f28d0d352"}
    if tei.get_ethosn_api_version() == 2011:
        _compile_hash = {"53f126cf654d4cf61ebb23c767f6740b"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"851665c060cf4719248919d17325ae02"}
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
    _compile_hash = {"cd335229a2052f30273f127a233bd319", "95dedc29d911cdc6b28207ca08e42470"}
    if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
        _compile_hash = {"deee52e136327436411fc725624ae2ea", "6526509d3cbee014e38c79e22bb29d7f"}
    if tei.get_ethosn_api_version() == 2011:
        _compile_hash = {"6e8c4586bdd26527c642a4f016f52284", "057c5efb094c79fbe4483b561147f1d2"}
        if tei.get_ethosn_variant() == "Ethos-N78_1TOPS_2PLE_RATIO":
            _compile_hash = {"dc687e60a4b6750fe740853f22aeb2dc", "1949d86100004eca41099c8e6fa919ab"}
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
