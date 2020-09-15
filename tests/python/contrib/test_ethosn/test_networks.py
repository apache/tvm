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
from tvm.relay.op.contrib.ethosn import ethosn_available, Available
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
    run=True,
    host_ops=0,
    npu_partitions=1,
):
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

    outputs = []
    inputs = {}
    for input_name in input_dict:
        input_shape = input_dict[input_name]
        inputs[input_name] = tei.get_real_image(input_shape[1], input_shape[2])

    for npu in [False, True]:
        mod, params = get_model()
        graph, lib, params = tei.build(
            mod, params, npu=npu, expected_host_ops=host_ops, npu_partitions=npu_partitions
        )
        if npu:
            tei.assert_lib_hash(lib, compile_hash)
        if run:
            outputs.append(tei.run(graph, lib, params, inputs, output_count, npu=npu))

    if run:
        tei.verify(outputs, 1, verify_saturation=False)


def test_mobilenet_v1():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    hw = ethosn_available()
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        model_sub_path="mobilenet_v1_1.0_224_quant.tflite",
        input_dict={"input": (1, 224, 224, 3)},
        compile_hash="81637c89339201a07dc96e3b5dbf836a",
        output_count=1,
        run=(hw == Available.SW_AND_HW),
        host_ops=3,
        npu_partitions=1,
    )


def test_inception_v3():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/tflite_11_05_08/inception_v3_quant.tgz",
        model_sub_path="inception_v3_quant.tflite",
        input_dict={"input": (1, 299, 299, 3)},
        compile_hash="de0e175af610ebd45ccb03d170dc9664",
        output_count=1,
        run=False,
        host_ops=0,
        npu_partitions=1,
    )


def test_inception_v4():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/inception_v4_299_quant_20181026.tgz",
        model_sub_path="inception_v4_299_quant.tflite",
        input_dict={"input": (1, 299, 299, 3)},
        compile_hash="06bf6cb56344f3904bcb108e54edfe87",
        output_count=1,
        run=False,
        host_ops=3,
        npu_partitions=1,
    )


def test_ssd_mobilenet_v1():
    # If this test is failing due to a hash mismatch, please notify @mbaret and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        model_sub_path="detect.tflite",
        input_dict={"normalized_input_image_tensor": (1, 300, 300, 3)},
        compile_hash="6211d96103880b016baa85e638abddef",
        output_count=4,
        run=False,
        host_ops=28,
        npu_partitions=2,
    )
