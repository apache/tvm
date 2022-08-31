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
    # If this test is failing due to a hash mismatch, please notify @lhutton1 and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"50186822915909303e813205db80e032"}
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
    # If this test is failing due to a hash mismatch, please notify @lhutton1 and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"9245965b2c01e7f3d9b478e38a186eb4", "4225fa951c145bb1e48e28cad6a3bdd4"}
    _test_image_network(
        model_url="https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/"
        "models/Quantized/resnet_50_quantized.tflite",
        model_sub_path="resnet_50_quantized.tflite",
        input_dict={"input": (1, 224, 224, 3)},
        compile_hash=_compile_hash,
        output_count=1,
        host_ops=10,
        npu_partitions=2,
    )


@requires_ethosn
def test_inception_v3():
    # If this test is failing due to a hash mismatch, please notify @lhutton1 and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"a5a2b5d2b618de754bf9a01033a020c0"}
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
    # If this test is failing due to a hash mismatch, please notify @lhutton1 and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"61b4ade41898d7cb2451dbdc3340aced"}
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
    # If this test is failing due to a hash mismatch, please notify @lhutton1 and
    # @Leo-arm. The hash is there to catch any changes in the behaviour of the
    # codegen, which could come about from either a change in Support Library
    # version or a change in the Ethos-N codegen. To update this requires running
    # on hardware that isn't available in CI.
    _compile_hash = {"789906c7d8ac787809b303d82781fc9d", "6b699f94795785d31b39940a5cf84a81"}
    _test_image_network(
        model_url="https://storage.googleapis.com/download.tensorflow.org/"
        "models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        model_sub_path="detect.tflite",
        input_dict={"normalized_input_image_tensor": (1, 300, 300, 3)},
        compile_hash=_compile_hash,
        output_count=4,
        host_ops=27,
        npu_partitions=2,
    )
