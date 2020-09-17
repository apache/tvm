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
import os
import tarfile

import pytest

from tvm.ir.module import IRModule

from tvm.driver import tvmc
from tvm.driver.tvmc.common import TVMCException


def test_get_frontends_contains_only_strings():
    sut = tvmc.frontends.get_frontend_names()
    assert all([type(x) is str for x in sut]) is True


def test_get_frontend_by_name_valid():
    # some CI environments wont offer TensorFlow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    sut = tvmc.frontends.get_frontend_by_name("keras")
    assert type(sut) is tvmc.frontends.KerasFrontend


def test_get_frontend_by_name_invalid():
    with pytest.raises(TVMCException):
        tvmc.frontends.get_frontend_by_name("unsupported_thing")


def test_guess_frontend_tflite():
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    sut = tvmc.frontends.guess_frontend("a_model.tflite")
    assert type(sut) is tvmc.frontends.TFLiteFrontend


def test_guess_frontend_onnx():
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    sut = tvmc.frontends.guess_frontend("a_model.onnx")
    assert type(sut) is tvmc.frontends.OnnxFrontend


def test_guess_frontend_pytorch():
    # some CI environments wont offer pytorch, so skip in case it is not present
    pytest.importorskip("torch")

    sut = tvmc.frontends.guess_frontend("a_model.pth")
    assert type(sut) is tvmc.frontends.PyTorchFrontend


def test_guess_frontend_keras():
    # some CI environments wont offer TensorFlow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    sut = tvmc.frontends.guess_frontend("a_model.h5")
    assert type(sut) is tvmc.frontends.KerasFrontend


def test_guess_frontend_tensorflow():
    # some CI environments wont offer TensorFlow, so skip in case it is not present
    pytest.importorskip("tensorflow")

    sut = tvmc.frontends.guess_frontend("a_model.pb")
    assert type(sut) is tvmc.frontends.TensorflowFrontend


def test_guess_frontend_invalid():
    with pytest.raises(TVMCException):
        tvmc.frontends.guess_frontend("not/a/file.txt")


def test_load_model__invalid_path__no_language():
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    with pytest.raises(FileNotFoundError):
        tvmc.frontends.load_model("not/a/file.tflite")


def test_load_model__invalid_path__with_language():
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    with pytest.raises(FileNotFoundError):
        tvmc.frontends.load_model("not/a/file.txt", model_format="onnx")


def test_load_model__tflite(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    mod, params = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant)
    assert type(mod) is IRModule
    assert type(params) is dict
    # check whether one known value is part of the params dict
    assert "_param_1" in params.keys()


def test_load_model__keras(keras_resnet50):
    # some CI environments wont offer TensorFlow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    mod, params = tvmc.frontends.load_model(keras_resnet50)
    assert type(mod) is IRModule
    assert type(params) is dict
    ## check whether one known value is part of the params dict
    assert "_param_1" in params.keys()


def test_load_model__onnx(onnx_resnet50):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    mod, params = tvmc.frontends.load_model(onnx_resnet50)
    assert type(mod) is IRModule
    assert type(params) is dict
    ## check whether one known value is part of the params dict
    assert "resnetv24_batchnorm0_gamma" in params.keys()


def test_load_model__pb(pb_mobilenet_v1_1_quant):
    # some CI environments wont offer TensorFlow, so skip in case it is not present
    pytest.importorskip("tensorflow")

    mod, params = tvmc.frontends.load_model(pb_mobilenet_v1_1_quant)
    assert type(mod) is IRModule
    assert type(params) is dict
    # check whether one known value is part of the params dict
    assert "MobilenetV1/Conv2d_0/weights" in params.keys()


def test_load_model___wrong_language__to_keras(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TensorFlow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    with pytest.raises(OSError):
        tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant, model_format="keras")


def test_load_model___wrong_language__to_tflite(keras_resnet50):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    with pytest.raises(TVMCException):
        tvmc.frontends.load_model(keras_resnet50, model_format="tflite")


def test_load_model___wrong_language__to_onnx(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    from google.protobuf.message import DecodeError

    with pytest.raises(DecodeError):
        tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant, model_format="onnx")


def test_load_model___wrong_language__to_pytorch(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer pytorch, so skip in case it is not present
    pytest.importorskip("torch")

    with pytest.raises(RuntimeError) as e:
        tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant, model_format="pytorch")
