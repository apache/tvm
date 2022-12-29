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

import platform
import pytest
import builtins
import importlib

import tvm
from unittest import mock
from tvm.ir.module import IRModule

from tvm.driver import tvmc
from tvm.driver.tvmc import TVMCException, TVMCImportError
from tvm.driver.tvmc.model import TVMCModel


orig_import = importlib.import_module


def mock_error_on_name(name):
    def mock_imports(module_name, package=None):
        if module_name == name:
            raise ImportError()
        return orig_import(module_name, package)

    return mock_imports


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


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
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


def test_guess_frontend_paddle():
    # some CI environments wont offer Paddle, so skip in case it is not present
    pytest.importorskip("paddle")

    sut = tvmc.frontends.guess_frontend("a_model.pdmodel")
    assert type(sut) is tvmc.frontends.PaddleFrontend


def test_guess_frontend_relay():

    sut = tvmc.frontends.guess_frontend("relay.relay")
    assert type(sut) is tvmc.frontends.RelayFrontend


def test_guess_frontend_invalid():
    with pytest.raises(TVMCException):
        tvmc.frontends.guess_frontend("not/a/file.txt")


def test_load_model__invalid_path__no_language():
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    with pytest.raises(FileNotFoundError):
        tvmc.load("not/a/file.tflite")


def test_load_model__invalid_path__with_language():
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    with pytest.raises(FileNotFoundError):
        tvmc.load("not/a/file.txt", model_format="onnx")


def test_load_model__tflite(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict
    # check whether one known value is part of the params dict
    assert "_param_1" in tvmc_model.params.keys()


@pytest.mark.parametrize("load_model_kwargs", [{}, {"layout": "NCHW"}])
def test_load_model__keras(keras_resnet50, load_model_kwargs):
    # some CI environments wont offer TensorFlow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    tvmc_model = tvmc.frontends.load_model(keras_resnet50, **load_model_kwargs)
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict
    ## check whether one known value is part of the params dict
    assert "_param_1" in tvmc_model.params.keys()


def verify_load_model__onnx(model, **kwargs):
    tvmc_model = tvmc.frontends.load_model(model, **kwargs)
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict
    return tvmc_model


def test_load_model__onnx(onnx_resnet50):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")
    tvmc_model = verify_load_model__onnx(onnx_resnet50, freeze_params=False)
    # check whether one known value is part of the params dict
    assert "resnetv24_batchnorm0_gamma" in tvmc_model.params.keys()
    tvmc_model = verify_load_model__onnx(onnx_resnet50, freeze_params=True)
    # check that the parameter dict is empty, implying that they have been folded into constants
    assert tvmc_model.params == {}


def test_load_model__pb(pb_mobilenet_v1_1_quant):
    # some CI environments wont offer TensorFlow, so skip in case it is not present
    pytest.importorskip("tensorflow")

    tvmc_model = tvmc.load(pb_mobilenet_v1_1_quant)
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict
    # check whether one known value is part of the params dict
    assert "MobilenetV1/Conv2d_0/weights" in tvmc_model.params.keys()


def test_load_model__paddle(paddle_resnet50):
    # some CI environments wont offer Paddle, so skip in case it is not present
    pytest.importorskip("paddle")

    tvmc_model = tvmc.load(paddle_resnet50, model_format="paddle")
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict


def test_load_model__relay(relay_text_conv2d):
    tvmc_model = tvmc.load(relay_text_conv2d, model_format="relay")
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict


def test_load_model___wrong_language__to_keras(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TensorFlow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    with pytest.raises(OSError):
        tvmc.load(tflite_mobilenet_v1_1_quant, model_format="keras")


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
        tvmc.load(tflite_mobilenet_v1_1_quant, model_format="onnx")


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_load_model__pth(pytorch_resnet18):
    # some CI environments wont offer torch, so skip in case it is not present
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    tvmc_model = tvmc.load(pytorch_resnet18, shape_dict={"input": [1, 3, 224, 224]})
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict
    # check whether one known value is part of the params dict
    assert "layer1.0.conv1.weight" in tvmc_model.params.keys()


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_load_quantized_model__pth(pytorch_mobilenetv2_quantized):
    # some CI environments wont offer torch, so skip in case it is not present
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    tvmc_model = tvmc.load(pytorch_mobilenetv2_quantized, shape_dict={"input": [1, 3, 224, 224]})
    assert type(tvmc_model) is TVMCModel
    assert type(tvmc_model.mod) is IRModule
    assert type(tvmc_model.params) is dict

    # checking weights remain quantized and are not float32
    for p in tvmc_model.params.values():
        assert p.dtype in ["int8", "uint8", "int32"]  # int32 for bias


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_load_model___wrong_language__to_pytorch(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer pytorch, so skip in case it is not present
    pytest.importorskip("torch")

    with pytest.raises(RuntimeError) as e:
        tvmc.load(
            tflite_mobilenet_v1_1_quant,
            model_format="pytorch",
            shape_dict={"input": [1, 3, 224, 224]},
        )


def test_compile_tflite_module_nhwc_to_nchw(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    tvmc_model = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant)
    before = tvmc_model.mod

    expected_layout = "NCHW"
    with tvm.transform.PassContext(opt_level=3):
        after = tvmc.transform.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NHWC"
                and node.attrs.dst_layout == "NCHW"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert any(layout_transform_calls), "Expected 'layout_transform NHWC->NCHW' not found"


def test_compile_onnx_module_nchw_to_nhwc(onnx_resnet50):
    # some CI environments wont offer ONNX, so skip in case it is not present
    pytest.importorskip("onnx")

    tvmc_model = tvmc.frontends.load_model(onnx_resnet50)
    before = tvmc_model.mod

    expected_layout = "NHWC"
    with tvm.transform.PassContext(opt_level=3):
        after = tvmc.transform.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NCHW"
                and node.attrs.dst_layout == "NHWC"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert any(layout_transform_calls), "Expected 'layout_transform NCWH->NHWC' not found"


def test_compile_paddle_module_nchw_to_nhwc(paddle_resnet50):
    # some CI environments wont offer Paddle, so skip in case it is not present
    pytest.importorskip("paddle")

    tvmc_model = tvmc.frontends.load_model(paddle_resnet50, "paddle")
    before = tvmc_model.mod

    expected_layout = "NHWC"
    with tvm.transform.PassContext(opt_level=3):
        after = tvmc.transform.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NCHW"
                and node.attrs.dst_layout == "NHWC"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert any(layout_transform_calls), "Expected 'layout_transform NCWH->NHWC' not found"


def test_compile_tflite_module__same_layout__nhwc_to_nhwc(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer TFLite, so skip in case it is not present
    pytest.importorskip("tflite")

    tvmc_model = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant)
    before = tvmc_model.mod

    expected_layout = "NHWC"

    with tvm.transform.PassContext(opt_level=3):
        after = tvmc.transform.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NHWC"
                and node.attrs.dst_layout == "NHWC"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert not any(layout_transform_calls), "Unexpected 'layout_transform' call"


def test_compile_onnx_module__same_layout__nchw_to_nchw(onnx_resnet50):
    # some CI environments wont offer ONNX, so skip in case it is not present
    pytest.importorskip("onnx")

    tvmc_model = tvmc.frontends.load_model(onnx_resnet50)
    before = tvmc_model.mod

    expected_layout = "NCHW"

    with tvm.transform.PassContext(opt_level=3):
        after = tvmc.transform.convert_graph_layout(before, expected_layout)

    layout_transform_calls = []

    def _is_layout_transform(node):
        if isinstance(node, tvm.relay.expr.Call):
            layout_transform_calls.append(
                node.op.name == "layout_transform"
                and node.attrs.src_layout == "NCHW"
                and node.attrs.dst_layout == "NCHW"
            )

    tvm.relay.analysis.post_order_visit(after["main"], _is_layout_transform)

    assert not any(layout_transform_calls), "Unexpected 'layout_transform' call"


def test_import_keras_friendly_message(keras_resnet50, monkeypatch):
    # keras is part of tensorflow
    monkeypatch.setattr("importlib.import_module", mock_error_on_name("tensorflow"))

    with pytest.raises(TVMCImportError, match="tensorflow") as e:
        _ = tvmc.frontends.load_model(keras_resnet50, model_format="keras")


def test_import_onnx_friendly_message(onnx_resnet50, monkeypatch):
    monkeypatch.setattr("importlib.import_module", mock_error_on_name("onnx"))

    with pytest.raises(TVMCImportError, match="onnx") as e:
        _ = tvmc.frontends.load_model(onnx_resnet50, model_format="onnx")


def test_import_tensorflow_friendly_message(pb_mobilenet_v1_1_quant, monkeypatch):
    monkeypatch.setattr("importlib.import_module", mock_error_on_name("tensorflow"))

    with pytest.raises(TVMCImportError, match="tensorflow") as e:
        _ = tvmc.frontends.load_model(pb_mobilenet_v1_1_quant, model_format="pb")


@pytest.mark.skipif(
    platform.machine() == "aarch64",
    reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
)
def test_import_torch_friendly_message(pytorch_resnet18, monkeypatch):
    monkeypatch.setattr("importlib.import_module", mock_error_on_name("torch"))

    with pytest.raises(TVMCImportError, match="torch") as e:
        _ = tvmc.frontends.load_model(pytorch_resnet18, model_format="pytorch")


def test_import_tflite_friendly_message(tflite_mobilenet_v1_1_quant, monkeypatch):
    monkeypatch.setattr("importlib.import_module", mock_error_on_name("tflite.Model"))

    with pytest.raises(TVMCImportError, match="tflite.Model") as e:
        _ = tvmc.frontends.load_model(tflite_mobilenet_v1_1_quant, model_format="tflite")
