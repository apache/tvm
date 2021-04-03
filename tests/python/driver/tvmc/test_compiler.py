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
import argparse
import os
import shutil
from os import path

from unittest import mock
import pytest

import tvm

from tvm.relay.op.contrib.ethosn import ethosn_available
from tvm.contrib.target.vitis_ai import vitis_ai_available

from tvm.driver import tvmc


def test_save_dumps(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data")
    dump_formats = {"relay": "fake relay", "ll": "fake llvm", "asm": "fake asm"}
    tvmc.compiler.save_dumps("fake_module", dump_formats, dump_root=tmpdir)

    assert path.exists("{}/{}".format(tmpdir, "fake_module.ll"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.asm"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.relay"))


# End to end tests for compilation


def verify_compile_tflite_module(model, shape_dict=None):
    pytest.importorskip("tflite")
    mod, params = tvmc.load(model, shape_dict=shape_dict)
    graph, lib, params, dumps = tvmc.compile(
        mod, params, target="llvm", dump_code="ll", alter_layout="NCHW"
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict


def test_compile_tflite_module(tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer tflite, so skip in case it is not present
    pytest.importorskip("tflite")
    # Check default compilation.
    verify_compile_tflite_module(tflite_mobilenet_v1_1_quant)
    # Check with manual shape override
    shape_string = "input:[1,224,224,3]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    verify_compile_tflite_module(tflite_mobilenet_v1_1_quant, shape_dict)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_aarch64_tflite_module(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")

    mod, params = tvmc.load(tflite_mobilenet_v1_1_quant)
    graph, lib, params, dumps = tvmc.compile(
        mod,
        params,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr='+neon'",
        dump_code="asm",
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict


def test_compile_keras__save_module(keras_resnet50, tmpdir_factory):
    # some CI environments wont offer tensorflow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    mod, params = tvmc.load(keras_resnet50)
    graph, lib, params, dumps = tvmc.compile(mod, params, target="llvm", dump_code="ll")

    expected_temp_dir = tmpdir_factory.mktemp("saved_output")
    expected_file_name = "saved.tar"
    module_file = os.path.join(expected_temp_dir, expected_file_name)
    tvmc.compiler.save_module(module_file, graph, lib, params)

    assert os.path.exists(module_file), "output file {0} should exist".format(module_file)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_aarch64_keras_module(keras_resnet50):
    # some CI environments wont offer tensorflow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    mod, params = tvmc.load(keras_resnet50)
    graph, lib, params, dumps = tvmc.compile(
        mod,
        params,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr='+neon'",
        dump_code="asm",
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict
    assert "asm" in dumps.keys()


def verify_compile_onnx_module(model, shape_dict=None):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")
    mod, params = tvmc.load(model, shape_dict=shape_dict)
    graph, lib, params, dumps = tvmc.compile(mod, params, target="llvm", dump_code="ll")

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict
    assert "ll" in dumps.keys()


def test_compile_onnx_module(onnx_resnet50):
    # Test default compilation
    verify_compile_onnx_module(onnx_resnet50)
    # Test with manual shape dict
    shape_string = "data:[1,3,200,200]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    verify_compile_onnx_module(onnx_resnet50, shape_dict)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_aarch64_onnx_module(onnx_resnet50):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    mod, params = tvmc.load(onnx_resnet50)
    graph, lib, params, dumps = tvmc.compile(
        mod,
        params,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
        dump_code="asm",
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict
    assert "asm" in dumps.keys()


@tvm.testing.requires_opencl
def test_compile_opencl(tflite_mobilenet_v1_0_25_128):
    pytest.importorskip("tflite")
    mod, params = tvmc.load(tflite_mobilenet_v1_0_25_128)
    graph, lib, params, dumps = tvmc.compile(
        mod,
        params,
        target="opencl --host=llvm",
        alter_layout="NCHW",
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict


@pytest.mark.skipif(
    not ethosn_available(),
    reason="--target=ethos-n77 is not available. TVM built with 'USE_ETHOSN OFF'",
)
def test_compile_tflite_module_with_external_codegen(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")
    mod, params = tvmc.load(tflite_mobilenet_v1_1_quant)
    graph, lib, params, dumps = tvmc.compile(
        mod, params, target="ethos-n77, llvm", dump_code="relay"
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict


@pytest.mark.skipif(
    not vitis_ai_available(),
    reason="--target=vitis-ai is not available. TVM built with 'USE_VITIS_AI OFF'",
)
def test_compile_tflite_module_with_external_codegen_vitis_ai(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")

    mod, params = tvmc.load(tflite_mobilenet_v1_1_quant)
    graph, lib, params, dumps = tvmc.compiler.compile_model(
        mod,
        params,
        target="vitis-ai -dpu=DPUCZDX8G-zcu104 -export_runtime_module=vitis_ai.rtmod, llvm",
        dump_code="relay",
    )

    # check for output types
    assert type(graph) is str
    assert type(lib) is tvm.runtime.module.Module
    assert type(params) is dict
    assert type(dumps) is dict


@mock.patch("tvm.relay.build")
@mock.patch("tvm.driver.tvmc.composite_target.get_codegen_by_target")
@mock.patch("tvm.driver.tvmc.load")
@mock.patch("tvm.transform.PassContext")
def test_compile_check_configs_composite_target(mock_pc, mock_fe, mock_ct, mock_relay):
    mock_codegen = {}
    mock_codegen["config_key"] = "relay.ext.mock.options"
    mock_codegen["pass_pipeline"] = lambda *args, **kwargs: None

    mock_fe.return_value = (None, None)
    mock_ct.return_value = mock_codegen
    mock_relay.return_value = mock.MagicMock()

    mod, params = tvmc.load("no_file_needed")
    graph, lib, params, dumps = tvmc.compile(mod, params, target="mockcodegen -testopt=value, llvm")

    mock_pc.assert_called_once_with(
        opt_level=3, config={"relay.ext.mock.options": {"testopt": "value"}}
    )
