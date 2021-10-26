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
import re
import shutil
import tarfile
from os import path

from unittest import mock
import pytest

import tvm
import tvm.testing

from tvm.contrib.target.vitis_ai import vitis_ai_available

from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCPackage

from tvm.contrib import utils


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
    tvmc_model = tvmc.load(model, shape_dict=shape_dict)
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", dump_code="ll", desired_layout="NCHW")
    dumps_path = tvmc_package.package_path + ".ll"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


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

    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr='+neon'",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_options_aarch64_tflite_module(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")

    fake_sysroot_dir = utils.tempdir().relpath("")

    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr='+neon'",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
        cross_options="--sysroot=" + fake_sysroot_dir,
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


def test_compile_keras__save_module(keras_resnet50, tmpdir_factory):
    # some CI environments wont offer tensorflow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    expected_temp_dir = tmpdir_factory.mktemp("saved_output")
    expected_file_name = "saved.tar"
    module_file = os.path.join(expected_temp_dir, expected_file_name)

    tvmc_model = tvmc.load(keras_resnet50)
    tvmc.compile(tvmc_model, target="llvm", dump_code="ll", package_path=module_file)

    assert os.path.exists(module_file), "output file {0} should exist".format(module_file)

    # Test that we can load back in a module.
    tvmc_package = TVMCPackage(package_path=module_file)
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.params) is bytearray


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_aarch64_keras_module(keras_resnet50):
    # some CI environments wont offer tensorflow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    tvmc_model = tvmc.load(keras_resnet50)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr='+neon'",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_options_aarch64_keras_module(keras_resnet50):
    # some CI environments wont offer tensorflow/Keras, so skip in case it is not present
    pytest.importorskip("tensorflow")

    fake_sysroot_dir = utils.tempdir().relpath("")

    tvmc_model = tvmc.load(keras_resnet50)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr='+neon'",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
        cross_options="--sysroot=" + fake_sysroot_dir,
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


def verify_compile_onnx_module(model, shape_dict=None):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")
    tvmc_model = tvmc.load(model, shape_dict=shape_dict)
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", dump_code="ll")
    dumps_path = tvmc_package.package_path + ".ll"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


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

    tvmc_model = tvmc.load(onnx_resnet50)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_options_aarch64_onnx_module(onnx_resnet50):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")

    fake_sysroot_dir = utils.tempdir().relpath("")

    tvmc_model = tvmc.load(onnx_resnet50)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
        cross_options="--sysroot=" + fake_sysroot_dir,
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


def verify_compile_paddle_module(model, shape_dict=None):
    pytest.importorskip("paddle")
    tvmc_model = tvmc.load(model, "paddle", shape_dict=shape_dict)
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", dump_code="ll", desired_layout="NCHW")
    dumps_path = tvmc_package.package_path + ".ll"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


def test_compile_paddle_module(paddle_resnet50):
    # some CI environments wont offer Paddle, so skip in case it is not present
    pytest.importorskip("paddle")
    # Check default compilation.
    verify_compile_paddle_module(paddle_resnet50)
    # Check with manual shape override
    shape_string = "inputs:[1,3,224,224]"
    shape_dict = tvmc.common.parse_shape_string(shape_string)
    verify_compile_paddle_module(paddle_resnet50, shape_dict)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_aarch64_paddle_module(paddle_resnet50):
    # some CI environments wont offer paddle, so skip in case it is not present
    pytest.importorskip("paddle")

    tvmc_model = tvmc.load(paddle_resnet50, "paddle")
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


# This test will be skipped if the AArch64 cross-compilation toolchain is not installed.
@pytest.mark.skipif(
    not shutil.which("aarch64-linux-gnu-gcc"), reason="cross-compilation toolchain not installed"
)
def test_cross_compile_options_aarch64_paddle_module(paddle_resnet50):
    # some CI environments wont offer paddle, so skip in case it is not present
    pytest.importorskip("paddle")

    fake_sysroot_dir = utils.tempdir().relpath("")

    tvmc_model = tvmc.load(paddle_resnet50, "paddle")
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon",
        dump_code="asm",
        cross="aarch64-linux-gnu-gcc",
        cross_options="--sysroot=" + fake_sysroot_dir,
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


@tvm.testing.requires_opencl
def test_compile_opencl(tflite_mobilenet_v1_0_25_128):
    pytest.importorskip("tflite")
    tvmc_model = tvmc.load(tflite_mobilenet_v1_0_25_128)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="opencl --host=llvm",
        desired_layout="NCHW",
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


@tvm.testing.requires_ethosn
def test_compile_tflite_module_with_external_codegen(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")
    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    tvmc_package = tvmc.compile(tvmc_model, target="ethos-n77, llvm", dump_code="relay")
    dumps_path = tvmc_package.package_path + ".relay"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


def test_compile_tflite_module_with_external_codegen_cmsisnn(
    tmpdir_factory, tflite_cnn_s_quantized
):
    pytest.importorskip("tflite")

    output_dir = tmpdir_factory.mktemp("mlf")
    tvmc_model = tvmc.load(tflite_cnn_s_quantized)

    output_file_name = f"{output_dir}/file.tar"

    tvmc_package = tvmc.compiler.compile_model(
        tvmc_model,
        target=f"cmsis-nn, c -runtime=c --system-lib --link-params -mcpu=cortex-m55 -executor=aot",
        output_format="mlf",
        package_path=output_file_name,
        pass_context_configs=["tir.disable_vectorize=true"],
    )

    # check whether an MLF package was created
    assert os.path.exists(output_file_name)

    # check whether the expected number of C sources are in the tarfile
    with tarfile.open(output_file_name) as mlf_package:
        c_source_files = [
            name
            for name in mlf_package.getnames()
            if re.match(r"\./codegen/host/src/\D+\d+\.c", name)
        ]
        assert len(c_source_files) == 3


@pytest.mark.skipif(
    not vitis_ai_available(),
    reason="--target=vitis-ai is not available. TVM built with 'USE_VITIS_AI OFF'",
)
def test_compile_tflite_module_with_external_codegen_vitis_ai(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")

    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    tvmc_package = tvmc.compiler.compile_model(
        tvmc_model,
        target="vitis-ai -dpu=DPUCZDX8G-zcu104 -export_runtime_module=vitis_ai.rtmod, llvm",
        dump_code="relay",
    )
    dumps_path = tvmc_package.package_path + ".relay"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


def test_compile_tflite_module_with_external_codegen_ethosu(
    tmpdir_factory, tflite_mobilenet_v1_1_quant
):
    pytest.importorskip("tflite")
    pytest.importorskip("ethosu.vela")
    ACCEL_TYPES = ["ethos-u55-256", "ethos-u55-128", "ethos-u55-64", "ethos-u55-32"]

    output_dir = tmpdir_factory.mktemp("mlf")

    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)

    for accel_type in ACCEL_TYPES:
        output_file_name = f"{output_dir}/file_{accel_type}.tar"

        tvmc_package = tvmc.compiler.compile_model(
            tvmc_model,
            target=f"ethos-u -accelerator_config={accel_type}, c -runtime=c --system-lib --link-params -mcpu=cortex-m55 -executor=aot",
            output_format="mlf",
            package_path=output_file_name,
            pass_context_configs=["tir.disable_vectorize=true"],
        )

        # check whether an MLF package was created
        assert os.path.exists(output_file_name)

        # check whether the expected number of C sources are in the tarfile
        with tarfile.open(output_file_name) as mlf_package:
            c_source_files = [
                name
                for name in mlf_package.getnames()
                if re.match(r"\./codegen/host/src/\D+\d+\.c", name)
            ]
            # The number of c_source_files depends on the number of fused subgraphs that
            # get offloaded to the NPU, e.g. conv2d->depthwise_conv2d->conv2d gets offloaded
            # as a single subgraph if both of these operators are supported by the NPU.
            # Currently there are two source files for CPU execution and two offload graphs
            assert len(c_source_files) == 4


@mock.patch("tvm.relay.build")
@mock.patch("tvm.driver.tvmc.composite_target.get_codegen_by_target")
@mock.patch("tvm.driver.tvmc.load")
@mock.patch("tvm.transform.PassContext")
@mock.patch("tvm.driver.tvmc.model.TVMCPackage.__init__", return_value=None)
def test_compile_check_configs_composite_target(mock_pkg, mock_pc, mock_fe, mock_ct, mock_relay):
    mock_codegen = {}
    mock_codegen["config_key"] = "relay.ext.mock.options"
    mock_codegen["pass_pipeline"] = lambda *args, **kwargs: None

    mock_fe.return_value = mock.MagicMock()
    mock_ct.return_value = mock_codegen
    mock_relay.return_value = mock.MagicMock()

    tvmc_model = tvmc.load("no_file_needed")
    tvmc.compile(tvmc_model, target="mockcodegen -testopt=value, llvm")

    mock_pc.assert_called_once_with(
        opt_level=3,
        config={"relay.ext.mock.options": {"testopt": "value"}},
        disabled_pass=None,
    )
