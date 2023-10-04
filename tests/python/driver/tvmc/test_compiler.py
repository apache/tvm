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
import numpy as np
import shutil
import tarfile
from os import path

from unittest import mock
import pytest

import tvm
from tvm.ir.memory_pools import WorkspacePoolInfo, WorkspaceMemoryPools
from tvm.target import Target
import tvm.testing
from tvm.relay.op.contrib.ethosn import ethosn_available
from tvm.relay.backend import Runtime, Executor
from tvm import relay

from tvm.contrib.target.vitis_ai import vitis_ai_available

from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCPackage

from tvm.contrib import utils


def test_save_dumps(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("data")
    dump_formats = {"relay": "fake relay", "tir": "fake tir", "ll": "fake llvm", "asm": "fake asm"}
    tvmc.compiler.save_dumps("fake_module", dump_formats, dump_root=tmpdir)

    assert path.exists("{}/{}".format(tmpdir, "fake_module.ll"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.asm"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.tir"))
    assert path.exists("{}/{}".format(tmpdir, "fake_module.relay"))


def test_save_dump_offloads_ethosu(tmp_path_factory):

    tflite = pytest.importorskip("tflite")
    tensorflow = pytest.importorskip("tensorflow")
    pytest.importorskip("ethosu.vela")

    import tensorflow as tf
    import tflite.Model
    from tvm.driver.tvmc.model import TVMCModel

    inp = (224, 224, 9)
    input_shape = (1, *inp)
    kernel_shape = (3, 3)
    padding = (1, 1, 1, 1)
    padding_out = (1, 33, 33, 1)

    @tf.function
    def simple_net(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], input_shape[3], 3]
        weights = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weight_shape[2] = 3
        weights1 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weights2 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        op = tf.nn.conv2d(
            x,
            filters=weights,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op1 = tf.nn.conv2d(
            op,
            filters=weights1,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op2 = tf.nn.conv2d(
            op,
            filters=weights2,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op = tf.concat([op1, op2], 1)
        op = tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[1]], [padding_out[2], padding[3]], [0, 0]],
            "CONSTANT",
        )
        return op

    from tests.python.contrib.test_ethosu.infra import get_tflite_graph

    _, tflite_graph = get_tflite_graph(simple_net, [input_shape])
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)

    tvmc_model = TVMCModel(mod, params)

    output_dir = tmp_path_factory.mktemp("tmp")
    output_file_name = os.path.join(str(output_dir), "list.txt")

    tvmc.compiler.compile_model(
        tvmc_model,
        target="ethos-u,cmsis-nn,c",
        runtime=Runtime("crt"),
        tuning_records="",
        package_path="module.tar",
        executor=Executor("aot", {"unpacked-api": 1, "interface-api": "c", "link-params": True}),
        cross="",
        cross_options="",
        output_format="mlf",
        dump_offloads=output_file_name,
        disabled_pass=[""],
        pass_context_configs=[
            "tir.disable_vectorize=1",
            "tir.usmp.enable=1",
            "tir.usmp.algorithm=hill_climb",
            "tir.disable_storage_rewrite=1",
            "relay.frontend.fill_span=1",
        ],
        additional_target_options={
            "c": {"mcpu": "cortex-m55"},
            "cmsis-nn": {"mcpu": "cortex-m55"},
            "ethos-u": {
                "accelerator_config": "ethos-u55-256",
            },
        },
    )

    expected = [
        r"Total number of operators and distribution by targets",
        r"Total: 11",
        r"ethos-u: 10",
        r"generic: 1",
        r"",
        r"ethos-u        <-     ethos-u.qnn_conv2d",
        r'ethos-u        <-          %0 = qnn.conv2d(%x, %v_param_1, -128, 0, 0.00392157f, meta[relay.Constant][0], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"ethos-u        <-          %1 = nn.bias_add(%0, %v_param_2, axis=3)",
        r'ethos-u        <-          %2 = qnn.requantize(%1, meta[relay.Constant][1], 0, 0.11364f, -128, axis=3, out_dtype="int8")',
        r"ethos-u        <-     ethos-u.qnn_conv2d",
        r'ethos-u        <-          %3 = qnn.conv2d(%2, %v_param_3, -128, 0, 0.11364f, meta[relay.Constant][2], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"ethos-u        <-          %4 = nn.bias_add(%3, %v_param_4, axis=3)",
        r'ethos-u        <-          %7 = qnn.requantize(%4, meta[relay.Constant][3], 0, 1.56803f, -128, axis=3, out_dtype="int8")',
        r"ethos-u        <-     ethos-u.qnn_conv2d",
        r'ethos-u        <-          %5 = qnn.conv2d(%2, %v_param_5, -128, 0, 0.11364f, meta[relay.Constant][4], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"ethos-u        <-          %6 = nn.bias_add(%5, %v_param_6, axis=3)",
        r'ethos-u        <-          %8 = qnn.requantize(%6, meta[relay.Constant][5], 0, 1.20538f, -128, axis=3, out_dtype="int8")',
        r"                      %9 = (%7, %8)",
        r"                      %10 = (1.59778f, 1.59778f)",
        r"                      %11 = (-128, -128)",
        r"ethos-u        <-     ethos-u.concat",
        r"ethos-u        <-          %12 = qnn.concatenate(%9, %10, %11, 1.59778f, -128, axis=1)",
        r"generic        <-     nn.pad(%12, -128f, pad_width=[[0, 0], [1, 33], [33, 1], [0, 0]])",
    ]

    file_path = os.path.abspath(output_file_name)
    # check that file file_path was created
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        for i, file_string in enumerate(f):
            r_output = re.search(r"(.*)\(", file_string.strip(), re.DOTALL)
            r_expected = re.search(r"(.*)\(", expected[i].strip(), re.DOTALL)
            # check that there is the same sequence of operations and composites,
            # combined with target names
            if r_output and r_expected:
                assert r_output.group(0) == r_expected.group(0)
            else:
                assert r_output == r_expected


def test_save_dump_offloads_cmsis(tmp_path_factory):

    tflite = pytest.importorskip("tflite")
    tensorflow = pytest.importorskip("tensorflow")
    pytest.importorskip("ethosu.vela")

    import tensorflow as tf
    from tvm.driver.tvmc.model import TVMCModel

    inp = (224, 224, 9)
    input_shape = (1, *inp)
    kernel_shape = (3, 3)
    padding = (1, 1, 1, 1)
    padding_out = (1, 33, 33, 1)

    @tf.function
    def simple_net(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], input_shape[3], 3]
        weights = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weight_shape[2] = 3
        weights1 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weights2 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        op = tf.nn.conv2d(
            x,
            filters=weights,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op1 = tf.nn.conv2d(
            op,
            filters=weights1,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op2 = tf.nn.conv2d(
            op,
            filters=weights2,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op = tf.concat([op1, op2], 1)
        op = tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[1]], [padding_out[2], padding[3]], [0, 0]],
            "CONSTANT",
        )
        return op

    from tests.python.contrib.test_ethosu.infra import get_tflite_graph

    _, tflite_graph = get_tflite_graph(simple_net, [input_shape])
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)

    tvmc_model = TVMCModel(mod, params)

    output_dir = tmp_path_factory.mktemp("tmp")
    output_file_name = os.path.join(str(output_dir), "list.txt")

    tvmc.compiler.compile_model(
        tvmc_model,
        target="cmsis-nn,c",
        runtime=Runtime("crt"),
        tuning_records="",
        package_path="module.tar",
        executor=Executor("aot", {"unpacked-api": 1, "interface-api": "c", "link-params": True}),
        cross="",
        cross_options="",
        output_format="mlf",
        dump_offloads=output_file_name,
        disabled_pass=[""],
        pass_context_configs=[
            "tir.disable_vectorize=1",
            "tir.usmp.enable=1",
            "tir.usmp.algorithm=hill_climb",
            "tir.disable_storage_rewrite=1",
            "relay.frontend.fill_span=1",
        ],
        additional_target_options={
            "c": {"mcpu": "cortex-m55"},
            "cmsis-nn": {"mcpu": "cortex-m55"},
        },
    )

    expected = [
        r"Total number of operators and distribution by targets",
        r"Total: 11",
        r"cmsis-nn: 9",
        r"generic: 2",
        r"",
        r"cmsis-nn       <-     cmsis-nn.qnn_conv2d",
        r'cmsis-nn       <-          %0 = qnn.conv2d(%x, %v_param_1, -128, 0, 0.00392157f, meta[relay.Constant][0], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"cmsis-nn       <-          %1 = nn.bias_add(%0, %v_param_2, axis=3)",
        r'cmsis-nn       <-          %2 = qnn.requantize(%1, meta[relay.Constant][1], 0, 0.115114f, -128, axis=3, out_dtype="int8")',
        r"cmsis-nn       <-     cmsis-nn.qnn_conv2d",
        r'cmsis-nn       <-          %3 = qnn.conv2d(%2, %v_param_3, -128, 0, 0.115114f, meta[relay.Constant][2], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"cmsis-nn       <-          %4 = nn.bias_add(%3, %v_param_4, axis=3)",
        r'cmsis-nn       <-          %7 = qnn.requantize(%4, meta[relay.Constant][3], 0, 1.59328f, -128, axis=3, out_dtype="int8")',
        r"cmsis-nn       <-     cmsis-nn.qnn_conv2d",
        r'cmsis-nn       <-          %5 = qnn.conv2d(%2, %v_param_5, -128, 0, 0.115114f, meta[relay.Constant][4], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"cmsis-nn       <-          %6 = nn.bias_add(%5, %v_param_6, axis=3)",
        r'cmsis-nn       <-          %8 = qnn.requantize(%6, meta[relay.Constant][5], 0, 1.59328f, -128, axis=3, out_dtype="int8")',
        r"                      %9 = (%7, %8)",
        r"                      %10 = (1.59328f, 1.59328f)",
        r"                      %11 = (-128, -128)",
        r"generic        <-     %12 = qnn.concatenate(%9, %10, %11, 1.59328f, -128, axis=1)",
        r"generic        <-     nn.pad(%12, -128f, pad_width=[[0, 0], [1, 33], [33, 1], [0, 0]])",
    ]

    file_path = os.path.abspath(output_file_name)
    # check that file file_path was created
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        for i, file_string in enumerate(f):
            r_output = re.search(r"(.*)\(", file_string.replace("\n", ""), re.DOTALL)
            r_expected = re.search(r"(.*)\(", expected[i], re.DOTALL)
            # check that there is the same sequence of operations and composites,
            # combined with target names
            if r_output and r_expected:
                assert r_output.group(0) == r_expected.group(0)
            else:
                assert file_string.replace("\n", "") == expected[i]


def test_save_dump_offloads_generic(tmp_path_factory):

    tflite = pytest.importorskip("tflite")
    tensorflow = pytest.importorskip("tensorflow")
    pytest.importorskip("ethosu.vela")

    import tensorflow as tf
    from tvm.driver.tvmc.model import TVMCModel

    inp = (224, 224, 9)
    input_shape = (1, *inp)
    kernel_shape = (3, 3)
    padding = (1, 1, 1, 1)
    padding_out = (1, 33, 33, 1)

    @tf.function
    def simple_net(x):
        weight_shape = [kernel_shape[0], kernel_shape[1], input_shape[3], 3]
        weights = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weight_shape[2] = 3
        weights1 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        weights2 = tf.constant(np.random.uniform(size=weight_shape), dtype=tf.float32)
        op = tf.nn.conv2d(
            x,
            filters=weights,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op1 = tf.nn.conv2d(
            op,
            filters=weights1,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op2 = tf.nn.conv2d(
            op,
            filters=weights2,
            strides=1,
            padding="SAME",
            data_format="NHWC",
            dilations=1,
        )
        op = tf.concat([op1, op2], 1)
        op = tf.pad(
            op,
            [[0, 0], [padding[0], padding_out[1]], [padding_out[2], padding[3]], [0, 0]],
            "CONSTANT",
        )
        return op

    from tests.python.contrib.test_ethosu.infra import get_tflite_graph

    _, tflite_graph = get_tflite_graph(simple_net, [input_shape])
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_graph, 0)
    mod, params = relay.frontend.from_tflite(tflite_model)

    tvmc_model = TVMCModel(mod, params)

    output_dir = tmp_path_factory.mktemp("tmp")
    output_file_name = os.path.join(str(output_dir), "list.txt")

    tvmc.compiler.compile_model(
        tvmc_model,
        target="c",
        runtime=Runtime("crt"),
        tuning_records="",
        package_path="module.tar",
        executor=Executor("aot", {"unpacked-api": 1, "interface-api": "c", "link-params": True}),
        cross="",
        cross_options="",
        output_format="mlf",
        dump_offloads=output_file_name,
        disabled_pass=[""],
        pass_context_configs=[
            "tir.disable_vectorize=1",
            "tir.usmp.enable=1",
            "tir.usmp.algorithm=hill_climb",
            "tir.disable_storage_rewrite=1",
            "relay.frontend.fill_span=1",
        ],
        additional_target_options={
            "c": {"mcpu": "cortex-m55"},
        },
    )

    expected = [
        r"Total number of operators and distribution by targets",
        r"Total: 11",
        r"generic: 11",
        r"",
        r'generic        <-     %0 = qnn.conv2d(%x, %v_param_1, -128, 0, 0.00392157f, meta[relay.Constant][0], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"generic        <-     %1 = nn.bias_add(%0, %v_param_2, axis=3)",
        r'generic        <-     %2 = qnn.requantize(%1, meta[relay.Constant][1], 0, 0.109484f, -128, axis=3, out_dtype="int8")',
        r'generic        <-     %3 = qnn.conv2d(%2, %v_param_3, -128, 0, 0.109484f, meta[relay.Constant][2], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"generic        <-     %4 = nn.bias_add(%3, %v_param_4, axis=3)",
        r'generic        <-     %5 = qnn.conv2d(%2, %v_param_5, -128, 0, 0.109484f, meta[relay.Constant][4], padding=[1, 1, 1, 1], channels=3, kernel_size=[3, 3], data_layout="NHWC", kernel_layout="HWIO", out_dtype="int32")',
        r"generic        <-     %6 = nn.bias_add(%5, %v_param_6, axis=3)",
        r'generic        <-     %7 = qnn.requantize(%4, meta[relay.Constant][3], 0, 1.45572f, -128, axis=3, out_dtype="int8")',
        r'generic        <-     %8 = qnn.requantize(%6, meta[relay.Constant][5], 0, 1.45572f, -128, axis=3, out_dtype="int8")',
        r"                      %9 = (%7, %8)",
        r"                      %10 = (1.45572f, 1.45572f)",
        r"                      %11 = (-128, -128)",
        r"generic        <-     %12 = qnn.concatenate(%9, %10, %11, 1.45572f, -128, axis=1)",
        r"generic        <-     nn.pad(%12, -128f, pad_width=[[0, 0], [1, 33], [33, 1], [0, 0]])",
    ]

    file_path = os.path.abspath(output_file_name)
    # check that file file_path was created
    assert os.path.exists(file_path)
    with open(file_path, "r") as f:
        for i, file_string in enumerate(f):
            r_output = re.search(r"(.*)\(", file_string.replace("\n", ""), re.DOTALL)
            r_expected = re.search(r"(.*)\(", expected[i], re.DOTALL)
            # check that there is the same sequence of operations and composites,
            # combined with target names
            if r_output and r_expected:
                assert r_output.group(0) == r_expected.group(0)
            else:
                assert file_string.replace("\n", "") == expected[i]


# End to end tests for compilation


def verify_tvmc_package(tvmc_package, dumps_path, use_vm=False):
    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert os.path.exists(dumps_path)
    assert type(tvmc_package.lib_path) is str

    if use_vm:
        assert tvmc_package.graph is None
        assert tvmc_package.params is None
    else:
        assert type(tvmc_package.graph) is str
        assert type(tvmc_package.params) is bytearray


def verify_compile_tflite_module(model, shape_dict=None, use_vm=False):
    pytest.importorskip("tflite")
    tvmc_model = tvmc.load(model, shape_dict=shape_dict)
    tvmc_package = tvmc.compile(
        tvmc_model,
        target="llvm",
        dump_code="ll",
        desired_layout="NCHW",
        use_vm=use_vm,
    )
    dumps_path = tvmc_package.package_path + ".ll"
    verify_tvmc_package(tvmc_package, dumps_path, use_vm=use_vm)


@pytest.mark.parametrize("use_vm", [True, False])
def test_compile_tflite_module(use_vm, tflite_mobilenet_v1_1_quant):
    # some CI environments wont offer tflite, so skip in case it is not present
    pytest.importorskip("tflite")
    # Check default compilation.
    verify_compile_tflite_module(tflite_mobilenet_v1_1_quant)
    # Check with manual shape override
    shape_string = "input:[1,224,224,3]"
    shape_dict = tvmc.shape_parser.parse_shape_string(shape_string)
    verify_compile_tflite_module(tflite_mobilenet_v1_1_quant, shape_dict, use_vm=use_vm)


def test_single_tir_dump(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")
    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", dump_code="tir")
    dumps_path = tvmc_package.package_path + ".tir"
    assert os.path.exists(dumps_path)
    with open(dumps_path) as f:
        assert "tir" in f.read()


def test_code_dumps(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")
    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    dump_code = ["asm", "ll", "tir", "relay"]
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", dump_code=dump_code)
    for ext in dump_code:
        dumps_path = tvmc_package.package_path + "." + ext
        assert os.path.exists(dumps_path)
        with open(dumps_path) as f:
            assert len(f.read()) > 0


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


def verify_compile_onnx_module(model, shape_dict=None, use_vm=False):
    # some CI environments wont offer onnx, so skip in case it is not present
    pytest.importorskip("onnx")
    tvmc_model = tvmc.load(model, shape_dict=shape_dict)
    tvmc_package = tvmc.compile(tvmc_model, target="llvm", dump_code="ll", use_vm=use_vm)
    dumps_path = tvmc_package.package_path + ".ll"
    verify_tvmc_package(tvmc_package, dumps_path, use_vm=use_vm)


@pytest.mark.parametrize("use_vm", [True, False])
def test_compile_onnx_module(use_vm, onnx_resnet50):
    # Test default compilation
    verify_compile_onnx_module(onnx_resnet50)
    # Test with manual shape dict
    shape_string = "data:[1,3,200,200]"
    shape_dict = tvmc.shape_parser.parse_shape_string(shape_string)
    verify_compile_onnx_module(onnx_resnet50, shape_dict, use_vm=use_vm)


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
    shape_dict = tvmc.shape_parser.parse_shape_string(shape_string)
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
        target="opencl -host=llvm",
        desired_layout="NCHW",
        dump_code="asm",
    )
    dumps_path = tvmc_package.package_path + ".asm"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)
    assert path.exists("{}.{}".format(tvmc_package.package_path, "opencl"))


@tvm.testing.requires_cmsisnn
def test_compile_tflite_module_with_external_codegen_cmsisnn(
    tmpdir_factory, tflite_cnn_s_quantized
):
    pytest.importorskip("tflite")

    output_dir = tmpdir_factory.mktemp("mlf")
    tvmc_model = tvmc.load(tflite_cnn_s_quantized)

    output_file_name = f"{output_dir}/file.tar"

    tvmc.compiler.compile_model(
        tvmc_model,
        target=f"cmsis-nn, c -mcpu=cortex-m55",
        runtime=Runtime("crt", {"system-lib": True}),
        executor=Executor("aot"),
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
        assert len(c_source_files) == 4


@tvm.testing.requires_ethosn
def test_compile_tflite_module_with_external_codegen_ethos_n78(tflite_mobilenet_v1_1_quant):
    pytest.importorskip("tflite")
    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    tvmc_package = tvmc.compile(tvmc_model, target="ethos-n -variant=n78, llvm", dump_code="relay")
    dumps_path = tvmc_package.package_path + ".relay"

    # check for output types
    assert type(tvmc_package) is TVMCPackage
    assert type(tvmc_package.graph) is str
    assert type(tvmc_package.lib_path) is str
    assert type(tvmc_package.params) is bytearray
    assert os.path.exists(dumps_path)


@tvm.testing.requires_vitis_ai
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

        tvmc.compiler.compile_model(
            tvmc_model,
            target=f"ethos-u -accelerator_config={accel_type}, c -mcpu=cortex-m55",
            runtime=Runtime("crt"),
            executor=Executor("aot", {"unpacked-api": True}),
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
            # Currently there are three source files for CPU execution and one offload graph
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

    assert mock_pc.call_count == 1
    codegen_compile_context = mock.call(
        config={"relay.ext.mock.options": {"testopt": "value"}},
        opt_level=3,
        disabled_pass=None,
        instruments=None,
    )
    mock_pc.assert_has_calls(
        [
            codegen_compile_context,
            codegen_compile_context.__enter__(),
            codegen_compile_context.__exit__(None, None, None),
        ]
    )


def test_compile_tflite_module_with_mod_name(tmpdir_factory, tflite_cnn_s_quantized):
    pytest.importorskip("tflite")

    output_dir = tmpdir_factory.mktemp("mlf")
    tvmc_model = tvmc.load(tflite_cnn_s_quantized)

    output_file_name = f"{output_dir}/file.tar"

    tvmc.compiler.compile_model(
        tvmc_model,
        target=f"c -mcpu=cortex-m55",
        runtime=Runtime("crt", {"system-lib": True}),
        executor=Executor("aot"),
        output_format="mlf",
        package_path=output_file_name,
        pass_context_configs=["tir.disable_vectorize=true"],
        mod_name="classify",
    )

    # check that an MLF package was created
    assert os.path.exists(output_file_name)

    with tarfile.open(output_file_name) as mlf_package:
        # check that the C source files have been named classify_lib*.c
        c_source_files = [
            name
            for name in mlf_package.getnames()
            if re.match(r"\./codegen/host/src/classify_lib\d+\.c", name)
        ]
        assert len(c_source_files) > 0

        # check that "default" doesn't occur in any of the C source files
        # check that function names are of the form "tvmgen_classify_*"
        for file_name in c_source_files:
            with mlf_package.extractfile(file_name) as f:
                content = f.read()
                assert b"default" not in content
                assert b"tvmgen_classify_" in content

        # check that tvmgen_classify_run() function exists
        with mlf_package.extractfile("./codegen/host/src/classify_lib0.c") as f:
            content = f.read()
            assert b"tvmgen_classify_run(" in content


@tvm.testing.requires_cmsisnn
def test_compile_tflite_module_with_mod_name_and_cmsisnn(tmpdir_factory, tflite_cnn_s_quantized):
    pytest.importorskip("tflite")

    output_dir = tmpdir_factory.mktemp("mlf")
    tvmc_model = tvmc.load(tflite_cnn_s_quantized)

    output_file_name = f"{output_dir}/file.tar"

    tvmc.compiler.compile_model(
        tvmc_model,
        target=f"cmsis-nn, c -mcpu=cortex-m55",
        runtime=Runtime("crt", {"system-lib": True}),
        executor=Executor("aot"),
        output_format="mlf",
        package_path=output_file_name,
        pass_context_configs=["tir.disable_vectorize=true"],
        mod_name="classify",
    )

    # check that an MLF package was created
    assert os.path.exists(output_file_name)

    with tarfile.open(output_file_name) as mlf_package:
        # check that the C source files have been named classify_lib*.c
        c_source_files = [
            name
            for name in mlf_package.getnames()
            if re.match(r"\./codegen/host/src/classify_lib\d+\.c", name)
        ]
        assert len(c_source_files) > 0

        # check that "default" doesn't occur in any of the C source files
        # check that function names are of the form "tvmgen_classify_*"
        for file_name in c_source_files:
            with mlf_package.extractfile(file_name) as f:
                content = f.read()
                assert b"default" not in content
                assert b"tvmgen_classify_" in content

        # check that tvmgen_classify_run() function exists
        with mlf_package.extractfile("./codegen/host/src/classify_lib0.c") as f:
            content = f.read()
            assert b"tvmgen_classify_run(" in content

        # check that CMSIS-NN function names are of the form "tvmgen_classify_cmsis_nn_main_*"
        with mlf_package.extractfile("./codegen/host/src/classify_lib2.c") as f:
            content = f.read()
            assert b"tvmgen_classify_cmsis_nn_main_" in content


def test_compile_tflite_module_with_mod_name_and_ethosu(
    tmpdir_factory, tflite_mobilenet_v1_1_quant
):
    pytest.importorskip("tflite")
    pytest.importorskip("ethosu.vela")

    output_dir = tmpdir_factory.mktemp("mlf")
    tvmc_model = tvmc.load(tflite_mobilenet_v1_1_quant)
    output_file_name = f"{output_dir}/file.tar"

    tvmc.compiler.compile_model(
        tvmc_model,
        target=f"ethos-u -accelerator_config=ethos-u55-256, c -mcpu=cortex-m55",
        runtime=Runtime("crt"),
        executor=Executor("aot", {"unpacked-api": True}),
        output_format="mlf",
        package_path=output_file_name,
        pass_context_configs=["tir.disable_vectorize=true"],
        mod_name="classify",
    )

    # check that an MLF package was created
    assert os.path.exists(output_file_name)

    with tarfile.open(output_file_name) as mlf_package:
        # check that the C source files have been named classify_lib*.c
        c_source_files = [
            name
            for name in mlf_package.getnames()
            if re.match(r"\./codegen/host/src/classify_lib\d+\.c", name)
        ]
        assert len(c_source_files) > 0

        # check that "default" doesn't occur in any of the C source files
        # check that function names are of the form "tvmgen_classify_*"
        for file_name in c_source_files:
            with mlf_package.extractfile(file_name) as f:
                content = f.read()
                assert b"default" not in content
                assert b"tvmgen_classify_" in content

        # check that tvmgen_classify_run() function exists
        with mlf_package.extractfile("./codegen/host/src/classify_lib0.c") as f:
            content = f.read()
            assert b"tvmgen_classify_run(" in content

        # check that microNPU function names are of the form "tvmgen_classify_ethos_u_main_*"
        with mlf_package.extractfile("./codegen/host/src/classify_lib2.c") as f:
            content = f.read()
            assert b"tvmgen_classify_ethos_u_main_" in content


@mock.patch("tvm.relay.build")
@mock.patch("tvm.driver.tvmc.load")
@mock.patch("tvm.driver.tvmc.model.TVMCPackage.__init__", return_value=None)
def test_compile_check_workspace_pools(mock_pkg, mock_fe, mock_relay):
    mock_fe.return_value = mock.MagicMock()
    mock_relay.return_value = mock.MagicMock()
    memory_pools = WorkspaceMemoryPools(
        [WorkspacePoolInfo(pool_name="sram", targets=[Target("llvm")])]
    )
    tvmc_model = tvmc.load("no_file_needed")
    tvmc.compile(
        tvmc_model,
        target="llvm,c",
        workspace_pools=memory_pools,
    )

    assert mock_relay.call_count == 1
    assert mock_relay.call_args_list[0][1]["workspace_memory_pools"] == memory_pools


def test_compile_check_pass_instrument(keras_resnet50):
    pytest.importorskip("tensorflow")

    @tvm.instrument.pass_instrument
    class PassesCounter:
        def __init__(self):
            self.run_before_count = 0
            self.run_after_count = 0

        def run_before_pass(self, mod, info):
            self.run_before_count = self.run_before_count + 1

        def run_after_pass(self, mod, info):
            self.run_after_count = self.run_after_count + 1

    passes_counter = PassesCounter()
    tvmc_model = tvmc.load(keras_resnet50)
    tvmc.compile(tvmc_model, target="llvm", instruments=[passes_counter])
    assert passes_counter.run_after_count > 0
    assert passes_counter.run_after_count == passes_counter.run_before_count


if __name__ == "__main__":
    tvm.testing.main()
