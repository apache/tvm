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

import logging
import os
import pathlib
import sys
import tarfile
import tempfile

from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError
import json

import pytest
import numpy as np

import requests

from test_utils import create_header_file
from test_utils import build_project
from test_utils import get_message

import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
from tvm import relay

from tvm.contrib.download import download_testdata
from tvm.micro.interface_api import generate_c_interface_header

import conftest


_LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _open_tflite_model():
    # Import TFLite model

    model_url = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/model/mnist_model_quant.tflite"
    model_path = download_testdata(model_url, "mnist_model_quant.tflite", module="model")

    tflite_model_buf = open(model_path, "rb").read()

    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    relay_mod, params = relay.frontend.from_tflite(tflite_model)

    return relay_mod, params

def _get_test_data(testdata_dir):

    from PIL import Image

    image_files = ["digit-2.jpg"]

    for file in image_files:
        img = Image.open(testdata_dir / file).resize((28, 28))
        img = np.asarray(img).astype("uint8")
        sample = np.reshape(img, -1)

    output_shape = (1, 10)

    return sample, output_shape


def _apply_desired_layout_simd(relay_mod):

    desired_layouts = {'qnn.conv2d': ['NHWC', 'HWOI'], 'nn.conv2d': ['NHWC', 'HWOI']}

    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])

    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)

def _apply_desired_layout_no_simd(relay_mod):

    desired_layouts = {'qnn.conv2d': ['NHWC', 'HWIO'], 'nn.conv2d': ['NHWC', 'HWIO']}

    seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)])

    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)


def _loadCMSIS(temp_dir):
    REPO_PATH = "ARM-software/CMSIS_5"
    BRANCH = "master"
    API_PATH_URL = f"https://api.github.com/repos/{REPO_PATH}/git/trees"
    RAW_PATH_URL = f"https://raw.githubusercontent.com/{REPO_PATH}/{BRANCH}"

    url = "https://api.github.com/repos/ARM-software/CMSIS_5/git/trees/master?recursive=1"
    r = requests.get(url)
    res = r.json()

    include_trees = {}

    for file in res["tree"]:
        if (file["path"] in {"CMSIS/DSP/Include", "CMSIS/DSP/Include/dsp", "CMSIS/NN/Include"}):
            include_trees.update({file["path"]: file["sha"]})

    for path, sha in include_trees.items():
        url = f"{API_PATH_URL}/{sha}"
        content = json.load(urlopen(url))
        temp_path = f"{temp_dir}"
        if path == "CMSIS/DSP/Include/dsp":
            temp_path = f"{temp_dir}/dsp"
            if not os.path.isdir(temp_path):
                os.makedirs(temp_path)
        for item in content['tree']:
            if item["type"] == "blob":
                file_name = item["path"]
                file_url = f"{RAW_PATH_URL}/{path}/{file_name}"
                print(file_name, "   ", file_url)
                try:
                    urlretrieve(file_url, f"{temp_path}/{file_name}")
                except HTTPError as e:
                    print(f"Failed to download {file_url}: {e}")


def _generate_project(temp_dir, board, west_cmd, lowered, build_config, sample, output_shape):

    with tempfile.NamedTemporaryFile() as tar_temp_file:
        with tarfile.open(tar_temp_file.name, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = os.path.join(tar_temp_dir, "include")
                os.mkdir(model_files_path)
                _loadCMSIS(model_files_path)
                tf.add(model_files_path, arcname=os.path.relpath(model_files_path, tar_temp_dir))
                header_path = generate_c_interface_header(
                    lowered.libmod_name, ["input_1"], ["output"], model_files_path
                )
                tf.add(header_path, arcname=os.path.relpath(header_path, tar_temp_dir))

            create_header_file("input_data", sample, "include", tf)
            create_header_file("output_data", np.zeros(shape=output_shape, dtype="float32"), "include", tf)

        project, _ = build_project(
            temp_dir,
            board,
            west_cmd,
            lowered,
            build_config,
            extra_files_tar=tar_temp_file.name,
        )

    return project


def _run_model(temp_dir, board, west_cmd, lowered, build_config, sample, output_shape):

    project = _generate_project(temp_dir, board, west_cmd, lowered, build_config, sample, output_shape)

    project.flash()

    with project.transport() as transport:
        timeout_read = 60
        transport.write(b"start\n", timeout_sec=5)
        result_line = get_message(transport, "#result", timeout_sec=timeout_read)

    result_line = result_line.strip("\n")
    result_line = result_line.split(":")
    result = int(result_line[1])
    time = int(result_line[2])
    logging.info(f"Result: {result}\ttime: {time} ms")

    return result, time


@tvm.testing.requires_micro
def test_armv7m_intrinsic(temp_dir, board, west_cmd, tvm_debug):
    """Testing a ARM v7m SIMD extension."""

    if board not in [
        "mps2_an521",
        "stm32f746xx_disco",
        "nucleo_f746zg",
        "nucleo_l4r5zi",
    ]:
        pytest.skip(msg="Platform does not support ARM v7m SIMD extenion.")

    model = conftest.ZEPHYR_BOARDS[board]

    build_config = {"debug": tvm_debug}

    this_dir = pathlib.Path(os.path.dirname(__file__))
    testdata_dir = this_dir.parent / "testdata" / "mnist"

    relay_mod, params = _open_tflite_model()

    sample, output_shape = _get_test_data(testdata_dir)

    relay_mod_simd = _apply_desired_layout_simd(relay_mod)
    # kernel layout "HWIO" is not supported by arm_cpu SIMD extension (see tvm\python\relay\op\strategy\arm_cpu.py)
    relay_mod_no_simd = _apply_desired_layout_no_simd(relay_mod)

    target = tvm.target.target.micro(
        model, options=["-keys=arm_cpu,cpu", "-link-params=1", "--executor=aot", "--unpacked-api=1", "--interface-api=c"]
    )

    temp_dir_simd = temp_dir / "simd"
    temp_dir_no_simd = temp_dir / "nosimd"

    os.makedirs(temp_dir_simd, exist_ok=True)
    os.makedirs(temp_dir_no_simd, exist_ok=True)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered_simd = relay.build(relay_mod_simd, target, params=params)
        lowered_no_simd = relay.build(relay_mod_no_simd, target, params=params)
        result_simd, time_simd = _run_model(temp_dir_simd, board, west_cmd, lowered_simd, build_config, sample, output_shape)
        result_no_simd, time_no_simd = _run_model(temp_dir_no_simd, board, west_cmd, lowered_no_simd, build_config, sample, output_shape)

    assert result_no_simd == result_simd
    assert time_no_simd > time_simd


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
