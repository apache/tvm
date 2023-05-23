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
import pathlib

import pytest
import numpy as np

import tvm
import tvm.rpc
import tvm.micro
import tvm.testing
from tvm import relay

from tvm.contrib.download import download_testdata
from tvm.relay.backend import Executor, Runtime

from . import utils


def _open_tflite_model():
    # Import TFLite model

    model_url = "https://github.com/tlc-pack/web-data/raw/b2f3c02427b67267a00fd968ba1fce28fc833028/testdata/microTVM/model/mnist_model_quant.tflite"
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

    desired_layouts = {"qnn.conv2d": ["NHWC", "HWOI"], "nn.conv2d": ["NHWC", "HWOI"]}

    seq = tvm.transform.Sequential(
        [relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)]
    )

    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)


def _apply_desired_layout_no_simd(relay_mod):

    desired_layouts = {"qnn.conv2d": ["NHWC", "HWIO"], "nn.conv2d": ["NHWC", "HWIO"]}

    seq = tvm.transform.Sequential(
        [relay.transform.RemoveUnusedFunctions(), relay.transform.ConvertLayout(desired_layouts)]
    )

    with tvm.transform.PassContext(opt_level=3):
        return seq(relay_mod)


@tvm.testing.requires_micro
@pytest.mark.skip_boards(
    ["mps2_an521", "stm32f746g_disco", "nucleo_f746zg", "nucleo_l4r5zi", "nrf5340dk_nrf5340_cpuapp"]
)
@pytest.mark.xfail(reason="due https://github.com/apache/tvm/issues/12619")
def test_armv7m_intrinsic(workspace_dir, board, microtvm_debug, serial_number):
    """Testing a ARM v7m SIMD extension."""
    build_config = {"debug": microtvm_debug}

    this_dir = pathlib.Path(os.path.dirname(__file__))
    testdata_dir = this_dir.parent / "testdata" / "mnist"

    relay_mod, params = _open_tflite_model()

    sample, output_shape = _get_test_data(testdata_dir)

    relay_mod_simd = _apply_desired_layout_simd(relay_mod)
    # kernel layout "HWIO" is not supported by arm_cpu SIMD extension (see tvm\python\relay\op\strategy\arm_cpu.py)
    relay_mod_no_simd = _apply_desired_layout_no_simd(relay_mod)

    target = tvm.target.target.micro(utils.ZEPHYR_BOARDS[board]["model"], options=["-keys=cpu"])
    target_simd = tvm.target.target.micro(
        utils.ZEPHYR_BOARDS[board]["model"], options=["-keys=arm_cpu,cpu"]
    )

    executor = Executor("aot", {"unpacked-api": True, "interface-api": "c"})
    runtime = Runtime("crt")

    workspace_dir_simd = workspace_dir / "simd"
    workspace_dir_no_simd = workspace_dir / "nosimd"

    os.makedirs(workspace_dir_simd, exist_ok=True)
    os.makedirs(workspace_dir_no_simd, exist_ok=True)

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        lowered_simd = relay.build(
            relay_mod_simd, target_simd, params=params, runtime=runtime, executor=executor
        )
        lowered_no_simd = relay.build(
            relay_mod_no_simd, target, params=params, runtime=runtime, executor=executor
        )

        simd_project, _ = utils.generate_project(
            workspace_dir_simd,
            board,
            lowered_simd,
            build_config,
            sample,
            output_shape,
            "float32",
            True,
            serial_number,
        )
        result_simd, time_simd = utils.run_model(simd_project)

        no_simd_project, _ = utils.generate_project(
            workspace_dir_no_simd,
            board,
            lowered_no_simd,
            build_config,
            sample,
            output_shape,
            "float32",
            False,
            serial_number,
        )
        result_no_simd, time_no_simd = utils.run_model(no_simd_project)

    assert result_no_simd == result_simd == 2

    # Time performance measurements on QEMU emulator are always equal to zero.
    if board not in [
        "mps2_an521",
        "mps3_an547",
    ]:
        assert time_no_simd > time_simd


if __name__ == "__main__":
    tvm.testing.main()
