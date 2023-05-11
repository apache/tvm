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
import pytest
import subprocess
import shlex
import sys
import logging
import tempfile
import pathlib
import sys
import os
import shutil

import tvm
import tvm.testing
from tvm.contrib.download import download_testdata

TVMC_COMMAND = [sys.executable, "-m", "tvm.driver.tvmc"]

MODEL_URL = "https://github.com/tensorflow/tflite-micro/raw/a56087ffa2703b4d5632f024a8a4c899815c31bb/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite"
MODEL_FILE = "micro_speech.tflite"

executor = tvm.testing.parameter("aot", "graph")
use_local_path = tvm.testing.parameter(True, False)

# TODO(mehrdadh): replace this with _main from tvm.driver.tvmc.main
# Issue: https://github.com/apache/tvm/issues/9612
def _run_tvmc(cmd_args: list, *args, **kwargs):
    """Run a tvmc command and return the results"""
    cmd_args_list = TVMC_COMMAND + cmd_args
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    logging.debug("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args_list))
    return subprocess.check_call(cmd_args_list, *args, **kwargs)


def create_project_command(project_path: str, mlf_path: str, platform: str, board: str) -> list:
    """Returns create project command with tvmc micro."""
    cmd = [
        "micro",
        "create-project",
        project_path,
        mlf_path,
        platform,
        "--project-option",
        "project_type=host_driven",
        f"board={board}",
    ]

    if platform == "zephyr":
        # TODO: 4096 is driven by experiment on nucleo_l4r5zi. We should cleanup this after we have
        # better memory management.
        cmd.append("config_main_stack_size=4096")
    return cmd


def compile_command(
    model_path: str, target: tvm.target.Target, tar_path: pathlib.Path, executor: str
):
    runtime = "crt"

    cmd = [
        "compile",
        model_path,
        f"--target={target}",
        f"--runtime={runtime}",
        f"--runtime-crt-system-lib",
        str(1),
        f"--executor={executor}",
    ]

    if executor == "graph":
        cmd += [
            "--executor-graph-link-params",
            str(0),
        ]

    cmd += [
        "--output",
        str(tar_path),
        "--output-format",
        "mlf",
        "--pass-config",
        "tir.disable_vectorize=1",
    ]
    if executor == "graph":
        cmd += ["--disabled-pass=AlterOpLayout"]

    cmd_str = ""
    for item in cmd:
        cmd_str += item
        cmd_str += " "
    return cmd


def get_workspace_dir(use_local_path: bool) -> pathlib.Path:
    if use_local_path:
        out_dir_temp = pathlib.Path(os.path.abspath("./tvmc_relative_path_test"))
        if os.path.isdir(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        os.mkdir(out_dir_temp)
    else:
        out_dir_temp = tvm.contrib.utils.tempdir()

    return out_dir_temp


@tvm.testing.requires_micro
def test_tvmc_exist(platform, board):
    cmd_result = _run_tvmc(["micro", "-h"])
    assert cmd_result == 0


@tvm.testing.requires_micro
def test_tvmc_model_build_only(platform, board, executor, use_local_path):
    target = tvm.micro.testing.get_target(platform, board)
    output_dir = get_workspace_dir(use_local_path)

    model_path = download_testdata(MODEL_URL, MODEL_FILE, module="model")
    tar_path = str(output_dir / "model.tar")
    project_dir = str(output_dir / "project")

    cmd_result = _run_tvmc(compile_command(model_path, target, tar_path, executor))

    assert cmd_result == 0, "tvmc failed in step: compile"

    cmd_result = _run_tvmc(create_project_command(project_dir, tar_path, platform, board))
    assert cmd_result == 0, "tvmc micro failed in step: create-project"

    build_cmd = ["micro", "build", project_dir, platform]
    cmd_result = _run_tvmc(build_cmd)
    assert cmd_result == 0, "tvmc micro failed in step: build"
    if use_local_path:
        shutil.rmtree(output_dir)


@pytest.mark.skip("Flaky, https://github.com/apache/tvm/issues/14004")
@pytest.mark.requires_hardware
@tvm.testing.requires_micro
@pytest.mark.skip_boards(
    ["nucleo_l4r5zi", "nucleo_f746zg", "stm32f746g_disco", "nrf5340dk_nrf5340_cpuapp"]
)
def test_tvmc_model_run(platform, board, executor, use_local_path):
    target = tvm.micro.testing.get_target(platform, board)

    output_dir = get_workspace_dir(use_local_path)

    model_path = model_path = download_testdata(MODEL_URL, MODEL_FILE, module="data")
    tar_path = str(output_dir / "model.tar")
    project_dir = str(output_dir / "project")

    cmd_result = _run_tvmc(compile_command(model_path, target, tar_path, executor))
    assert cmd_result == 0, "tvmc failed in step: compile"

    cmd_result = _run_tvmc(create_project_command(project_dir, tar_path, platform, board))
    assert cmd_result == 0, "tvmc micro failed in step: create-project"

    build_cmd = ["micro", "build", project_dir, platform]
    cmd_result = _run_tvmc(build_cmd)

    assert cmd_result == 0, "tvmc micro failed in step: build"

    flash_cmd = ["micro", "flash", project_dir, platform]
    cmd_result = _run_tvmc(flash_cmd)
    assert cmd_result == 0, "tvmc micro failed in step: flash"

    run_cmd = [
        "run",
        "--device",
        "micro",
        project_dir,
    ]
    run_cmd += ["--fill-mode", "random"]
    cmd_result = _run_tvmc(run_cmd)
    assert cmd_result == 0, "tvmc micro failed in step: run"
    if use_local_path:
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    tvm.testing.main()
