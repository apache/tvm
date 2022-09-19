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

MODEL_URL = "https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite"
MODEL_FILE = "micro_speech.tflite"

# TODO(mehrdadh): replace this with _main from tvm.driver.tvmc.main
# Issue: https://github.com/apache/tvm/issues/9612
def _run_tvmc(cmd_args: list, *args, **kwargs):
    """Run a tvmc command and return the results"""
    cmd_args_list = TVMC_COMMAND + cmd_args
    cwd_str = "" if "cwd" not in kwargs else f" (in cwd: {kwargs['cwd']})"
    logging.debug("run%s: %s", cwd_str, " ".join(shlex.quote(a) for a in cmd_args_list))
    return subprocess.check_call(cmd_args_list, *args, **kwargs)


@tvm.testing.requires_micro
def test_tvmc_exist(platform, board):
    cmd_result = _run_tvmc(["micro", "-h"])
    assert cmd_result == 0


@tvm.testing.requires_micro
@pytest.mark.parametrize(
    "output_dir,",
    [pathlib.Path("./tvmc_relative_path_test"), pathlib.Path(tempfile.mkdtemp())],
)
def test_tvmc_model_build_only(platform, board, output_dir):
    target = tvm.micro.testing.get_target(platform, board)

    if not os.path.isabs(output_dir):
        out_dir_temp = os.path.abspath(output_dir)
        if os.path.isdir(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        os.mkdir(out_dir_temp)

    model_path = download_testdata(MODEL_URL, MODEL_FILE, module="data")
    tar_path = str(output_dir / "model.tar")
    project_dir = str(output_dir / "project")

    runtime = "crt"
    executor = "graph"

    cmd_result = _run_tvmc(
        [
            "compile",
            model_path,
            f"--target={target}",
            f"--runtime={runtime}",
            f"--runtime-crt-system-lib",
            str(1),
            f"--executor={executor}",
            "--executor-graph-link-params",
            str(0),
            "--output",
            tar_path,
            "--output-format",
            "mlf",
            "--pass-config",
            "tir.disable_vectorize=1",
            "--disabled-pass=AlterOpLayout",
        ]
    )
    assert cmd_result == 0, "tvmc failed in step: compile"

    create_project_cmd = [
        "micro",
        "create-project",
        project_dir,
        tar_path,
        platform,
        "--project-option",
        "project_type=host_driven",
        f"board={board}",
    ]

    cmd_result = _run_tvmc(create_project_cmd)
    assert cmd_result == 0, "tvmc micro failed in step: create-project"

    build_cmd = ["micro", "build", project_dir, platform]
    cmd_result = _run_tvmc(build_cmd)
    assert cmd_result == 0, "tvmc micro failed in step: build"
    shutil.rmtree(output_dir)


@pytest.mark.requires_hardware
@tvm.testing.requires_micro
@pytest.mark.parametrize(
    "output_dir,",
    [pathlib.Path("./tvmc_relative_path_test"), pathlib.Path(tempfile.mkdtemp())],
)
def test_tvmc_model_run(platform, board, output_dir):
    target = tvm.micro.testing.get_target(platform, board)

    if not os.path.isabs(output_dir):
        out_dir_temp = os.path.abspath(output_dir)
        if os.path.isdir(out_dir_temp):
            shutil.rmtree(out_dir_temp)
        os.mkdir(out_dir_temp)

    model_path = model_path = download_testdata(MODEL_URL, MODEL_FILE, module="data")
    tar_path = str(output_dir / "model.tar")
    project_dir = str(output_dir / "project")

    runtime = "crt"
    executor = "graph"

    cmd_result = _run_tvmc(
        [
            "compile",
            model_path,
            f"--target={target}",
            f"--runtime={runtime}",
            f"--runtime-crt-system-lib",
            str(1),
            f"--executor={executor}",
            "--executor-graph-link-params",
            str(0),
            "--output",
            tar_path,
            "--output-format",
            "mlf",
            "--pass-config",
            "tir.disable_vectorize=1",
            "--disabled-pass=AlterOpLayout",
        ]
    )
    assert cmd_result == 0, "tvmc failed in step: compile"

    create_project_cmd = [
        "micro",
        "create-project",
        project_dir,
        tar_path,
        platform,
        "--project-option",
        "project_type=host_driven",
        f"board={board}",
    ]

    cmd_result = _run_tvmc(create_project_cmd)
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
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    tvm.testing.main()
