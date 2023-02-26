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
import io
import os
import json
import pathlib
import tarfile
import tempfile
import logging

import numpy as np

from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError
import json
import requests

import tvm.micro
from tvm.micro import export_model_library_format
from tvm.micro.testing.utils import create_header_file
from tvm.micro.testing.utils import (
    mlf_extract_workspace_size_bytes,
    aot_transport_init_wait,
    aot_transport_find_message,
)

TEMPLATE_PROJECT_DIR = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))

_LOG = logging.getLogger(__name__)


def zephyr_boards() -> dict:
    """Returns Zephyr board properties"""
    with open(TEMPLATE_PROJECT_DIR / "boards.json") as f:
        board_properties = json.load(f)
    return board_properties


ZEPHYR_BOARDS = zephyr_boards()


def build_project(
    temp_dir, zephyr_board, mod, build_config, serial_number, simd=False, extra_files_tar=None
):
    project_dir = temp_dir / "project"

    with tempfile.TemporaryDirectory() as tar_temp_dir:
        model_tar_path = pathlib.Path(tar_temp_dir) / "model.tar"
        export_model_library_format(mod, model_tar_path)

        workspace_size = mlf_extract_workspace_size_bytes(model_tar_path)
        project_options = {
            "extra_files_tar": extra_files_tar,
            "project_type": "aot_standalone_demo",
            "verbose": bool(build_config.get("debug")),
            "board": zephyr_board,
            "serial_number": serial_number,
            "compile_definitions": [
                # TODO(mehrdadh): It fails without offset.
                f"-DWORKSPACE_SIZE={workspace_size + 128}",
            ],
        }
        if simd:
            project_options["config_main_stack_size"] = 1536

        project = tvm.micro.project.generate_project_from_mlf(
            str(TEMPLATE_PROJECT_DIR), project_dir, model_tar_path, project_options
        )
        project.build()
    return project, project_dir


# TODO move CMSIS integration to microtvm_api_server.py
# see https://discuss.tvm.apache.org/t/tvm-capturing-dependent-libraries-of-code-generated-tir-initially-for-use-in-model-library-format/11080
def loadCMSIS(temp_dir):
    REPO_PATH = "ARM-software/CMSIS_5"
    BRANCH = "master"
    API_PATH_URL = f"https://api.github.com/repos/{REPO_PATH}/git/trees"
    RAW_PATH_URL = f"https://raw.githubusercontent.com/{REPO_PATH}/{BRANCH}"

    url = "https://api.github.com/repos/ARM-software/CMSIS_5/git/trees/master?recursive=1"
    r = requests.get(url)
    res = r.json()

    include_trees = {}

    for file in res["tree"]:
        if file["path"] in {"CMSIS/DSP/Include", "CMSIS/DSP/Include/dsp", "CMSIS/NN/Include"}:
            include_trees.update({file["path"]: file["sha"]})

    for path, sha in include_trees.items():
        url = f"{API_PATH_URL}/{sha}"
        content = json.load(urlopen(url))
        temp_path = f"{temp_dir}"
        if path == "CMSIS/DSP/Include/dsp":
            temp_path = f"{temp_dir}/dsp"
            if not os.path.isdir(temp_path):
                os.makedirs(temp_path)
        for item in content["tree"]:
            if item["type"] == "blob":
                file_name = item["path"]
                file_url = f"{RAW_PATH_URL}/{path}/{file_name}"
                print(file_name, "   ", file_url)
                try:
                    urlretrieve(file_url, f"{temp_path}/{file_name}")
                except HTTPError as e:
                    print(f"Failed to download {file_url}: {e}")


def run_model(project):
    project.flash()

    with project.transport() as transport:
        aot_transport_init_wait(transport)
        transport.write(b"infer%", timeout_sec=5)
        result_line = aot_transport_find_message(transport, "result", timeout_sec=60)

    result_line = result_line.strip("\n")
    result_line = result_line.split(":")
    result = int(result_line[1])
    time = int(result_line[2])
    _LOG.info(f"Result: {result}\ttime: {time} ms")

    return result, time


def generate_project(
    temp_dir,
    board,
    lowered,
    build_config,
    sample,
    output_shape,
    output_type,
    load_cmsis,
    serial_number,
):
    with tempfile.NamedTemporaryFile() as tar_temp_file:
        with tarfile.open(tar_temp_file.name, "w:gz") as tf:
            with tempfile.TemporaryDirectory() as tar_temp_dir:
                model_files_path = pathlib.Path(tar_temp_dir) / "include"
                model_files_path.mkdir(parents=True)
                if load_cmsis:
                    loadCMSIS(model_files_path)
                    tf.add(
                        model_files_path, arcname=os.path.relpath(model_files_path, tar_temp_dir)
                    )
            create_header_file("input_data", sample, "include/tvm", tf)
            create_header_file(
                "output_data", np.zeros(shape=output_shape, dtype=output_type), "include/tvm", tf
            )

        project, project_dir = build_project(
            temp_dir,
            board,
            lowered,
            build_config,
            serial_number,
            simd=load_cmsis,
            extra_files_tar=tar_temp_file.name,
        )

    return project, project_dir
