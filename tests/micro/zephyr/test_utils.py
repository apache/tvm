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
from typing import Union

import numpy as np

from urllib.request import urlopen, urlretrieve
from urllib.error import HTTPError
import json
import requests

import tvm.micro
from tvm.micro import export_model_library_format


TEMPLATE_PROJECT_DIR = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "apps"
    / "microtvm"
    / "zephyr"
    / "template_project"
).resolve()

BOARDS = TEMPLATE_PROJECT_DIR / "boards.json"


def zephyr_boards() -> dict:
    """Returns a dict mapping board to target model"""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    boards_model = {board: info["model"] for board, info in board_properties.items()}
    return boards_model


ZEPHYR_BOARDS = zephyr_boards()


def qemu_boards(board: str):
    """Returns True if board is QEMU."""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    qemu_boards = [name for name, board in board_properties.items() if board["is_qemu"]]
    return board in qemu_boards


def has_fpu(board: str):
    """Returns True if board has FPU."""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    fpu_boards = [name for name, board in board_properties.items() if board["fpu"]]
    return board in fpu_boards

def extract_workspace_size_bytes(tar_path: Union[pathlib.Path, str], extract_path: Union[pathlib.Path, str]):
    tar_file = str(tar_path)
    base_path = str(extract_path)
    t = tarfile.open(tar_file)
    t.extractall(base_path)

    with open(os.path.join(base_path, "metadata.json")) as json_f:
        metadata = json.load(json_f)
        return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]

def build_project(temp_dir, zephyr_board, west_cmd, mod, build_config, extra_files_tar=None):
    project_dir = temp_dir / "project"
    
    with tempfile.TemporaryDirectory() as tar_temp_dir:
        model_tar_path = pathlib.Path(tar_temp_dir) / "model.tar" 
        export_model_library_format(mod, model_tar_path)

        workspace_size = extract_workspace_size_bytes(model_tar_path, tar_temp_dir)
        project = tvm.micro.project.generate_project_from_mlf(
            str(TEMPLATE_PROJECT_DIR),
            project_dir,
            model_tar_path,
            {
                "extra_files_tar": extra_files_tar,
                "project_type": "aot_demo",
                "west_cmd": west_cmd,
                "verbose": bool(build_config.get("debug")),
                "zephyr_board": zephyr_board,
                "compile_definitions": [
                    f"-DWORKSPACE_SIZE={workspace_size}",
                ],
            },
        )
        project.build()
    return project, project_dir


def create_header_file(tensor_name, npy_data, output_path, tar_file):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs).
    """
    header_file = io.StringIO()
    header_file.write("#include <stddef.h>\n")
    header_file.write("#include <stdint.h>\n")
    header_file.write("#include <dlpack/dlpack.h>\n")
    header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

    if npy_data.dtype == "int8":
        header_file.write(f"int8_t {tensor_name}[] =")
    elif npy_data.dtype == "int32":
        header_file.write(f"int32_t {tensor_name}[] = ")
    elif npy_data.dtype == "uint8":
        header_file.write(f"uint8_t {tensor_name}[] = ")
    elif npy_data.dtype == "float32":
        header_file.write(f"float {tensor_name}[] = ")
    else:
        raise ValueError("Data type not expected.")

    header_file.write("{")
    for i in np.ndindex(npy_data.shape):
        header_file.write(f"{npy_data[i]}, ")
    header_file.write("};\n\n")

    header_file_bytes = bytes(header_file.getvalue(), "utf-8")
    raw_path = pathlib.Path(output_path) / f"{tensor_name}.h"
    ti = tarfile.TarInfo(name=str(raw_path))
    ti.size = len(header_file_bytes)
    ti.mode = 0o644
    ti.type = tarfile.REGTYPE
    tar_file.addfile(ti, io.BytesIO(header_file_bytes))


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
