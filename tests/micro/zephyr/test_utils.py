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
import logging
import pathlib
import tarfile

import numpy as np

import tvm.micro


def build_project(temp_dir, zephyr_board, west_cmd, mod, build_config, extra_files_tar=None):
    template_project_dir = (
        pathlib.Path(__file__).parent
        / ".."
        / ".."
        / ".."
        / "apps"
        / "microtvm"
        / "zephyr"
        / "template_project"
    ).resolve()
    project_dir = temp_dir / "project"
    project = tvm.micro.generate_project(
        str(template_project_dir),
        mod,
        project_dir,
        {
            "extra_files_tar": extra_files_tar,
            "project_type": "aot_demo",
            "west_cmd": west_cmd,
            "verbose": bool(build_config.get("debug")),
            "zephyr_board": zephyr_board,
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


def _read_line(fd, timeout_sec: int):
    data = ""
    new_line = False
    while True:
        if new_line:
            break
        new_data = fd.read(1, timeout_sec=timeout_sec)
        logging.debug(f"read data: {new_data}")
        for item in new_data:
            new_c = chr(item)
            data = data + new_c
            if new_c == "\n":
                new_line = True
                break
    return data


def get_message(fd, expr: str, timeout_sec: int):
    while True:
        data = _read_line(fd, timeout_sec)
        logging.debug(f"new line: {data}")
        if expr in data:
            return data
