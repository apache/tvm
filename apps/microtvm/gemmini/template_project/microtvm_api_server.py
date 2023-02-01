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
"""
MicroTVM API Server for Gemmini baremetal tests on the Spike simulator
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

import atexit
import collections
import functools
import json
import logging
import os
import os.path
import pathlib
import re
import shlex
import shutil
import shlex, subprocess
import sys
import tarfile
import tempfile
import time
from string import Template
import re
from distutils.dir_util import copy_tree
import subprocess
import serial

# import serial.tools.list_ports
from tvm.micro.project_api import server

from subprocess import PIPE

_LOG = logging.getLogger(__name__)

MODEL_LIBRARY_FORMAT_RELPATH = pathlib.Path("src") / "model" / "model.tar"
API_SERVER_DIR = pathlib.Path(os.path.dirname(__file__) or os.path.getcwd())
BUILD_DIR = API_SERVER_DIR / "build"
MODEL_LIBRARY_FORMAT_PATH = API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH

IS_TEMPLATE = not (API_SERVER_DIR / MODEL_LIBRARY_FORMAT_RELPATH).exists()

PROJECT_TYPES = [
    "dense_example",
    "conv2d_example",
    "dwconv2d_example",
    "add_example",
    "maxpool2d_example",
    "mobilenet_example",
]

PROJECT_OPTIONS = [
    server.ProjectOption(
        "project_type",
        required=["generate_project"],
        choices=tuple(PROJECT_TYPES),
        type="str",
        help="Type of project to generate.",
    )
]


class Handler(server.ProjectAPIHandler):
    def __init__(self):
        super(Handler, self).__init__()
        self._proc = None
        self._port = None
        self._transport = None
        self._project_dir = None
        self._qemu_instance = None

    def server_info_query(self, tvm_version):
        return server.ServerInfo(
            platform_name="gemmini",
            is_template=IS_TEMPLATE,
            model_library_format_path="" if IS_TEMPLATE else MODEL_LIBRARY_FORMAT_PATH,
            project_options=PROJECT_OPTIONS,
        )

    def _copy_project_files(self, api_server_dir, project_dir, project_type):
        """Copies the files for project_type into project_dir.

        Notes
        -----
        template_dir is NOT a project type, and that directory is never copied
        in this function. template_dir only holds this file and its unit tests,
        so this file is copied separately in generate_project.

        """
        for item in (API_SERVER_DIR / "src" / project_type).iterdir():
            dest = project_dir / "src" / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        shutil.copy2(project_dir / "src" / "Makefile.template", project_dir / "src" / "Makefile")

        test_name = project_type.replace("_example", "")
        new_line = f"tests = {test_name}\n"
        with open(project_dir / "src" / "Makefile", "r") as original:
            data = original.read()
        with open(project_dir / "src" / "Makefile", "w") as modified:
            modified.write(new_line + data)

    CRT_COPY_ITEMS = ("include", "src")

    def _copy_standalone_crt(self, source_dir, standalone_crt_dir):
        output_crt_dir = source_dir / "standalone_crt"
        for item in self.CRT_COPY_ITEMS:
            src_path = os.path.join(standalone_crt_dir, item)
            dst_path = output_crt_dir / item
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)

    def _disassemble_mlf(self, mlf_tar_path, source_dir):
        with tempfile.TemporaryDirectory() as mlf_unpacking_dir_str:
            mlf_unpacking_dir = pathlib.Path(mlf_unpacking_dir_str)
            with tarfile.open(mlf_tar_path, "r:") as tar:
                tar.extractall(mlf_unpacking_dir)

            model_dir = source_dir / "model"
            model_dir.mkdir()

            # Copy C files from model. The filesnames and quantity
            # depend on the target string, so we just copy all c files
            source_dir = mlf_unpacking_dir / "codegen" / "host" / "src"
            for file in source_dir.rglob(f"*.c"):
                shutil.copy(file, model_dir)

            source_dir = mlf_unpacking_dir / "codegen" / "host" / "include"
            for file in source_dir.rglob(f"*.h"):
                shutil.copy(file, model_dir)

            # Return metadata.json for use in templating
            with open(os.path.join(mlf_unpacking_dir, "metadata.json")) as f:
                metadata = json.load(f)
        return metadata

    CPP_FILE_EXTENSION_SYNONYMS = ("cc", "cxx")

    def _convert_includes(self, project_dir, source_dir):
        """Changes all #include statements in project_dir to be relevant to their
        containing file's location.

        """
        for ext in ("c", "h", "cpp"):
            for filename in source_dir.rglob(f"*.{ext}"):
                with filename.open("rb") as src_file:
                    lines = src_file.readlines()
                    with filename.open("wb") as dst_file:
                        for i, line in enumerate(lines):
                            line_str = str(line, "utf-8")
                            # Check if line has an include
                            result = re.search(r"#include\s*[<\"]([^>]*)[>\"]", line_str)
                            if not result:
                                dst_file.write(line)
                            else:
                                new_include = self._find_modified_include_path(
                                    project_dir, filename, result.groups()[0]
                                )
                                updated_line = f'#include "{new_include}"\n'
                                dst_file.write(updated_line.encode("utf-8"))

    # Most of the files we used to be able to point to directly are under "src/standalone_crt/include/".
    # Howver, crt_config.h lives under "src/standalone_crt/crt_config/", and more exceptions might
    # be added in the future.
    POSSIBLE_BASE_PATHS = ["src/standalone_crt/include/", "src/standalone_crt/crt_config/"]

    def _find_modified_include_path(self, project_dir, file_path, include_path):
        """Takes a single #include path, and returns the location it should point to.

        Examples
        --------
        >>> _find_modified_include_path(
        ...     "/path/to/project/dir"
        ...     "/path/to/project/dir/src/standalone_crt/src/runtime/crt/common/ndarray.c"
        ...     "tvm/runtime/crt/platform.h"
        ... )
        "../../../../../../src/standalone_crt/include/tvm/runtime/crt/platform.h"

        """
        if include_path.endswith(".inc"):
            include_path = re.sub(r"\.[a-z]+$", ".h", include_path)

        # Change includes referencing .cc and .cxx files to point to the renamed .cpp file
        if include_path.endswith(self.CPP_FILE_EXTENSION_SYNONYMS):
            include_path = re.sub(r"\.[a-z]+$", ".cpp", include_path)

        # If the include already works, don't modify it
        if (file_path.parents[0] / include_path).exists():
            return include_path

        relative_path = file_path.relative_to(project_dir)
        up_dirs_path = "../" * str(relative_path).count("/")

        for base_path in self.POSSIBLE_BASE_PATHS:
            full_potential_path = project_dir / base_path / include_path
            if full_potential_path.exists():
                return up_dirs_path + base_path + include_path

        # If we can't find the file, just leave it untouched
        # It's probably a standard C/C++ header
        return include_path

    def _copy_debug_data_files(self, project_dir):
        if os.path.isdir(str(project_dir / ".." / "include")):
            copy_tree(str(project_dir / ".." / "include"), str(project_dir / "src" / "model"))

    def generate_project(self, model_library_format_path, standalone_crt_dir, project_dir, options):

        # Reference key directories with pathlib
        project_dir = pathlib.Path(project_dir)
        project_dir.mkdir()
        source_dir = project_dir / "src"
        source_dir.mkdir()

        # Copies files from the template folder to project_dir
        shutil.copy2(API_SERVER_DIR / "microtvm_api_server.py", project_dir)
        self._copy_project_files(API_SERVER_DIR, project_dir, options["project_type"])

        # Copy standalone_crt into src folder
        self._copy_standalone_crt(source_dir, standalone_crt_dir)

        # Populate crt-config.h
        crt_config_dir = project_dir / "src" / "standalone_crt" / "crt_config"
        crt_config_dir.mkdir()
        shutil.copy2(
            API_SERVER_DIR / "crt_config" / "crt_config.h", crt_config_dir / "crt_config.h"
        )

        # Unpack the MLF and copy the relevant files
        metadata = self._disassemble_mlf(model_library_format_path, source_dir)
        shutil.copy2(model_library_format_path, project_dir / MODEL_LIBRARY_FORMAT_RELPATH)

        self._copy_debug_data_files(project_dir)

        # Recursively change includes
        self._convert_includes(project_dir, source_dir)

    def build(self, options):
        subprocess.call(
            "cd src && ./build.sh",
            shell=True,
        )

    def flash(self, options):
        test_name = options["project_type"].split("_")[0]
        subprocess.call(
            "cd src/build && spike --extension=gemmini %s" % (test_name + "-baremetal",),
            shell=True,
        )

    def open_transport(self, options):
        pass

    def close_transport(self):
        pass

    def read_transport(self, n, timeout_sec):
        pass

    def write_transport(self, data, timeout_sec):
        pass


if __name__ == "__main__":
    server.main(Handler())
