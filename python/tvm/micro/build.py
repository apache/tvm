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

"""Defines top-level glue functions for building microTVM artifacts."""

import json
import logging
import os
import contextlib
import enum
from pathlib import Path
import shutil

from typing import Union
from .._ffi import libinfo
from .. import rpc as _rpc


_LOG = logging.getLogger(__name__)


STANDALONE_CRT_DIR = None


class MicroTVMTemplateProject(enum.Enum):
    ZEPHYR = "zephyr"
    ARDUINO = "arduino"
    CRT = "crt"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class CrtNotFoundError(Exception):
    """Raised when the standalone CRT dirtree cannot be found."""


class MicroTVMTemplateProjectNotFoundError(Exception):
    """Raised when the microTVM template project dirtree cannot be found."""


def get_standalone_crt_dir() -> str:
    """Find the standalone_crt directory.

    Though the C runtime source lives in the tvm tree, it is intended to be distributed with any
    binary build of TVM. This source tree is intended to be integrated into user projects to run
    models targeted with --runtime=c.

    Returns
    -------
    str :
        The path to the standalone_crt
    """
    global STANDALONE_CRT_DIR
    if STANDALONE_CRT_DIR is None:
        for path in libinfo.find_lib_path():
            crt_path = os.path.join(os.path.dirname(path), "standalone_crt")
            if os.path.isdir(crt_path):
                STANDALONE_CRT_DIR = crt_path
                break

        else:
            raise CrtNotFoundError()

    return STANDALONE_CRT_DIR


def get_microtvm_template_projects(platform: str) -> str:
    """Find microTVM template project directory for specific platform.

    Parameters
    ----------
    platform : str
        Platform type which should be defined in MicroTVMTemplateProject.

    Returns
    -------
    str :
        Path to template project directory for platform.
    """
    if platform not in MicroTVMTemplateProject.list():
        raise ValueError(f"platform {platform} is not supported.")

    microtvm_template_projects = None
    for path in libinfo.find_lib_path():
        template_path = os.path.join(os.path.dirname(path), "microtvm_template_projects")
        if os.path.isdir(template_path):
            microtvm_template_projects = template_path
            break
    else:
        raise MicroTVMTemplateProjectNotFoundError()

    return os.path.join(microtvm_template_projects, platform)


def copy_crt_config_header(platform: str, output_path: Path):
    """Copy crt_config header file for a platform to destinatin.

    Parameters
    ----------
    platform : str
        Platform type which should be defined in MicroTVMTemplateProject.

    output_path: Path
        Output path for crt_config header file.
    """
    crt_config_path = Path(get_microtvm_template_projects(platform)) / "crt_config" / "crt_config.h"
    shutil.copy(crt_config_path, output_path)


class AutoTvmModuleLoader:
    """MicroTVM AutoTVM Module Loader

    Parameters
    ----------
    template_project_dir : Union[os.PathLike, str]
        project template path

    project_options : dict
        project generation option

    project_dir: str
        if use_existing is False: The path to save the generated microTVM Project.
        if use_existing is True: The path to a generated microTVM Project for debugging.

    use_existing: bool
        skips the project generation and opens transport to the project at the project_dir address.
    """

    def __init__(
        self,
        template_project_dir: Union[os.PathLike, str],
        project_options: dict = None,
        project_dir: Union[os.PathLike, str] = None,
        use_existing: bool = False,
    ):
        self._project_options = project_options
        self._use_existing = use_existing

        if isinstance(template_project_dir, (os.PathLike, str)):
            self._template_project_dir = str(template_project_dir)
        elif not isinstance(template_project_dir, str):
            raise TypeError(f"Incorrect type {type(template_project_dir)}.")

        if isinstance(project_dir, (os.PathLike, str)):
            self._project_dir = str(project_dir)
        else:
            self._project_dir = None

    @contextlib.contextmanager
    def __call__(self, remote_kw, build_result):
        with open(build_result.filename, "rb") as build_file:
            build_result_bin = build_file.read()

        # In case we are tuning on multiple physical boards (with Meta-schedule), the tracker
        # device_key is the serial_number of the board that wil be used in generating micro session.
        # For CRT projects, and in cases that the serial number is not provided
        # (including tuning with AutoTVM), the serial number field doesn't change.
        if "board" in self._project_options and "$local$device" not in remote_kw["device_key"]:
            self._project_options["serial_number"] = remote_kw["device_key"]

        tracker = _rpc.connect_tracker(remote_kw["host"], remote_kw["port"])
        remote = tracker.request(
            remote_kw["device_key"],
            priority=remote_kw["priority"],
            session_timeout=remote_kw["timeout"],
            session_constructor_args=[
                "tvm.micro.compile_and_create_micro_session",
                build_result_bin,
                self._template_project_dir,
                json.dumps(self._project_options),
                self._project_dir,
                self._use_existing,
            ],
        )
        system_lib = remote.get_function("runtime.SystemLib")()
        yield remote, system_lib


def autotvm_build_func():
    """A dummy build function which causes autotvm to use a different export format."""


# A sentinel value for the output format.
autotvm_build_func.output_format = ".model-library-format"
