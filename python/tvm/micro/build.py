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
import pathlib
import contextlib
import enum

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

    if platform == MicroTVMTemplateProject.CRT.value:
        return os.path.join(get_standalone_crt_dir(), "template", "host")

    microtvm_template_projects = None
    for path in libinfo.find_lib_path():
        template_path = os.path.join(os.path.dirname(path), "microtvm_template_projects")
        if os.path.isdir(template_path):
            microtvm_template_projects = template_path
            break
    else:
        raise MicroTVMTemplateProjectNotFoundError()

    return os.path.join(microtvm_template_projects, platform)


class AutoTvmModuleLoader:
    """MicroTVM AutoTVM Module Loader

    Parameters
    ----------
    template_project_dir : Union[pathlib.Path, str]
        project template path

    project_options : dict
        project generation option
    """

    def __init__(
        self, template_project_dir: Union[pathlib.Path, str], project_options: dict = None
    ):
        self._project_options = project_options

        if isinstance(template_project_dir, (pathlib.Path, str)):
            self._template_project_dir = str(template_project_dir)
        elif not isinstance(template_project_dir, str):
            raise TypeError(f"Incorrect type {type(template_project_dir)}.")

    @contextlib.contextmanager
    def __call__(self, remote_kw, build_result):
        with open(build_result.filename, "rb") as build_file:
            build_result_bin = build_file.read()

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
            ],
        )
        system_lib = remote.get_function("runtime.SystemLib")()
        yield remote, system_lib


def autotvm_build_func():
    """A dummy build function which causes autotvm to use a different export format."""


# A sentinel value for the output format.
autotvm_build_func.output_format = ".model-library-format"
