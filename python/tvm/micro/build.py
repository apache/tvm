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

from .._ffi import libinfo
from .. import rpc as _rpc


_LOG = logging.getLogger(__name__)


STANDALONE_CRT_DIR = None


class CrtNotFoundError(Exception):
    """Raised when the standalone CRT dirtree cannot be found."""


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


class AutoTvmModuleLoader:
    """MicroTVM AutoTVM Module Loader

    Parameters
    ----------
    template_project_dir : str
        project template path

    project_options : dict
        project generation option
    """

    def __init__(self, template_project_dir: str, project_options: dict = None):
        self._project_options = project_options

        if isinstance(template_project_dir, pathlib.Path):
            self._template_project_dir = str(template_project_dir)
        elif not isinstance(template_project_dir, str):
            raise TypeError(f"Incorrect type {type(template_project_dir)}.")

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
        try:
            remote.get_function("tvm.micro.destroy_micro_session")()
        except tvm.error.TVMError as exception:
            _LOG.warning("Error destroying remote session: %s", str(exception), exc_info=1)


def autotvm_build_func():
    """A dummy build function which causes autotvm to use a different export format."""


# A sentinel value for the output format.
autotvm_build_func.output_format = ".model-library-format"
