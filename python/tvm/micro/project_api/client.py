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
# pylint: disable=consider-using-with
"""
Project API client.
"""
import base64
import io
import json
import logging
import platform
import os
import pathlib
import subprocess
import sys
import typing

from . import server

_LOG = logging.getLogger(__name__)


class ProjectAPIErrorBase(Exception):
    """Base class for all Project API errors."""


class ConnectionShutdownError(ProjectAPIErrorBase):
    """Raised when a request is made but the connection has been closed."""


class MalformedReplyError(ProjectAPIErrorBase):
    """Raised when the server responds with an invalid reply."""


class MismatchedIdError(ProjectAPIErrorBase):
    """Raised when the reply ID does not match the request."""


class ProjectAPIServerNotFoundError(ProjectAPIErrorBase):
    """Raised when the Project API server can't be found in the repo."""


class UnsupportedProtocolVersionError(ProjectAPIErrorBase):
    """Raised when the protocol version returned by the API server is unsupported."""


class RPCError(ProjectAPIErrorBase):
    def __init__(self, request, error):
        ProjectAPIErrorBase.__init__()
        self.request = request
        self.error = error

    def __str__(self):
        return f"Calling project API method {self.request['method']}:" "\n" f"{self.error}"


class ProjectAPIClient:
    """A client for the Project API."""

    def __init__(
        self,
        read_file: typing.BinaryIO,
        write_file: typing.BinaryIO,
        testonly_did_write_request: typing.Optional[typing.Callable] = None,
    ):
        self.read_file = io.TextIOWrapper(read_file, encoding="UTF-8", errors="strict")
        self.write_file = io.TextIOWrapper(
            write_file, encoding="UTF-8", errors="strict", write_through=True
        )
        self.testonly_did_write_request = testonly_did_write_request
        self.next_request_id = 1

    @property
    def is_shutdown(self):
        return self.read_file.closed

    def shutdown(self):
        if self.is_shutdown:  # pylint: disable=using-constant-test
            return

        self.read_file.close()
        self.write_file.close()

    def _request_reply(self, method, params):
        if self.is_shutdown:  # pylint: disable=using-constant-test
            raise ConnectionShutdownError("connection already closed")

        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.next_request_id,
        }
        self.next_request_id += 1

        request_str = json.dumps(request)
        self.write_file.write(request_str)
        _LOG.debug("send -> %s", request_str)
        self.write_file.write("\n")
        if self.testonly_did_write_request:
            self.testonly_did_write_request()  # Allow test to assert on server processing.
        reply_line = self.read_file.readline()
        _LOG.debug("recv <- %s", reply_line)
        if not reply_line:
            self.shutdown()
            raise ConnectionShutdownError("got EOF reading reply from API server")

        reply = json.loads(reply_line)

        if reply.get("jsonrpc") != "2.0":
            raise MalformedReplyError(
                f"Server reply should include 'jsonrpc': '2.0'; "
                f"saw jsonrpc={reply.get('jsonrpc')!r}"
            )

        if reply["id"] != request["id"]:
            raise MismatchedIdError(
                f"Reply id ({reply['id']}) does not equal request id ({request['id']}"
            )

        if "error" in reply:
            raise server.JSONRPCError.from_json(f"calling method {method}", reply["error"])

        if "result" not in reply:
            raise MalformedReplyError(f"Expected 'result' key in server reply, got {reply!r}")

        return reply["result"]

    def server_info_query(self, tvm_version: str):
        reply = self._request_reply("server_info_query", {"tvm_version": tvm_version})
        if reply["protocol_version"] != server.ProjectAPIServer._PROTOCOL_VERSION:
            raise UnsupportedProtocolVersionError(
                f'microTVM API Server supports protocol version {reply["protocol_version"]}; '
                f"want {server.ProjectAPIServer._PROTOCOL_VERSION}"
            )

        return reply

    def generate_project(
        self,
        model_library_format_path: str,
        standalone_crt_dir: str,
        project_dir: str,
        options: dict = None,
    ):
        return self._request_reply(
            "generate_project",
            {
                "model_library_format_path": model_library_format_path,
                "standalone_crt_dir": standalone_crt_dir,
                "project_dir": project_dir,
                "options": (options if options is not None else {}),
            },
        )

    def build(self, options: dict = None):
        return self._request_reply("build", {"options": (options if options is not None else {})})

    def flash(self, options: dict = None):
        return self._request_reply("flash", {"options": (options if options is not None else {})})

    def open_transport(self, options: dict = None):
        return self._request_reply(
            "open_transport", {"options": (options if options is not None else {})}
        )

    def close_transport(self):
        return self._request_reply("close_transport", {})

    def read_transport(self, n, timeout_sec):
        reply = self._request_reply("read_transport", {"n": n, "timeout_sec": timeout_sec})
        reply["data"] = base64.b85decode(reply["data"])
        return reply

    def write_transport(self, data, timeout_sec):
        return self._request_reply(
            "write_transport",
            {"data": str(base64.b85encode(data), "utf-8"), "timeout_sec": timeout_sec},
        )


# NOTE: windows support untested
SERVER_LAUNCH_SCRIPT_FILENAME = (
    f"launch_microtvm_api_server.{'sh' if platform.system() != 'Windows' else '.bat'}"
)


SERVER_PYTHON_FILENAME = "microtvm_api_server.py"


def instantiate_from_dir(project_dir: typing.Union[pathlib.Path, str], debug: bool = False):
    """Launch server located in project_dir, and instantiate a Project API Client
    connected to it."""
    proc_args = None
    project_dir = pathlib.Path(project_dir)

    python_script = project_dir / SERVER_PYTHON_FILENAME
    if python_script.is_file():
        proc_args = [sys.executable, str(python_script)]

    launch_script = project_dir / SERVER_LAUNCH_SCRIPT_FILENAME
    if launch_script.is_file():
        proc_args = [str(launch_script), str(python_script)]

    if proc_args is None:
        raise ProjectAPIServerNotFoundError(
            f"No Project API server found in project directory: {project_dir}"
            "\n"
            f"Tried: {SERVER_LAUNCH_SCRIPT_FILENAME}, {SERVER_PYTHON_FILENAME}"
        )

    api_server_read_fd, tvm_write_fd = os.pipe()
    tvm_read_fd, api_server_write_fd = os.pipe()

    proc_args.extend(["--read-fd", str(api_server_read_fd), "--write-fd", str(api_server_write_fd)])
    if debug:
        proc_args.append("--debug")

    api_server_proc = subprocess.Popen(  # pylint: disable=unused-variable
        proc_args, bufsize=0, pass_fds=(api_server_read_fd, api_server_write_fd), cwd=project_dir
    )
    os.close(api_server_read_fd)
    os.close(api_server_write_fd)

    return ProjectAPIClient(
        os.fdopen(tvm_read_fd, "rb", buffering=0), os.fdopen(tvm_write_fd, "wb", buffering=0)
    )
