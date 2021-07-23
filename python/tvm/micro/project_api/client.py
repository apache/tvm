import base64
import io
import json
import logging
import os
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


class RPCError(ProjectAPIErrorBase):

    def __init__(self, request, error):
        self.request = request
        self.error = error

    def __str__(self):
        return (f"Calling project API method {self.request['method']}:" "\n"
                f"{self.error}")


class ProjectAPIClient:
    """A client for the Project API."""

    def __init__(self, read_file : typing.BinaryIO, write_file : typing.BinaryIO,
                 testonly_did_write_request : typing.Optional[typing.Callable] = None):
        self.read_file = io.TextIOWrapper(read_file, encoding='UTF-8', errors='strict')
        self.write_file = io.TextIOWrapper(write_file, encoding='UTF-8', errors='strict', write_through=True)
        self.testonly_did_write_request = testonly_did_write_request
        self.next_request_id = 1

    @property
    def is_shutdown(self):
        return self.read_file is None

    def shutdown(self):
        if self.is_shutdown:
            return

        self.read_file.close()
        self.write_file.close()

    def _request_reply(self, method, params):
        if self.is_shutdown:
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
        self.write_file.write('\n')
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
                f"saw jsonrpc={reply.get('jsonrpc')!r}")

        if reply["id"] != request["id"]:
            raise MismatchedIdError(
                f"Reply id ({reply['id']}) does not equal request id ({request['id']}")

        if "error" in reply:
            raise server.JSONRPCError.from_json(f"calling method {method}", reply["error"])
        elif "result" not in reply:
            raise MalformedReplyError(f"Expected 'result' key in server reply, got {reply!r}")

        return reply["result"]

    def server_info_query(self):
        return self._request_reply("server_info_query", {})

    def generate_project(self, model_library_format_path : str, standalone_crt_dir : str, project_dir : str, options : dict = None):
        return self._request_reply("generate_project", {"model_library_format_path": model_library_format_path,
                                                        "standalone_crt_dir": standalone_crt_dir,
                                                        "project_dir": project_dir,
                                                        "options": (options if options is not None else {})})

    def build(self, options : dict = None):
        return self._request_reply("build", {"options": (options if options is not None else {})})

    def flash(self, options : dict = None):
        return self._request_reply("flash", {"options": (options if options is not None else {})})

    def open_transport(self, options : dict = None):
        return self._request_reply("open_transport", {"options": (options if options is not None else {})})

    def close_transport(self):
        return self._request_reply("close_transport", {})

    def read_transport(self, n, timeout_sec):
        reply = self._request_reply("read_transport", {"n": n, "timeout_sec": timeout_sec})
        reply['data'] = base64.b85decode(reply['data'])
        return reply

    def write_transport(self, data, timeout_sec):
        return self._request_reply("write_transport", {"data": str(base64.b85encode(data), 'utf-8'),
                                                       "timeout_sec": timeout_sec})


# NOTE: windows support untested
SERVER_LAUNCH_SCRIPT_FILENAME = f"launch_microtvm_api_server.{'sh' if os.system != 'win32' else '.bat'}"


SERVER_PYTHON_FILENAME = "microtvm_api_server.py"


def instantiate_from_dir(project_dir : str, debug : bool = False):
    """Launch server located in project_dir, and instantiate a Project API Client connected to it."""
    args = None

    launch_script = os.path.join(project_dir, SERVER_LAUNCH_SCRIPT_FILENAME)
    if os.path.exists(launch_script):
        args = [launch_script]

    python_script = os.path.join(project_dir, SERVER_PYTHON_FILENAME)
    if os.path.exists(python_script):
        args = [sys.executable, python_script]

    if args is None:
        raise ProjectAPIServerNotFoundError(
            f"No Project API server found in project directory: {project_dir}" "\n"
            f"Tried: {SERVER_LAUNCH_SCRIPT_FILENAME}, {SERVER_PYTHON_FILENAME}")

    api_server_read_fd, tvm_write_fd = os.pipe()
    tvm_read_fd, api_server_write_fd = os.pipe()

    args.extend(["--read-fd", str(api_server_read_fd),
                 "--write-fd", str(api_server_write_fd)])
    if debug:
      args.append("--debug")

    api_server_proc = subprocess.Popen(args, bufsize=0, pass_fds=(api_server_read_fd, api_server_write_fd),
                                       cwd=project_dir)
    os.close(api_server_read_fd)
    os.close(api_server_write_fd)

    return ProjectAPIClient(os.fdopen(tvm_read_fd, 'rb', buffering=0), os.fdopen(tvm_write_fd, 'wb', buffering=0))
