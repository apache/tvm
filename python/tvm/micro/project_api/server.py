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

"""Defines a basic Project API server template.

This file is meant to be imported or copied into Project API servers, so it should not have any
imports or dependencies outside of things strictly required to run the API server.
"""

import abc
import argparse
import base64
import collections
import enum
import io
import json
import logging
import os
import pathlib
import re
import select
import sys
import time
import traceback
import typing


_LOG = logging.getLogger(__name__)


_ProjectOption = collections.namedtuple(
    "ProjectOption", ("name", "choices", "default", "type", "required", "optional", "help")
)


class ProjectOption(_ProjectOption):
    """Class used to keep the metadata associated to project options."""

    def __new__(cls, name, **kw):
        """Override __new__ to force all options except name to be specified as kwargs."""
        assert "name" not in kw
        assert (
            "required" in kw or "optional" in kw
        ), "at least one of 'required' or 'optional' must be specified."
        assert "type" in kw, "'type' field must be specified."

        kw["name"] = name
        for param in ["choices", "default", "required", "optional"]:
            kw.setdefault(param, None)

        return super().__new__(cls, **kw)

    def replace(self, attributes):
        """Update attributes associated to the project option."""
        updated_option = self
        return updated_option._replace(**attributes)


ServerInfo = collections.namedtuple(
    "ServerInfo", ("platform_name", "is_template", "model_library_format_path", "project_options")
)


# Timeouts supported by the underlying C++ MicroSession.
#
# session_start_retry_timeout_sec : float
#     Number of seconds to wait for the device to send a kSessionStartReply after sending the
#     initial session start message. After this time elapses another
#     kSessionTerminated-kSessionStartInit train is sent. 0 disables this.
# session_start_timeout_sec : float
#     Total number of seconds to wait for the session to be established. After this time, the
#     client gives up trying to establish a session and raises an exception.
# session_established_timeout_sec : float
#     Number of seconds to wait for a reply message after a session has been established. 0
#     disables this.
TransportTimeouts = collections.namedtuple(
    "TransportTimeouts",
    [
        "session_start_retry_timeout_sec",
        "session_start_timeout_sec",
        "session_established_timeout_sec",
    ],
)


class ErrorCode(enum.IntEnum):
    """Enumerates error codes which can be returned. Includes JSON-RPC standard and custom codes."""

    # Custom (in reserved error code space).
    SERVER_ERROR = -32000  # A generic error was raised while processing the request.

    # JSON-RPC standard
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class JSONRPCError(Exception):
    """An error class with properties that meet the JSON-RPC error spec."""

    def __init__(self, code, message, data, client_context=None):
        Exception.__init__(self)
        self.code = code
        self.message = message
        self.data = data
        self.client_context = client_context

    def to_json(self):
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data,
        }

    def __str__(self):
        data_str = ""
        if self.data:
            if isinstance(self.data, dict) and self.data.get("traceback"):
                data_str = f'\n{self.data["traceback"]}'
            else:
                data_str = f"\n{self.data!r}"
        return f"JSON-RPC error # {self.code}: {self.message}" + data_str

    @classmethod
    def from_json(cls, client_context, json_error):
        """Convert an encapsulated ServerError into JSON-RPC compliant format."""
        found_server_error = False
        try:
            if ErrorCode(json_error["code"]) == ErrorCode.SERVER_ERROR:
                found_server_error = True
        except ValueError:
            ServerError.from_json(client_context, json_error)

        if found_server_error:
            return ServerError.from_json(client_context, json_error)

        return cls(
            json_error["code"],
            json_error["message"],
            json_error.get("data", None),
            client_context=client_context,
        )


class ServerError(JSONRPCError):
    """Superclass for JSON-RPC errors which occur while processing valid requests."""

    @classmethod
    def from_exception(cls, exc, **kw):
        to_return = cls(**kw)
        to_return.set_traceback(traceback.TracebackException.from_exception(exc).format())
        return to_return

    def __init__(self, message=None, data=None, client_context=None):
        if self.__class__ == ServerError:
            assert message is not None, "Plain ServerError must have message="
        else:
            assert (
                message is None
            ), f"ServerError subclasses must not supply message=; got {message!r}"
            message = self.__class__.__name__

        super(ServerError, self).__init__(ErrorCode.SERVER_ERROR, message, data)
        self.client_context = client_context

    def __str__(self):
        context_str = f"{self.client_context}: " if self.client_context is not None else ""
        super_str = super(ServerError, self).__str__()
        return context_str + super_str

    def set_traceback(self, traceback):  # pylint: disable=redefined-outer-name
        """Format a traceback to be embedded in the JSON-RPC format."""

        if self.data is None:
            self.data = {}

        if "traceback" not in self.data:
            # NOTE: TVM's FFI layer reorders Python stack traces several times and strips
            # intermediary lines that start with "Traceback". This logic adds a comment to the first
            # stack frame to explicitly identify the first stack frame line that occurs on the
            # server.
            traceback_list = list(traceback)

            # The traceback list contains one entry per stack frame, and each entry contains 1-2
            # lines:
            #    File "path/to/file", line 123, in <method>:
            #      <copy of the line>
            # We want to place a comment on the first line of the outermost frame to indicate this
            # is the server-side stack frame.
            first_frame_list = traceback_list[1].split("\n")
            self.data["traceback"] = (
                traceback_list[0]
                + f"{first_frame_list[0]}  # <--- Outermost server-side stack frame\n"
                + "\n".join(first_frame_list[1:])
                + "".join(traceback_list[2:])
            )

    @classmethod
    def from_json(cls, client_context, json_error):
        assert json_error["code"] == ErrorCode.SERVER_ERROR

        for sub_cls in cls.__subclasses__():
            if sub_cls.__name__ == json_error["message"]:
                return sub_cls(
                    data=json_error.get("data"),
                    client_context=client_context,
                )

        return cls(
            json_error["message"], data=json_error.get("data"), client_context=client_context
        )


class TransportClosedError(ServerError):
    """Raised when a transport can no longer be used due to underlying I/O problems."""


class IoTimeoutError(ServerError):
    """Raised when the I/O operation could not be completed before the timeout.

    Specifically:
     - when no data could be read before the timeout
     - when some of the write data could be written before the timeout

    Note the asymmetric behavior of read() vs write(), since in one case the total length of the
    data to transfer is known.
    """


class UnsupportedTVMVersionError(ServerError):
    """Raised when the version of TVM supplied to server_info_query is unsupported."""


class ProjectAPIHandler(metaclass=abc.ABCMeta):
    """The interface class for all Project API implementations.

    Extend this class in your microtvm_api_server.py and implement each function defined here.
    """

    @abc.abstractmethod
    def server_info_query(self, tvm_version: str) -> ServerInfo:
        """Initial request issued by TVM to retrieve metadata about this API server and project.

        Should this API server not

        Parameters
        ----------
        tvm_version : str
            The value of tvm.__version__.

        Returns
        -------
        ServerInfo :
            A ServerInfo namedtuple containing the metadata needed by TVM.

        Raises
        ------
        UnsupportedTVMVersionError :
           When tvm_version indicates a known-unsupported version of TVM.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_project(
        self,
        model_library_format_path: pathlib.Path,
        standalone_crt_dir: pathlib.Path,
        project_dir: pathlib.Path,
        options: dict,
    ):
        """Generate a project from the given artifacts, copying ourselves to that project.

        Parameters
        ----------
        model_library_format_path : pathlib.Path
            Path to the Model Library Format tar archive.
        standalone_crt_dir : pathlib.Path
            Path to the root directory of the "standalone_crt" TVM build artifact. This contains the
            TVM C runtime.
        project_dir : pathlib.Path
            Path to a nonexistent directory which should be created and filled with the generated
            project.
        options : dict
            Dict mapping option name to ProjectOption.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def build(self, options: dict):
        """Build the project, enabling the flash() call to made.

        Parameters
        ----------
        options : Dict[str, ProjectOption]
            ProjectOption which may influence the build, keyed by option name.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def flash(self, options: dict):
        """Program the project onto the device.

        Parameters
        ----------
        options : Dict[str, ProjectOption]
            ProjectOption which may influence the programming process, keyed by option name.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def open_transport(self, options: dict) -> TransportTimeouts:
        """Open resources needed for the transport layer.

        This function might e.g. open files or serial ports needed in write_transport or
        read_transport.

        Calling this function enables the write_transport and read_transport calls. If the
        transport is not open, this method is a no-op.

        Parameters
        ----------
        options : Dict[str, ProjectOption]
            ProjectOption which may influence the programming process, keyed by option name.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def close_transport(self):
        """Close resources needed to operate the transport layer.

        This function might e.g. close files or serial ports needed in write_transport or
        read_transport.

        Calling this function disables the write_transport and read_transport calls. If the
        transport is not open, this method is a no-op.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    # pylint: disable=unidiomatic-typecheck
    def read_transport(self, n: int, timeout_sec: typing.Union[float, type(None)]) -> bytes:
        """Read data from the transport.

        Parameters
        ----------
        n : int
            The exact number of bytes to read from the transport.
        timeout_sec : Union[float, None]
            Number of seconds to wait for at least one byte to be written before timing out. If
            timeout_sec is 0, write should attempt to service the request in a non-blocking fashion.
            If timeout_sec is None, write should block until all `n` bytes of data can be returned.

        Returns
        -------
        bytes :
            Data read from the channel. Should be exactly `n` bytes long.

        Raises
        ------
        TransportClosedError :
            When the transport layer determines that the transport can no longer send or receive
            data due to an underlying I/O problem (i.e. file descriptor closed, cable removed, etc).

        IoTimeoutError :
            When `timeout_sec` elapses without receiving any data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def write_transport(self, data: bytes, timeout_sec: float):
        """Write data to the transport.

        This function should either write all bytes in `data` or raise an exception.

        Parameters
        ----------
        data : bytes
            The data to write over the channel.
        timeout_sec : Union[float, None]
            Number of seconds to wait for all bytes to be written before timing out. If timeout_sec
            is 0, write should attempt to service the request in a non-blocking fashion. If
            timeout_sec is None, write should block until it has written all data.

        Raises
        ------
        TransportClosedError :
            When the transport layer determines that the transport can no longer send or receive
            data due to an underlying I/O problem (i.e. file descriptor closed, cable removed, etc).

        IoTimeoutError :
            When `timeout_sec` elapses without receiving any data.
        """
        raise NotImplementedError()


class ProjectAPIServer:
    """Base class for Project API Servers.

    This API server implements communication using JSON-RPC 2.0:
        https://www.jsonrpc.org/specification

    Suggested use of this class is to import this module or copy this file into Project Generator
    implementations, then instantiate it with server.start().

    This RPC server is single-threaded, blocking, and one-request-at-a-time. Don't get anxious.
    """

    _PROTOCOL_VERSION = 1

    def __init__(
        self, read_file: typing.BinaryIO, write_file: typing.BinaryIO, handler: ProjectAPIHandler
    ):
        """Initialize a new ProjectAPIServer.

        Parameters
        ----------
        read_file : BinaryIO
            A file-like object used to read binary data from the client.
        write_file : BinaryIO
            A file-like object used to write binary data to the client.
        handler : ProjectAPIHandler
            A class which extends the abstract class ProjectAPIHandler and implements the server RPC
            functions.
        """
        self._read_file = io.TextIOWrapper(read_file, encoding="UTF-8", errors="strict")
        self._write_file = io.TextIOWrapper(
            write_file, encoding="UTF-8", errors="strict", write_through=True
        )
        self._handler = handler

    def serve_forever(self):
        """Serve requests until no more are available."""
        has_more = True
        while has_more:
            has_more = self.serve_one_request()

    def serve_one_request(self):
        """Read, process, and reply to a single request from read_file.

        When errors occur reading the request line or loading the request into JSON, they are
        propagated to the caller (the stream is then likely corrupted and no further requests
        should be served. When errors occur past this point, they are caught and send back to the
        client.

        Return
        ----------
        bool :
            True when more data could be read from read_file, False otherwise.
        """
        try:
            line = self._read_file.readline()
            _LOG.debug("read request <- %s", line)
            if not line:
                return False

            request = json.loads(line)

        except EOFError:
            _LOG.error("EOF")
            return False

        except Exception as exc:  # pylint: disable=broad-except
            _LOG.error("Caught error reading request", exc_info=1)
            return False

        did_validate = False
        try:
            self._validate_request(request)
            did_validate = True
            self._dispatch_request(request)
        except JSONRPCError as exc:
            if isinstance(exc, ServerError):
                exc.set_traceback(traceback.TracebackException.from_exception(exc).format())
            request_id = None if not did_validate else request.get("id")
            self._reply_error(request_id, exc)
            return did_validate
        except Exception as exc:  # pylint: disable=broad-except
            message = "validating request"
            if did_validate:
                message = f"calling method {request['method']}"

            exc = ServerError.from_exception(exc, message=message)
            request_id = None if not isinstance(request, dict) else request.get("id")
            self._reply_error(request_id, exc)
            return did_validate

        return True

    VALID_METHOD_RE = re.compile("^[a-zA-Z0-9_]+$")

    def _validate_request(self, request):
        if not isinstance(request, dict):
            raise JSONRPCError(
                ErrorCode.INVALID_REQUEST, f"request: want dict; got {request!r}", None
            )

        jsonrpc = request.get("jsonrpc")
        if jsonrpc != "2.0":
            raise JSONRPCError(
                ErrorCode.INVALID_REQUEST, f'request["jsonrpc"]: want "2.0"; got {jsonrpc!r}', None
            )

        method = request.get("method")
        if not isinstance(method, str):
            raise JSONRPCError(
                ErrorCode.INVALID_REQUEST, f'request["method"]: want str; got {method!r}', None
            )

        if not self.VALID_METHOD_RE.match(method):
            raise JSONRPCError(
                ErrorCode.INVALID_REQUEST,
                f'request["method"]: should match regex {self.VALID_METHOD_RE.pattern}; '
                f"got {method!r}",
                None,
            )

        params = request.get("params")
        if not isinstance(params, dict):
            raise JSONRPCError(
                ErrorCode.INVALID_REQUEST, f'request["params"]: want dict; got {type(params)}', None
            )

        request_id = request.get("id")
        # pylint: disable=unidiomatic-typecheck
        if not isinstance(request_id, (str, int, type(None))):
            raise JSONRPCError(
                ErrorCode.INVALID_REQUEST,
                f'request["id"]: want str, number, null; got {request_id!r}',
                None,
            )

    def _dispatch_request(self, request):
        method = request["method"]

        interface_method = getattr(ProjectAPIHandler, method, None)
        if interface_method is None:
            raise JSONRPCError(
                ErrorCode.METHOD_NOT_FOUND, f'{request["method"]}: no such method', None
            )

        has_preprocessing = True
        dispatch_method = getattr(self, f"_dispatch_{method}", None)
        if dispatch_method is None:
            dispatch_method = getattr(self._handler, method)
            has_preprocessing = False

        request_params = request["params"]
        params = {}

        for var_name, var_type in typing.get_type_hints(interface_method).items():
            if var_name in ("self", "return"):
                continue

            # NOTE: types can only be JSON-compatible types, so var_type is expected to be of type
            # 'type'.
            if var_name not in request_params:
                raise JSONRPCError(
                    ErrorCode.INVALID_PARAMS,
                    f'method {request["method"]}: parameter {var_name} not given',
                    None,
                )

            param = request_params[var_name]
            if not has_preprocessing and not isinstance(param, var_type):
                raise JSONRPCError(
                    ErrorCode.INVALID_PARAMS,
                    f'method {request["method"]}: parameter {var_name}: want {var_type!r}, '
                    f"got {type(param)!r}",
                    None,
                )

            params[var_name] = param

        extra_params = [p for p in request["params"] if p not in params]
        if extra_params:
            raise JSONRPCError(
                ErrorCode.INVALID_PARAMS,
                f'{request["method"]}: extra parameters: {", ".join(extra_params)}',
                None,
            )

        return_value = dispatch_method(**params)
        self._write_reply(request["id"], result=return_value)

    def _write_reply(self, request_id, result=None, error=None):
        reply_dict = {
            "jsonrpc": "2.0",
            "id": request_id,
        }

        if error is not None:
            assert (
                result is None
            ), f"Want either result= or error=, got result={result!r} and error={error!r})"
            reply_dict["error"] = error
        else:
            reply_dict["result"] = result

        reply_str = json.dumps(reply_dict)
        _LOG.debug("write reply -> %r", reply_dict)
        self._write_file.write(reply_str)
        self._write_file.write("\n")

    def _reply_error(self, request_id, exception):
        self._write_reply(request_id, error=exception.to_json())

    def _dispatch_generate_project(
        self, model_library_format_path, standalone_crt_dir, project_dir, options
    ):
        return self._handler.generate_project(
            pathlib.Path(model_library_format_path),
            pathlib.Path(standalone_crt_dir),
            pathlib.Path(project_dir),
            options,
        )

    def _dispatch_server_info_query(self, tvm_version):
        query_reply = self._handler.server_info_query(tvm_version)
        to_return = query_reply._asdict()
        if to_return["model_library_format_path"] is not None:
            to_return["model_library_format_path"] = str(to_return["model_library_format_path"])
        to_return.setdefault("protocol_version", self._PROTOCOL_VERSION)
        to_return["project_options"] = [o._asdict() for o in query_reply.project_options]
        return to_return

    def _dispatch_open_transport(self, options):
        reply = self._handler.open_transport(options)
        return {"timeouts": reply._asdict()}

    def _dispatch_read_transport(self, n, timeout_sec):
        reply_data = self._handler.read_transport(n, timeout_sec)
        return {"data": str(base64.b85encode(reply_data), "utf-8")}

    def _dispatch_write_transport(self, data, timeout_sec):
        self._handler.write_transport(base64.b85decode(data), timeout_sec)


def _await_nonblocking_ready(rlist, wlist, timeout_sec=None, end_time=None):
    if end_time is None:
        return True

    if timeout_sec is None:
        timeout_sec = max(0, end_time - time.monotonic())
    rlist, wlist, xlist = select.select(rlist, wlist, rlist + wlist, timeout_sec)
    if not rlist and not wlist and not xlist:
        raise IoTimeoutError()

    return True


def read_with_timeout(fd, n, timeout_sec):  # pylint: disable=invalid-name
    """Read data from a file descriptor, with timeout.

    This function is intended as a helper function for implementations of ProjectAPIHandler
    read_transport. Tested on Linux and OS X. Not tested on Windows.

    Parameters
    ----------
    fd : int
        File descriptor to read from. Must be opened in non-blocking mode (e.g. with O_NONBLOCK)
        if timeout_sec is not None.

    n : int
        Maximum number of bytes to read.

    timeout_sec : float or None
        If not None, maximum number of seconds to wait before raising IoTimeoutError.

    Returns
    -------
    bytes :
        If at least one byte was received before timeout_sec, returns a bytes object with length
        in [1, n]. If timeout_sec is None, returns the equivalent of os.read(fd, n).

    Raises
    ------
    IoTimeoutException :
        When timeout_sec is not None and that number of seconds elapses before any data is read.
    """
    end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

    while True:
        _await_nonblocking_ready([fd], [], end_time=end_time)
        try:
            to_return = os.read(fd, n)
            break
        except BlockingIOError:
            pass

    # When EOF is reached, close the file.
    if not to_return:
        os.close(fd)
        raise TransportClosedError()

    return to_return


def write_with_timeout(fd, data, timeout_sec):  # pylint: disable=invalid-name
    """Write data to a file descriptor, with timeout.

    This function is intended as a helper function for implementations of ProjectAPIHandler
    write_transport. Tested on Linux and OS X. Not tested on Windows.

    Parameters
    ----------
    fd : int
        File descriptor to read from. Must be opened in non-blocking mode (e.g. with O_NONBLOCK)
        if timeout_sec is not None.

    data : bytes
        Data to write.

    timeout_sec : float or None
        If not None, maximum number of seconds to wait before raising IoTimeoutError.

    Returns
    -------
    int :
        The number of bytes written to the file descriptor, if any bytes were written. A value
        in [1, len(data)]. If timeout_sec is None, returns the equivalent of os.write(fd, data).

    Raises
    ------
    IoTimeoutException :
        When timeout_sec is not None and that number of seconds elapses before any data is read.
    """
    end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

    num_written = 0
    while data:
        try:
            _await_nonblocking_ready([], [fd], end_time=end_time)
        except IoTimeoutError as exc:
            if num_written:
                return num_written

            raise exc

        num_written_this_cycle = os.write(fd, data)

        if not num_written_this_cycle:
            os.close(fd)
            raise base.TransportClosedError()

        data = data[num_written_this_cycle:]
        num_written += num_written_this_cycle

    return num_written


def default_project_options(**kw) -> typing.List[ProjectOption]:
    """Get default Project Options

    Attributes of any default option can be updated. Here is an example
    when attribute `optional` from `verbose` option needs to be updates:

        default_project_options(verbose={"optional": ["build"]})

    This will update the `optional` attribute of `verbose` ProjectOption
    to be `["build"]`.

    Returns
    -------
    options: List[ProjectOption]
        A list of default ProjectOption with modifications.
    """
    options = [
        ProjectOption(
            "verbose",
            optional=["generate_project"],
            type="bool",
            default=False,
            help="Run build with verbose output.",
        ),
        ProjectOption(
            "project_type",
            required=["generate_project"],
            type="str",
            help="Type of project to generate.",
        ),
        ProjectOption(
            "board",
            required=["generate_project"],
            type="str",
            help="Name of the board to build for.",
        ),
        ProjectOption(
            "cmsis_path",
            optional=["generate_project"],
            type="str",
            default=None,
            help="Path to the CMSIS directory.",
        ),
        ProjectOption(
            "warning_as_error",
            optional=["generate_project"],
            type="bool",
            default=False,
            help="Treat warnings as errors and raise an Exception.",
        ),
        ProjectOption(
            "compile_definitions",
            optional=["generate_project"],
            type="str",
            default=None,
            help="Extra definitions added project compile.",
        ),
        ProjectOption(
            "extra_files_tar",
            optional=["generate_project"],
            type="str",
            default=None,
            help="If given, during generate_project, "
            "uncompress the tarball at this path into the project dir.",
        ),
    ]
    for name, config in kw.items():
        option_found = False
        for ind, option in enumerate(options):
            if option.name == name:
                options[ind] = option.replace(config)
                option_found = True
                break
        if not option_found:
            raise ValueError("Option {} was not found in default ProjectOptions.".format(name))

    return options


def main(handler: ProjectAPIHandler, argv: typing.List[str] = None):
    """Start a Project API server.

    Parameters
    ----------
    argv : list[str]
        Command-line parameters to this program. If not given, sys.argv is used.
    handler : ProjectAPIHandler
        Handler class that implements the API server RPC calls.
    """
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Generic TVM Project API server entry point")
    parser.add_argument(
        "--read-fd",
        type=int,
        required=True,
        help="Numeric file descriptor where RPC requests should be read.",
    )
    parser.add_argument(
        "--write-fd",
        type=int,
        required=True,
        help="Numeric file descriptor where RPC replies should be written.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="When given, configure logging at DEBUG level."
    )
    args = parser.parse_args()

    logging.basicConfig(level="DEBUG" if args.debug else "INFO", stream=sys.stderr)

    read_file = os.fdopen(args.read_fd, "rb", buffering=0)
    write_file = os.fdopen(args.write_fd, "wb", buffering=0)

    server = ProjectAPIServer(read_file, write_file, handler)
    server.serve_forever()
