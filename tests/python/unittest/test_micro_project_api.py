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
import json
import sys
import unittest
from unittest import mock

import pytest

from tvm.micro import project_api


class BaseTestHandler(project_api.server.ProjectAPIHandler):

    DEFAULT_TEST_SERVER_INFO = project_api.server.ServerInfo(
        platform_name="platform_name",
        is_template=True,
        model_library_format_path="./model-library-format-path.sh",
        project_options=[
            project_api.server.ProjectOption(name="foo", help="Option foo"),
            project_api.server.ProjectOption(name="bar", choices=["qux"], help="Option bar"),
        ],
    )

    # Overridable by test case.
    TEST_SERVER_INFO = None

    def server_info_query(self):
        return (
            self.TEST_SERVER_INFO
            if self.TEST_SERVER_INFO is not None
            else self.DEFAULT_TEST_SERVER_INFO
        )

    def generate_project(self, model_library_format_path, crt_path, project_path, options):
        assert False, "generate_project is not implemented for this test"

    def build(self, options):
        assert False, "build is not implemented for this test"

    def flash(self, options):
        assert False, "flash is not implemented for this test"

    def open_transport(self, options):
        assert False, "open_transport is not implemented for this test"

    def close_transport(self, options):
        assert False, "open_transport is not implemented for this test"

    def read_transport(self, n, timeout_sec):
        assert False, "read_transport is not implemented for this test"

    def write_transport(self, data, timeout_sec):
        assert False, "write_transport is not implemented for this test"


class Transport:

    def readable(self):
        return True

    def writable(self):
        return True

    def seekable(self):
        return False

    closed = False

    def __init__(self):
        self.data = bytearray()
        self.rpos = 0

        self.items = []

    def read(self, size=-1):
        to_read = len(self.data) - self.rpos
        if size != -1:
            to_read = min(size, to_read)

        rpos = self.rpos
        self.rpos += to_read
        return self.data[rpos : self.rpos]

    def write(self, data):
        self.data.extend(data)


class ClientServerFixture:

    def __init__(self, handler):
        self.handler = handler
        self.client_to_server = Transport()
        self.server_to_client = Transport()

        self.server = project_api.server.ProjectAPIServer(self.client_to_server, self.server_to_client, handler)
        self.client = project_api.client.ProjectAPIClient(
            self.server_to_client, self.client_to_server, testonly_did_write_request=self._process_server_request
        )

        self.expect_failure = False

    def _process_server_request(self):
        assert self.server.serve_one_request() == (not self.expect_failure), "Server failed to process request"


def test_server_info_query():
    fixture = ClientServerFixture(BaseTestHandler())

    # Examine reply explicitly because these are the defaults for all derivative test cases.
    reply = fixture.client.server_info_query()
    assert reply["protocol_version"] == 1
    assert reply["platform_name"] == "platform_name"
    assert reply["is_template"] == True
    assert reply["model_library_format_path"] == "./model-library-format-path.sh"
    assert reply["project_options"] == [
        {"name": "foo", "choices": None, "help": "Option foo"},
        {"name": "bar", "choices": ["qux"], "help": "Option bar"},
    ]

    fixture.handler.TEST_SERVER_INFO = project_api.server.ServerInfo(
        platform_name="foo_bar",
        is_template=False,
        model_library_format_path=None,
        project_options=[])
    reply = fixture.client.server_info_query()
    expected_reply = dict(fixture.handler.TEST_SERVER_INFO._asdict())
    expected_reply["protocol_version"] = 1
    assert reply == expected_reply


def test_base_test_handler():
    """All methods should raise AssertionError on BaseTestHandler."""
    fixture = ClientServerFixture(BaseTestHandler())

    for method in dir(fixture.handler):
        if method.startswith("_") or not callable(method) or method == "server_info_query":
            continue

        with self.assertThrows(AssertionError) as exc_info:
            getattr(fixture.client, method)()

            assert (exc_info.exception) == f"{method} is not implemented for this test"


def test_build():
    with mock.patch.object(BaseTestHandler, "build", return_value=None) as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        fixture.client.build(options={"bar": "baz"})

        fixture.handler.build.assert_called_once_with(options={"bar": "baz"})


def test_flash():
    with mock.patch.object(BaseTestHandler, "flash", return_value=None) as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        fixture.client.flash(options={"bar": "baz"})
        fixture.handler.flash.assert_called_once_with(options={"bar": "baz"})


def test_open_transport():
    timeouts = project_api.server.TransportTimeouts(
        session_start_retry_timeout_sec=1.0,
        session_start_timeout_sec=2.0,
        session_established_timeout_sec=3.0,
    )

    with mock.patch.object(BaseTestHandler, "open_transport", return_value=timeouts) as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        assert fixture.client.open_transport(options={"bar": "baz"}) == {"timeouts": dict(timeouts._asdict())}
        fixture.handler.open_transport.assert_called_once_with({"bar": "baz"})


def test_close_transport():
    with mock.patch.object(BaseTestHandler, "close_transport", return_value=None) as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        fixture.client.close_transport()
        fixture.handler.close_transport.assert_called_once_with()


def test_read_transport():
    with mock.patch.object(BaseTestHandler, "read_transport", return_value=b"foo\x1b") as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        assert fixture.client.read_transport(128, timeout_sec=5.0) == {"data": b"foo\x1b"}

        fixture.handler.read_transport.assert_called_with(128, 5.0)

        fixture.handler.read_transport.side_effect = project_api.server.IoTimeoutError
        with pytest.raises(project_api.server.IoTimeoutError) as exc_info:
            fixture.client.read_transport(256, timeout_sec=10.0)

        fixture.handler.read_transport.assert_called_with(256, 10.0)

        fixture.handler.read_transport.side_effect = project_api.server.TransportClosedError
        with pytest.raises(project_api.server.TransportClosedError) as exc_info:
            fixture.client.read_transport(512, timeout_sec=15.0)

        fixture.handler.read_transport.assert_called_with(512, 15.0)

        assert fixture.handler.read_transport.call_count == 3


def test_write_transport():
    with mock.patch.object(BaseTestHandler, "write_transport", return_value=None) as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        assert fixture.client.write_transport(b"foo", timeout_sec=5.0) is None
        fixture.handler.write_transport.assert_called_with(b"foo", 5.0)

        fixture.handler.write_transport.side_effect = project_api.server.IoTimeoutError
        with pytest.raises(project_api.server.IoTimeoutError) as exc_info:
            fixture.client.write_transport(b"bar", timeout_sec=10.0)

        fixture.handler.write_transport.assert_called_with(b"bar", 10.0)

        fixture.handler.write_transport.side_effect = project_api.server.TransportClosedError
        with pytest.raises(project_api.server.TransportClosedError) as exc_info:
            fixture.client.write_transport(b"baz", timeout_sec=15.0)

        fixture.handler.write_transport.assert_called_with(b"baz", 15.0)

        assert fixture.handler.write_transport.call_count == 3


class ProjectAPITestError(Exception):
    """An error raised in test."""


def test_method_raises_error():
    with mock.patch.object(BaseTestHandler, "close_transport", side_effect=ProjectAPITestError) as patch:
        fixture = ClientServerFixture(BaseTestHandler())
        with pytest.raises(project_api.server.ServerError) as exc_info:
            fixture.client.close_transport()

        fixture.handler.close_transport.assert_called_once_with()
        assert "ProjectAPITestError" in str(exc_info.value)


def test_method_not_found():
    fixture = ClientServerFixture(BaseTestHandler())

    with pytest.raises(project_api.server.JSONRPCError) as exc_info:
        fixture.client._request_reply("invalid_method", {"bar": None})

    assert exc_info.value.code == project_api.server.ErrorCode.METHOD_NOT_FOUND


def test_extra_param():
    fixture = ClientServerFixture(BaseTestHandler())

    # test one with has_preprocssing and one without
    assert hasattr(fixture.server, '_dispatch_build') == False
    with pytest.raises(project_api.server.JSONRPCError) as exc_info:
        fixture.client._request_reply("build", {"invalid_param_name": None, "options": {}})

    assert exc_info.value.code == project_api.server.ErrorCode.INVALID_PARAMS
    assert "build: extra parameters: invalid_param_name" in str(exc_info.value)

    assert hasattr(fixture.server, '_dispatch_open_transport') == True
    with pytest.raises(project_api.server.JSONRPCError) as exc_info:
        fixture.client._request_reply("open_transport", {"invalid_param_name": None, "options": {}})

    assert exc_info.value.code == project_api.server.ErrorCode.INVALID_PARAMS
    assert "open_transport: extra parameters: invalid_param_name" in str(exc_info.value)


def test_missing_param():
    fixture = ClientServerFixture(BaseTestHandler())

    # test one with has_preprocssing and one without
    assert hasattr(fixture.server, '_dispatch_build') == False
    with pytest.raises(project_api.server.JSONRPCError) as exc_info:
        fixture.client._request_reply("build", {})

    assert exc_info.value.code == project_api.server.ErrorCode.INVALID_PARAMS
    assert "build: parameter options not given" in str(exc_info.value)

    assert hasattr(fixture.server, '_dispatch_open_transport') == True
    with pytest.raises(project_api.server.JSONRPCError) as exc_info:
        fixture.client._request_reply("open_transport", {})

    assert exc_info.value.code == project_api.server.ErrorCode.INVALID_PARAMS
    assert "open_transport: parameter options not given" in str(exc_info.value)


def test_incorrect_param_type():
    fixture = ClientServerFixture(BaseTestHandler())

    # The error message given at the JSON-RPC server level doesn't make sense when preprocessing is
    # used. Only test without preprocessing here.
    assert hasattr(fixture.server, '_dispatch_build') == False
    with pytest.raises(project_api.server.JSONRPCError) as exc_info:
        fixture.client._request_reply("build", {"options": None})

    assert exc_info.value.code == project_api.server.ErrorCode.INVALID_PARAMS
    assert "build: parameter options: want <class 'dict'>, got <class 'NoneType'>" in str(exc_info.value)


def test_invalid_request():
    fixture = ClientServerFixture(BaseTestHandler())

    # Invalid JSON does not get a reply.
    fixture.client_to_server.write(b"foobar\n")
    assert fixture.server.serve_one_request() == False
    assert fixture.server_to_client.read() == b""

    # EOF causes a clean return
    assert fixture.server.serve_one_request() == False
    assert fixture.server_to_client.read() == b""

    def _request_reply(request):
        fixture.client_to_server.write(request + b"\n")
        assert fixture.server.serve_one_request() == False
        return json.loads(fixture.server_to_client.read())

    # Parseable JSON with the wrong schema gets a reply.
    assert _request_reply(b"1") == {
        "error": {"code": project_api.server.ErrorCode.INVALID_REQUEST,
                  "data": None,
                  "message": "request: want dict; got 1"},
        "id": None,
        "jsonrpc": "2.0",
    }

    # Incorrect JSON-RPC spec version.
    assert _request_reply(b'{"jsonrpc": 1.0}') == {
        "error": {"code": project_api.server.ErrorCode.INVALID_REQUEST,
                  "data": None,
                  "message": 'request["jsonrpc"]: want "2.0"; got 1.0'},
        "id": None,
        "jsonrpc": "2.0",
    }

    # Method not a str
    assert _request_reply(b'{"jsonrpc": "2.0", "method": 123}') == {
        "error": {"code": project_api.server.ErrorCode.INVALID_REQUEST,
                  "data": None,
                  "message": 'request["method"]: want str; got 123'},
        "id": None,
        "jsonrpc": "2.0",
    }

    # Method name has invalid characters
    assert _request_reply(b'{"jsonrpc": "2.0", "method": "bar!"}') == {
        "error": {"code": project_api.server.ErrorCode.INVALID_REQUEST,
                  "data": None,
                  "message": 'request["method"]: should match regex ^[a-zA-Z0-9_]+$; got \'bar!\''},
        "id": None,
        "jsonrpc": "2.0",
    }

    # params not a dict
    assert _request_reply(b'{"jsonrpc": "2.0", "method": "bar", "params": 123}') == {
        "error": {"code": project_api.server.ErrorCode.INVALID_REQUEST,
                  "data": None,
                  "message": 'request["params"]: want dict; got <class \'int\'>'},
        "id": None,
        "jsonrpc": "2.0",
    }

    # id not valid
    assert _request_reply(b'{"jsonrpc": "2.0", "method": "bar", "params": {}, "id": {}}') == {
        "error": {"code": project_api.server.ErrorCode.INVALID_REQUEST,
                  "data": None,
                  "message": 'request["id"]: want str, number, null; got {}'},
        "id": None,
        "jsonrpc": "2.0",
    }


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
