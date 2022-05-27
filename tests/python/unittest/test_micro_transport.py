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

"""Tests for common micro transports."""

import logging
import sys
import unittest

import pytest

import tvm.testing


# Implementing as a fixture so that the tvm.micro import doesn't occur
# until fixture setup time.  This is necessary for pytest's collection
# phase to work when USE_MICRO=OFF, while still explicitly listing the
# tests as skipped.
@tvm.testing.fixture
def transport():
    import tvm.micro

    class MockTransport_Impl(tvm.micro.transport.Transport):
        def __init__(self):
            self.exc = None
            self.to_return = None

        def _raise_or_return(self):
            if self.exc is not None:
                to_raise = self.exc
                self.exc = None
                raise to_raise
            elif self.to_return is not None:
                to_return = self.to_return
                self.to_return = None
                return to_return
            else:
                assert False, "should not get here"

        def open(self):
            pass

        def close(self):
            pass

        def timeouts(self):
            raise NotImplementedError()

        def read(self, n, timeout_sec):
            return self._raise_or_return()

        def write(self, data, timeout_sec):
            return self._raise_or_return()

    return MockTransport_Impl()


@tvm.testing.fixture
def transport_logger(transport):
    logger = logging.getLogger("transport_logger_test")
    return tvm.micro.transport.TransportLogger("foo", transport, logger=logger)


@tvm.testing.fixture
def get_latest_log(caplog):
    def inner():
        return caplog.records[-1].getMessage()

    with caplog.at_level(logging.INFO, "transport_logger_test"):
        yield inner


@tvm.testing.requires_micro
def test_open(transport_logger, get_latest_log):
    transport_logger.open()
    assert get_latest_log() == "foo: opening transport"


@tvm.testing.requires_micro
def test_close(transport_logger, get_latest_log):
    transport_logger.close()
    assert get_latest_log() == "foo: closing transport"


@tvm.testing.requires_micro
def test_read_normal(transport, transport_logger, get_latest_log):
    transport.to_return = b"data"
    transport_logger.read(23, 3.0)
    assert get_latest_log() == (
        "foo: read { 3.00s}   23 B -> [  4 B]: 64 61 74 61"
        "                                      data"
    )


@tvm.testing.requires_micro
def test_read_multiline(transport, transport_logger, get_latest_log):
    transport.to_return = b"data" * 6
    transport_logger.read(23, 3.0)
    assert get_latest_log() == (
        "foo: read { 3.00s}   23 B -> [ 24 B]:\n"
        "0000  64 61 74 61 64 61 74 61 64 61 74 61 64 61 74 61  datadatadatadata\n"
        "0010  64 61 74 61 64 61 74 61                          datadata"
    )


@tvm.testing.requires_micro
def test_read_no_timeout_prints(transport, transport_logger, get_latest_log):
    transport.to_return = b"data"
    transport_logger.read(15, None)
    assert get_latest_log() == (
        "foo: read { None }   15 B -> [  4 B]: 64 61 74 61"
        "                                      data"
    )


@tvm.testing.requires_micro
def test_read_io_timeout(transport, transport_logger, get_latest_log):
    # IoTimeoutError includes the timeout value.
    transport.exc = tvm.micro.transport.IoTimeoutError()
    with pytest.raises(tvm.micro.transport.IoTimeoutError):
        transport_logger.read(23, 0.0)

    assert get_latest_log() == ("foo: read { 0.00s}   23 B -> [IoTimeoutError  0.00s]")


@tvm.testing.requires_micro
def test_read_other_exception(transport, transport_logger, get_latest_log):
    # Other exceptions are logged by name.
    transport.exc = tvm.micro.transport.TransportClosedError()
    with pytest.raises(tvm.micro.transport.TransportClosedError):
        transport_logger.read(8, 0.0)

    assert get_latest_log() == ("foo: read { 0.00s}    8 B -> [err: TransportClosedError]")


@tvm.testing.requires_micro
def test_read_keyboard_interrupt(transport, transport_logger, get_latest_log):
    # KeyboardInterrupt produces no log record.
    transport.exc = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        transport_logger.read(8, 0.0)

    with pytest.raises(IndexError):
        get_latest_log()


@tvm.testing.requires_micro
def test_write_normal(transport, transport_logger, get_latest_log):
    transport.to_return = 3
    transport_logger.write(b"data", 3.0)
    assert get_latest_log() == (
        "foo: write { 3.00s}        <- [  4 B]: 64 61 74 61"
        "                                      data"
    )


@tvm.testing.requires_micro
def test_write_multiline(transport, transport_logger, get_latest_log):
    # Normal log, multi-line data written.
    transport.to_return = 20
    transport_logger.write(b"data" * 6, 3.0)
    assert get_latest_log() == (
        "foo: write { 3.00s}        <- [ 24 B]:\n"
        "0000  64 61 74 61 64 61 74 61 64 61 74 61 64 61 74 61  datadatadatadata\n"
        "0010  64 61 74 61 64 61 74 61                          datadata"
    )


@tvm.testing.requires_micro
def test_write_no_timeout_prints(transport, transport_logger, get_latest_log):
    transport.to_return = 3
    transport_logger.write(b"data", None)
    assert get_latest_log() == (
        "foo: write { None }        <- [  4 B]: 64 61 74 61"
        "                                      data"
    )


@tvm.testing.requires_micro
def test_write_io_timeout(transport, transport_logger, get_latest_log):
    # IoTimeoutError includes the timeout value.
    transport.exc = tvm.micro.transport.IoTimeoutError()
    with pytest.raises(tvm.micro.transport.IoTimeoutError):
        transport_logger.write(b"data", 0.0)

    assert get_latest_log() == ("foo: write { 0.00s}       <- [  4 B]: [IoTimeoutError  0.00s]")


@tvm.testing.requires_micro
def test_write_other_exception(transport, transport_logger, get_latest_log):
    # Other exceptions are logged by name.
    transport.exc = tvm.micro.transport.TransportClosedError()
    with pytest.raises(tvm.micro.transport.TransportClosedError):
        transport_logger.write(b"data", 0.0)

    assert get_latest_log() == ("foo: write { 0.00s}       <- [  4 B]: [err: TransportClosedError]")


@tvm.testing.requires_micro
def test_write_keyboard_interrupt(transport, transport_logger, get_latest_log):
    # KeyboardInterrupt produces no log record.
    transport.exc = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        transport_logger.write(b"data", 0.0)

    with pytest.raises(IndexError):
        get_latest_log()


if __name__ == "__main__":
    tvm.testing.main()
