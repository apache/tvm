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


@tvm.testing.requires_micro
class TransportLoggerTests(unittest.TestCase):
    import tvm.micro

    class TestTransport(tvm.micro.transport.Transport):
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

    def test_transport_logger(self):
        """Tests the TransportLogger class."""

        logger = logging.getLogger("transport_logger_test")
        with self.assertLogs(logger) as test_log:
            transport = self.TestTransport()
            transport_logger = tvm.micro.transport.TransportLogger("foo", transport, logger=logger)

            transport_logger.open()
            assert test_log.records[-1].getMessage() == "foo: opening transport"

            ########### read() tests ##########

            # Normal log, single-line data returned.
            transport.to_return = b"data"
            transport_logger.read(23, 3.0)
            assert test_log.records[-1].getMessage() == (
                "foo: read { 3.00s}   23 B -> [  4 B]: 64 61 74 61"
                "                                      data"
            )

            # Normal log, multi-line data returned.
            transport.to_return = b"data" * 6
            transport_logger.read(23, 3.0)
            assert test_log.records[-1].getMessage() == (
                "foo: read { 3.00s}   23 B -> [ 24 B]:\n"
                "0000  64 61 74 61 64 61 74 61 64 61 74 61 64 61 74 61  datadatadatadata\n"
                "0010  64 61 74 61 64 61 74 61                          datadata"
            )

            # Lack of timeout prints.
            transport.to_return = b"data"
            transport_logger.read(15, None)
            assert test_log.records[-1].getMessage() == (
                "foo: read { None }   15 B -> [  4 B]: 64 61 74 61"
                "                                      data"
            )

            # IoTimeoutError includes the timeout value.
            transport.exc = tvm.micro.transport.IoTimeoutError()
            with self.assertRaises(tvm.micro.transport.IoTimeoutError):
                transport_logger.read(23, 0.0)

            assert test_log.records[-1].getMessage() == (
                "foo: read { 0.00s}   23 B -> [IoTimeoutError  0.00s]"
            )

            # Other exceptions are logged by name.
            transport.exc = tvm.micro.transport.TransportClosedError()
            with self.assertRaises(tvm.micro.transport.TransportClosedError):
                transport_logger.read(8, 0.0)

            assert test_log.records[-1].getMessage() == (
                "foo: read { 0.00s}    8 B -> [err: TransportClosedError]"
            )

            # KeyboardInterrupt produces no log record.
            before_len = len(test_log.records)
            transport.exc = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                transport_logger.read(8, 0.0)

            assert len(test_log.records) == before_len

            ########### write() tests ##########

            # Normal log, single-line data written.
            transport.to_return = 3
            transport_logger.write(b"data", 3.0)
            assert test_log.records[-1].getMessage() == (
                "foo: write { 3.00s}        <- [  4 B]: 64 61 74 61"
                "                                      data"
            )

            # Normal log, multi-line data written.
            transport.to_return = 20
            transport_logger.write(b"data" * 6, 3.0)
            assert test_log.records[-1].getMessage() == (
                "foo: write { 3.00s}        <- [ 24 B]:\n"
                "0000  64 61 74 61 64 61 74 61 64 61 74 61 64 61 74 61  datadatadatadata\n"
                "0010  64 61 74 61 64 61 74 61                          datadata"
            )

            # Lack of timeout prints.
            transport.to_return = 3
            transport_logger.write(b"data", None)
            assert test_log.records[-1].getMessage() == (
                "foo: write { None }        <- [  4 B]: 64 61 74 61"
                "                                      data"
            )

            # IoTimeoutError includes the timeout value.
            transport.exc = tvm.micro.transport.IoTimeoutError()
            with self.assertRaises(tvm.micro.transport.IoTimeoutError):
                transport_logger.write(b"data", 0.0)

            assert test_log.records[-1].getMessage() == (
                "foo: write { 0.00s}       <- [  4 B]: [IoTimeoutError  0.00s]"
            )

            # Other exceptions are logged by name.
            transport.exc = tvm.micro.transport.TransportClosedError()
            with self.assertRaises(tvm.micro.transport.TransportClosedError):
                transport_logger.write(b"data", 0.0)

            assert test_log.records[-1].getMessage() == (
                "foo: write { 0.00s}       <- [  4 B]: [err: TransportClosedError]"
            )

            # KeyboardInterrupt produces no log record.
            before_len = len(test_log.records)
            transport.exc = KeyboardInterrupt()
            with self.assertRaises(KeyboardInterrupt):
                transport_logger.write(b"data", 0.0)

            assert len(test_log.records) == before_len

            transport_logger.close()
            assert test_log.records[-1].getMessage() == "foo: closing transport"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
