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

"""Defines an implementation of Transport that uses file descriptors."""

import fcntl
import os
import select
import time
from . import base


class FdConfigurationError(Exception):
    """Raised when specified file descriptors can't be placed in non-blocking mode."""


class FdTransport(base.Transport):
    """A Transport implementation that implements timeouts using non-blocking I/O."""

    @classmethod
    def _validate_configure_fd(cls, file_descriptor):
        file_descriptor = (
            file_descriptor if isinstance(file_descriptor, int) else file_descriptor.fileno()
        )
        flag = fcntl.fcntl(file_descriptor, fcntl.F_GETFL)
        if flag & os.O_NONBLOCK != 0:
            return file_descriptor

        fcntl.fcntl(file_descriptor, fcntl.F_SETFL, os.O_NONBLOCK | flag)
        new_flag = fcntl.fcntl(file_descriptor, fcntl.F_GETFL)
        if (new_flag & os.O_NONBLOCK) == 0:
            raise FdConfigurationError(
                f"Cannot set file descriptor {file_descriptor} to non-blocking"
            )
        return file_descriptor

    def __init__(self, read_fd, write_fd, timeouts):
        self.read_fd = self._validate_configure_fd(read_fd)
        self.write_fd = self._validate_configure_fd(write_fd)
        self._timeouts = timeouts

    def timeouts(self):
        return self._timeouts

    def open(self):
        pass

    def close(self):
        if self.read_fd is not None:
            os.close(self.read_fd)
            self.read_fd = None

        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

    def _await_ready(self, rlist, wlist, timeout_sec=None, end_time=None):
        if end_time is None:
            return True

        if timeout_sec is None:
            timeout_sec = max(0, end_time - time.monotonic())
        rlist, wlist, xlist = select.select(rlist, wlist, rlist + wlist, timeout_sec)
        if not rlist and not wlist and not xlist:
            raise base.IoTimeoutError()

        return True

    def read(self, n, timeout_sec):
        if self.read_fd is None:
            raise base.TransportClosedError()

        end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

        while True:
            self._await_ready([self.read_fd], [], end_time=end_time)
            try:
                to_return = os.read(self.read_fd, n)
                break
            except BlockingIOError:
                pass

        if not to_return:
            self.close()
            raise base.TransportClosedError()

        return to_return

    def write(self, data, timeout_sec):
        if self.write_fd is None:
            raise base.TransportClosedError()

        end_time = None if timeout_sec is None else time.monotonic() + timeout_sec

        data_len = len(data)
        while data:
            self._await_ready(end_time, [], [self.write_fd])
            num_written = os.write(self.write_fd, data)
            if not num_written:
                self.close()
                raise base.TransportClosedError()

            data = data[num_written:]

        return data_len
