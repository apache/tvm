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

"""Defines an implementation of Transport that uses subprocesses."""

import logging
import time
from . import base


_LOG = logging.getLogger(__name__)


class WakeupTransport(base.Transport):
    """A Transport implementation that waits for a "wakeup sequence" from the remote end."""

    def __init__(self, child_transport, wakeup_sequence):
        self.child_transport = child_transport
        self.wakeup_sequence = bytes(wakeup_sequence)
        self.wakeup_sequence_buffer = bytearray()
        self.line_start_index = 0
        self.found_wakeup_sequence = False

    def open(self):
        return self.child_transport.open()

    def close(self):
        return self.child_transport.close()

    def timeouts(self):
        return self.child_transport.timeouts()

    def _await_wakeup(self, end_time):
        def _time_remaining():
            if end_time is None:
                return None
            return max(0, end_time - time.monotonic())

        if not self.found_wakeup_sequence:
            while self.wakeup_sequence not in self.wakeup_sequence_buffer:
                x = self.child_transport.read(1, _time_remaining())
                self.wakeup_sequence_buffer.extend(x)
                if x[0] in (b"\n", b"\xff"):
                    _LOG.debug("%s", self.wakeup_sequence_buffer[self.line_start_index : -1])
                    self.line_start_index = len(self.wakeup_sequence_buffer)

            _LOG.info("remote side woke up!")
            self.found_wakeup_sequence = True
            time.sleep(0.2)

        return _time_remaining()

    def read(self, n, timeout_sec):
        if not self.found_wakeup_sequence:
            end_time = None if timeout_sec is None else time.monotonic() + timeout_sec
            timeout_sec = self._await_wakeup(end_time)

        return self.child_transport.read(n, timeout_sec)

    def write(self, data, timeout_sec):
        if not self.found_wakeup_sequence:
            end_time = None if timeout_sec is None else time.monotonic() + timeout_sec
            timeout_sec = self._await_wakeup(end_time)

        return self.child_transport.write(data, timeout_sec)
