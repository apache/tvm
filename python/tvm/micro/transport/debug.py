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

"""Defines a wrapper Transport class that launches a debugger before opening."""

from .base import Transport, TransportTimeouts


class DebugWrapperTransport(Transport):
    """A Transport wrapper class that launches a debugger before opening the transport.

    This is primiarly useful when debugging the other end of a SubprocessTransport. It allows you
    to pipe data through the GDB process to drive the subprocess with a debugger attached.
    """

    def __init__(self, debugger, transport, disable_session_start_retry=False):
        self.debugger = debugger
        self.transport = transport
        self.disable_session_start_retry = disable_session_start_retry

    def timeouts(self):
        child_timeouts = self.transport.timeouts()
        return TransportTimeouts(
            session_start_retry_timeout_sec=(
                0
                if self.disable_session_start_retry
                else child_timeouts.session_start_retry_timeout_sec
            ),
            session_start_timeout_sec=0,
            session_established_timeout_sec=0,
        )

    def open(self):
        self.debugger.start()

        try:
            self.transport.open()
        except Exception:
            self.debugger.stop()
            raise

    def write(self, data, timeout_sec):
        return self.transport.write(data, timeout_sec)

    def read(self, n, timeout_sec):
        return self.transport.read(n, timeout_sec)

    def close(self):
        self.transport.close()
        self.debugger.stop()
