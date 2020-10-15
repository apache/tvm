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

import subprocess
from . import base
from . import file_descriptor


class SubprocessFdTransport(file_descriptor.FdTransport):
    def timeouts(self):
        raise NotImplementedError()


class SubprocessTransport(base.Transport):
    """A Transport implementation that uses a subprocess's stdin/stdout as the channel."""

    def __init__(self, args, max_startup_latency_sec=5.0, max_latency_sec=5.0, **kwargs):
        self.max_startup_latency_sec = max_startup_latency_sec
        self.max_latency_sec = max_latency_sec
        self.args = args
        self.kwargs = kwargs
        self.popen = None
        self.child_transport = None

    def timeouts(self):
        return base.TransportTimeouts(
            session_start_retry_timeout_sec=0,
            session_start_timeout_sec=self.max_startup_latency_sec,
            session_established_timeout_sec=self.max_latency_sec,
        )

    def open(self):
        self.kwargs["stdout"] = subprocess.PIPE
        self.kwargs["stdin"] = subprocess.PIPE
        self.kwargs["bufsize"] = 0
        self.popen = subprocess.Popen(self.args, **self.kwargs)
        self.child_transport = SubprocessFdTransport(
            self.popen.stdout, self.popen.stdin, self.timeouts()
        )

    def write(self, data, timeout_sec):
        return self.child_transport.write(data, timeout_sec)

    def read(self, n, timeout_sec):
        return self.child_transport.read(n, timeout_sec)

    def close(self):
        if self.child_transport is not None:
            self.child_transport.close()

        self.popen.terminate()
