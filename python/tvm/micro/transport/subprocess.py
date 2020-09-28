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

from .base import Transport


class SubprocessTransport(Transport):
    """A Transport implementation that uses a subprocess's stdin/stdout as the channel."""

    def __init__(self, args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.popen = None

    def open(self):
        self.kwargs["stdout"] = subprocess.PIPE
        self.kwargs["stdin"] = subprocess.PIPE
        self.kwargs["bufsize"] = 0
        self.popen = subprocess.Popen(self.args, **self.kwargs)
        self.stdin = self.popen.stdin
        self.stdout = self.popen.stdout

    def write(self, data):
        to_return = self.stdin.write(data)
        self.stdin.flush()

        return to_return

    def read(self, n):
        return self.stdout.read(n)

    def close(self):
        self.stdin.close()
        self.stdout.close()
        self.popen.terminate()
