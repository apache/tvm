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
"""RPC tracker and server running locally"""
from tvm.rpc.tracker import Tracker
from tvm.rpc.server import Server


class LocalRPC:
    """A pair of RPC tracker/server running locally

    Parameters
    ----------
    tracker_host : str
        The host URL of the tracker
    tracker_port : int
        The port of the tracker
    tracker_key: str
        The key used in the tracker to refer to a worker
    """

    tracker_host: str
    tracker_port: int
    tracker_key: str

    def __init__(
        self,
        tracker_key: str = "key",
        silent: bool = False,
        no_fork: bool = False,
    ) -> None:
        self.tracker = Tracker(
            silent=silent,
            port=9190,
            port_end=12345,
        )
        self.server = Server(
            host="0.0.0.0",
            is_proxy=False,
            tracker_addr=(self.tracker.host, self.tracker.port),
            key=tracker_key,
            silent=silent,
            no_fork=no_fork,
            port=9190,
            port_end=12345,
        )
        self.tracker_host = self.tracker.host
        self.tracker_port = self.tracker.port
        self.tracker_key = tracker_key

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        if hasattr(self, "server"):
            del self.server
        if hasattr(self, "tracker"):
            del self.tracker
