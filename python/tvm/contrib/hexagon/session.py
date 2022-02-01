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

"""Defines a Session class for Hexagon devices."""

import os
from tvm import rpc as _rpc


class Session:
    """Hexagon Device Session

    Parameters
    ----------
    remote_kw : dict
        Remote configs for RPC tracker.

    session_name : str
        Hexagon RPC session name.
    """

    def __init__(
        self,
        remote_kw: dict,
        session_name: str = "hexagon-rpc",
        remote_stack_size_bytes: int = 128 * 1024,
    ):
        self._session_name = session_name
        self._remote_stack_size_bytes = remote_stack_size_bytes
        self._remote_kw = remote_kw
        self._rpc = None
        self.device = None

    def __enter__(self):
        if self.device:
            # Already initialized
            return self

        tracker = _rpc.connect_tracker(self._remote_kw["host"], self._remote_kw["port"])
        try:
            self._rpc = tracker.request(
                self._remote_kw["key"],
                priority=self._remote_kw["priority"],
                session_timeout=self._remote_kw["timeout"],
                session_constructor_args=[
                    "tvm.contrib.hexagon.create_hexagon_session",
                    self._session_name,
                    self._remote_stack_size_bytes,
                    os.environ.get("HEXAGON_SIM_ARGS", ""),
                ],
            )
            self.device = self._rpc.hexagon(0)
            return self

        except RuntimeError as exception:
            raise exception

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def load_module(self, path: str):
        assert isinstance(path, str), f"Invalid path type, {type(path)} != str"
        return self._rpc.get_function("tvm.hexagon.load_module")(path)
