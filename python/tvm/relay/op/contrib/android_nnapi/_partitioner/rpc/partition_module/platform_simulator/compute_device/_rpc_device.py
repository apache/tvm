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
"""Base class for RPC-based ComputeDevice
"""
from ._compute_device import ComputeDevice


class RPCDevice(ComputeDevice):  # pylint: disable=abstract-method
    """Base class for RPC-based ComputeDevice.

    Parameters
    ----------
    options: dict
        The partitioner options dict.

    tracker: tvm.rpc.TrackerSession
        The tracker managing RPC devices used for profiling.
    """

    def __init__(self, options, tracker):
        super().__init__()
        self._options = options
        self._tracker = tracker

        self._remote_key = options["tvm"]["rpc"]["remote_key"]
        self._remote_profile_run = options["tvm"]["rpc"]["profile_run"]

        self._target_triple = options["target"]["llvm_triple"]
        self._tvm_target = f"llvm -mtriple={ self._target_triple }"
