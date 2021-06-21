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
"""Partition Relay IR graph for Android NNAPI based on RPC profiling."""
from .partitioner import Partitioner as _Partitioner


def rpc_partition(mod, params, tracker, options={}):  # pylint: disable=dangerous-default-value
    """Partition Relay IR graph into NNAPI convertible graph.

    Parameters
    ----------
    mod: tvm.IRModule
        The graph to be partitioned.

    trackers: tvm.rpc.TrackerSession
        The tracker client managing RPC device sessions.

    options["target"]["api_level"]: int
        The targeting API level of Android. Defaults to 29.

    options["target"]["llvm_triple"]: str
        The LLVM triple describing the target. Defaults to "aarch64-linux-android29".

    options["tvm"]["rpc"]["remote_key"]: str
        The key under which the profiling device is registered in the tracker.
        Defaults to "android".

    options["tvm"]["rpc"]["profile_run"]: int
        The remote profile cycle count for an operation. Defaults to 10.

    Returns
    -------
    mod: tvm.IRModule
        The partitioned graph.
    """
    return _Partitioner(tracker, options).partition(mod, params)
