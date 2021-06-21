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
"""Partition a Relay IR graph into subgraphs compiled by
TVM/Android NNAPI compilers using RPC profiling."""
import copy
import tvm.relay.transform
from .. import _base
from .partition_module import PartitionModule


class Partitioner:
    """Partition a Relay IR graph into subgraphs compiled by
    TVM/Android NNAPI compilers using RPC profiling.

    Parameters
    ----------
    tracker: tvm.rpc.TrackerSession
        The tracker client managing RPC device sessions.

    options: dict
        The partitioner option dict.
    """

    DEFAULT_OPTIONS = {
        "target": {
            "api_level": 29,
            "llvm_triple": "aarch64-linux-android29",
        },
        "tvm": {
            "external_compiler": "android_nnapi",
            "rpc": {
                "profile_run": 10,
                "remote_key": "android",
            },
        },
    }

    def __init__(self, tracker, options):
        self._tracker = tracker
        self._options = self._expand_options(options)

    def partition(self, mod, params):
        """Partition a Relay IR graph

        Parameters
        ----------
        mod: tvm.IRModule
            The graph to be partitioned

        params: dict of str to tvm.runtime.NDArray
            The input parameters to the graph

        Returns
        -------
        mod: tvm.IRModule
            The partitioned graph

        params: dict of str to tvm.runtime.NDArray
            The transformed input parameters to the graph

        """
        assert isinstance(mod, tvm.IRModule)
        mod = _base.pre_partition_transform(mod)
        mod = PartitionModule(self._tracker, self._options)(mod)
        mod, params = _base.post_partition_transform(
            mod,
            params,
            android_nnapi_level=self._options["target"]["api_level"],
            external_compiler=self._options["tvm"]["external_compiler"],
        )
        return mod, params

    @staticmethod
    def _expand_options(options):
        ret = options

        def _recursive_merge(cur_opts, def_opts):
            for k, v in def_opts.items():
                if k in cur_opts:
                    if isinstance(v, dict):
                        assert isinstance(cur_opts[k], dict)
                        _recursive_merge(cur_opts[k], v)
                    else:
                        assert isinstance(cur_opts[k], (float, int, str))
                else:
                    cur_opts[k] = copy.deepcopy(v)

        _recursive_merge(ret, Partitioner.DEFAULT_OPTIONS)

        return ret
