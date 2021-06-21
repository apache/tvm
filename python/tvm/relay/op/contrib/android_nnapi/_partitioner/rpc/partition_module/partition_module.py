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
"""Partition graphs in a given Relay module into those for tvm/android_nnapi compilers."""
import tvm
from .collect_branching_nodes import CollectBranchingNodes
from .platform_simulator import PlatformSimulator
from .export_decision_marker import ExportDecisionMarker
from .annotate_for_relay_compiler import AnnotateForRelayCompiler


class PartitionModule:
    """Partition graphs in a given Relay module into those for tvm/android_nnapi compilers.

    Parameters
    ----------
    tracker: tvm.rpc.TrackerSession
        The tracker client managing RPC device sessions.

    options: dict
        The partitioner option dict.
    """

    def __init__(self, tracker, options):
        self._tracker = tracker
        self._options = options

    def __call__(self, mod):
        """Partition graphs in a given Relay module into those for tvm/android_nnapi compilers.

        Parameters
        ----------
        mod: tvm.IRModule
            The partition target module.

        Returns
        -------
        mod: tvm.IRModule
            The partitioned module.
        """
        assert isinstance(mod, tvm.IRModule)
        gvs = mod.get_global_vars()
        for gvar in gvs:
            func = mod[gvar]
            branching_nodes = CollectBranchingNodes().collect(func)
            psim = PlatformSimulator(self._tracker, self._options, branching_nodes)
            psim.calculate_cost(func)
            edm = ExportDecisionMarker(self._options, psim.node_transfers)
            edm.mark(func)
            assert all(
                [
                    edm.node_is_exported(n, "tvm") == edm.EXPORT_RESULT["YES"]
                    for n in branching_nodes
                ]
            )
            func = AnnotateForRelayCompiler(self._options, edm).annotate(func)
            mod[gvar] = func
        mod = tvm.relay.transform.PartitionGraph()(mod)
        return mod
