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
"""Partitioner base class of the Universal Modular Accelerator Interface (UMA)"""

from typing import Callable, Dict, List, Tuple, Optional

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.register import register_pattern_table
from .utils import PassPhase


PatternTable = List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, Callable]]


class UMAPartitioner:
    """Partitioner base class of the Universal Modular Accelerator Interface (UMA)."""

    def __init__(self, target_name: str, merge_compiler_regions: bool = True) -> None:
        self.target_name = target_name
        self.merge_compiler_regions = merge_compiler_regions

        self._relay_passes: List[Tuple[PassPhase, tvm.transform.Pass]] = []
        self._patterns: PatternTable = []

    def add_pattern(
        self,
        name: str,
        pattern: tvm.relay.dataflow_pattern.DFPattern,
        predicate: Optional[Callable] = None,
    ) -> None:
        """Add pattern to UMA partitioner

        Parameters
        ----------
        name : str
            relay name of pattern

        pattern: tvm.relay.dataflow_pattern.DFPattern
            pattern description as DFPattern

        predicate: Optional[Callable]
            Optional predicate

        """

        name = self.target_name + "." + name
        if predicate:
            self._patterns.append((name, pattern, predicate))
        else:
            self._patterns.append((name, pattern))

    def _pattern_table(self) -> PatternTable:
        return self._patterns

    def register(self) -> None:
        """Register all relevant relay-to-relay functions."""
        register_pattern_table(self.target_name, self._pattern_table)

    def partition(
        self, mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
    ) -> tvm.IRModule:
        """Partition the relay graph in parts supported and unsupported by the
        target hardware accelerator.

        Parameters
        ----------
        mod : tvm.IRModule
            The relay module to be partitioned.

        params: Optional[Dict[str, tvm.runtime.NDArray]]

        Returns
        -------
        out : tvm.IRModule
            The partitioned relay module.

        """
        if params:
            mod["main"] = bind_params_by_name(mod["main"], params)

        pass_sequence = []
        pass_sequence.extend(
            [p[1] for p in self._relay_passes if p[0] == PassPhase.PRE_PARTITIONING]
        )
        pass_sequence.append(relay.transform.MergeComposite(self._pattern_table()))
        pass_sequence.append(relay.transform.AnnotateTarget(self.target_name))
        if self.merge_compiler_regions:
            pass_sequence.append(relay.transform.MergeCompilerRegions())
        pass_sequence.append(relay.transform.PartitionGraph())
        pass_sequence.extend(
            [p[1] for p in self._relay_passes if p[0] == PassPhase.POST_PARTITIONING_0]
        )

        sequential_passes = tvm.transform.Sequential(pass_sequence)
        mod = sequential_passes(mod)

        # Defunctionalize the partitioned functions to allow lowering
        for gvar, func in mod.functions.items():
            mod.update_func(gvar, relay.transform.Defunctionalization(func, mod))

        post_partition_passes_1 = tvm.transform.Sequential(
            [p[1] for p in self._relay_passes if p[0] == PassPhase.POST_PARTITIONING_1]
        )
        mod = post_partition_passes_1(mod)

        return mod
