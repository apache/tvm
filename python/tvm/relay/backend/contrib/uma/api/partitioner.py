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

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.register import register_pattern_table

from typing import Dict, List, Tuple, Optional


class UMAPartitioner(object):
    def __init__(
        self, target_name: str, variant: str = "", merge_compiler_regions: bool = True
    ) -> None:
        self.target_name = target_name
        self.variant = variant
        self.merge_compiler_regions = merge_compiler_regions

        self._relay_passes: List[Tuple[int, tvm.transform.Pass]] = []
        self._patterns: List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, List[str]]] = []

    def _pattern_table(self):
        return [
            (self.target_name + "." + pattern[0], pattern[1])
            for pattern in self._patterns
            if self.variant in pattern[2] or not pattern[2]
        ]

    def register(self) -> None:
        register_pattern_table(self.target_name, self._pattern_table)

    def partition(
        self, mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
    ) -> tvm.IRModule:
        """Partition the relay graph in by the NPU supported and unsupported parts.

        Parameters
        ----------
        mod : tvm.IRModule
            The relay module to be partitioned.

        Returns
        -------
        out : tvm.IRModule
            The partitioned relay module.

        """
        if params:
            mod["main"] = bind_params_by_name(mod["main"], params)

        mod = relay.transform.InferType()(mod)
        mod = tvm.transform.Sequential([p[1] for p in self._relay_passes if p[0] == 0])(mod)
        mod = relay.transform.MergeComposite(self._pattern_table())(mod)
        mod = relay.transform.AnnotateTarget(self.target_name)(mod)
        if self.merge_compiler_regions:
            mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        mod = relay.transform.InferType()(mod)
        mod = tvm.transform.Sequential([p[1] for p in self._relay_passes if p[0] == 1])(mod)
        mod = relay.transform.InferType()(mod)
        # Defunctionalize the partitioned functions to allow lowering
        for gv, func in mod.functions.items():
            mod.update_func(gv, relay.transform.Defunctionalization(func, mod))
        mod = tvm.transform.Sequential([p[1] for p in self._relay_passes if p[0] == 2])(mod)

        return mod
