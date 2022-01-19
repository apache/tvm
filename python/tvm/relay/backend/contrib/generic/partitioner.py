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
"""Generic relay partitioner for NPUs"""

from tvm.relay.build_module import bind_params_by_name
from typing import Dict, List, Tuple, Optional
import tvm
from tvm import relay
from abc import abstractmethod

from tvm.relay.op.contrib.register import register_pattern_table


class GenericPartitioner(object):
    def __init__(self, variant: str = "") -> None:
        self._variant = variant

        self._relay_passes: List[Tuple[int, tvm.transform.Pass]] = []
        self._patterns: List[Tuple[str, tvm.relay.dataflow_pattern.DFPattern, List[str]]] = []

        self._register_relay_passes()
        self._register_patterns()
        register_pattern_table(self.target_name, self._pattern_table)

    @property
    @abstractmethod
    def target_name(self) -> str:
        """Name of the hardware target.

        Returns
        -------
        out : str
            The hardware target name.
        """

    @abstractmethod
    def _register_relay_passes(self) -> None:
        """Register a set of relay passes which are applied during lowering.

        Example
        -------
        Here is an example of how two passes can be registered.

        .. code-block:: python

            def _register_relay_passes(self):
                self._register_relay_pass(pass_0)
                self._register_relay_pass(pass_1)

        Use `pass` if no relay pass should be registerd.

        .. code-block:: python

            def _register_relay_passes(self):
                pass

        """

    @abstractmethod
    def _register_patterns(self) -> None:
        """Register a set of relay graph patterns which used for partitioning.

        Example
        -------
        Here is an example of how two patterns can be registered.

        .. code-block:: python

            def _register_patterns(self):
                self._register_pattern(pattern_0)
                self._register_pattern(pattern_1)
        """

    def _register_relay_pass(self, stage: int, relay_pass: tvm.transform.Pass) -> None:
        self._relay_passes.append((stage, relay_pass))

    def _register_pattern(
        self,
        name: str,
        pattern: tvm.relay.dataflow_pattern.DFPattern,
        variants: Optional[List[str]] = None,
    ):
        self._patterns.append((name, pattern, [] if variants is None else variants))

    def _pattern_table(self):
        return [
            (self.target_name + "." + pattern[0], pattern[1])
            for pattern in self._patterns
            if self._variant in pattern[2] or not pattern[2]
        ]

    def __call__(self, mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]]) -> tvm.IRModule:
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

        pattern = relay.op.contrib.get_pattern_table(self.target_name)
        mod = relay.transform.InferType()(mod)
        mod = tvm.transform.Sequential([p[1] for p in self._relay_passes if p[0] == 0])(mod)
        mod = relay.transform.MergeComposite(pattern)(mod)
        mod = relay.transform.AnnotateTarget(self.target_name)(mod)
        mod = relay.transform.MergeCompilerRegions()(mod)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.PartitionGraph()(mod)
        mod = relay.transform.InferType()(mod)
        mod = tvm.transform.Sequential([p[1] for p in self._relay_passes if p[0] == 1])(mod)
        mod = relay.transform.InferType()(mod)
        # Defunctionalize the partitioned functions to allow lowering
        for gv, func in mod.functions.items():
            mod.update_func(
                gv, relay.transform.Defunctionalization(func, mod)
            )
        mod = tvm.transform.Sequential([p[1] for p in self._relay_passes if p[0] == 2])(mod)

        return mod
