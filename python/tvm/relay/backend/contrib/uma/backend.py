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
"""Backend base class of the Universal Modular Accelerator Interface (UMA)"""

import tvm

from abc import abstractmethod
from typing import List, Dict, Callable, Optional

from tvm.relay.backend.contrib.uma.api.partitioner import UMAPartitioner
from tvm.relay.backend.contrib.uma.api.lower import UMALower
from tvm.relay.backend.contrib.uma.api.codegen import UMACodegen


class UMABackend(object):
    def __init__(self, variant: str = "", merge_compiler_region: bool = True) -> None:
        self._relay_to_relay = UMAPartitioner(self.target_name, variant, merge_compiler_region)
        self._relay_to_tir = UMALower(self.target_name)
        self._tir_to_runtime = UMACodegen(self.target_name)

    @property
    @abstractmethod
    def target_name(self) -> str:
        """Name of the hardware target.

        Returns
        -------
        out : str
            The hardware target name.
        """
        ...

    ############################################################################
    # Relay to Relay function registration
    ############################################################################
    def _register_relay_pass(self, stage: int, relay_pass: tvm.transform.Pass) -> None:
        self._relay_to_relay._relay_passes.append((stage, relay_pass))

    def _register_pattern(
        self,
        name: str,
        pattern: tvm.relay.dataflow_pattern.DFPattern,
        variants: Optional[List[str]] = None,
    ) -> None:
        self._relay_to_relay._patterns.append((name, pattern, [] if variants is None else variants))

    ############################################################################
    # Relay to TIR function registration
    ############################################################################
    def _register_operator_strategy(
        self,
        op: str,
        strat: Callable[
            [tvm.ir.Attrs, tvm.ir.Array, tvm.ir.TensorType, tvm.target.Target],
            tvm.relay.op.op.OpStrategy,
        ],
        plevel: Optional[int] = 11,
    ) -> None:
        self._relay_to_tir._operator_strategies.append((op, strat, plevel))

    def _register_tir_schedule(
        self, sch_func: Callable[[tvm.tir.Schedule], tvm.tir.Schedule]
    ) -> None:
        self._relay_to_tir._tir_schedules.append(sch_func)

    def _register_tir_pass(self, stage: int, tir_pass: tvm.tir.transform.PrimFuncPass) -> None:
        self._relay_to_tir._tir_passes.append((stage, tir_pass))

    ############################################################################
    # TIR to runtime function registration
    ############################################################################
    def _register_codegen(self, fmt: str = "c", **kwargs) -> None:
        self._tir_to_runtime._register_codegen(fmt, **kwargs)

    ############################################################################
    # Backend functions
    ############################################################################
    def register(self) -> None:

        registration_func = tvm.get_global_func("relay.backend.contrib.uma.RegisterTarget")
        registration_func(self.target_name)

        self._relay_to_relay.register()
        self._relay_to_tir.register()
        self._tir_to_runtime.register()

    def partition(
        self, mod: tvm.IRModule, params: Optional[Dict[str, tvm.runtime.NDArray]] = None
    ) -> tvm.IRModule:
        return self._relay_to_relay.partition(mod, params)
