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
"""Generic codegen for NPUs"""

from abc import abstractmethod
from typing import Dict, List
import tvm
from tvm import relay, te, tir


class GenericCodegen(object):
    def _lower_relay_to_tir(self, relay_prim_func: relay.Function) -> tvm.tir.PrimFunc:
        """Lower a Relay primitive function to a S-TIR primitive function.

        Parameters
        ----------
        prim_func : tvm.relay.Function
            The Relay function to lower.

        Returns
        -------
        out : tvm.tir.PrimFunc
            The lowered schedulable TensorIR primitive function.

        """
        f = tvm._ffi.get_global_func("relay.backend.LowerToTE")
        te_cached_func = f(relay_prim_func)
        tir_prim_func = te.create_prim_func_from_outputs(te_cached_func.outputs)
        tir_prim_func = tir_prim_func.with_attr(
            "global_symbol", relay_prim_func.attrs["global_symbol"]
        )
        return tir_prim_func

    def _lower_stir_to_nstir(self, schedule: tvm.tir.Schedule) -> tvm.tir.PrimFunc:
        mod = schedule.mod
        mod = self.apply_passes_before(mod)
        mod = tir.transform.StorageFlatten(64, False)(mod)
        mod = tir.transform.LowerInitBlock()(mod)
        mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        mod = tir.transform.ConvertBlocksToOpaque()(mod)
        mod = tir.transform.CompactBufferAllocation()(mod)
        mod = tir.transform.LowerMatchBuffer()(mod)
        mod = tir.transform.FlattenBuffer()(mod)
        mod = tir.transform.Simplify()(mod)
        mod = self.apply_passes_after(mod)
        prim_func = mod["main"]
        return prim_func

    @abstractmethod
    def apply_schedules(self, schedule: tvm.tir.Schedule) -> tvm.tir.Schedule:
        pass

    def apply_passes_before(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        return mod

    def apply_passes_after(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        return mod

    def relay_to_tir_func(self, ext_func: relay.Function) -> tvm.tir.PrimFunc:
        """
        This is the hook for python-based lowering of relay function
        that gets offloaded to the target NPU.

        Parameters
        ----------
        ext_func : relay.Function
            The partitioned relay function.

        Returns
        -------
        prim_func : tir.PrimFunc
            The scheduled PrimFunc.
        """
        prim_func = self._lower_relay_to_tir(ext_func)
        schedule = tir.Schedule(prim_func)
        schedule = self.apply_schedules(schedule)
        prim_func = self._lower_stir_to_nstir(schedule)
        return prim_func
