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
"""Lowering base class of the Universal Modular Accelerator Interface (UMA)"""

import tvm
from tvm import relay, te, tir

from typing import List, Tuple, Callable, Optional

from .utils import extract_constants


class UMALower(object):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name

        self._operator_strategies: List[
            Tuple[
                str,
                Callable[
                    [tvm.ir.Attrs, tvm.ir.Array, tvm.ir.TensorType, tvm.target.Target],
                    tvm.relay.op.op.OpStrategy,
                ],
                Optional[int],
            ]
        ] = []
        self._tir_schedules: List[Callable[[tvm.tir.Schedule], tvm.tir.Schedule]] = []
        self._tir_passes: List[Tuple[int, tvm.tir.transform.PrimFuncPass]] = []

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
        relay_prim_func, constants = extract_constants(relay_prim_func)
        f = tvm._ffi.get_global_func("relay.backend.LowerToTE")
        te_cached_func = f(relay_prim_func)
        tir_prim_func = te.create_prim_func_from_outputs(te_cached_func.outputs)
        tir_prim_func = tir_prim_func.with_attr(
            "global_symbol", relay_prim_func.attrs["global_symbol"]
        )
        tir_prim_func = tir_prim_func.with_attr("constants", constants)
        tir_prim_func = tir_prim_func.with_attr("relay_attrs", relay_prim_func.attrs)
        return tir_prim_func

    def _lower_stir_to_nstir(self, schedule: tvm.tir.Schedule) -> tvm.tir.PrimFunc:
        """Lower a S-TIR schedule to a NS-TIR primitive function.

        Parameters
        ----------
        schedule : tvm.tir.Schedule
            The schedule to lower.

        Returns
        -------
        out : tvm.tir.PrimFunc
            The lowered non-schedulable TensorIR primitive function.

        """
        with tvm.transform.PassContext(
            config={"tir.add_lower_pass": self._tir_passes}, opt_level=0
        ):
            mod = tvm.lower(schedule.mod)
        prim_func = mod["main"]
        return prim_func

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
        for sch_func in self._tir_schedules:
            schedule = sch_func(schedule)
        prim_func = self._lower_stir_to_nstir(schedule)
        return prim_func

    def register(self) -> None:
        tvm._ffi.register_func(
            "relay.ext.uma.relay_to_tir_func_{}".format(self.target_name), self.relay_to_tir_func
        )
