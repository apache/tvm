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

from typing import List, Tuple, Callable, Optional

import tvm
from tvm import relay, te
from tvm.relay.op.op import register_strategy
from . import _ffi_api
from .utils import PassPhase

OperatorStrategies = List[
    Tuple[
        str,
        Callable[
            [tvm.ir.Attrs, tvm.ir.Array, tvm.ir.TensorType, tvm.target.Target],
            tvm.relay.op.op.OpStrategy,
        ],
        Optional[int],
    ]
]


class UMALower:
    """Lowering base class of the Universal Modular Accelerator Interface (UMA)."""

    def __init__(self, target_name: str) -> None:
        self.target_name = target_name
        self._operator_strategies: OperatorStrategies = []
        self._tir_passes: List[Tuple[PassPhase, tvm.tir.transform.PrimFuncPass]] = []

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

        def _get_tensors(te_cached_func):
            return list(te_cached_func.inputs) + list(te_cached_func.outputs)

        lower_to_te = tvm._ffi.get_global_func("relay.backend.LowerToTE")
        te_cached_func = lower_to_te(relay_prim_func)
        x = _get_tensors(te_cached_func)
        tir_prim_func = te.create_prim_func(x)
        tir_prim_func = tir_prim_func.with_attr(
            "global_symbol", relay_prim_func.attrs["global_symbol"]
        )

        compiler_attr = relay_prim_func.attrs["Compiler"]
        target = tvm.target.Target.current()
        if target.kind.name != compiler_attr:
            target = tvm.target.Target(compiler_attr)

        tir_prim_func = tir_prim_func.with_attr("target", target)
        tir_prim_func = tir_prim_func.with_attr("relay_attrs", relay_prim_func.attrs)
        return tir_prim_func

    def _lower_stir_to_nstir(self, prim_func: tvm.tir.PrimFunc) -> tvm.tir.PrimFunc:
        """Lower a S-TIR primitive function to a NS-TIR primitive function.

        Parameters
        ----------
        prim_func : tvm.tir.PrimFunc
            The primitive function to lower.

        Returns
        -------
        out : tvm.tir.PrimFunc
            The lowered non-schedulable TensorIR primitive function.

        """
        curr_ctxt = tvm.transform.PassContext().current()
        assert "tir.add_lower_pass" not in curr_ctxt.config

        pass_map = {
            PassPhase.TIR_PHASE_0: 0,
            PassPhase.TIR_PHASE_1: 1,
            PassPhase.TIR_PHASE_2: 2,
            PassPhase.TIR_PHASE_3: 3,
        }
        lower_passes = [(pass_map[k], v) for k, v in self._tir_passes]

        with tvm.transform.PassContext(
            opt_level=curr_ctxt.opt_level,
            required_pass=curr_ctxt.required_pass,
            disabled_pass=curr_ctxt.disabled_pass,
            instruments=curr_ctxt.instruments,
            config={**dict(curr_ctxt.config), "tir.add_lower_pass": lower_passes},
        ):
            mod = tvm.lower(tvm.ir.IRModule.from_expr(prim_func))
        prim_func = mod[prim_func.attrs["global_symbol"]]
        return prim_func

    def relay_to_tir(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        """
        This is the hook for python-based lowering of a Relay module which lowers NPU
        external functions to TIR.

        Parameters
        ----------
        mod : tvm.ir.IRModule
            This is the Relay module.

        Returns
        -------
        mod : tvm.ir.IRModule
            The Relay module with scheduled NPU external functions.
        """
        mod = _ffi_api.OutlineCompilerFunctions(self.target_name)(mod)
        for gvar, func in mod.functions.items():
            if "Compiler" in func.attrs and func.attrs["Compiler"] == self.target_name:
                func = self._lower_relay_to_tir(func)
                func = self._lower_stir_to_nstir(func)
                mod.update_func(gvar, func)
        return mod

    def register(self) -> None:
        """Register all relevant relay-to-tir functions."""
        tvm._ffi.register_func(f"relay.ext.uma.{self.target_name}.relay_to_tir", self.relay_to_tir)
        for op, strategy, plevel in self._operator_strategies:
            register_strategy(op, strategy, plevel)
