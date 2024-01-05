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
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Dispatch sort and scan operators to platform dependent implementation."""

from tvm import topi, dlight
from tvm.ir import Op
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.target import Target
from tvm.contrib.thrust import can_use_thrust
from tvm.relax import Expr, GlobalVar, Function, Call, PyExprMutator, expr_functor, TensorStructInfo


@expr_functor.mutator
class SortScanDispatcher(PyExprMutator):
    """
    Dispatcher to dispatch sort and scan.

    """

    def __init__(self, mod):
        super().__init__(mod)

    def _get_target(self, expr: Expr) -> Target:
        sinfo = expr.struct_info
        # Get target information from TensorStructInfo
        if isinstance(sinfo, TensorStructInfo):
            vdevice = sinfo.vdevice
            if vdevice is not None:
                return vdevice.target
        # Return the target in current context
        target = Target.current()
        if target is None:
            raise ValueError(
                "Target not found. Please ensure that the target is annotated within the module, "
                "or alternatively, execute this within a specified target context."
            )
        return target

    def visit_call_(self, call: Call) -> Expr:
        if not isinstance(call.op, Op):
            return super().visit_call_(call)

        if call.op.name == "relax.sort":
            tgt = self._get_target(call)
            with tgt:
                if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                    return self.builder_.call_te(
                        topi.cuda.sort_thrust,
                        call.args[0],
                        call.attrs.axis,
                        not call.attrs.descending,
                    )
                return self.builder_.call_te(
                    topi.cuda.sort if tgt.kind.name == "cuda" else topi.sort,
                    call.args[0],
                    call.attrs.axis,
                    not call.attrs.descending,
                )

        if call.op.name in ("relax.cumprod", "relax.cumsum"):
            tgt = self._get_target(call)
            axis = int(call.attrs.axis) if call.attrs.axis is not None else call.attrs.axis
            te_func = topi.cuda.cumsum if tgt.kind.name == "cuda" else topi.cumsum
            if call.op.name == "relax.cumprod":
                te_func = topi.cuda.cumprod if tgt.kind.name == "cuda" else topi.cumprod
            with tgt:
                tir_call = self.builder_.call_te(
                    te_func,
                    call.args[0],
                    axis,
                    call.attrs.dtype,
                    call.attrs.exclusive,
                )

            if tgt.kind.name != "cuda":
                return tir_call

            # Apply dlight.gpu.Fallback() for scan on GPU
            gvar = tir_call.args[0]
            assert isinstance(gvar, GlobalVar)
            scan_prim_func = self.builder_.get()[gvar]
            sch = dlight.base.transform.apply_rules(
                scan_prim_func,
                tgt,
                [
                    dlight.gpu.Fallback(),
                ],
                False,
            )
            if sch is not None:
                assert len(sch) == 1
                self.builder_.update_func(gvar, sch[0].mod["main"].with_attr("tir.is_scheduled", 1))
            return tir_call
        return super().visit_call_(call)


@module_pass(opt_level=0, name="DispatchSortScan")
class DispatchSortScan:
    """
    Pass to dispatch scan and sort operators to platform dependent implementation.
    """

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        sort_scan_dispater = SortScanDispatcher(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, Function):
                func = sort_scan_dispater.visit_expr(func)
                sort_scan_dispater.builder_.update_func(gv, func)
        return sort_scan_dispater.builder_.finalize()
