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

from functools import reduce
from operator import mul

from tvm import DataType, dlight, relax, topi
from tvm.contrib.thrust import can_use_thrust
from tvm.ir import Op
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.relax import PyExprMutator, expr_functor
from tvm.target import Target


def is_gpu_target(target: Target) -> bool:
    """Check if the target is a GPU target."""
    return "gpu" in target.keys


@expr_functor.mutator
class SortScanDispatcher(PyExprMutator):
    """
    Dispatcher to dispatch sort and scan.

    """

    def __init__(self, mod):
        super().__init__(mod)

    def _get_target(self, sinfo: relax.StructInfo) -> Target:
        # Get target information from TensorStructInfo
        if isinstance(sinfo, relax.TensorStructInfo):
            vdevice = sinfo.vdevice
            if vdevice is not None:
                return vdevice.target
        elif isinstance(sinfo, relax.TupleStructInfo):
            for f in sinfo.fields:
                tgt = self._get_target(f)
                if tgt != Target.current():
                    return tgt
        # Return the target in current context
        target = Target.current()
        if target is None:
            raise ValueError(
                "Target not found. Please ensure that the target is annotated within the module, "
                "or alternatively, execute this within a specified target context."
            )
        return target

    def _apply_dlight_gpu_fallback(self, target: Target, tir_call: relax.Call) -> None:
        # Apply dlight.gpu.Fallback() on GPU
        gvar = tir_call.args[0]
        assert isinstance(gvar, relax.GlobalVar)
        scan_prim_func = self.builder_.get()[gvar]
        sch = dlight.base.transform._apply_rules(
            scan_prim_func,
            target,
            [
                dlight.gpu.Fallback(),
            ],
            False,
        )
        if sch is not None:
            assert len(sch) == 1
            self.builder_.update_func(gvar, sch[0].mod["main"].with_attr("tir.is_scheduled", 1))

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        if not isinstance(call.op, Op):
            return super().visit_call_(call)

        if call.op.name == "relax.sort":
            tgt = self._get_target(call.struct_info)
            te_func = topi.sort
            kwargs = {}
            with tgt:
                if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                    te_func = topi.cuda.sort_thrust
                    kwargs["workspace"] = self.allocate_workspace(call)
                elif is_gpu_target(tgt):
                    te_func = topi.cuda.sort
            return self.builder_.call_te(
                te_func, call.args[0], call.attrs.axis, not call.attrs.descending, **kwargs
            )
        if call.op.name == "relax.argsort":
            tgt = self._get_target(call.struct_info)
            te_func = topi.argsort
            kwargs = {}
            with tgt:
                if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                    te_func = topi.cuda.argsort_thrust
                    kwargs["workspace"] = self.allocate_workspace(call)
                elif is_gpu_target(tgt):
                    te_func = topi.cuda.argsort
            return self.builder_.call_te(
                te_func,
                call.args[0],
                axis=call.attrs.axis,
                is_ascend=not call.attrs.descending,
                dtype=call.attrs.dtype,
                **kwargs,
            )
        if call.op.name == "relax.topk":
            tgt = self._get_target(call.struct_info)
            te_func = topi.topk
            kwargs = {}
            if can_use_thrust(tgt, "tvm.contrib.thrust.sort"):
                te_func = topi.cuda.topk_thrust
                kwargs["workspace"] = self.allocate_workspace(call)
            elif is_gpu_target(tgt):
                te_func = topi.cuda.topk
            tir_call = self.builder_.call_te(
                te_func,
                call.args[0],
                k=call.attrs.k,
                axis=call.attrs.axis,
                ret_type=call.attrs.ret_type,
                is_ascend=not call.attrs.largest,
                dtype=call.attrs.dtype,
                **kwargs,
            )
            if not is_gpu_target(tgt):
                return tir_call
            # apply dlight gpu fallback
            self._apply_dlight_gpu_fallback(tgt, tir_call)
            return tir_call
        if call.op.name in ("relax.cumprod", "relax.cumsum"):
            tgt = self._get_target(call.struct_info)
            axis = int(call.attrs.axis) if call.attrs.axis is not None else call.attrs.axis
            kwargs = {}
            with tgt:
                if call.op.name == "relax.cumsum":
                    te_func = topi.cuda.cumsum if is_gpu_target(tgt) else topi.cumsum
                    if can_use_thrust(tgt, "tvm.contrib.thrust.sum_scan"):
                        kwargs["workspace"] = self.allocate_workspace(call)
                elif call.op.name == "relax.cumprod":
                    te_func = topi.cuda.cumprod if is_gpu_target(tgt) else topi.cumprod
                else:
                    raise ValueError(f"Unsupported op: {call.op.name}")
                tir_call = self.builder_.call_te(
                    te_func,
                    call.args[0],
                    axis,
                    call.attrs.dtype,
                    call.attrs.exclusive,
                    **kwargs,
                )
            if not is_gpu_target(tgt):
                return tir_call
            # apply dlight gpu fallback
            self._apply_dlight_gpu_fallback(tgt, tir_call)
            return tir_call
        return super().visit_call_(call)

    def estimate_thrust_workspace_size(self, call: relax.Call) -> int:
        """
        Estimate the workspace size for thrust sort/argsort/topk/cumsum
        """
        input_shape = call.args[0].struct_info.shape
        input_byte_per_elem = DataType(call.args[0].struct_info.dtype).bits // 8
        input_size = reduce(mul, input_shape, 1) * input_byte_per_elem
        # Most GPU algorithms take O(n) space or less, we choose 2N + 4MB as a safe estimation
        return 2 * input_size + 4 * 1024 * 1024

    def allocate_workspace(self, call: relax.Call) -> relax.Var:
        """
        Allocate workspace for thrust sort/argsort/topk.
        """
        workspace_size = self.estimate_thrust_workspace_size(call)
        alloc = relax.op.builtin.alloc_tensor(
            relax.ShapeExpr((workspace_size,)), "uint8", runtime_device_index=0
        )
        return self.builder_.emit(alloc)


@module_pass(opt_level=0, name="DispatchSortScan")
class DispatchSortScan:
    """
    Pass to dispatch scan and sort operators to platform dependent implementation.
    """

    def transform_module(self, mod: IRModule, ctx: PassContext) -> IRModule:
        sort_scan_dispater = SortScanDispatcher(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                func = sort_scan_dispater.visit_expr(func)
                sort_scan_dispater.builder_.update_func(gv, func)
        return sort_scan_dispater.builder_.finalize()
