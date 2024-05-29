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
from typing import Dict

from tvm import DataType, dlight, relax, topi
from tvm.contrib.thrust import can_use_thrust
from tvm.ir import GlobalVar, Op
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm.relax import expr_functor
from tvm.target import Target

from .utils import BackendDispatcher


@expr_functor.mutator
class SortScanDispatcher(BackendDispatcher):
    """Dispatcher to dispatch sort and scan."""

    calls_to_update: Dict[GlobalVar, Target]

    def __init__(self, mod):
        super().__init__(mod)
        self.calls_to_update = {}

    def apply_dlight_gpu_fallback(
        self,
    ) -> None:
        """Apply DLight rules for all the calls that need to be updated."""
        for gvar, target in self.calls_to_update.items():
            func = self.builder_.get()[gvar]
            sch = dlight.base.transform._apply_rules(
                func,
                target,
                rules=[dlight.gpu.Fallback()],
                tunable=False,
            )
            if sch is not None:
                assert len(sch) == 1
                self.builder_.update_func(gvar, sch[0].mod["main"].with_attr("tir.is_scheduled", 1))

    def _append_calls_to_update(self, tir_call: relax.Call, target: Target) -> None:
        gvar = tir_call.args[0]
        assert isinstance(gvar, GlobalVar)
        existing_tgt = self.calls_to_update.get(gvar, None)
        if existing_tgt is not None and existing_tgt != target:
            raise ValueError(
                f"Multiple targets detected for function {gvar}. "
                f"Existing target: {existing_tgt}, new target: {target}"
            )
        self.calls_to_update[gvar] = target

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
                elif self.is_gpu_target(tgt):
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
                elif self.is_gpu_target(tgt):
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
            elif self.is_gpu_target(tgt):
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
            self._append_calls_to_update(tir_call, tgt)
            return tir_call
        if call.op.name in ("relax.cumprod", "relax.cumsum"):
            tgt = self._get_target(call.struct_info)
            axis = int(call.attrs.axis) if call.attrs.axis is not None else call.attrs.axis
            shape = call.struct_info.shape
            # TODO(tvm-team): Support fully dynamic case with `shape=None`
            if shape is None:
                raise ValueError("non-symbolic shape is not supported for now")
            kwargs = {}
            if (
                shape is not None
                and (axis == -1 or axis == len(shape) - 1)
                and self.is_gpu_target(tgt)
                and not can_use_thrust(tgt, "tvm.contrib.thrust.sum_scan")
                and call.op.name == "relax.cumsum"
                and call.attrs.exclusive == 0
            ):
                from tvm.relax.backend_tir import (  # pylint: disable=import-outside-toplevel
                    gpu_2d_continuous_cumsum,
                )

                dim = 1
                for i in range(len(shape) - 1):
                    dim *= shape[i]
                in_dtype = call.args[0].struct_info.dtype
                out_dtype = call.attrs.dtype
                out_dtype = out_dtype or in_dtype
                cumsum_2d_shape = relax.ShapeExpr([dim, shape[-1]])
                reshape = relax.call_pure_packed(
                    "vm.builtin.reshape",
                    call.args[0],
                    cumsum_2d_shape,
                    sinfo_args=relax.TensorStructInfo(cumsum_2d_shape, out_dtype),
                )
                gv = self.builder_.add_func(
                    gpu_2d_continuous_cumsum(in_dtype=in_dtype, out_dtype=out_dtype),
                    "gpu_2d_continuous_cumsum",
                )
                cumsum = relax.call_tir(
                    gv,
                    reshape,
                    out_sinfo=relax.TensorStructInfo(cumsum_2d_shape, out_dtype),
                )
                return relax.call_pure_packed(
                    "vm.builtin.reshape",
                    cumsum,
                    shape,
                    sinfo_args=call.struct_info,
                )

            with tgt:
                if call.op.name == "relax.cumsum":
                    te_func = topi.cuda.cumsum if self.is_gpu_target(tgt) else topi.cumsum
                    if can_use_thrust(tgt, "tvm.contrib.thrust.sum_scan"):
                        kwargs["workspace"] = self.allocate_workspace(call)
                elif call.op.name == "relax.cumprod":
                    te_func = topi.cuda.cumprod if self.is_gpu_target(tgt) else topi.cumprod
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
            self._append_calls_to_update(tir_call, tgt)
            return tir_call
        return super().visit_call_(call)

    def estimate_thrust_workspace_size(self, call: relax.Call) -> int:
        """
        Estimate the workspace size for thrust sort/argsort/topk/cumsum
        """
        input_shape = call.args[0].struct_info.shape
        input_byte_per_elem = DataType(call.args[0].struct_info.dtype).bits // 8
        int64_byte_per_elem = DataType("int64").bits // 8
        int32_byte_per_elem = DataType("int32").bits // 8
        num_elem = reduce(mul, input_shape, 1)
        input_size = num_elem * input_byte_per_elem
        # Most GPU algorithms take O(n) space or less, we choose 8N + 8MB as a safe estimation
        # for algorithm workspace.
        # The current thrust sort implementation may need extra int64 and int32 arrays
        # for temporary data, so we further add this part to the workspace.
        return (
            8 * input_size
            + 8 * 1024 * 1024
            + num_elem * (int64_byte_per_elem + int32_byte_per_elem)
        )

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
        sort_scan_dispater.apply_dlight_gpu_fallback()
        return sort_scan_dispater.builder_.finalize()
