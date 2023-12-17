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

from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm import relax, topi
from tvm.relax import Expr, Function, Call, PyExprMutator, expr_functor
from tvm.target import Target

from tvm.topi.utils import prod, swap


@expr_functor.mutator
class SortScanDispatcher(PyExprMutator):
    """
    Converts the expensive non linear functions to their fast but approximate counterparts.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """

    def __init__(self, mod):
        super().__init__(mod)

    def _get_target(self, expr: Expr) -> Target:
        sinfo = expr.struct_info
        # Get target information from TensorStructInfo
        if isinstance(sinfo, relax.TensorStructInfo):
            vdevice = sinfo.vdevice
            if vdevice is not None:
                return vdevice.target
        # Return the target in current context
        target = Target.current()
        if target is None:
            raise ValueError(
                "Target not found. Please ensure that the target is annotated within the module, \
                or alternatively, execute this within a specified target context."
            )
        return target

    def visit_call_(self, call: Call) -> Expr:
        if call.op.name == "relax.sort":
            relax.op.sort
            tgt = self._get_target(call)
            axis = call.attrs.axis
            with tgt:
                return self.builder_.call_te(
                    topi.cuda.sort if tgt.kind.name == "cuda" else topi.sort,
                    call.args[0],
                    int(axis) if axis is not None else axis,
                    int(not call.attrs.descending),
                )

        if call.op.name == "relax.cumsum":
            tgt = self._get_target(call)
            with tgt:
                return self.builder_.call_te(
                    topi.cuda.cumsum if tgt.kind.name == "cuda" else topi.cumsum,
                    call.args[0],
                    int(call.attrs.axis),
                    call.attrs.dtype,
                )

        return super().visit_call_(call)


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
        return sort_scan_dispater.builder_.get()
