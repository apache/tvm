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
"""Dispatch platform dependent operators to related implementation."""

from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm import relax
from tvm.relax import Expr, Function
from tvm.relax.dpl import is_op, rewrite_call, wildcard, has_target

from ..transform import function_pass
from tvm.topi.utils import prod, swap


@function_pass(opt_level=0)
class DispatchOps:
    """
    Pass to dispatch operators to platform dependent implementation.
    """

    def __init__(self):
        self.input = wildcard()
        # cumsum on cpu will be legalized
        self.cumsum_cpu = is_op("relax.cumsum")(self.input) & has_target("llvm")
        self.cumsum_gpu = is_op("relax.cumsum")(self.input) & has_target("cuda")
        self.sort_cpu = is_op("relax.sort")(self.input) & has_target("llvm")
        self.sort_gpu = is_op("relax.sort")(self.input) & has_target("cuda")
        # if no target is specified, default will be on GPU
        self.sort = is_op("relax.sort")(self.input)
        self.cumsum = is_op("relax.cumsum")(self.input)
        self.pattern = (
            self.cumsum_gpu
            | self.cumsum_cpu
            | self.sort_gpu
            | self.sort_cpu
            | self.sort
            | self.cumsum
        )

    def transform_function(self, func: Expr, mod: IRModule, ctx: PassContext) -> IRModule:
        """
        Tranformation function to replace operations with target dependent extern implementation

        Parameters
        ----------
        func: Expr
            The relax function to be optimized

        mod: IRModule
            The ir module

        ctx: PassContext
            Relax pass context
        """

        self.mod = mod
        updated_func = func
        for _, func in mod.functions_items():
            # Skip non-relax functions
            if not isinstance(func, Function):
                continue

            def rewriter(expr, matches):
                arg = matches[self.input]

                if self.cumsum_gpu in matches or (
                    self.cumsum in matches and self.cumsum_cpu not in matches
                ):
                    axis = matches[self.cumsum_gpu].attrs.axis
                    output_dtype = matches[self.cumsum_gpu].attrs.dtype
                    if output_dtype is None:
                        output_dtype = out_sinfo.dtype
                    out_sinfo = arg.struct_info
                    if axis is None:
                        axis = 0
                        new_shape = (prod(arg.struct_info.shape),)
                        arg = relax.op.reshape(arg, new_shape)
                        out_sinfo = relax.TensorStructInfo(
                            new_shape, output_dtype, out_sinfo.vdevice
                        )
                    return relax.op.call_dps_packed(
                        "tvm.contrib.thrust.sum_scan",
                        [arg, int(axis)],
                        out_sinfo=out_sinfo,
                    )

                elif self.sort_cpu in matches:
                    axis = int(matches[self.sort_cpu].attrs.axis)
                    is_ascend = int(matches[self.sort_cpu].attrs.is_ascend)
                    out_sinfo = arg.struct_info
                    if axis is None:
                        axis = 0
                        new_shape = (prod(arg.struct_info.shape),)
                        out_sinfo = relax.TensorStructInfo(
                            new_shape, arg.struct_info.dtype, out_sinfo.vdevice
                        )

                    return relax.call_dps_packed(
                        "tvm.contrib.sort.sort",
                        [arg, axis, is_ascend],
                        out_sinfo=out_sinfo,
                    )

                elif self.sort_gpu in matches or self.sort in matches:
                    axis = matches[self.sort_gpu].attrs.axis
                    if axis is None:
                        axis = -1
                    axis = int(axis)

                    is_ascend = matches[self.sort_gpu].attrs.is_ascend
                    if is_ascend is None:
                        is_ascend = True
                    out_sinfo = arg.struct_info
                    ndim = arg.struct_info.ndim

                    axis = ndim + axis if axis < 0 else axis
                    if axis != ndim - 1:
                        # Prepare for sorting along axis -1.
                        axes = swap(list(range(ndim)), axis)
                        arg = relax.op.permute_dims(arg, axes)
                        new_shape = [out_sinfo.shape[i] for i in axes]
                        out_sinfo = relax.TensorStructInfo(
                            new_shape, out_sinfo.dtype, out_sinfo.vdevice
                        )

                    out = relax.op.call_dps_packed(
                        "tvm.contrib.thrust.sort",
                        [arg, int(is_ascend)],
                        out_sinfo=out_sinfo,
                    )
                    if axis != ndim - 1:
                        # Prepare for sorting along axis -1.
                        axes = swap(list(range(ndim)), axis)
                        out = relax.op.permute_dims(out, axes)
                    return out

                return expr

            updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func
