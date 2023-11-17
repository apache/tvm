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


@function_pass(opt_level=0)
class DispatchOps:
    """
    Pass to dispatch operators to platform dependent implementation.
    """

    def __init__(self):
        self.input = wildcard()
        # cumsum on cpu will be legalized
        self.cumsum_gpu = is_op("relax.cumsum")(self.input) & has_target("cuda")
        self.sort_gpu = is_op("relax.sort")(self.input) & has_target("cuda")
        self.sort_cpu = is_op("relax.sort")(self.input) & has_target("llvm")
        self.pattern = self.cumsum_gpu | self.sort_gpu | self.sort_cpu

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
                print("got here 70, expr: ", expr)
                arg = matches[self.input]
                print("got arg: ", arg)

                if self.cumsum_gpu in matches:
                    print("86 matches[self.no_op_reshape]: ", matches[self.cumsum_gpu])
                    return relax.call_dps_packed(
                        "tvm.contrib.thrust.sum_scan",
                        [arg],
                        out_sinfo=arg.struct_info,
                    )
                elif self.sort_gpu in matches:
                    print("86 matches[self.no_op_reshape]: ", matches[self.sort_gpu])
                    return relax.call_dps_packed(
                        "tvm.contrib.thrust.sort",
                        [arg],
                        out_sinfo=arg.struct_info,
                    )
                elif self.sort_cpu in matches:
                    return relax.call_dps_packed(
                        "tvm.contrib.sort.sort",
                        [arg],
                        out_sinfo=arg.struct_info,
                    )

                return expr

            updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func


# Option 0): add a global dict for it: {op, target, condition, dps_packed},
#    condition is some specific setting like the value of k in topk
#    Q: how to work it with pattern match?
#
# Option 1): normal python mod pass, straightforward, but not easy to hack like topk
#
# Option 2): c++ pass, not easy to be updated. Don't go
#
# How to handle with target? don't require RealizeVDevice, just specify the vdevice in inputs
# but vdevice is necessary, we could have default for it
#
# Sample map
# cumsum - cpu => ignore for legalization
# cumsum - gpu =>  relax.call_dps_packed(
#                "tvm.contrib.thrust.sum_scan",
#                [data],
#                out_sinfo=data.struct_info,
#            )
# f32_233 = wildcard().has_shape((2, 3, 3)) & has_dtype("float32") # and pattern
# is_op("nn.conv2d")(xp, yp).has_attr({"kernel_size": [4, 3]}).match(conv2d)
