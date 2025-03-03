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
# pylint: disable=invalid-name, unused-argument, missing-function-docstring, abstract-method
"""Relax Remove Redundant Reshape ops"""
from tvm import IRModule, relax
from tvm.ir import structural_equal
from tvm.ir.transform import PassContext
from tvm.relax import Expr
from tvm.relax.dpl import is_op, rewrite_call, wildcard

from . import function_pass


@function_pass(opt_level=0)
class RemoveRedundantReshape:
    """
    Transformation pass to remove redundant reshape operator
    """

    def __init__(self):
        self.input1 = wildcard()
        shape1 = wildcard()
        pattern_redundant_reshape = is_op("relax.reshape")(self.input1, shape1)
        self.no_op_reshape = pattern_redundant_reshape
        shape2 = wildcard()
        self.repeated_reshape = is_op("relax.reshape")(pattern_redundant_reshape, shape2)
        self.pattern = self.repeated_reshape | self.no_op_reshape

    def transform_function(self, func: Expr, mod: IRModule, ctx: PassContext) -> IRModule:
        """
        Tarnsformation function to remove redundant reshape
        where tensors before and after reshape are of same dimentions.

        Parameters
        --------------
        func: Expr
            The relax function to be optimized

        mod: IRModule
            The IR module

        ctx: PassContext
            Relax pass context
        """

        updated_func = func

        # Skip primitive functions
        if "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
            return updated_func

        def rewriter(expr, matches):
            arg = matches[self.input1]

            if self.repeated_reshape in matches:
                output_shape = matches[self.repeated_reshape].args[1]
                return relax.op.reshape(arg, output_shape)

            elif self.no_op_reshape in matches:
                output_shape = matches[self.no_op_reshape].args[1]
                if arg.struct_info.shape and structural_equal(arg.struct_info.shape, output_shape):
                    return arg
            return expr

        updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func
